"""Emit int8 TFLite submodels for the Axon compiler.

Every NPU piece of the layered pipeline is a single conv/FC over a frame tile:
frames map to the WIDTH axis, channels to the channel axis, so an FC over T
frames is a 1x1 CONV_2D on a [1, 1, T, C] tensor. (Frames on the height axis
fails: the Axon compiler rejects pointwise convolution with width < 4 on
packed input.) Convs use VALID padding; the firmware supplies the k=3 halo
frames explicitly when tiling.

This currently emits one instance of each distinct submodel shape (the Axon
compiler acceptance test). The full per-layer export + manifest lands with the
firmware tape format.

Run whisper_ref.py first (needs out/ref.npz for the calibration clip).
"""

import os
import sys

import numpy as np

import common
from common import submodel_names

FRAME_TILE = 64  # frames per NPU invocation (encoder)


def emit(name, w, b, stride, t_in, acts, out_dir):
    """w [out, in, k] (k=1 for FC), b or None -> int8 tflite at out_dir.

    `acts` is the REAL input activation record [frames, C_in] (float
    pipeline); the converter calibrates input AND output scales from
    contiguous windows of it. Random-uniform representative data
    underestimated real output ranges badly enough to saturate block-3's
    fc2 partials.
    """
    import tensorflow as tf

    cout, cin, k = w.shape
    inp = tf.keras.Input(shape=(1, t_in, cin), batch_size=1)
    conv = tf.keras.layers.Conv2D(cout, (1, k), strides=(1, stride),
                                  padding="valid", use_bias=b is not None)
    m = tf.keras.Model(inp, conv(inp))
    kernel = w.transpose(2, 1, 0)[None, :, :, :]  # [1, k, in, out]
    conv.set_weights([kernel, b] if b is not None else [kernel])

    assert acts.shape[1] == cin and acts.shape[0] >= t_in
    rng = np.random.default_rng(7)

    def rep():
        for s in rng.integers(0, acts.shape[0] - t_in + 1, 16):
            yield [acts[s:s + t_in][None, None].astype(np.float32)]

    cv = tf.lite.TFLiteConverter.from_keras_model(m)
    cv.optimizations = [tf.lite.Optimize.DEFAULT]
    cv.representative_dataset = rep
    cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cv.inference_input_type = tf.int8
    cv.inference_output_type = tf.int8
    data = cv.convert()
    path = os.path.join(out_dir, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(data)

    # Golden vector for the on-device selftest: one random int8 input and the
    # interpreter's output. The Axon engine is bit-exact vs the interpreter
    # (established by the KWS project), so byte equality is the pass bar.
    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    q_in = rng.integers(-128, 128, ind["shape"], dtype=np.int8)
    interp.set_tensor(ind["index"], q_in)
    interp.invoke()
    q_out = interp.get_tensor(outd["index"])
    # Device activations are channel-planar [C][W] (hardware-verified:
    # bit-exact with the transpose, garbage without), vs TFLite's [W][C].
    q_in[0, 0].T.tofile(os.path.join(out_dir, f"{name}.input.bin"))
    q_out[0, 0].T.tofile(os.path.join(out_dir, f"{name}.expect.bin"))

    print(f"  {name}.tflite  {len(data) / 1024:.1f} KB  "
          f"in [1,1,{t_in},{cin}] -> out {cout} (stride {stride}), "
          f"golden {q_in.size}B -> {q_out.size}B")
    return path


def capture_acts(sd):
    """Float-pipeline activations at every submodel input site: an encoder
    pass plus a greedy decode (decoder sites accumulate one row per step;
    exact token-suppression parity is irrelevant for calibration)."""
    import layered

    ref = np.load(os.path.join(common.OUT, "ref.npz"))
    sites = ["enc.mel", "enc.gelu1", "enc.out"]
    for l in range(common.N_LAYERS):
        sites += [f"enc.b{l}.{s}" for s in ("ln1", "ctx", "ln2", "gelu")]
        sites += [f"dec.b{l}.{s}"
                  for s in ("ln1", "ctx", "xln", "xctx", "ln2", "gelu")]
    eng = layered.Engine(sd, "float", record=set(sites))
    enc = layered.encoder(eng, ref["mel_chunk"], common.AUDIO_CTX)
    suppress = np.concatenate([ref["non_speech"],
                               np.arange(int(ref["timestamp_begin"]), 51864)])
    layered.greedy_decode(eng, enc, list(ref["sot_sequence"]),
                          int(ref["eot"]), suppress, ref["blank"])
    return eng.recorded


def main():
    _, sd = common.load_weights()
    out_dir = os.path.join(common.OUT, "submodels")
    os.makedirs(out_dir, exist_ok=True)
    acts = capture_acts(sd)

    t = FRAME_TILE
    only = sys.argv[1:] or None  # optional submodel-name filter
    print("emitting Axon submodels (convs + all encoder blocks):")

    def maybe_emit(name, *args):
        if only is None or name in only:
            emit(name, *args, out_dir)

    # conv1: k=3 s=1, 80 -> 384; halo-padded input tile in MEL frames
    # (2 per encoder frame).
    maybe_emit("wconv1", sd["encoder.conv1.weight"], sd["encoder.conv1.bias"],
               1, t + 2, acts["enc.mel"])
    # conv2: k=3 s=2, 384 -> 384, output-channel tiles of 128 (slot cap)
    for i, part in enumerate("abc"):
        rows = slice(128 * i, 128 * (i + 1))
        maybe_emit(f"wconv2{part}", sd["encoder.conv2.weight"][rows],
                   sd["encoder.conv2.bias"][rows],
                   2, 2 * t + 2, acts["enc.gelu1"])

    for l in range(common.N_LAYERS):
        p = f"encoder.blocks.{l}."
        n = submodel_names(l)
        # attention projections: FC 384 -> 384 over frame tiles
        for key, w, b, site in [
            ("q", "attn.query.weight", "attn.query.bias", f"enc.b{l}.ln1"),
            ("k", "attn.key.weight", None, f"enc.b{l}.ln1"),
            ("v", "attn.value.weight", "attn.value.bias", f"enc.b{l}.ln1"),
            ("out", "attn.out.weight", "attn.out.bias", f"enc.b{l}.ctx"),
        ]:
            maybe_emit(n[key], sd[p + w][:, :, None],
                       sd[p + b] if b else None, 1, t, acts[site])
        # mlp fc1 output-channel tiles (bias travels with its rows)
        for i, part in enumerate("abcd"):
            rows = slice(384 * i, 384 * (i + 1))
            maybe_emit(n[f"fc1{part}"],
                       sd[p + "mlp.0.weight"][rows][:, :, None],
                       sd[p + "mlp.0.bias"][rows],
                       1, t, acts[f"enc.b{l}.ln2"])
        # mlp fc2 input-dim partials (partial 0 carries the bias)
        for i in range(4):
            cols = slice(384 * i, 384 * (i + 1))
            maybe_emit(n[f"fc2p{i}"],
                       sd[p + "mlp.2.weight"][:, cols][:, :, None],
                       sd[p + "mlp.2.bias"] if i == 0 else None,
                       1, t, acts[f"enc.b{l}.gelu"][:, cols])

    # Decoder submodels. Token-rate pieces run at width 4 (pointwise conv
    # rejects width < 4; only column 0 is meaningful); cross K/V run over
    # encoder frame tiles like the encoder pieces.
    for l in range(common.N_LAYERS):
        p = f"decoder.blocks.{l}."
        n = common.decoder_submodel_names(l)
        for key, w, b, site, t_in in [
            ("q", "attn.query.weight", "attn.query.bias",
             f"dec.b{l}.ln1", 4),
            ("k", "attn.key.weight", None, f"dec.b{l}.ln1", 4),
            ("v", "attn.value.weight", "attn.value.bias",
             f"dec.b{l}.ln1", 4),
            ("out", "attn.out.weight", "attn.out.bias",
             f"dec.b{l}.ctx", 4),
            ("xq", "cross_attn.query.weight", "cross_attn.query.bias",
             f"dec.b{l}.xln", 4),
            ("xout", "cross_attn.out.weight", "cross_attn.out.bias",
             f"dec.b{l}.xctx", 4),
            ("xk", "cross_attn.key.weight", None, "enc.out", t),
            ("xv", "cross_attn.value.weight", "cross_attn.value.bias",
             "enc.out", t),
        ]:
            maybe_emit(n[key], sd[p + w][:, :, None],
                       sd[p + b] if b else None, 1, t_in, acts[site])
        for i, part in enumerate("abcd"):
            rows = slice(384 * i, 384 * (i + 1))
            maybe_emit(n[f"fc1{part}"],
                       sd[p + "mlp.0.weight"][rows][:, :, None],
                       sd[p + "mlp.0.bias"][rows],
                       1, 4, acts[f"dec.b{l}.ln2"])
        for i in range(4):
            cols = slice(384 * i, 384 * (i + 1))
            maybe_emit(n[f"fc2p{i}"],
                       sd[p + "mlp.2.weight"][:, cols][:, :, None],
                       sd[p + "mlp.2.bias"] if i == 0 else None,
                       1, 4, acts[f"dec.b{l}.gelu"][:, cols])
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
