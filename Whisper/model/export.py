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

Run simulate.py first (needs out/scales.json for representative ranges).
"""

import json
import os

import numpy as np

import common

FRAME_TILE = 64  # frames per NPU invocation (encoder)


def in_range(scales, site):
    s = scales[site]
    lvl = 2 ** s["bits"]
    return ((-lvl // 2 - s["zp"]) * s["scale"],
            (lvl // 2 - 1 - s["zp"]) * s["scale"])


def emit(name, w, b, stride, t_in, lo, hi, out_dir):
    """w [out, in, k] (k=1 for FC), b or None -> int8 tflite at out_dir."""
    import tensorflow as tf

    cout, cin, k = w.shape
    inp = tf.keras.Input(shape=(1, t_in, cin), batch_size=1)
    conv = tf.keras.layers.Conv2D(cout, (1, k), strides=(1, stride),
                                  padding="valid", use_bias=b is not None)
    m = tf.keras.Model(inp, conv(inp))
    kernel = w.transpose(2, 1, 0)[None, :, :, :]  # [1, k, in, out]
    conv.set_weights([kernel, b] if b is not None else [kernel])

    rng = np.random.default_rng(7)

    def rep():
        for _ in range(8):
            yield [rng.uniform(lo, hi, (1, 1, t_in, cin)).astype(np.float32)]

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


def main():
    _, sd = common.load_weights()
    with open(os.path.join(common.OUT, "scales.json")) as f:
        scales = json.load(f)
    out_dir = os.path.join(common.OUT, "submodels")
    os.makedirs(out_dir, exist_ok=True)

    t = FRAME_TILE
    print("emitting Axon submodel shape set:")
    # conv1: k=3 s=1, 80 -> 384; halo-padded input tile in MEL frames
    # (2 per encoder frame).
    emit("wconv1", sd["encoder.conv1.weight"], sd["encoder.conv1.bias"],
         1, t + 2, *in_range(scales, "enc.mel"), out_dir)
    # conv2: k=3 s=2, 384 -> 384, output-channel tile of 128 (weight slot cap)
    emit("wconv2a", sd["encoder.conv2.weight"][:128],
         sd["encoder.conv2.bias"][:128],
         2, 2 * t + 2, *in_range(scales, "enc.gelu1"), out_dir)
    # attention projection: FC 384 -> 384 (q/k/v/out and cross flavors)
    emit("wq0", sd["encoder.blocks.0.attn.query.weight"][:, :, None],
         sd["encoder.blocks.0.attn.query.bias"],
         1, t, *in_range(scales, "enc.b0.ln1"), out_dir)
    # mlp fc1 output-channel tile: FC 384 -> 384 (of 1536)
    emit("wfc1a", sd["encoder.blocks.0.mlp.0.weight"][:384][:, :, None],
         sd["encoder.blocks.0.mlp.0.bias"][:384],
         1, t, *in_range(scales, "enc.b0.ln2"), out_dir)
    # mlp fc2 input-dim partial 0: FC 384 -> 384 (of 1536 in), carries bias
    emit("wfc2p0", sd["encoder.blocks.0.mlp.2.weight"][:, :384][:, :, None],
         sd["encoder.blocks.0.mlp.2.bias"],
         1, t, *in_range(scales, "enc.b0.gelu"), out_dir)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
