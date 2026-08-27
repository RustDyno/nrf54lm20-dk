"""4-bit decoder-weight quality gate (TODO.md item 2, model-side).

Simulates the device plan: per-token decoder weights (d{l} q/k/v/out/
xq/xout/fc1a-d/fc2p0-3) and the LM-head embedding stay on the SAME int8
grid the NPU/blobs already use, but every value is constrained to a
4-bit code: groups of G consecutive weights along the input dim share a
scale gs = max|w|/7, stored nibble = round(w/gs) in [-7,7], device
reconstruction w' = clip(round(nibble*gs)) via a per-group 15-entry LUT.
Quantization params, biases, blob command streams are all unchanged --
only weight VALUES move, so this run measures exactly the quality left
after the SD stream is halved.

Patching: the weight tensor bytes are located in each .tflite by exact
byte search (147456-byte strings, unique) and replaced in a copy under
out/submodels-q4g{G}/. Cross K/V submodels (per-utterance, cheap) are
left at int8.

Measures, all with the DEVICE LM head (int8 rows x f32 hidden x row
scale), against the int8 baseline on the enc_out.npz utterance:
  - teacher-forced next-token agreement (decoder 4-bit, emb int8)
  - free-running transcript (decoder 4-bit, emb int8)
  - LM-head argmax flips on baseline hiddens (emb 4-bit only)
  - free-running transcript with BOTH 4-bit (the full plan)

Run: pixi run python quant4_check.py
"""

import os
import shutil

import numpy as np

import common
import decode_model
import tape

GROUPS = [64, 32, 16]
PER_TOKEN_KINDS = (["q", "k", "v", "out", "xq", "xout"]
                   + [f"fc1{p}" for p in "abcd"]
                   + [f"fc2p{j}" for j in range(4)])


def q4_group(w8, g):
    """int8 -> nearest 4-bit-coded int8 via the shared int-exact spec.
    (g other than quant4.G kept for the sweep by transient override.)"""
    import quant4

    saved = quant4.G
    try:
        quant4.G = g
        return quant4.requant(w8)
    finally:
        quant4.G = saved


def find_weight(interp):
    """The one big int8 constant (the 384x384 filter) of a submodel."""
    hits = []
    for d in interp.get_tensor_details():
        if d["dtype"] == np.int8 and int(np.prod(d["shape"])) == 384 * 384:
            try:
                hits.append((d["index"], interp.get_tensor(d["index"])))
            except ValueError:
                pass  # activation tensor without data
    assert len(hits) == 1, f"expected one filter, got {len(hits)}"
    return hits[0][1]


def patch_tflite(src, dst, g):
    """Copy src -> dst with the filter tensor requantized to 4-bit codes.
    Returns (rms error in int8 LSB, max abs error)."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=src)
    interp.allocate_tensors()
    w = find_weight(interp)
    w2 = q4_group(w, g)
    raw = open(src, "rb").read()
    b = w.tobytes()
    assert raw.count(b) == 1, "filter bytes not unique in file"
    open(dst, "wb").write(raw.replace(b, w2.tobytes()))
    d = w2.astype(np.float32) - w.astype(np.float32)
    return float(np.sqrt((d * d).mean())), int(np.abs(d).max())


class SM:
    """tape.Submodel with an explicit path (for the patched copies)."""

    def __init__(self, path):
        import tensorflow as tf

        self.name = os.path.basename(path)
        self.interp = tf.lite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.ind = self.interp.get_input_details()[0]
        self.outd = self.interp.get_output_details()[0]
        self.in_q = self.ind["quantization"]
        self.out_q = self.outd["quantization"]

    run = tape.Submodel.run


def build_mods(g):
    """Patched per-token submodels for group size g; xk/xv stay original.
    Returns (mods, per-kind rms table)."""
    sub = os.path.join(common.OUT, "submodels")
    pdir = os.path.join(common.OUT, f"submodels-q4g{g}")
    os.makedirs(pdir, exist_ok=True)
    mods, rms = {}, {}
    for l in range(common.N_LAYERS):
        names = common.decoder_submodel_names(l)
        for k, name in names.items():
            src = os.path.join(sub, f"{name}.tflite")
            if k in PER_TOKEN_KINDS:
                dst = os.path.join(pdir, f"{name}.tflite")
                if not os.path.exists(dst):
                    r, mx = patch_tflite(src, dst, g)
                    rms[name] = (r, mx)
                mods[(l, k)] = SM(dst)
            else:
                mods[(l, k)] = SM(src)
    return mods, rms


def run_decode(sd, scales, mods, xkv, ref, suppress, lm, force=None,
               max_tokens=decode_model.MAX_TOKENS):
    """One decode with LM-head callable `lm(hid) -> logits`. If `force`
    is given, feed those gold tokens and score predictions against them.
    Returns (predicted tokens, hiddens, agreement list)."""
    dec = decode_model.Decoder(sd, scales, mods, xkv)
    seq = [int(t) for t in ref["sot_sequence"]]
    eot = int(ref["eot"])
    blank = ref["blank"]
    for i, t in enumerate(seq[:-1]):
        dec.step(t, i)
    out, hids, agree = [], [], []
    for step in range(max_tokens):
        hid = dec.step(seq[-1], len(seq) - 1)
        hids.append(hid.copy())
        logits = lm(hid)
        logits[suppress] = -np.inf
        if step == 0:
            logits[blank] = -np.inf
        nxt = int(np.argmax(logits))
        if force is None:
            if nxt == eot:
                break
            out.append(nxt)
            seq.append(nxt)
        else:
            out.append(nxt)
            gold = force[step] if step < len(force) else eot
            agree.append(nxt == gold)
            if gold == eot:
                break
            seq.append(gold)
    return out, hids, agree


def main():
    sd, scales, enc, ref, mods0 = decode_model.load_all()
    suppress, tok = decode_model.build_suppress(ref)
    emb_q, emb_rs = decode_model.quantize_emb(sd)
    embf = emb_q.astype(np.float32)

    def lm8(hid):
        return (embf @ hid) * emb_rs

    print("cross K/V (int8, shared by all runs) ...")
    xkv = decode_model.cross_kv(enc, mods0)

    print("baseline int8 decode (device LM head) ...")
    base, hids, _ = run_decode(sd, scales, mods0, xkv, ref, suppress, lm8)
    print(f"  int8 transcript: {tok.decode(base)!r}")
    print(f"  reference:       {str(ref['text'])!r}")

    for g in GROUPS:
        print(f"\n=== group size {g} "
              f"(scales overhead 1/{g} -> {100 / g:.1f}%) ===")
        mods, rms = build_mods(g)
        if rms:
            worst = sorted(rms.items(), key=lambda kv: -kv[1][0])[:3]
            avg = np.mean([r for r, _ in rms.values()])
            print(f"  weight err (int8 LSB): avg rms {avg:.2f}, worst "
                  + ", ".join(f"{n} {r:.2f}/max{mx}" for n, (r, mx) in worst))

        _, _, agree = run_decode(sd, scales, mods, xkv, ref, suppress, lm8,
                                 force=base)
        print(f"  teacher-forced agreement: {sum(agree)}/{len(agree)}")

        free, _, _ = run_decode(sd, scales, mods, xkv, ref, suppress, lm8)
        print(f"  free transcript (emb int8): {tok.decode(free)!r}"
              f"{'  [== int8]' if free == base else ''}")

        # LM head alone: 4-bit embedding rows on the baseline hiddens.
        emb4 = q4_group(emb_q, g).astype(np.float32)

        def lm4(hid):
            return (emb4 @ hid) * emb_rs

        flips = 0
        for i, (hid, gold) in enumerate(zip(hids, base + [int(ref["eot"])])):
            logits = lm4(hid)
            logits[suppress] = -np.inf
            if i == 0:
                logits[ref["blank"]] = -np.inf
            flips += int(np.argmax(logits)) != gold
        print(f"  emb-4bit argmax flips on int8 hiddens: "
              f"{flips}/{len(hids)}")

        both, _, _ = run_decode(sd, scales, mods, xkv, ref, suppress, lm4)
        print(f"  free transcript (BOTH 4-bit): {tok.decode(both)!r}"
              f"{'  [== int8]' if both == base else ''}")
        del mods

    # keep out/ tidy; the patched dirs are cheap to regenerate
    for g in GROUPS:
        shutil.rmtree(os.path.join(common.OUT, f"submodels-q4g{g}"),
                      ignore_errors=True)


if __name__ == "__main__":
    main()
