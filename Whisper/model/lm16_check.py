"""LM-head int16-hidden gate.

The device LM head dots the 4-bit-coded int8 embedding rows with the
hidden vector quantized to a per-token int16 grid (hs = max|hid| / 32767,
hq = round(hid / hs)) and forms logit = i32_dot * (row_scale * hs) --
exactly what firmware app.rs lm_head computes. This compares that against
the previous device head (same rows, f32 hidden) on the golden decode:
argmax flips must be 0 and the free-running transcript must not change.

Run: pixi run python lm16_check.py [--int8]

--int8 gates the "embc8" head instead: the int8 rows themselves (no
4-bit coding) dotted with the int16 hidden vector, compared against the
same rows with the f32 hidden vector (the pre-4-bit device head). The
transcript is also compared with the 4-bit head's.
"""

import sys

import numpy as np

import decode_model as dm
import quant4_check as q4c

VOCAB_KEEP = 12288  # make_sd_image.VOCAB_KEEP: rows >= this never reach the card


def main():
    sd, scales, enc, ref, mods = dm.load_all()
    suppress, tok = dm.build_suppress(ref)
    emb_q, emb_rs = dm.quantize_emb(sd)
    int8 = "--int8" in sys.argv
    emb4 = q4c.q4_group(emb_q, 64)  # the rows as the card carries them
    emb4f = emb4.astype(np.float32)
    emb4i = emb4.astype(np.int32)
    rows_f = emb_q.astype(np.float32) if int8 else emb4f
    rows_i = emb_q.astype(np.int32) if int8 else emb4i
    what = "int8 rows" if int8 else "4-bit rows"
    eot = int(ref["eot"])

    def mask(logits, first):
        logits[suppress] = -np.inf
        keep_eot = logits[eot]
        logits[VOCAB_KEEP:] = -np.inf
        logits[eot] = keep_eot  # the card keeps EOT despite its high id
        if first:
            logits[ref["blank"]] = -np.inf
        return logits

    def lm_f32(hid):
        return (rows_f @ hid) * emb_rs

    def lm_f32_q4(hid):
        return (emb4f @ hid) * emb_rs

    def lm_i16(hid):
        hmax = float(np.abs(hid).max())
        hs = np.float32(hmax / 32767.0) if hmax > 0 else np.float32(1.0)
        hq = np.clip(np.round(hid / hs), -32767, 32767).astype(np.int32)
        acc = rows_i @ hq
        return acc.astype(np.float32) * (emb_rs * hs).astype(np.float32)

    print("cross K/V ...")
    xkv = dm.cross_kv(enc, mods)
    print(f"golden decode ({what}, f32 hidden) ...")
    base, hids, _ = q4c.run_decode(sd, scales, mods, xkv, ref, suppress, lm_f32)
    print(f"  transcript: {tok.decode(base)!r}")
    if int8:
        print("4-bit head for comparison ...")
        base4, _, _ = q4c.run_decode(sd, scales, mods, xkv, ref, suppress, lm_f32_q4)
        print(f"  transcript: {tok.decode(base4)!r}  "
              f"[{'== int8 rows' if base4 == base else 'DIFFERS from int8 rows'}]")

    flips, min_gap, max_dev = 0, np.inf, 0.0
    for i, (hid, gold) in enumerate(zip(hids, base + [eot])):
        l32 = mask(lm_f32(hid), i == 0)
        l16 = mask(lm_i16(hid), i == 0)
        a32, a16 = int(np.argmax(l32)), int(np.argmax(l16))
        flips += a16 != gold
        top2 = np.sort(l16[np.isfinite(l16)])[-2:]
        min_gap = min(min_gap, float(top2[1] - top2[0]))
        fin = np.isfinite(l32)
        max_dev = max(max_dev, float(np.abs(l16[fin] - l32[fin]).max()))
        if a16 != a32:
            print(f"  step {i}: f32 argmax {a32} vs int16 {a16}")
    print(f"int16 hidden, {what}: {flips}/{len(hids)} argmax flips, "
          f"max |logit dev| {max_dev:.4f}, min top-2 gap {min_gap:.3f}")

    print("free-running decode with the int16 head ...")
    free, _, _ = q4c.run_decode(sd, scales, mods, xkv, ref, suppress, lm_i16)
    same = free == base
    print(f"  transcript: {tok.decode(free)!r}  [{'== f32 head' if same else 'DIFFERS'}]")
    if flips or not same:
        raise SystemExit("lm16_check: FAILED")
    print("lm16_check: OK")


if __name__ == "__main__":
    main()
