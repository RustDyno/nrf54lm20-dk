"""Feasibility ladder for the layered pipeline, run entirely on the host.

Stage 1  float, full 1500-frame context: layered decomposition vs
         openai-whisper (encoder parity + identical transcript).
Stage 2  float, truncated AUDIO_CTX context (the device configuration).
Stage 3  calibration pass (min/max at every quantization site).
Stage 4  int8 simulation with TFLite/Axon semantics: the device-accurate run.

Writes out/scales.json (the exporter/firmware quantization contract).
"""

import json
import os
import time

import numpy as np

import common
import layered


def build_suppress(tok):
    """Mirror openai-whisper's default suppress list (suppress_tokens=-1),
    plus all timestamp tokens (the device pipeline never emits them)."""
    s = set(tok.non_speech_tokens)
    s.update([tok.transcribe, tok.translate, tok.sot, tok.sot_prev, tok.sot_lm])
    if tok.no_speech is not None:
        s.add(tok.no_speech)
    s.update(range(tok.timestamp_begin, tok.timestamp_begin + 1501))
    return np.array(sorted(t for t in s if t < 51864), dtype=np.int64)


def main():
    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    model, sd = common.load_weights()
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        language="en", task="transcribe")
    ref = np.load(os.path.join(common.OUT, "ref.npz"))
    sot = list(ref["sot_sequence"])
    eot = int(ref["eot"])
    suppress = build_suppress(tok)
    blank = ref["blank"]
    ref_tokens = list(ref["tokens"])
    print(f"reference: {ref['text']}")

    def run(eng, mel, ctx, gaps=None):
        enc = layered.encoder(eng, mel, ctx)
        toks = layered.greedy_decode(eng, enc, sot, eot, suppress, blank,
                                     log_gaps=gaps)
        return enc, toks

    # --- stage 1: float, full context, vs openai-whisper -------------------
    t0 = time.time()
    eng = layered.Engine(sd, "float")
    enc, toks = run(eng, ref["mel_full"], common.FULL_AUDIO_CTX)
    with torch.no_grad():
        enc_ref = model.encoder(torch.from_numpy(ref["mel_full"])[None]).numpy()[0]
    err = np.abs(enc.x - enc_ref).max()
    match = "MATCH" if toks == ref_tokens else "MISMATCH"
    print(f"\n[stage 1] float / full ctx ({time.time() - t0:.0f} s)")
    print(f"  encoder max |diff| vs torch: {err:.2e}")
    print(f"  transcript ({match}): {tok.decode(toks)!r}")

    # --- stage 2: float, truncated context ---------------------------------
    t0 = time.time()
    eng = layered.Engine(sd, "float")
    _, toks_f = run(eng, ref["mel_chunk"], common.AUDIO_CTX)
    print(f"\n[stage 2] float / audio_ctx={common.AUDIO_CTX} "
          f"({time.time() - t0:.0f} s)")
    print(f"  transcript: {tok.decode(toks_f)!r}")

    # --- stage 3: calibration ----------------------------------------------
    t0 = time.time()
    cal = layered.Engine(sd, "calib")
    run(cal, ref["mel_chunk"], common.AUDIO_CTX)
    scales = cal.finish_calib()
    with open(os.path.join(common.OUT, "scales.json"), "w") as f:
        json.dump({k: {"scale": s, "zp": z, "bits": b}
                   for k, (s, z, b) in scales.items()},
                  f, indent=1, sort_keys=True)
    print(f"\n[stage 3] calibrated {len(scales)} sites "
          f"({time.time() - t0:.0f} s) -> out/scales.json")

    # --- stage 4: int8 simulation ------------------------------------------
    t0 = time.time()
    q = layered.Engine(sd, "int8", scales)
    gaps = []
    enc_q, toks_q = run(q, ref["mel_chunk"], common.AUDIO_CTX, gaps=gaps)
    eng2 = layered.Engine(sd, "float")
    enc_f = layered.encoder(eng2, ref["mel_chunk"], common.AUDIO_CTX)
    rms = float(np.sqrt(np.mean((enc_q.x - enc_f.x) ** 2)))
    ref_rms = float(np.sqrt(np.mean(enc_f.x ** 2)))
    match = "MATCH" if toks_q == toks_f else "MISMATCH vs stage 2"
    print(f"\n[stage 4] int8 / audio_ctx={common.AUDIO_CTX} "
          f"({time.time() - t0:.0f} s)")
    print(f"  encoder-out rms err {rms:.4f} (signal rms {ref_rms:.4f}, "
          f"snr {20 * np.log10(ref_rms / max(rms, 1e-9)):.1f} dB)")
    print(f"  argmax top1-top2 logit gap: min {min(gaps):.2f} "
          f"median {np.median(gaps):.2f}")
    print(f"  transcript ({match}): {tok.decode(toks_q)!r}")


if __name__ == "__main__":
    main()
