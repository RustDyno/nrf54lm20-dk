# TODO

Speed queue (2026-08-27). Baseline: the 2026-08-26 verified run, ~2 min
speak-to-done, sd[...] stats in NOTES.md.

## 1. Application-class SD card (zero code)

Reads run at 3.3 MB/s everywhere, but writes crawl at 103-278 KB/s
(card page-program latency; the read/write interleave defeats CMD25
coalescing). Roughly 22 s of every utterance is SD writes (encoder
scratch 17 s + cross K/V 5 s).

Test: dd the same sd.img to an A1/A2-rated card and rerun. The two `wr`
numbers in the `sd[encoder]` / `sd[cross]` lines are the only ones to
watch; expect most of the 22 s back if the card is the limit.

## 2. Bench-verify the two implemented-blind passes

Both are committed, host-verified, and awaiting one hardware session:

- Mel FFT + streaming mel (reflash only, any card): expect no
  "overruns streaming mel" warning and the record-end -> "encoder..."
  gap collapsing from ~22 s to ~1 s.
- 4-bit decoder weights + embedding (needs the NEW sd.img dd'd AND the
  new firmware together): expect sd[decode] rd to roughly halve
  (~104 -> ~55-60 MB) and ~4.8 -> ~3 s/token. First boot verifies every
  expanded blob against sums4 (hard error -907 on drift, so a bad pack
  cannot silently garble transcripts). After a re-dd the decode-only
  feature needs one full run first to repopulate the S_XKV scratch.

Rollback levers if something misbehaves: TRY_STREAM_MEL=false (mel),
old card image or deleting the embp4/LAY4 entries (4-bit; the firmware
falls back to raw paths automatically with the old card).

## 3. Later: 4-bit encoder weights

The loader is generic (any "LAY4" entry expands in the slot), so this
is make_sd_image-side only -- but it needs its own quality gate first
(full-pipeline simulate.py transcript with patched encoder tflites),
since quant4_check.py only gated the decoder. Encoder reads ~27 MB per
utterance; packing would cut ~8 MB of that.
