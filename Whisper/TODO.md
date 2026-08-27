# TODO

Speed queue after the mel FFT + streaming re-enable (2026-08-27, awaiting
a bench run). Baseline: the 2026-08-26 verified run, ~2 min speak-to-done,
sd[...] stats in NOTES.md.

## 1. Application-class SD card (zero code)

Reads run at 3.3 MB/s everywhere, but writes crawl at 103-278 KB/s
(card page-program latency; the read/write interleave defeats CMD25
coalescing). Roughly 22 s of every utterance is SD writes (encoder
scratch 17 s + cross K/V 5 s).

Test: dd the same sd.img to an A1/A2-rated card and rerun. The two `wr`
numbers in the `sd[encoder]` / `sd[cross]` lines are the only ones to
watch; expect most of the 22 s back if the card is the limit.

## 2. 4-bit decoder weights (model-side)

Decode reads 104.6 MB per utterance (all decoder weights re-streamed
every token, plus 4.7 MB LM head per sampled token) at 3.3 MB/s =
SD-bound at ~4.8 s/token. Packing the decoder weights and LM-head
embedding to int4 halves the stream; the device unpacks int4->int8 into
the slot before the NPU run (CPU cost trivial next to the read).

- Validate quality first in the Python mirror (per-group scales;
  decode_model.py transcript parity + teacher-forcing accuracy).
- Needs new blobs + card image (make-blob.sh, make_sd_image.py). fwid
  is pinned, so firmware and card no longer regenerate in lockstep.
