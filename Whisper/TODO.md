# TODO

## 0. ACCURACY: fix the microphone level (was "inaccurate on device")

Localized 2026-09-03 with the mock rig (TODO_complete). The pipeline is
fine -- fed the reference clip the device reproduces the reference
transcript and a mel matching the host mirror to 2 int8 steps. The
microphone is ~63 dB too quiet: PCM peak 58/32768 (-55 dBFS) where the
clip is 25647 (-2.1 dBFS), which pins the mel at the int8 floor
([-128, -91], 24 of 256 codes).

The level matters because whisper's mel normalization is only partly
relative: the floor is `max - 8` but the offset is an absolute
`(log_spec + 4) / 4`, so a quiet input shifts the whole normalized mel
down instead of being absorbed.

pdm.rs runs GAINL/GAINR at 0x28 = 0 dB; the register tops out at 0x50 =
+20 dB, which is only 10x and will not close a 1400x gap on its own. So
decide between:
  a. PDM gain to +20 dB AND a fixed software gain on the PCM, calibrated
     against the clip's level; or
  b. normalize per utterance (scale the PCM so its peak matches the
     calibration clip's) -- robust to speaking distance, but changes the
     mel the encoder was calibrated on, so re-gate with simulate.py; or
  c. check first whether the PDM decimation output is simply being
     shifted down too far -- 58 counts of ambient noise is low even for
     0 dB, so measure a known-loud source before adding gain.

Measure with: mockusb serve --no-audio, then
`pixi run python mock_compare.py out/mock-work.img` (it writes the
recorded audio out as a wav). Do (c) before (a) or (b).


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

## RESOLVED 2026-09-01: constant-token decode = in-place 4-bit
## expansion (unpack_slot)

The since-the-4-bit-change freeze (one token every step -- " Pitt"/"uss"
-- regardless of audio) is the IN-PLACE per-token weight expansion
`unpack_slot`. Proven: a raw-int8 decoder image decodes correctly on
device (" Yes.", output8.log) with a trace bit-exact to the host golden,
while the 4-bit card freezes. The width-4 NPU conv, the packed weights
(card d0q == quant4.requant, maxdiff 0), and the q4 spec are all fine;
the 4-bit LM-head embp4 (DISJOINT q4::unpack) also works. Only the
per-token decoder blobs' in-place expansion feeds the NPU slightly-wrong
weights (broad ~2 LSB, sum-preserving so raw_sum can't catch), which on
top of a marginal G=64 quantization tips the knife-edge decode into a
constant token. Full localization trail (runs 1-8) in NOTES "SOLVED
LOCALIZATION" / "RESOLVED (run 8)".

DONE -- FIX A (guaranteed): decoder ships RAW int8. make_sd_image now
skips per-token packing by default (Q4_PACK=1 re-enables it once the
in-place path is fixed); out/sd.img is the int8 image (packed backup
out/sd-q4.img + sd-index-q4.json); firmware debug instrumentation
stripped. The card flashed for output8 already runs this. Cost: ~4 MB
more decode stream/token (USB host mode already made storage a
non-bottleneck).

FOLLOW-UPS to recover the 4-bit SD win / robustness (pick per priority):
- B. Disjoint decoder expansion like embp4 (expand LAY4 into a spare
  buffer, not in place) -- keeps 4-bit but needs ~152 KB scratch the
  in-place trick avoided. Verify with the dtr/dtr8 debug (git-revert of
  this cleanup): device dtr8 q must become -24 -13 -14 -8, self L0
  10360 14570 5078 11024.
- C. Cheap probe first: add a barrier / D-cache clean of the slot after
  unpack_slot before slot::run, re-enable Q4_PACK, retest. Only helps if
  it is CPU-write / NPU-DMA ordering (no D-cache is enabled today, so
  low odds, but a 1-line experiment). If B or C works, it also confirms
  the mechanism.
- D. Re-gate quant4 quality on MULTIPLE clips and move to G=32 -- the
  JFK-only quant4_check gate missed that G=64 degenerates on other audio
  (host sim: int8 & G=32 clean, G=64 "Cool cool"). Orthogonal to B/C but
  needed before trusting any 4-bit decoder.

The separate "device mel is ~6 log10 units low, int8 floor-pinned" note
that used to sit here is now item 0 above: measured with the mock rig and
localized to the microphone level, not R_MAX.

## 2. Bench-verify 4-bit decode (mel FFT+streaming already verified)

2026-08-27 bench: streaming mel VERIFIED (no overruns, sd[mel] 142 ms
rd / 321 ms wr, mel fully hidden behind capture). The 4-bit run then
hardfaulted at the first decode token -- root cause was NOT the 4-bit
data: attn_head's 6.4 KB stack frame at decode depth dipped below
_stack_end and corrupted the Axon driver state at the top of .bss
(gl_axon_instances -> wild register write BFAR=0x7f0005e0). Fixed:
attention scratch + unpack staging now live in the idle interlayer,
SUMS4 static replaced by an in-header raw_sum (-1 KB .bss), and
MSPLIM is armed so any future overflow is an immediate STKOF fault.

Next session: REFLASH + RE-DD TOGETHER (the LAY4 header grew to five
words -- the previously dd'd card errors -906 against the new
firmware). Expect sd[decode] rd to roughly halve (~104 -> ~55-60 MB)
and ~4.8 -> ~3 s/token; first use of each blob verifies the expansion
(-907 on drift). After the dd, decode-only needs one full run first to
repopulate the S_XKV scratch.

Rollback levers: TRY_STREAM_MEL=false (mel); the pre-q4 card image +
this firmware (raw fallbacks work) for the 4-bit path.

## 3. DONE 2026-09-01: USB host mode verified end to end (see
TODO_complete.md; findings in NOTES.md). Storage is no longer the
bottleneck -- the M33 CPU share (attention, 4-bit unpack, LM-head
dots) now dominates encoder and decode wall time, which reorders
this queue: CPU kernels beat further storage work.

## 4. Later: 4-bit encoder weights

The loader is generic (any "LAY4" entry expands in the slot), so this
is make_sd_image-side only -- but it needs its own quality gate first
(full-pipeline simulate.py transcript with patched encoder tflites),
since quant4_check.py only gated the decoder. Encoder reads ~27 MB per
utterance; packing would cut ~8 MB of that.
