# TODO

## 0. ACCURACY: the microphone needs ~31 dB of gain (PROVEN END TO END)

Localized AND proven 2026-09-03 with the mock rig. The mic, the mel code,
the encoder, the quantization and the NPU path are ALL fine; the input is
simply ~31 dB too quiet.

Proof: recorded a live mic window (`--features mic-check`), multiplied the
samples by 36 on the host, injected the result back through the SAME
firmware and image -- the device transcribed it correctly
(" 1 2 3 4 5 1 2 3 4 5 ...", the operator counting). The mel went from
floor-pinned int8 [-128,-91] to a healthy [-119,127] against the
reference clip's [-128,127]. Nothing in the firmware changed.

Measured levels (100 ms frames, speech-active frames only):

    source            peak    active rms   noise floor
    mic (as recorded)  5568         156          3.4
    jfk reference     25647        5615        254.5

So peak is only 13 dB down but SUSTAINED level is 31 dB down -- the mic
capture has a much higher crest factor (one transient at 5568 while
speech sits near 156). The spectrum of the loudest second is a ~128 Hz
fundamental with harmonics at 258/516 Hz and 59% of energy in
300-3400 Hz: clean speech, just quiet. Audio sounds undistorted.

Why the level matters at all, given whisper normalizes: the floor is
relative (`max - 8`) but the offset is ABSOLUTE (`(log_spec + 4) / 4`),
so a quiet input shifts the whole normalized mel down instead of being
absorbed.

Fixing it -- 36x is 31 dB and pdm.rs GAINL/GAINR only reach +20 dB
(0x28 = 0 dB now, 0x50 = +20 dB max), so the register alone cannot do it:
  a. PDM gain to +20 dB plus ~3.6x in software. Simple, but a fixed
     software gain clips loud speakers: 36x on this clip already clips
     0.23% of samples on one transient.
  b. Per-utterance normalization (scale so a high percentile -- NOT the
     raw peak, which was 35x the speech rms here -- hits ~0.7 FS). This
     is closest to what whisper actually expects, since its reference
     pipeline is fed float audio peaking near full scale, and it is
     robust to speaking distance. Re-gate with simulate.py because it
     changes the mel the encoder was calibrated on.
  c. Still worth one measurement first: 58 counts of ambient and 156 of
     speech is low even for 0 dB gain, so check whether the PDM
     decimation output is being shifted down further than intended
     before compensating for it downstream.

Reproduce: `mockusb serve --no-audio` + `cargo run --release --features
mic-check` records 12 s windows, prints peak/dBFS/rms each, and stops at
the first one above the noise floor with the audio left in S_PCM. Then
`pixi run python mock_compare.py out/mock-work.img` writes it out as a
wav.

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
