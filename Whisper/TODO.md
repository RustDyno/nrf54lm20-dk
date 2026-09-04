# TODO

## 0. DONE 2026-09-03: microphone level fixed on the board

Shipped, verified on hardware (see TODO_complete). Two levers:

- pdm.rs GAIN 0x28 (0 dB) -> 0x28+24 (+12 dB), applied inside the
  peripheral ahead of the 16-bit output so it keeps detail a later
  software scale cannot recover. Measured: no clipping, 0 overruns.
- app.rs mel_lift(): the remainder taken out per utterance in the
  log-mel domain, where it cannot clip. whisper's normalization clamps
  relative to the utterance peak but applies an ABSOLUTE +4.0 offset, so
  it does not normalize level; mel_lift adds the shift that puts this
  utterance's peak where the calibration clip's was (MEL_TARGET_MAX
  1.845, from ref.npz mel_chunk max 1.46126 * 4 - 4). A gain g on the
  samples shifts log10 power by 2*log10(g) uniformly, so this is exactly
  equivalent to recording louder -- but pass 1 already computed the
  peak, so it is free. Clamped to [-2.0, +4.0] log10 so silence is not
  lifted to speech level and a shout is brought down.

Still open, minor: the reference clip and a live mic window now both
land at the same mel peak, but the mic's noise floor is much lower
relative to speech (3.4 vs 254 at 0 dB gain), so a lifted quiet
utterance presents a cleaner-than-calibration background. Worth a
simulate.py pass over several recorded clips to confirm the encoder is
happy with that, and to re-check MEL_TARGET_MAX against more than one
reference. MEL_AUTOLEVEL = false restores stock whisper behaviour for
comparison.

Speed queue (2026-08-27). Baseline: the 2026-08-26 verified run, ~2 min
speak-to-done, sd[...] stats in NOTES.md.

## 5. DONE 2026-09-04: speed pass 4, CPU kernels on the DSP extension

Profile source: firmware/output9.log on the USB stick, written up in
speedup.md section 9. A ctx-600 utterance spends 85 s of its 123 s
encoder in the scalar attention kernel (~8 cycles/MAC) and 1.5 s of
every 2.9 s decode step in the LM head. Storage is ~20-30 percent.

Accuracy contract for each item (asked and answered before starting):
bit-exact = same integer/float operations in a different order or
instruction; gated = a numeric change with a host gate that must show
zero argmax flips / transcript parity; risk = a real behavioral change.

- [x] 5.1 Attention on SMLAD (bit-exact). K transposed to key-major
      int16 [key][64], V expanded to int16 [64][640], both in the weight
      SLOT (idle during CPU attention; the next blob reloads in 18 ms),
      2 queries x 2 keys register blocking, one 40 KB read per head for
      Q/K/V and one write for the context tiles. Same i32 sums as
      today. Target 21 s -> ~5 s per block.
- [x] 5.2 Softmax tail. VCVTA for round-half-away (bit-exact), the
      /(1/256) as an exact *256 (bit-exact), expf via exp2 polynomial
      (gated: attncheck reports every ctx byte that moves; transcript
      parity on the rig). Switchable per call site.
- [x] 5.3 LM head. int16 hidden vector (per-utterance scale) x int8
      rows on SMLAD (gated: lm16_check.py must show 0 argmax flips on
      the golden decode), row scales and ids stored inside each packed
      chunk so a chunk is ONE read (layout only), unpack straight to
      int16 (bit-exact). New image entry "embc4"; embp4 dropped.
- [x] 5.4 Blob byte-sum with USADA8 (bit-exact; the check stays).
- [x] 5.5 Decode cross-attention: read a head's 10 tile blocks in one
      command into the slot, SMLAD kernel (bit-exact). Replaces 480
      4 KB reads per token with 48.
- [x] 5.6 Stop recording at silence (risk: a pause longer than the
      timeout ends the utterance). Online VAD on the streaming mel
      chunks; stop after 3 silent 0.64 s chunks past the last speech
      chunk and never before 7 chunks, which keeps every frame the
      encoder touches (ctx floor, margin, tile round-up, conv halo)
      real recorded audio -- the encoder input is then identical to
      the 12 s capture's for the same VAD endpoint.
- [x] 5.7 Host gates: attncheck (new kernel vs golden, both exp paths),
      exp ulp sweep, unpack16 vs unpack8, USADA8 sum vs byte loop,
      lm16_check.py.
- [x] 5.8 Rig run (mock USB, JFK clip): transcript must match the
      reference; encoder output compared block-for-block with the
      previous work image; new cpu[phase] lines for the timings.

Result (mock rig, JFK clip): encoder 100 -> 38 s, decode 2.4 -> 1.08 s
per step, transcript identical, 0 of 5480 scratch blocks differ from
the previous firmware (mock_diff.py). Summary in TODO_complete.md,
design notes in NOTES.md, numbers in speedup.md section 10. The silence
stop (5.6) is built and reasoned about but not yet exercised with a
live microphone.

Left in the queue, not part of this pass:
- Fast USB stick (zero code, needs hardware): reads sit at a flat
  9-10 MB/s at every transfer size, so the limit is bandwidth.
- Overlap DMA with compute (medium; only pays once 5.1-5.5 are in).
- 4-bit decoder blobs (item 1 follow-ups B/D; accuracy regate first).
- Speculative / batched decode positions (large; exact by construction
  but the draft's acceptance rate is unmeasured).

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

# Model

Swap to Moonshine Tiny, Vosk Small Models, Next-Gen Kaldi / Sherpa-ONNX Zipformer, NVIDIA NeMo FastConformer-CTC Tiny?