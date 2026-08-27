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

## 3. Later: 4-bit encoder weights

The loader is generic (any "LAY4" entry expands in the slot), so this
is make_sd_image-side only -- but it needs its own quality gate first
(full-pipeline simulate.py transcript with patched encoder tflites),
since quant4_check.py only gated the decoder. Encoder reads ~27 MB per
utterance; packing would cut ~8 MB of that.
