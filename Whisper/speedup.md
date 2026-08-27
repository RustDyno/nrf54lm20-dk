# Whisper speed analysis: where the time goes and how to cut it

Status: 3.1 (SPIM00 at 32 MHz), 3.4 (VAD endpointing), the attention
kernel rewrite, and the mel FFT (section 7) are IMPLEMENTED; everything
up to the attention rewrite is hardware-verified (2026-08-26 run, ~2 min
speak-to-done from ~6.5). 3.5a (mel during recording) is re-enabled on
the strength of the FFT and awaits a bench run. 3.2a (LM-head norm-bound
early exit) was implemented, MEASURED on the mirror, and rejected:
Whisper LM-head cosines are so small that even the loosest row bound
sits ~3x above the best logit (best 21.5 vs minimum bound 56.5, 0 of
12228 rows prunable) -- see NOTES.md. The LM-head lever is therefore
amortization across positions (3.3), not screening. Remaining queue:
TODO.md (A1/A2 card, 4-bit decoder weights).

Baseline (commit 5f2ae02): host-driven decode runs at ~2.3 min/token over
SWD; the standalone SD build is estimated at ~10 min per 12 s utterance.
Both modes are almost entirely **bus-bound**: the NPU works for milliseconds
per token and the M33 already runs at its 128 MHz maximum. Every meaningful
speedup is either (a) a faster byte pipe or (b) moving fewer bytes per token.

## 1. Where the time goes

### Standalone (SD on SPIM22 at 8 MHz, ~0.9 MB/s reads, slower writes)

Derived from app.rs loop structure and the image layout:

| Phase | SD traffic | Est. time |
|---|---|---|
| record (fixed) | 0.75 MB write | 12 s wall |
| mel (2 passes) | ~2.5 MB r/w | ~5 s |
| encoder (conv + 4 blocks) | ~52 MB (60/40 r/w) | ~90-110 s |
| cross K/V | ~5.3 MB | ~8 s |
| decode, per token | ~16 MB | ~18-20 s |

Per-token breakdown (the target that matters most):

| Component | Bytes/token |
|---|---|
| 56 decoder weight blobs (14 per block x 4) | 9.3 MB |
| LM head: pruned embedding scan (12230 x 384) | 4.7 MB |
| cross-K/V head reassembly (re-read every token) | 2.0 MB |
| LN gamma/beta, LUTs, embedding rows | ~0.1 MB |

At ~25 tokens/utterance: ~2 min encoder + ~8 min decode = the ~10 min
estimate. Decode compute per token is trivial by comparison: ~5 ms NPU,
~0.3 s CPU (LM head dot products + attention).

### Host-driven (SWD via probe-rs at 2 MHz, 74 KB/s)

9.3 MB weights + ~2 MB cross-KV per token at 74 KB/s = ~2.3 min/token.
The link is 12x slower than the SD card; everything else is noise.

## 2. Verified hardware facts that open levers

From the nRF54LM20A datasheet v1.0 and the DK user guide v0.7.0:

- **SPIM00 (HS-SPI) does 32 MHz SCK**; the six SPIM2x instances cap at
  8 MHz. SPIM22 is already at its maximum (DIV_FAST=2).
- SPIM00's pins are **P2.00-P2.05**, which the DK routes either to the
  on-board 8 MB NOR (MX25R6435F) or to **headers P4/P17** via analog
  switches driven by the board controller (Board Configurator app).
- The chip has a **USBHS peripheral: DesignWare (DWC2) USB 2.0 device,
  480 Mbps high-speed, 16 endpoints, DMA**. Zephyr supports it on this DK
  (udc_dwc2), so there is reference driver code to crib from.
- **UARTE00 (HS-UART) does 4 Mbps**; the 1 Mbaud figure in NOTES applies
  to the VCOM path, not the peripheral.
- The NOR is QSPI-capable; quad access needs the **sQSPI SoftPeripheral on
  the VPR (FLPR) coprocessor** - plain SPIM00 single-lane otherwise.

## 3. Ranked levers - standalone build

### 3.1 Move the SD card to SPIM00 at 32 MHz (4x bus) - DONE (built blind)

Rewire the 4 breakout lines from P3.x to the P2.x pins on headers P4/P17
(board controller must release them from the NOR), point sd.rs at the
SPIM00 instance, and set the fast-domain divider (SCKDIV: CK/4 = 32 MHz).
The driver logic is unchanged; SPIM00 has the same EasyDMA register shape
(plus extras like DCX that we ignore).

Effect: decode ~18 s/token -> ~5 s/token; utterance ~10 min -> ~3 min.
This is the single biggest cheap win. Note SD cards in SPI mode commonly
tolerate 25-50 MHz; if a given card misbehaves at 32 MHz, CK/5 = 25.6 MHz
is the fallback.

### 3.2 LM head: stop scanning the whole vocabulary

4.7 MB/token to argmax 12230 rows. Options, updated after measurement:

a. **Norm-bound early exit - MEASURED, REJECTED.** The exact scheme (rows
   sorted by b_r = scale_r * ||row_r||_2, stop when b_r * ||hid|| < best)
   prunes 0 of 12228 rows: LM-head cosine similarities are tiny, so every
   row's bound (min 56.5) dwarfs the best logit (21.5). An int4-coarse
   first pass with exact re-check dies the same way (the residual bound is
   ~5x the typical argmax gap), and its statistical variant only breaks
   even once the bus runs at 32 MHz. Details in NOTES.md.
b. **VOCAB_KEEP dial** (exists already): 12230 -> 8k rows is a direct
   proportional cut, at the cost of rarer words.
c. **Amortize across positions**: the LM-head stream scores k hidden
   vectors for the price of one scan -- falls out of 3.3 for free.

### 3.3 Use all 4 width columns per pass - medium/large effort, up to ~3x

Token-rate submodels already run at width 4 because the Axon compiler
rejects pointwise conv below width 4 - columns 1-3 are computed and thrown
away. Every pointwise stage would process 4 token positions for the exact
same 9.3 MB of weight streaming.

- **Free win now: batch the SOT warmup.** The forced sot-sequence tokens
  (n_sot ~ 2-3) are known in advance; run them through the decoder in one
  width-4 pass instead of n_sot-1 sequential passes. Saves 1-2 full
  passes (~20-40 s) per utterance. Needs: per-column causal masking in
  the self-attention call and appending several KV columns per pass.
- **Speculative decoding.** Draft 3 cheap continuation tokens, verify all
  4 positions in one pass, accept the agreeing prefix. The LM head
  amortizes perfectly (score 4 hidden vectors per chunk while it streams
  once), so a pass costs the same bytes as today and yields 1-4 accepted
  tokens. Even a frequency/bigram draft over the pruned vocab could hit
  1.5-2 accepted/pass on English; the ceiling is 4. This is the only
  lever that divides the 9.3 MB/token weight floor without new hardware.
  Same kernel changes as the SOT batch, plus draft + accept/rollback of
  KV columns (RAM cache: trivial; nothing else is stateful).

### 3.4 Dynamic audio context (VAD endpointing) - medium effort

Everything frame-tiled scales with the number of 64-frame tiles: encoder
weights are loaded once per (blob, layer) but activations re-page per
tile, and cross-attention reads all CTX=600 keys per token. The blobs are
per-tile and the CPU attention kernel already takes the key count as a
parameter, so a runtime tile count works without recompiling anything.
End recording at silence (simple energy VAD on the mel pass), round up to
a tile: a 5 s utterance runs ~2.4x less encoder and ~2.4x less cross-KV
per token. Also stops the fixed 12 s record wait early.

### 3.5 Overlap and micro-optimizations - small effort each

- **Mel during record - DONE (built blind)**: pass 1 now streams during
  the recording with chunk-sized PDM buffers (bit-exact vs the sequential
  pass, which remains as an automatic fallback), hiding the whole pass
  and dropping the PCM round-trip (~15 s/utterance).
- **Cross-KV layout**: store the assembled planar [64, CTX] per
  (layer, matrix, head) once after cross_kv() instead of reassembling
  from 10 scattered 4 KB head blocks per use: same bytes, ~1/10th the
  commands, no repack loop. (True fix is 3.3: amortize across positions.)
- **SD driver**: read data+CRC in one DMA per block, keep CMD18 streams
  open across sequential region reads, ACMD23 pre-erase before CMD25
  bursts. Each recv1() today is a full EasyDMA setup for 1 byte on the
  wire. Worth ~10-20% together, more on the write path.
- **Blob compression** (LZ4-class, decompress into the slot): int8
  weights are high-entropy, expect only ~15-20%. Low priority.

### 3.6 The on-board NOR as a second, parallel bus - medium effort

8 MB MX25R6435F on SPIM00. If the SD stays on SPIM22, the NOR can hold
the LM head embedding (4.7 MB) plus the hottest blobs, streamed at
4 MB/s while the SD streams the rest at 0.9 MB/s in parallel (needs
async/interrupt DMA on both). If the SD moves to SPIM00 (3.1), the NOR
shares that bus and adds little unless driven quad via the FLPR sQSPI
SoftPeripheral (~2-4x again on its 8 MB, but bare-metal VPR bring-up is
a project of its own). Prefer 3.1 first; revisit the NOR only if the
per-token floor still hurts.

### Combined standalone outlook

Implemented so far (3.1 + 3.5a): ~16 MB/token at ~3.5 MB/s = ~5 s/token,
mel hidden behind the mic -> utterance ~10 min -> ~3 min. Adding the SOT
batch and speculative decode at ~2 accepted/pass (which also halves the
LM-head and cross-KV shares): ~1-1.5 min. With VAD on short utterances:
tens of seconds. Still not real time, but a different product category
("say a sentence, wait half a minute").

## 4. Ranked levers - host-driven mode

- **USBHS device (DWC2, 480 Mbps)** - large effort, transformative.
  Even a lazy bulk implementation at 20-30 MB/s makes the link faster
  than every other stage; token time collapses to compute + protocol
  (~1-2 s/token) and tape verification runs (43 MB) go from 49 min to
  minutes. Zephyr's udc_dwc2 driver and the STM32 synopsys-usb-otg Rust
  crate are working references for the same core. This also future-proofs
  every sibling project's host tooling.
- **SEGGER's own J-Link driver** (libjlinkarm FFI or JLinkExe batch) -
  small/medium effort: SEGGER's memory-write path is typically 5-10x
  probe-rs's (74 KB/s -> 0.5-1 MB/s), and it may clock the OB above the
  2 MHz probe-rs accepts. Keep probe-rs for flash+verify; use the SEGGER
  path only for bulk streaming.
- **UARTE00 at 4 Mbps** with a $10 USB-UART bridge (FT232H/CP2102N) on
  two header pins - small effort, ~400 KB/s (~5x SWD): double-buffered
  EasyDMA RX into the mailbox protocol. The pragmatic middle option if
  USBHS is postponed.

## 5. What will NOT help

- NPU optimizations: the Axon is idle >95% of the time in both modes.
- CPU clock: already 128 MHz; CPU glue is a rounding error next to IO.
- Bigger frame tiles / interlayer tuning: encoder activation paging is
  second-order once the bus is 4x faster.
- SPIM22 tuning: already at its 8 MHz hardware maximum.

## 6. Suggested order (updated)

1. ~~SPIM00 rewire + divider (3.1)~~ DONE, built blind.
2. ~~LM head norm-bound early exit (3.2a)~~ measured, rejected.
3. ~~Mel-during-record (3.5a)~~ DONE, built blind.
4. SOT warmup batch (3.3a), then speculative decode (3.3b) - the big
   algorithmic swing, and the only thing that divides the 9.3 MB/token
   weight floor (it also amortizes the LM head and cross-KV).
5. VAD/dynamic context (3.4) for short utterances.
6. USBHS (4) when host-driven development speed matters more than
   standalone polish.

Quality gates for anything touching quantization or schedule semantics:
decode_model.py transcript parity, teacher-forcing accuracy, and the
tape.py per-stage SNR ladder (TAPE_REF=float|int8).

## 7. 2026-08-27: mel DFT -> mixed-radix FFT, streaming mel back on

The ~22 s sequential mel pass was the O(N^2) direct DFT: 201 bins x
400 samples x 1200 frames. Rewritten as a 400-point DIT FFT
(400 = 5*5*4*4, real radix-4 leaves, in-place radix-4/5 combines).
Every twiddle W_400^m comes from the existing 400-entry cos table (sin
via the +300 index shift), so the SD image and tables are unchanged.
Mel filter rows are also trimmed to their nonzero spans (bit-exact:
the skipped products are exact +0.0).

tools/melcheck (host comparator compiling the real mel.rs against a
frozen copy of the old DFT, tables read from the built sd.img, device
chunking mirrored): sequential/streaming/whole drives bitwise
self-consistent; int8 mel (enc.mel quant) differs in at most 16/96000
cells by +-1 LSB across six signal types -- inside the tol=1 the tape's
own device-vs-whisper check allows. Host wall clock 13.5x
(179 -> 13 ms per 1200-frame pass).

Projection on the M33: ~22 s -> ~1.6 s sequential; a 64-frame streaming
chunk ~85 ms against its 640 ms budget (the old DFT needed more than
the full period -- that WAS the deterministic 17-overrun failure), so
TRY_STREAM_MEL is true again and pass 1 hides behind the 12 s capture.
The overrun -> sequential fallback stays. Expected critical-path win vs
the verified run: the whole mel gap between recording end and encoder
start (~22 s), leaving pass 2 (~1 s of SD).

## 8. 2026-08-27: 4-bit decoder weights + embedding (TODO item 2)

Decode is SD-bound at ~4.8 s/token (all 56 per-token decoder blobs
re-streamed every step, 4.7 MB LM-head embedding every sampled token).
Now every per-token decoder blob and the pruned embedding are stored
4-bit on the card: groups of 64 int8 weights share a u8 amax, nibbles
reconstruct as w' = sign*min(127,(|nib|*amax*2+7)/14) -- pure-integer
round-half-away, one spec in model/quant4.py mirrored by firmware
q4.rs, cross-checked bit for bit by tools/q4check over shared vectors.

Quality gate (model/quant4_check.py, device LM-head math): at G=64 the
JFK transcript is byte-identical to int8, 24/24 teacher-forced
agreement, 0 embedding argmax flips; weight rms error ~4 int8 LSB.
G=32/16 gain nothing (G=16's one-token drift is a knife-edge comma).

Key structural findings that made this cheap:
- Every decoder blob embeds its tflite filter tensor VERBATIM as the
  final 147456 bytes, and requantization changes no scales/zero-points/
  biases, so the Axon converter never re-runs: make_sd_image locates
  the weights by byte-search and packs "LAY4" entries; blobs on disk
  are untouched (tape/SWD flows keep using the raw .bin files).
- The firmware expands packed entries in place in the slot (packed
  bytes moved to the slot tail, writer can never catch the reader,
  ~56 KB margin) and bounces packed embedding chunks through the
  interlayer buffer (transient use between NPU runs is safe).
- First use of each expanded blob is verified against a raw-content
  sum ("sums4" asset): packer/unpacker drift is error -907, not a
  garbage transcript. The old card still works with the new firmware
  (no LAY4/embp4 entries -> raw paths).

Image: 42.5 MB (was 44.0), 6.2 MB less SD read per token cycle
(4.0 MB blobs + 2.2 MB embedding). Unpack costs ~0.3 s/step CPU against
~1.2-1.9 s/step of reads saved; expect ~4.8 -> ~3 s/token. Awaiting a
bench run (new dd + reflash together).

## Sources

- nRF54LM20A/B datasheet v1.0 (SPIM00 32 MHz, UARTE00 4 Mbps, USBHS
  DWC2 480 Mbps): https://files.seeedstudio.com/wiki/XIAO_nRF54LM20A/getting_start/RES/nRF54LM20A_nRF54LM20B_Datasheet_v1.0.pdf
- nRF54LM20 DK HW user guide v0.7.0 (P2.00-P2.05 NOR/header switching,
  USB connectors): https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/8897/nRF54LM20_DK_HW_User_Guide_v0.7.0.pdf
- Zephyr nRF54LM20 DK board docs (udc_dwc2 USBHS support): https://docs.zephyrproject.org/latest/boards/nordic/nrf54lm20dk/doc/index.html
