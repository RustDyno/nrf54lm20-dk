# Whisper speed analysis: where the time goes and how to cut it

2026-09-04 UPDATE: section 9's levers 1-6 are IMPLEMENTED and verified
on the mock rig (section 10): encoder 100 -> 38 s, decode 2.4 -> 1.08 s
per step, transcript and every scratch region bit-identical. Sections
1-6 describe the SD-era ranking and are kept for history.

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

First bench attempt hardfaulted at decode token 1 -- not the 4-bit
data: attn_head's 6.4 KB stack frame at decode depth crossed
_stack_end into the Axon driver's .bss state (the +1 KB SUMS4 static
had eaten the last of an always-razor-thin margin). Fix: attention and
unpack scratch moved to the idle interlayer buffer, raw_sum folded
into the LAY4 header (SUMS4 dropped, .bss -1 KB), MSPLIM armed at
_stack_end so overflow is a precise STKOF fault from now on. Same
session verified streaming mel: sd[mel] 142 ms rd / 321 ms wr, no
overruns -- the ~22 s mel pass is fully hidden behind capture.

## Sources

- nRF54LM20A/B datasheet v1.0 (SPIM00 32 MHz, UARTE00 4 Mbps, USBHS
  DWC2 480 Mbps): https://files.seeedstudio.com/wiki/XIAO_nRF54LM20A/getting_start/RES/nRF54LM20A_nRF54LM20B_Datasheet_v1.0.pdf
- nRF54LM20 DK HW user guide v0.7.0 (P2.00-P2.05 NOR/header switching,
  USB connectors): https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/8897/nRF54LM20_DK_HW_User_Guide_v0.7.0.pdf
- Zephyr nRF54LM20 DK board docs (udc_dwc2 USBHS support): https://docs.zephyrproject.org/latest/boards/nordic/nrf54lm20dk/doc/index.html

## 9. 2026-09-03: profile on the USB stick; the M33 is the bottleneck

Source: firmware/output9.log (the 14cd:1212 USB stick in host mode,
reads ~9-10 MB/s, writes ~1 MB/s). Commits after that run touched only
the microphone path (PDM gain, mel lift, mock rig), not the encoder,
decoder or LM head, so the compute profile still stands. Phase boundaries
come from the host timestamps on the "npu <blob>" lines (RTT is
NoBlockSkip, so logging does not stall the firmware; the 120 ms bursts
are the host's poll cadence). The sd[phase] lines give the storage
share of each phase directly.

### Encoder, ctx 600 (10 tiles): 122.8 s wall, storage 24.7 s

| Phase (per block unless noted) | Wall | What it is |
|---|---|---|
| conv1 + conv2 + posenc (once) | 4.7 s | 50 NPU runs, halo assembly, IO |
| q/k/v projections | 1.2-2.0 s | 30 NPU runs, 180 4 KB head writes |
| v -> out gap = ATTENTION | 21.2 s | 60 attn_head calls on the CPU |
| out-proj + res_add + ln2 | 0.8-1.7 s | 10 NPU runs, IO |
| MLP (fc1 x4, fc2p x4) | 3.1-4.0 s | 80 NPU runs, 960 KB partial writes |
| recombination + next ln1 | 0.9-1.6 s | 3.8 MB of partial/residual IO |
| final LN (once) | 1.75 s | |

Attention: 4 x 21.2 s = 85 s = 71 percent of the encoder. Per block
that is 295 M MACs + 2.3 M expf + 2.3 M roundf/div quantizations in
~2.7 G cycles: ~8 cycles per MAC. The kernel is scalar (ldrsb, ldr,
mla, str, loop per MAC at opt-level s); the M33 DSP extension does two
16-bit MACs per cycle (SMLAD) and the acc read-modify-write pattern of
the rank-1 formulation is the wrong shape for it. Everything else in
the encoder (NPU ~530 runs, ~9 MB reads of activations, 15.8 MB of
writes at ~1 MB/s) totals ~38 s.

The ctx-192 run (3 tiles) scales as predicted: attention 2.2-3.0 s per
block, whole encoder 20.8 s, storage 7.7 s.

### Decode: 2.9-3.0 s per sampled token (6 tokens: 23 s)

| Component | Time/step | Basis |
|---|---|---|
| four decoder blocks (56 blobs, 9.3 MB) | 1.3-1.4 s | d0q -> d3fc1a timestamps |
| of which storage | ~1.0 s | 9.3 MB blobs + 1.9 MB cross-KV pages at ~10 MB/s |
| of which blob byte-sum check | ~0.2 s | 9.3 MB at ~3 cycles/byte |
| LM head (hid -> lm: row) | 1.53 s | measured twice, identical |
| SOT warm-up (once per utterance) | 1.4-1.6 s | one block pass, no LM head |

The LM head streams the 4-bit embedding (2.35 MB, ~0.25 s) but spends
the rest on the CPU and on command count: 192 chunks, each a packed
read plus a separate 1-block row-scale read, a nibble unpack through a
16-entry table, and f32 dot products with a bounds-checked index
(~8 cycles per element for 4.7 M elements). NPU time per step is
negligible (56 runs at width 4, well under 0.1 s).

### Ranked levers (savings for a ctx-600, 10-token utterance, ~160 s)

1. Attention kernel on the DSP extension: transpose K to [key][dim]
   int16 and expand V to int16 once per head (153 KB; the weight SLOT
   is idle during CPU attention and reloading the next blob costs
   18 ms), then SMLAD dot products with 2x2 register blocking. Same
   i32 arithmetic, so bit-exact by construction. ~8 -> ~1.2
   cycles/MAC: 21 s -> ~5 s per block. SAVES ~65 s. Small/medium
   effort, one file plus a scratch reservation.
2. Softmax tail: expf via a table or a short exp2 polynomial, and the
   probability quantization via VRINTA/VCVTA (1 instruction, roundf
   semantics) with the /(1/256) as an exact *256. ~2 s/block -> ~0.5.
   SAVES ~6 s. Small. Not bit-identical to libm::expf in the last ulp;
   the tape's 1-LSB tolerance and the transcript gate cover it.
3. LM head: int16 hid x int8 rows on SMLAD (the original design was
   int8 x int8 -> i32 anyway), row scales stored inside each packed
   chunk (one read per chunk, not two), unpack written as a word loop.
   1.5 s -> ~0.5 s/token. SAVES ~10 s. Small.
4. Blob byte-sum on the USB backend: the bulk protocol already CRC16s
   every packet; skip it there, or sum words with USAD8 (4 bytes per
   cycle). SAVES ~2 s (0.2 s/token) plus ~0.2 s in the encoder. Trivial.
5. Cross-KV planar on the card (section 3.5): 480 4 KB reads per token
   -> 48 reads of 38 KB, no repack loops. SAVES ~2-3 s. Small.
6. Stop recording at silence: the VAD already runs chunk by chunk on
   the streaming mel pass, but the capture always runs the full 12 s
   (16.4 s from "speak now" to encoder start on a 2.3 s utterance).
   SAVES up to ~9 s on short utterances. Small.
7. Try a fast stick (zero code): reads are a flat ~9-10 MB/s at every
   transfer size (4 KB head blocks to 166 KB blobs), so the limit is
   bandwidth, not per-command overhead: the bargain 14cd:1212 stick or
   the polled DMA loop. A USB 2.0 stick can read 30-40 MB/s and write
   10+ MB/s; that would cut the 16 s of encoder scratch writes and
   the ~1.3 s/token of decode reads by 2-3x if the driver keeps up.
8. Overlap storage with compute (DWC2 buffer DMA runs autonomously;
   start a transfer, do CPU work, poll later): worth up to the
   storage share (~25 s encoder, ~1 s/token) once 1-3 are in. Medium.
9. 4-bit decoder blobs (TODO B/D): -4.6 MB/token of stream (~0.5 s)
   against ~0.2-0.3 s of unpack. Net ~0.3 s/token. Medium, and the
   G=32 quality regate comes first.
10. Speculative decoding (section 3.3b): the only lever that divides
    the 9.3 MB/token weight floor; acceptance rate with a cheap draft
    is unmeasured. Large.

Projection after 1-6: encoder ~45 s, decode ~1.7 s/token, so the same
utterance lands at ~85 s; a fast stick (7) or IO overlap (8) takes the
encoder toward ~30 s.

Also cheap: the whole crate builds at opt-level "s". Kernels-only
opt-level 3 is not expressible in stable Rust, but a crate-wide switch
is a zero-effort experiment now that the blobs bind only to pinned
data addresses (text-only changes keep the card valid).

## 10. 2026-09-04: levers 1-6 implemented; measured on the rig

Mock rig, JFK clip, ctx 600, same session before/after (NOTES.md has the
design of each kernel):

| phase | before | after |
|---|---|---|
| encoder | 100 s | 38 s (CPU attention 80 -> 20.6 s) |
| cross K/V | 2.3 s | 2.2 s |
| decode per step (25 steps) | 2.4 s | 1.08 s |
| of which storage (rig, ~24 MB/s) | 0.64 s | 0.52 s |
| of which LM head (dots + unpack) | ~1.5 s | 0.24 s |
| of which cross-attention paging | ~0.3 s | 0.15 s |
| speak-to-done | ~168 s | ~71 s |

Accuracy: transcript identical; mock_diff.py finds 0 of 5480 scratch
blocks (mel, residual, encoder output, cross K/V) differing from the
previous firmware's run, fast exponential included. Host gates:
attncheck 0 bytes off the golden, exp within 2 ulp, lm16_check 0/24
argmax flips (max logit deviation 0.0014 against a 0.89 minimum gap).

Remaining decode step on the rig: storage 0.52, unpack 0.16, attention
0.15 (mostly the per-head transposition at one query), LM dots 0.08,
byte-sums 0.05, NPU/LN/misc ~0.12. On the stick the storage term is
~1.3 s, so the stick is now the decode floor: levers 7 (faster stick)
and 8 (IO overlap) are next; 10 (batched positions) divides everything.

## 11. 2026-09-04: would a different model be faster?

Candidates: Moonshine Tiny, Vosk small (Kaldi TDNN-F), Sherpa-ONNX
Zipformer transducer (~20M), NeMo Conformer/FastConformer-CTC small.

The decode floor after section 10 is the ~9.3 MB of decoder weights
streamed per token (1.3 s on the stick). That floor belongs to every
encoder-decoder model whose decoder does not fit in 510 KB RAM; no
kernel lever removes it, levers 7-10 only shrink it. A model with no
autoregressive decoder deletes it: ~47 s of a 25-token utterance.

| model | decode per token | encoder attention vs Whisper | verdict |
|---|---|---|---|
| Moonshine Tiny | ~10 MB stream (6 dec layers, d=288, 32k vocab) | ~20 pct cheaper, variable length | no gain |
| Vosk small | WFST beam search, ~30 MB graph, random access | linear time, no attention | poor fit for RAM and stick |
| Zipformer transducer | predictor + joiner, ~100s of KB, resident | 4-16x cheaper (25 Hz, downsampled stacks) | yes, largest win, hardest port |
| Conformer-CTC small (13M) | none, argmax per frame, 128 vocab | 2x at 4x subsampling, ~8x for FastConformer | yes, cleanest port |

- Moonshine: raw-audio conv stem, no 30 s padding, but full-rate
  attention and the same weight-stream decoder. Dynamic ctx already
  gives the short-utterance benefit. RoPE + SwiGLU are new CPU glue.
- Vosk: the acoustic model maps to Axon convs, but its outputs are
  context-dependent phone states; there is no greedy path without the
  graph, and Viterbi over a 30 MB graph at 10 MB/s is hopeless.
- Zipformer: streaming variant runs chunk by chunk during capture, so
  post-speech latency approaches one chunk. SwooshR, BiasNorm,
  non-linear attention and bypass modules are all new glue with no
  Axon op; int8 quantization needs its own gate.
- Conformer-CTC: depthwise + pointwise convs land on the NPU like the
  current projections. Rel-pos attention doubles the CPU dot work per
  layer but at 150-300 frames instead of 600 it is net cheaper. Swish
  and GLU need LUTs like GELU. A public "FastConformer-CTC Tiny" EN
  checkpoint is unconfirmed; Conformer-CTC small (13M) is the known
  size at that scale.

Outlook on the stick, 25-token utterance: Whisper + fast stick + IO
overlap ~50-60 s; CTC at 4x subsampling ~20-30 s (encoder NPU passes
and scratch writes remain); streaming hides most of that behind the
recording.

Cost: export, Axon compile, blob link, firmware glue and the accuracy
gates are all Whisper-specific, so a port is weeks against days for
levers 7-10. Accuracy: the small CTC/Zipformer models report
LibriSpeech WER at or below tiny.en but are LibriSpeech-trained and
lose more on noisy/far-field audio than Whisper's 680k-hour training.

Decision 2026-09-04: finish the Whisper levers first (7-10); revisit
CTC if speak-to-done must go under ~30 s.

## 12. 2026-09-04: storage overlapped with compute (speed pass 5)

Split-phase transfers (storage.rs read_start / write_start / poll /
finish; the data phase and the CSW run on the DWC2 channel DMA, the
mock rig's payload and response on the endpoint DMA) under four
compute windows: encoder attention (context write + next head's K/V/Q
prefetch, pumped every 16 queries), encoder MLP tiles (partial write
under the fc1 run, next input under fc2p), decode cross-attention (K/V
prefetch), LM head (chunk double buffer). Design in NOTES.md.

Mock rig, JFK clip, ctx 600, same-session baseline (HEAD firmware):

| phase | before | after |
|---|---|---|
| encoder | 38.2 s (storage stall rd 3.29 + wr 0.80 s) | 37.6 s (rd 3.03 + wr 0.63 s) |
| cross K/V | 2.2 s | 2.3 s |
| decode per step (24 steps) | 1.10 s (storage 0.55 s) | 0.97 s (storage 0.42 s) |
| speak-to-done | 67.7 s | 63.8 s |

Accuracy: transcript identical; mock_diff.py 0 of 5480 scratch blocks
differ (mel, residual, encoder output, cross K/V). The sd[phase] ms
figures now count CPU time inside storage calls, i.e. the unhidden
stall, which is what the table compares.

Why the rig moves so little: its writes run at ~20 MB/s, so the 4.3 MB
of writes now hidden (context tiles 0.96 MB, fc2 partials 3.5 MB) cost
it 0.2 s. On the stick they cost ~4.3 s, plus ~0.35 s for the 3.5 MB of
hidden encoder reads and ~0.4 s/token for the 4.2 MB/token of decode
reads (cross K/V 1.9 MB, embedding 2.35 MB): roughly 5 s off the
encoder and 10 s off a 24-token decode, ~15 s of ~100 s. Unmeasured
until a stick run.

What stays synchronous on the stick, and why (from the 15.8 MB of
encoder writes): q/k/v projection head blocks 2.9 MB (six 4 KB writes
behind one NPU run; only one transfer can straddle a blocking
infer_sync, a queue would need a timer-interrupt pump), residual /
layernorm / recombination passes 5.7 MB (IO-bound loops with
milliseconds of CPU between read and write: nothing to hide under),
out-projection and conv stem 1.7 MB, last tile / last head 0.6 MB.
Reads: the fc1/fc2p blobs alternate every tile, 53 MB of the encoder's
83 MB of reads; the alternative (all fc1 tiles first, spilling 240 KB
per pass) trades them for writes, the wrong way on this stick.

Ranking after this pass: lever 7 (a stick with 10+ MB/s writes) is now
worth ~11 s of synchronous writes plus the 9.3 MB/token blob floor at
whatever its read speed is; lever 10 (batched positions) still divides
the decode floor; the projection-write queue is ~3 s for a medium
change.

### Stick run, 2026-09-04 (new stick: PNY USB 3.0 FD 154b:00ed, 128 GB)

JFK clip played through the laptop speakers into the board's microphone
(mel level correction +28.7 dB; the transcript has two acoustic errors
but the rig proves the arithmetic), VAD ctx 531 (9 tiles), 22 tokens:

| phase | wall | storage stall (unhidden) |
|---|---|---|
| encoder | 37.5 s | rd 75 MB / 2.6 s, wr 14.3 MB / 5.8 s |
| cross K/V | 3.9 s | wr 1.7 MB / 2.0 s |
| decode per step (23) | 0.80 s | 0.26 s |
| end of speech to transcript | 62.6 s | recording tail 2.0 s (silence stop, first live exercise) + 60.6 s |

The new stick reads at ~28 MB/s and writes at ~2 MB/s, against the old
one's 9-10 / 1. With the overlap in, decode on the stick (0.80 s/step)
is now FASTER than on the rig (0.97): the remaining decode step is
CPU (unpack 0.165, attention 0.14, LM dots 0.08, sums 0.05, NPU/LN
~0.1) plus 0.26 s of blob-read stall. The encoder is attention 17 s +
~12 s NPU/LN/IO + 8.4 s of write-dominated stall (the synchronous
passes listed above). The stick's first read after enumeration took
over 516 ms, so the read budget floor is now 2 s and the index read is
retried (app.rs run()).

Ranking on this stick: the synchronous encoder/cross writes (~7.8 s
of stall: projection head blocks, residual/LN/recombination passes,
cross K/V head blocks) are the largest storage term left; a queue
pumped from a timer interrupt would hide the projection and cross K/V
head-block writes (~4 s). Decode is CPU-bound again: the 4-bit unpack
(0.165 s) and the cross-attention transposition (0.14 s) are the next
kernels, then batched positions.

## 13. 2026-09-04: speed pass 6, build profile, layouts, int8 LM head, pumped writes, kernel order

Mock rig, JFK clip, ctx 582, 23 tokens (25 decoder passes), same-session
baseline of the pass-5 firmware. Every row: transcript identical,
mock_diff.py 0 of 5480 scratch blocks differ (cross K compared under its
new block transposition), per-step hidden vectors and token ids
identical. Wall times from the sd[phase] timestamps, "total" the
firmware's processing time from the end of the clip.

| change (cumulative) | encoder | cross | decode | /step | total |
|---|---|---|---|---|---|
| baseline (pass 5, opt-level s) | 39.0 s | 2.8 s | 26.0 s | 1.040 s | 67.9 s |
| 1. opt-level 3, both profiles | 38.3 | 2.5 | 22.6 | 0.903 | 63.4 |
| 2. cross K stored key-major, decode widens instead of transposing | 38.4 | 2.5 | 22.2 | 0.889 | 63.1 |
| 3. int8 LM-head rows (embc8), SXTB16 dot kernel | 38.2 | 2.5 | 19.1 | 0.765 | 59.9 |
| 4. head-block writes pumped from the NPU wait (projections, cross K/V) | 38.2 | 2.3 | 19.2 | 0.767 | 59.6 |
| 5a. dev profile without overflow checks / debug assertions | 36.4 | 2.2 | 18.7 | 0.748 | 57.3 |
| 5b. loads grouped in the 2x2 kernels, exact softmax restructure | 32.9 | 2.2 | 18.3 | 0.732 | 53.2 |

Where the encoder attention went (cpu[encoder] split, new this pass):
21.2 s = QK 6.5 + softmax 7.5 + PV 6.9 at the start; 16.7 s = QK 4.8 +
softmax 6.3 + PV 5.4 at the end. QK runs at 1.18 cycles per MAC (was
1.6), PV at 1.31 (was 1.7): the M33 issues consecutive loads one per
cycle after the first, so a step that loads its four operand words and
then does its four SMLADs beats the interleaved order the kernels had.
Softmax is now the largest term at ~99 cycles per key: two float passes
(exp with a degree-7 polynomial, then the quantization) over 8.1 M keys
per utterance. Decode attention 2.90 -> 2.30 s per utterance; of what is
left, ~1.2 s is widening K and V to int16 for the kernels.

Item 3 is the one numeric change: the int8 rows are the pre-4-bit
embedding, closer to the f32 model than the 4-bit-coded rows the LM head
used since pass 4. lm16_check.py --int8: 0/24 argmax flips against the
int8-row f32-hidden head, transcript equal to the 4-bit head's; on the
rig the best logits moved by up to 1.07 (of ~25) and no token changed.
The image carries both entries; the firmware prefers embc8 (LM_ROWS_INT8)
and falls back to embc4.

What the rig cannot show: item 4 hides writes that cost the rig 0.5 s and
the stick ~3.5 s (projection head blocks 2.9 MB, cross K/V 1.7 MB, at
~2.7 ms of command overhead plus 2 MB/s); item 3 is read-bound on the
rig (4.7 MB per token at ~22 MB/s) and should be ~0.08 s per step better
than embc4 on the 28 MB/s stick. Also found: the rig built the dev
profile with overflow checks on (release never had them), so earlier rig
CPU numbers were inflated by ~5 percent against the stick's.

Tried and dropped: the exp pass computing two keys in lockstep so the
two polynomial chains interleave (the compiler did interleave them, and
the arithmetic per key was unchanged): softmax 6.31 -> 6.25 s. The M33
FPU has no latency to hide; the softmax is bound by its ~99 instructions
per key, so only a cheaper exponential (a numeric change, to be gated)
or fewer keys would move it.

Remaining levers, in order: int8 K/V variants of the 2x2 kernels for
decode (removes the ~1.2 s of widening per utterance); 4x2 register
blocking of QK/PV (~10 percent of 10 s); a lower-degree or table
exponential for the softmax (gated: attncheck byte moves, transcript
parity; up to ~3 s); the residual/layernorm passes (5.7 MB of
synchronous writes, still nothing to hide them under); speculative
decode positions (large).

## 14. 2026-09-05: speed pass 7, table softmax, int8 decode attention, paired MLP tiles

Mock rig, JFK clip, ctx 582, 25 decoder passes, same-session baseline of
the pass-6 firmware. Every row: transcript identical, mock_diff.py 0 of
5480 scratch blocks differ, per-step hidden vectors and token ids
identical. Wall times from the sd[phase] timestamps, "total" the
firmware's processing time from the end of the clip.

| change (cumulative) | encoder | cross | decode | /step | total |
|---|---|---|---|---|---|
| baseline (pass 6) | 32.8 s | 2.3 s | 18.3 s | 0.733 s | 53.4 s |
| 1. softmax exp as an integer-indexed table (encoder) | 29.2 | 2.3 | 18.3 | 0.733 | 50.0 (est.) |
| 2. int8 K/V decode attention kernels, ~45 fewer small reads per token | 29.2 | 2.3 | 17.0 | 0.682 | 48.6 |
| 3. byte-sum 32 bytes per iteration | 29.2 | 2.3 | 17.1 | 0.683 | 48.7 |
| 4. encoder MLP tiles paired per blob load | 28.1 | 2.4 | 17.1 | 0.683 | 47.6 |

(Row 1 was measured together with row 2 in one run; its encoder column
is exact, the total is the baseline minus the encoder difference.)

Encoder CPU split (cpu[encoder]): attention 16.7 -> 13.1 s (QK 4.8 ->
4.9, softmax 6.3 -> 2.5, PV 5.4 -> 5.4), NPU 9.8 s unchanged, byte-sums
0.30 -> 0.14 s, storage stall 3.2 -> 2.5 s with the reads down from 83
to 57 MB (the fc1/fc2p reloads halved). Decode CPU: attention 2.30 ->
0.98 s (the K/V widening and the twin-query waste gone), LM head 2.0 s,
byte-sums 1.15 -> 0.99 s, NPU 2.32 s; storage stall 9.8 -> 10.1 s (less
CPU to hide the reads behind). The new "validate" figure in the cpu[]
line is 0 ms: the 1.66 ms per width-4 decode run is the driver's
inference path and the engine, not the model validation.

Numerics: the table softmax is a different evaluation of the same
function (each value within ~1.5 ulp of the true exponential of the
exact integer deficit, where the golden's float path rounds its
argument first). attncheck moved 0 of 64000 context bytes over five
score multipliers spanning the image's, and the rig's encoder output is
bit-identical, so on this clip it is not a numeric change at all; a
1-LSB probability flip on another clip would be absorbed like the fast
exponential's were. Everything else in the pass is a reordering.

Found on the way: with `+vfp2` (needed for the asm's FPU operands) LLVM
compiles f64 to double-precision VFP instructions the M33 does not have;
the first table build hard-faulted as an undefined instruction.
`-fp64` in the target features fixes it for good (NOTES.md, skill).

What the rig cannot show: the ~45 small reads per token now gone cost
the stick ~2.7 ms each (~3 s per utterance); the 26 MB of blob reloads
now gone cost it ~1 s. Both unmeasured there; the stick image needs a
re-dd for the "dc" entries (the firmware falls back without them) and,
still, for embc8.

Remaining levers, in order: the decode blob stream (9.3 MB/token, 0.4 s
of the 0.68 s step on the rig: 4-bit decoder blobs at G=32 with a
word-wise int8 expansion, regated on several clips, ~3 s; or batched
positions); the decode LM head reads (4.7 MB/token int8 rows, hidden
behind their dots only partly); the encoder attention kernels at ~1.2
cycles per MAC (a 2x3 blocking with remainder handling, ~1 s; or the QK
and PV matmuls as NPU blobs with the keys/values as runtime-patched
weights, several seconds but the int8 requantized scores need a gate);
the encoder's residual/layernorm passes (IO-bound on the stick, ~1.5 s
of CPU that a tile pipeline could hide).
