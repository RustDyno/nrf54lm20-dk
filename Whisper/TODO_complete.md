# Completed TODO items

## USB host mode: model image on a USB stick (2026-09-01)

Asked as "load model from usbstick"; landed as a full storage backend
plus its hardware verification in one day.

- The nRF54LM20's "device-only" USBHS is dual-role DWC2 v5.00b silicon
  (GHWCFG2 OTGMODE=2, 16 host channels). usb.rs forces host mode and
  speaks bulk-only mass storage: polled, buffer DMA, everything on host
  channel 0. storage.rs probes USB then SD; the same dd image works on
  either medium; app.rs/mailbox/host tooling backend-agnostic.
- Wiring: the chip cannot source VBUS, so 5 V from a 5V0:CONN header
  feeds J3's VBUS (recipes in README).
- Verified end to end: a generic 4 GB stick served a complete
  standalone utterance (streaming mel, ctx-600 encoder 83 MB, cross
  K/V, 32-token decode 283 MB). Reads 8.5-9.4 MB/s vs SD 3.3; writes
  1.0-2.7 MB/s vs SD 0.10-0.28. Decode 3.4 s/token at ctx 600 with
  storage only ~30 percent of it: the M33 is now the bottleneck.
- Bench-found and fixed: v4.20a+ soft-reset handshake (CSftRstDone,
  undocumented), wrapper STATUS.CORE never asserts (gate on GSNPSID),
  VREGUSB VBUSDETECTED edge event vs re-init, BOT state-machine desync
  after a timed-out write (mass-storage reset recovery), multi-second
  slow-stick write/ready budgets, CSW staging must not share the
  small-response buffer.
- Full notes in NOTES.md ("USB host mode" sections); protocol code
  host-tested by tools/usbcheck.

## Constant-token decode freeze fixed: ship int8 decoder (2026-09-01)

- Symptom: every run since the 4-bit decoder change (speed pass 4)
  decoded ONE token every step until the budget, regardless of audio
  (" Pitt", "uss"). Root cause: the IN-PLACE 4-bit weight expansion
  (unpack_slot) hands the NPU slightly-wrong per-token decoder weights
  (~2 LSB, sum-preserving so raw_sum can't catch), which on top of a
  marginal G=64 quantization tips the knife-edge decode into a fixed
  attractor.
- Localized by dumping the device scratch (S_MEL/S_EO/S_XKV) off the
  stick and decoding it through the host mirror (decode_model.py):
  encoder output and cross-K/V verified BIT-EXACT on silicon; the fault
  was purely the per-token decode transformer, and step-0 sub-layer
  fingerprints (temporary dtr/dtr8 firmware trace vs decode_golden*.py)
  put the first divergence at the q/k/v projection. A raw-int8 decoder
  image then decoded correctly (" Yes."), trace bit-exact to golden --
  proving the width-4 NPU conv and the weights are fine and isolating
  the bug to unpack_slot (the disjoint embp4 path was always fine).
- Fix A shipped: make_sd_image skips per-token 4-bit packing by default
  (Q4_PACK=1 to re-enable once the in-place path is fixed); decoder runs
  raw int8. Firmware decode debug instrumentation removed. Follow-ups to
  reclaim the 4-bit SD savings (disjoint expansion / barrier probe /
  re-gate at G=32) tracked in TODO.md item 0.

## Mock USB rig: run the whole pipeline on device from a PC (2026-09-03)

Asked as "make a mock usb to test with ... so you can test the whole
thing on device to improve".

- `firmware/src/usbdev.rs`: the same DWC2 core as usb.rs forced into
  DEVICE mode (its documented role), presenting CDC-ACM -- so Linux binds
  cdc-acm, the port is group `dialout`, and no udev rule or root is
  needed. `firmware/src/mockblk.rs` speaks 512-byte blocks over it and
  registers as `Backend::Mock` behind feature `mock-usb`. usb.rs's
  bring-up prologue is now shared as `usb::platform_up()`.
- `tools/mockusb`: host daemon serving `model/out/sd.img` from a work
  copy, injecting a wav/raw clip in place of the microphone, and
  surviving reflashes (it waits for a changed USB devnum).
- Audio injection is a runtime marker block in the image, so one firmware
  build both replays a clip and records live.
- `model/mock_compare.py`: reads the mel / encoder output / PCM straight
  out of the work file and diffs against `out/ref.npz`.
- VERIFIED ON HARDWARE: a full deterministic utterance in under 3 minutes
  (mel 2.5 s, encoder 100 s / 83 MB, 23-token decode), reproducing the
  reference transcript exactly. ~1-6 MB/s vs SWD's 0.074.

Bring-up findings that cost bench time are in NOTES ("Mock USB"): the
missing terminating ZLP, the host tty's line-discipline window eating a
frame's first five bytes (op 3 == VINTR flushes the input queue), the
u32 overflow in the DWT deadline past 33 s, and retry-induced desync.

## Accuracy localized to the microphone, not the model (2026-09-03)

The standing "inaccurate on device" problem is the AUDIO INPUT. Same
firmware, same image, two runs:

- injected reference clip: mel int8 range [-128, 127], dequantized
  [-0.541, 1.459] against the mirror's [-0.539, 1.461] -- max diff 0.018
  (2 int8 steps), rms 0.0024, correlation 0.99999. Transcript matches the
  reference token for token (only the known audio_ctx=600 comma missing).
- live microphone: PCM peak 58 / 32768 (-55 dBFS, rms 3.2) against the
  clip's 25647 (-2.1 dBFS, rms 4458) -- ~63 dB down. Mel pinned at the
  int8 floor, [-128, -91], using 24 of 256 codes.

That is the "~6 log10 units low" mel already noted (that first mic run
was ambient silence; with speech it is ~31 dB / 3.1 log10 units down).
Whisper's mel normalization has an absolute `(log_spec + 4) / 4` term, so
the shift is not absorbed.

PROVEN by a round trip the same day: a live mic window recorded off the
device (`--features mic-check`), multiplied by 36 on the host and injected
back through the SAME firmware and image, transcribed CORRECTLY
(" 1 2 3 4 5 ...") with the mel moving to [-119,127]. Nothing in the
firmware changed between the failing and passing runs, which exonerates
every stage after the microphone. Remaining work is choosing the gain
scheme -- TODO item 0.

## Microphone level fixed on the board (2026-09-03)

Asked as "can we boost the mic on the board?" -- yes, and it needed two
levers because neither alone is enough.

- pdm.rs GAIN 0 dB -> +12 dB (0x28 + 24). Digital gain inside the PDM
  peripheral, applied ahead of the 16-bit output, so unlike a software
  scale it keeps detail that would otherwise be truncated. Stopped at
  +12 of the available +20 dB because speech peaks were already
  -15.4 dBFS; measured after the change: peak 4292, 0 clipped samples,
  0 overruns.
- app.rs mel_lift() + mel.rs MelNormParams.lift: the remaining ~25 dB
  taken out per utterance in the log-mel domain, where it cannot clip.
  Equivalent to recording louder (a gain g shifts log10 power by
  2*log10(g) uniformly) but free, since mel pass 1 already accumulates
  the peak. Clamped to [-2.0, +4.0] log10.

VERIFIED, three ways:
- reference clip: correction computes as +0.000 and fp[mel8] is 0x138d4,
  BYTE-IDENTICAL to the pre-change verified run -- the calibration path
  is untouched.
- the raw mic recording that used to transcribe as nonsense (peak 1707,
  active rms 137, NO host-side gain) now gets +26.9 dB of correction
  automatically, a mel of [-128,127] against the reference's [-128,127],
  and transcribes CORRECTLY: " 1 2 3 4 5 1 2 3 4 5 ...".
- a second capture transcribed as " . . . ." -- but its spectrum has no
  voice fundamental (dominant 1378 Hz, only 9% of energy below 300 Hz)
  against the good capture's clear 128 Hz with harmonics, i.e. nothing
  was said in that window rather than a regression. Worth knowing: once
  the lift is in, a window with no speech gets amplified to full scale
  and decodes as repeated punctuation, so "loud" no longer implies
  "speech" and the mic-check trigger threshold is not a VAD.


## 2026-09-04: speed pass 4, CPU kernels on the DSP extension

Profiled on the USB stick (output9.log): 85 of 123 encoder seconds in
the scalar attention kernel, 1.5 of 2.9 s per decode step in the LM
head. Moved onto the Cortex-M33 DSP extension (firmware/src/dsp.rs,
inline asm with portable fallbacks):

- attention: keys transposed / values widened to int16 in the idle
  weight slot, 2x2 SMLAD blocks, one read per head for Q/K/V and one
  write for the context tiles (bit-exact: attncheck 0 bytes off)
- softmax tail: VCVTA rounding, exact *256, fast exp (2 ulp, moved
  nothing on the real utterance)
- LM head: int16 hidden x int16-expanded 4-bit rows on SMLAD, one read
  per 64-row chunk with scales and ids inside (new "embc4" entry;
  lm16_check.py 0/24 flips), reconstruction tables built once per decode
- blob byte-sum on USADA8; decode cross K/V paged in one read per head
- recording stops after 3 silent chunks past speech (never before 7),
  keeping every frame the encoder touches real

Verified on the mock rig (JFK clip): transcript identical, mel /
residual / encoder output / cross K/V scratch bit-identical to the
previous firmware (mock_diff.py, 0 of 5480 blocks), encoder 100 -> 38 s,
decode 2.4 -> 1.08 s per step, speak-to-done ~168 -> ~71 s. Image
rebuilt (embc4 replaces embp4; old cards need a re-dd).

## 2026-09-04: USB-only probe, latency after the transcript

- storage::init tries the USB stick three times and then stops; the SD
  card probe (and its pin diagnostics) is behind `--features sd-card`.
- After "Detected: ..." the firmware prints the time from the end of
  speech to the printed words, split into the recording tail after the
  last speech frame and the processing time (injected clips: processing
  from the end of the clip). SysTick now keeps a 10 ms uptime clock.

## 2026-09-04: speed pass 5, storage overlapped with compute

- storage.rs split-phase interface (read_start / write_start / poll /
  finish, one transfer in flight, -495 on a blocking call while one is
  pending); usb.rs runs the data phase and CSW on the channel DMA,
  mockblk.rs/usbdev.rs the payload and response on the endpoint DMA.
- Overlap sites: encoder attention (context write + next head K/V/Q
  prefetch through a small IoQueue pumped every 16 queries), encoder
  MLP tile pipeline (blob load split from the NPU run so the transfer
  starts between them), decode cross-attention K/V prefetch, LM head
  embedding-chunk double buffer.
- Rig (JFK clip): transcript identical, 0 of 5480 scratch blocks differ
  from a same-session baseline; speak-to-done 67.7 -> 63.8 s, decode
  1.10 -> 0.97 s/step. Stick projection ~15 s of ~100 s (speedup.md 12).
- Stick (new PNY USB 3.0, clip played into the mic, ctx 531): encoder
  37.5 s, decode 0.80 s/step, end of speech to transcript 62.6 s; the
  silence stop worked live (2.0 s tail). First-read budget raised to
  2 s and the index read retried (the stick timed out its first read).
