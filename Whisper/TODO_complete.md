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
