/* nRF54LM20B memory map with the Whisper layer slot carved out.
 *
 * Confirmed from the nRF54LM20B MDK (see ../../npu/memory.x for provenance):
 *   FLASH (RRAM): base 0x00000000, size 2036 KB
 *   RAM: 0x20000000, 510 KB usable (the top ~512 B bus-fault on the DK).
 *
 * SLOT is the runtime-loaded Axon model slot: the host streams per-layer
 * blobs (weights + command buffer + descriptor) into it over SWD. Blobs are
 * linked offline at exactly this address (tools/slot.ld) against the firmware
 * ELF, so the three addresses below and slot.ld MUST stay in sync, and blobs
 * must be regenerated whenever the firmware ELF changes.
 *
 * cortex-m-rt places the initial SP at the end of RAM, i.e. the stack grows
 * down from the bottom of SLOT.
 */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 2036K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 300K
  SLOT (rw)  : ORIGIN = 0x2004B000, LENGTH = 208K
}
