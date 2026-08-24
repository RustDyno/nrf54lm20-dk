/* nRF54LM20B memory map with the Whisper layer slot and arena carved out.
 *
 * Confirmed from the nRF54LM20B MDK (see ../../npu/memory.x for provenance):
 *   FLASH (RRAM): base 0x00000000, size 2036 KB
 *   RAM: 0x20000000, 510 KB usable (the top ~512 B bus-fault on the DK).
 *
 * SLOT is the runtime-loaded Axon model slot; ARENA holds activations and
 * kernel parameters. Both are at FIXED addresses so that (a) blobs can be
 * linked offline at the slot address (tools/slot.ld) and (b) the tape
 * generator (model/tape.py) can emit absolute addresses. These origins are
 * mirrored in src/slot.rs, src/main.rs, tools/slot.ld, and model/tape.py --
 * keep all of them in sync. Blobs must be regenerated whenever the firmware
 * ELF changes.
 *
 * cortex-m-rt places the initial SP at the end of RAM, i.e. the stack grows
 * down toward the interlayer buffer in .bss.
 */
MEMORY
{
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 2036K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 200K
  ARENA (rw) : ORIGIN = 0x20032000, LENGTH = 100K
  SLOT (rw)  : ORIGIN = 0x2004B000, LENGTH = 208K
}

SECTIONS
{
  .arena (NOLOAD) : { KEEP(*(.arena)) } > ARENA
}
