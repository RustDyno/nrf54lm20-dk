/* Layer-blob link script: place one compiled Axon model at the firmware's
 * SLOT region so every embedded pointer is absolute. Must match memory.x's
 * SLOT and src/slot.rs. The slot header (magic + descriptor pointer) is
 * pinned to the first bytes of the blob.
 */
MEMORY
{
  SLOT (rwx) : ORIGIN = 0x2004B000, LENGTH = 208K
}

SECTIONS
{
  .slot ORIGIN(SLOT) :
  {
    KEEP(*(.slot.header))
    *(.rodata*)
    *(.data*)
    . = ALIGN(4);
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
  } > SLOT

  /DISCARD/ : { *(.comment) *(.ARM.attributes) *(.ARM.exidx*) *(.debug*) }
}
