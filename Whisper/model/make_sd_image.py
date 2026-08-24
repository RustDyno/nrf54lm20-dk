"""Build the raw SD card image: every blob and decoder asset at a
block-aligned offset behind a simple index. No filesystem.

Layout (512-byte blocks):
  block 0..15   header: magic "WSPRIMG1", u32 version, u32 entry count,
                then 32-byte entries {name[24] zero-padded, u32 offset_blocks,
                u32 length_bytes} -- up to 254 entries
  block 16..    payloads, each starting on a block boundary

Write it to a card with a USB reader (fastest):
    sudo dd if=out/sd.img of=/dev/sdX bs=4M conf=fsync
or stream it through the DK (slow, ~10 min): whisper-host sdwrite.

Also emits out/tape-sdtest/tape.json: an SD smoke test (init, write a
pattern past the image, read it back, compare) for the tape player.
"""

import json
import os
import struct

import numpy as np

import common

BLOCK = 512
HEADER_BLOCKS = 16
CMD_SD_INIT, CMD_SD_READ, CMD_SD_WRITE = 20, 21, 22
ARENA_BASE = 0x2003_2000


def main():
    out = common.OUT
    entries = []  # (name, bytes)

    blobs = sorted(os.listdir(os.path.join(out, "blobs")))
    for b in blobs:
        if b.endswith(".bin"):
            with open(os.path.join(out, "blobs", b), "rb") as f:
                entries.append((b[:-4], f.read()))

    plan_dir = os.path.join(out, "decoder-plan")
    for name in sorted(os.listdir(plan_dir)):
        with open(os.path.join(plan_dir, name), "rb") as f:
            entries.append((name, f.read()))

    assert len(entries) <= (HEADER_BLOCKS * BLOCK - 16) // 32, "index full"
    for name, _ in entries:
        assert len(name) < 24, f"name too long: {name}"

    index = struct.pack("<8sII", b"WSPRIMG1", 1, len(entries))
    off_blocks = HEADER_BLOCKS
    payload = bytearray()
    for name, data in entries:
        index += struct.pack("<24sII", name.encode(), off_blocks, len(data))
        payload += data
        pad = (-len(data)) % BLOCK
        payload += b"\0" * pad
        off_blocks += (len(data) + pad) // BLOCK

    img = bytearray(index)
    img += b"\0" * (HEADER_BLOCKS * BLOCK - len(img))
    img += payload
    path = os.path.join(out, "sd.img")
    with open(path, "wb") as f:
        f.write(img)
    with open(os.path.join(out, "sd-index.json"), "w") as f:
        json.dump({name: {"lba": HEADER_BLOCKS + sum(
            (len(d) + (-len(d)) % BLOCK) // BLOCK for _, d in entries[:i]),
            "bytes": len(data)}
            for i, (name, data) in enumerate(entries)}, f, indent=1)
    print(f"{path}: {len(img) / 1e6:.1f} MB, {len(entries)} entries "
          f"({off_blocks} blocks)")

    # --- SD smoke-test tape -------------------------------------------------
    tdir = os.path.join(out, "tape-sdtest")
    os.makedirs(tdir, exist_ok=True)
    rng = np.random.default_rng(11)
    pattern = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    zeros = bytes(4096)
    for fn, data in [("pattern.bin", pattern), ("zeros.bin", zeros)]:
        with open(os.path.join(tdir, fn), "wb") as f:
            f.write(data)
    test_lba = off_blocks + 1024  # comfortably past the image
    steps = [
        {"op": "cmd", "code": CMD_SD_INIT, "args": []},
        {"op": "write", "file": "pattern.bin", "addr": ARENA_BASE},
        {"op": "cmd", "code": CMD_SD_WRITE, "args": [test_lba, ARENA_BASE, 8]},
        {"op": "write", "file": "zeros.bin", "addr": ARENA_BASE},
        {"op": "cmd", "code": CMD_SD_READ, "args": [test_lba, ARENA_BASE, 8]},
        {"op": "check", "file": "pattern.bin", "addr": ARENA_BASE,
         "label": "sd-roundtrip", "tol": 0},
        # header readback: the image magic should be present at block 0
        {"op": "cmd", "code": CMD_SD_READ, "args": [0, ARENA_BASE, 1]},
        {"op": "check", "file": "magic.bin", "addr": ARENA_BASE,
         "label": "sd-image-magic", "tol": 0},
    ]
    with open(os.path.join(tdir, "magic.bin"), "wb") as f:
        f.write(img[:16])
    with open(os.path.join(tdir, "tape.json"), "w") as f:
        json.dump({"steps": steps}, f, indent=1)
    print(f"sd test tape -> {tdir} (test lba {test_lba})")


if __name__ == "__main__":
    main()
