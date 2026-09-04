"""Block-for-block comparison of two mock-rig work images.

Two runs of the same clip through two firmware builds should leave the
same scratch behind wherever the arithmetic is unchanged. This reports,
per scratch region, how many 512-byte blocks and bytes differ, so a
kernel rewrite that claims bit-exactness can be held to it on silicon.

    pixi run python mock_diff.py before.img after.img
"""

import struct
import sys

import numpy as np

BLOCK = 512
HEADER_BLOCKS = 16

# Scratch layout, mirrored from firmware/src/app.rs (block offsets, blocks).
REGIONS = [
    ("mel8", 1600, 200),
    ("x16", 2880, 960),
    ("enc_out", 8960, 480),
    ("xkv", 9472, 8 * 480),
]


def scratch_lba(path):
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BLOCKS * BLOCK)
    if hdr[:8] != b"WSPRIMG1":
        raise SystemExit(f"{path}: not a Whisper model image")
    n = struct.unpack_from("<I", hdr, 12)[0]
    for i in range(n):
        e = 16 + i * 32
        if hdr[e:e + 24].rstrip(b"\0") == b"plan":
            lba = struct.unpack_from("<I", hdr, e + 24)[0]
            with open(path, "rb") as f:
                f.seek(lba * BLOCK)
                p = f.read(16)
            return struct.unpack_from("<I", p, 8)[0]
    raise SystemExit(f"{path}: no plan entry")


def region(path, base, off, blocks):
    with open(path, "rb") as f:
        f.seek((base + off) * BLOCK)
        return np.frombuffer(f.read(blocks * BLOCK), np.uint8)


def main():
    a, b = sys.argv[1], sys.argv[2]
    sa, sb = scratch_lba(a), scratch_lba(b)
    print(f"scratch at block {sa} / {sb}")
    for name, off, blocks in REGIONS:
        ra, rb = region(a, sa, off, blocks), region(b, sb, off, blocks)
        if len(ra) != len(rb):
            print(f"{name:8s}: size mismatch")
            continue
        diff = ra != rb
        nblk = int(diff.reshape(-1, BLOCK).any(1).sum())
        nbytes = int(diff.sum())
        line = f"{name:8s}: {nblk}/{blocks} blocks, {nbytes} bytes differ"
        if nbytes:
            d = np.abs(ra.astype(np.int16).view(np.int16) if False else
                       ra.astype(np.int8).astype(np.int16)
                       - rb.astype(np.int8).astype(np.int16))
            line += f", max |diff| {int(d.max())} (as int8)"
        print(line)


if __name__ == "__main__":
    main()
