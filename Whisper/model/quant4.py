"""Int-exact 4-bit weight coding, shared by the image builder and the
quality gate, and mirrored (integer for integer) by the firmware
unpacker. Keep the three in lockstep.

Groups of G=64 consecutive int8 weights share one u8 scale byte: the
group's max |w| (amax). Codes are nibbles -7..7 stored biased (+8, so
1..15; 0 never emitted):
    nib = sign(w) * min(7, (|w|*14 + amax) // (2*amax))      (amax>0)
    w'  = sign(nib) * min(127, (|nib|*amax*2 + 7) // 14)
Both are round-half-away-from-zero in pure integers, so numpy here and
i32 math on the M33 agree bit for bit. |w'| <= amax <= 127; the min(127)
only guards the never-emitted nibble 0 (-8) slot of the device LUT.

Packed blob entry (little-endian u32 header words):
    "LAY4" | raw_len | w_off | n_weights | raw_sum
    raw[0..w_off] verbatim | amax[n/64] u8 | nibbles[n/2]
(one weight region per blob, ending at raw_len; n_weights % 128 == 0;
raw_sum = wrapping u32 byte-sum of head + reconstructed weights, which
the firmware verifies after expanding).

Packed embedding ("embc4"): rows of 384 in chunks of 64 rows; per chunk
the rows' f32 scales (64), their u32 token ids (64), then amax[64*6] and
nibbles[64*192], padded to whole 512-byte blocks (26 per chunk) so one
block-aligned read brings everything the LM head needs for the chunk.
Pad rows are zero with scale 0 (skipped, like the -1 input-only marker).
"""

import struct

import numpy as np

G = 64
BLOCK = 512
MAGIC = 0x3459414C  # "LAY4"
EMB_ROW = 384
EMB_CHUNK_ROWS = 64
EMB_CHUNK_BLOCKS = 26
EMB_CHUNK_BYTES = EMB_CHUNK_BLOCKS * BLOCK
assert (EMB_CHUNK_ROWS * (4 + 4 + EMB_ROW // G + EMB_ROW // 2)
        <= EMB_CHUNK_BYTES)


def encode(w8):
    """int8[n] (n % 128 == 0) -> (amax u8[n/G], nibble bytes u8[n/2])."""
    w = np.asarray(w8, np.int32).reshape(-1, G)
    amax = np.abs(w).max(1)
    a = amax[:, None]
    nib = np.sign(w) * np.minimum(
        7, (np.abs(w) * 14 + a) // np.maximum(2 * a, 1))
    k = (nib + 8).astype(np.uint8).reshape(-1, 2)
    return amax.astype(np.uint8), (k[:, 0] | (k[:, 1] << 4))


def decode(amax, nibs):
    """Inverse of encode: the exact int8 values the device reconstructs."""
    k = np.empty(nibs.size * 2, np.int32)
    k[0::2] = nibs & 15
    k[1::2] = nibs >> 4
    nib = k - 8
    a = np.repeat(amax.astype(np.int32), G)
    v = np.sign(nib) * np.minimum(127, (np.abs(nib) * a * 2 + 7) // 14)
    return v.astype(np.int8)


def requant(w8):
    """int8 array -> nearest 4-bit-coded int8 (same shape), for patching
    the interpreter mirrors with exactly what the device will run."""
    flat = np.asarray(w8, np.int8).reshape(-1)
    assert flat.size % (2 * G) == 0, flat.size
    return decode(*encode(flat)).reshape(np.asarray(w8).shape)


def pack_blob(raw, w_off):
    """Blob bytes + weight-region offset -> packed entry bytes."""
    n = len(raw) - w_off
    assert n > 0 and n % (2 * G) == 0, (len(raw), w_off)
    amax, nibs = encode(np.frombuffer(raw[w_off:], np.int8))
    raw_sum = (sum(raw[:w_off]) + int(decode(amax, nibs).view(np.uint8)
                                      .astype(np.uint32).sum())) & 0xFFFFFFFF
    return (struct.pack("<IIIII", MAGIC, len(raw), w_off, n, raw_sum)
            + raw[:w_off] + amax.tobytes() + nibs.tobytes())


def pack_emb(q, scl, ids):
    """int8 [n, 384] pruned embedding + f32 row scales + u32 ids ->
    "embc4" entry bytes (firmware app.rs lm_head mirrors the layout)."""
    n = q.shape[0]
    rows = -(-n // EMB_CHUNK_ROWS) * EMB_CHUNK_ROWS
    qp = np.zeros((rows, EMB_ROW), np.int8)
    qp[:n] = q
    sp = np.zeros(rows, "<f4")
    sp[:n] = scl
    ip = np.zeros(rows, "<u4")
    ip[:n] = ids
    out = bytearray()
    for c in range(rows // EMB_CHUNK_ROWS):
        sl = slice(c * EMB_CHUNK_ROWS, (c + 1) * EMB_CHUNK_ROWS)
        amax, nibs = encode(qp[sl].reshape(-1))
        part = (sp[sl].tobytes() + ip[sl].tobytes() + amax.tobytes()
                + nibs.tobytes())
        assert len(part) <= EMB_CHUNK_BYTES
        out += part + b"\0" * (EMB_CHUNK_BYTES - len(part))
    return bytes(out)


if __name__ == "__main__":
    # Self-test plus cross-language vectors: tools/q4check compiles the
    # REAL firmware unpacker (q4.rs) and replays these bit for bit.
    import os

    rng = np.random.default_rng(4)
    blocks = []
    for i in range(64):
        n = int(rng.integers(1, 40)) * 2 * G
        w = rng.integers(-127, 128, n).astype(np.int8)
        if i % 4 == 0:
            w[:G * (i % 8 + 1)] = 0  # all-zero groups (amax == 0)
        amax, nibs = encode(w)
        rec = decode(amax, nibs)
        assert np.array_equal(rec, requant(w))
        err = np.abs(rec.astype(np.int32) - w.astype(np.int32)).max()
        assert err <= 10, err  # half a step: <= ceil(127/14)
        blocks.append(struct.pack("<I", n) + amax.tobytes() + nibs.tobytes()
                      + rec.tobytes())
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                        "q4vec.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(blocks)) + b"".join(blocks))
    print(f"{path}: {len(blocks)} vectors, self-test ok")
