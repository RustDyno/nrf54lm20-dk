"""Int-exact 4-bit weight coding, shared by the image builder and the
quality gate, and mirrored (integer for integer) by the firmware
unpacker. Keep the three in lockstep.

Two codings share the nibble arithmetic:

Legacy (LM-head chunks "embc4"): groups of G=64 consecutive int8 weights
share one u8 scale byte, the group's max |w| (amax). Codes are nibbles
-7..7 stored biased (+8, so 1..15; 0 never emitted):
    nib = sign(w) * min(7, (|w|*14 + amax) // (2*amax))      (amax>0)
    w'  = sign(nib) * min(127, (|nib|*amax*2 + 7) // 14)
Both are round-half-away-from-zero in pure integers, so numpy here and
i32 math on the M33 agree bit for bit. |w'| <= amax <= 127; the min(127)
only guards the never-emitted nibble 0 (-8) slot of the device LUT.

Level-coded (decoder blobs "LAY5"): groups of GL=16 weights, and the
group scale is an ODD value 1..127 stored as a 6-bit code c = scale >> 1
(scale = 2c + 1). The same nib / w' formulas apply with the scale in
place of amax; the packer picks, per group, the odd scale at or below
max|w| with the least squared reconstruction error (clipping one or two
large weights often beats a coarser step for the rest). The 64 possible
scales let the firmware expand two weights per table lookup from a
64 x 256 table of int8 pairs (32 KB) instead of one nibble at a time.

Legacy packed blob entry (little-endian u32 header words), no longer
written by the image builder but still accepted by the firmware:
    "LAY4" | raw_len | w_off | n_weights | raw_sum
    raw[0..w_off] verbatim | amax[n/64] u8 | nibbles[n/2]

Level-coded blob entry:
    "LAY5" | raw_len | w_off | n_weights | raw_sum | base
    raw[0..w_off] verbatim | codes[n/GL] u8 | nibbles[n/2]
base is the RAM address the raw blob was linked at (where the firmware
expands it); one weight region per blob, ending at raw_len;
n_weights % 64 == 0; raw_sum = wrapping u32 byte-sum of head +
reconstructed weights, which the firmware verifies after expanding.

The blobs a LAY5 entry is made from must have been COMPILED from the
requantized weights: the Axon compiler folds -zp_in * sum(w) per output
channel into the command stream's bias words, so patching only the
filter bytes of an int8 blob leaves that bias stale (a broad 1-2 LSB
per-channel error that byte sums cannot see). pack_blob checks that the
blob's filter bytes are already on the 4-bit grid.

Packed embedding ("embc4"): rows of 384 in chunks of 64 rows; per chunk
the rows' f32 scales (64), their u32 token ids (64), then amax[64*6] and
nibbles[64*192], padded to whole 512-byte blocks (26 per chunk) so one
block-aligned read brings everything the LM head needs for the chunk.
Pad rows are zero with scale 0 (skipped, like the -1 input-only marker).
"""

import os
import struct

import numpy as np

G = 64
# level-coded group (Q4_GL overrides for experiments; the firmware's
# q4::GL must match what the image was built with)
GL = int(os.environ.get("Q4_GL", "16"))
BLOCK = 512
MAGIC = 0x3459414C  # "LAY4"
MAGIC5 = 0x3559414C  # "LAY5"
HDR5 = 24
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


def _nib(w, a):
    """Nearest 4-bit code of int32 w [.., GL] under odd scale a [.., 1]."""
    return np.sign(w) * np.minimum(7, (np.abs(w) * 14 + a) // (2 * a))


def _rec(nib, a):
    return np.sign(nib) * np.minimum(127, (np.abs(nib) * a * 2 + 7) // 14)


# Scales tried per group, as fractions of max|w|: the smallest squared
# reconstruction error wins (clipping the largest weight or two is often
# cheaper than a coarse step for the other thirty). 1.0 alone is the
# plain amax coding.
SCALE_SEARCH = np.arange(1.0, 0.69, -0.02)


def encode_lvl(w8):
    """int8[n] (n % 64 == 0) -> (code u8[n/GL], nibble bytes u8[n/2]),
    level-coded: an odd scale per group (code = scale >> 1) chosen by
    squared-error search below max|w|."""
    w = np.asarray(w8, np.int32).reshape(-1, GL)
    amax = np.abs(w).max(1)
    best_a = np.maximum(amax, 1) | 1
    best_err = None
    for f in SCALE_SEARCH:
        a = np.maximum(np.ceil(amax * f).astype(np.int32), 1) | 1
        a = np.minimum(a, 127)
        ac = a[:, None]
        err = ((_rec(_nib(w, ac), ac) - w) ** 2).sum(1)
        if best_err is None:
            best_err, best_a = err, a
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_a = np.where(better, a, best_a)
    a = best_a[:, None]
    nib = _nib(w, a)
    assert np.abs(nib).max() <= 7
    k = (nib + 8).astype(np.uint8).reshape(-1, 2)
    return (best_a >> 1).astype(np.uint8), (k[:, 0] | (k[:, 1] << 4))


def encode_lvl_exact(w8):
    """encode_lvl for weights already on the 4-bit grid (a compiled blob's
    filter): every group must reproduce exactly. The error search does
    not always revisit the scale that produced a group (its max|w| may sit
    below that scale), so groups the search misses are rescanned over all
    odd scales for an exact match."""
    w = np.asarray(w8, np.int32).reshape(-1, GL)
    codes, nibs = encode_lvl(w)
    rec = decode_lvl(codes, nibs).reshape(-1, GL)
    bad = np.nonzero((rec != w).any(1))[0]
    for g in bad:
        for a in range(1, 128, 2):
            r = _rec(_nib(w[g], a), a)
            if np.array_equal(r, w[g]):
                codes[g] = a >> 1
                k = (_nib(w[g], a) + 8).astype(np.uint8).reshape(-1, 2)
                nibs[g * GL // 2:(g + 1) * GL // 2] = k[:, 0] | (k[:, 1] << 4)
                break
        else:
            raise AssertionError(f"group {g} is not on the 4-bit grid")
    assert np.array_equal(decode_lvl(codes, nibs).reshape(-1, GL), w)
    return codes, nibs


def decode_lvl(codes, nibs):
    """Inverse of encode_lvl: the exact int8 values the device expands."""
    k = np.empty(nibs.size * 2, np.int32)
    k[0::2] = nibs & 15
    k[1::2] = nibs >> 4
    nib = k - 8
    a = np.repeat(codes.astype(np.int32) * 2 + 1, GL)
    v = np.sign(nib) * np.minimum(127, (np.abs(nib) * a * 2 + 7) // 14)
    return v.astype(np.int8)


def requant_lvl(w8):
    """int8 array -> nearest level-coded int8 (same shape): what to put
    in the tflite before compiling the blob."""
    flat = np.asarray(w8, np.int8).reshape(-1)
    assert flat.size % (2 * GL) == 0, flat.size
    return decode_lvl(*encode_lvl(flat)).reshape(np.asarray(w8).shape)


def pack_blob_lvl(raw, w_off, base):
    """Compiled blob bytes (filters already level-coded) + weight-region
    offset + link address -> "LAY5" entry bytes."""
    n = len(raw) - w_off
    assert n > 0 and n % (2 * GL) == 0, (len(raw), w_off)
    w = np.frombuffer(raw[w_off:], np.int8)
    codes, nibs = encode_lvl_exact(w)
    raw_sum = sum(raw) & 0xFFFFFFFF
    return (struct.pack("<IIIIII", MAGIC5, len(raw), w_off, n, raw_sum, base)
            + raw[:w_off] + codes.tobytes() + nibs.tobytes())


def table_lvl():
    """The firmware's expansion table: u16[64][256], entry = low byte the
    weight of the low nibble, high byte the weight of the high nibble."""
    t = np.empty((64, 256), np.uint16)
    for c in range(64):
        a = 2 * c + 1
        lut = np.array([np.sign(n) * min(127, (abs(n) * a * 2 + 7) // 14)
                        for n in range(-8, 8)], np.int32).astype(np.int8)
        lo = lut[np.arange(256) & 15].astype(np.uint8).astype(np.uint16)
        hi = lut[np.arange(256) >> 4].astype(np.uint8).astype(np.uint16)
        t[c] = lo | (hi << 8)
    return t


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

    # level-coded vectors: n | codes | nibbles | reconstruction, plus the
    # expansion table the firmware must build identically
    blocks = []
    for i in range(64):
        n = int(rng.integers(1, 40)) * 2 * GL
        w = rng.integers(-127, 128, n).astype(np.int8)
        if i % 4 == 0:
            w[:GL * (i % 8 + 1)] = 0
        if i % 5 == 0:
            w[GL:2 * GL] = rng.integers(-3, 4, GL)  # tiny amax
        codes, nibs = encode_lvl(w)
        rec = decode_lvl(codes, nibs)
        assert np.array_equal(rec, requant_lvl(w))
        # a gridded array packs back to itself (the packer's contract)
        assert np.array_equal(decode_lvl(*encode_lvl_exact(rec)), rec)
        # the searched scale never loses to the plain amax coding
        w32 = w.astype(np.int32).reshape(-1, GL)
        a = (np.abs(w32).max(1) | 1)[:, None]
        plain = _rec(_nib(w32, a), a)
        assert np.abs(plain - w32).max() <= 10
        sse = ((rec.astype(np.int32).reshape(-1, GL) - w32) ** 2).sum(1)
        assert np.all(sse <= ((plain - w32) ** 2).sum(1))
        blocks.append(struct.pack("<I", n) + codes.tobytes() + nibs.tobytes()
                      + rec.tobytes())
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                        "q4lvec.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(blocks)) + b"".join(blocks)
                + table_lvl().astype("<u2").tobytes())
    print(f"{path}: {len(blocks)} level-coded vectors + table, self-test ok")
