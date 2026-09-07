"""Build the raw SD card image for the standalone firmware.

Layout (512-byte blocks):
  block 0..15   header: magic "WSPRIMG1", u32 version, u32 entry count,
                then 32-byte entries {name[24] zero-padded, u32 offset_blocks,
                u32 length_bytes} -- up to 254 entries
  block 16..    payloads, each starting on a block boundary

Contents: all 116 blobs, the decoder assets, encoder-side assets (LN
params, GELU LUTs, positional encodings, mel tables), the pruned
vocabulary tables, and the binary plan consumed by firmware app.rs
(Plan::load and write_plan below are ONE contract: same field order).

The vocabulary is pruned to ids < VOCAB_KEEP (GPT-2 BPE ids are roughly
frequency-ordered, so low ids cover common English) minus whisper's
suppress set, plus EOT; SOT-sequence tokens ride along for input
embedding only (marked with a negative row scale so argmax skips them).

Write the image with a USB reader:
    sudo dd if=out/sd.img of=/dev/sdX bs=4M conv=fsync
"""

import json
import os
import struct
import subprocess

import numpy as np

import common
import decode_model
import quant4
import tape
from tape import Submodel

BLOCK = 512
HEADER_BLOCKS = 16
CMD_SD_INIT, CMD_SD_READ, CMD_SD_WRITE = 20, 21, 22
ARENA_BASE = 0x2003_2000
SLOT_BYTES = 208 * 1024  # must match firmware memory.x SLOT
# firmware app.rs decode blob pipeline: per-token blobs are linked at the
# top DEC_BYTES of the slot and their packed entries land in LAND_BYTES
DEC_BYTES = 152 * 1024
DEC_BASE = 0x2004_B000 + SLOT_BYTES - DEC_BYTES  # 0x20059000
LAND_BYTES = 128 * 1024
PLAN_MAGIC = 0x4E4C_5057  # "WPLN"
VOCAB_KEEP = 12288
MAX_TOKENS = 32


def pq(q):
    return struct.pack("<fi", float(q[0]), int(q[1]))


def build_vocab(sd, ref, suppress, tok):
    """Pruned embedding tables + id map + printable pieces."""
    sot = [int(t) for t in ref["sot_sequence"]]
    eot = int(ref["eot"])
    allowed = sorted((set(range(VOCAB_KEEP)) - set(int(s) for s in suppress))
                     | {eot})
    kept = sorted(set(allowed) | set(sot))
    input_only = set(kept) - set(allowed)
    ids = np.array(kept, dtype=np.uint32)
    for t in [int(x) for x in ref["tokens"]]:
        assert t in set(allowed), f"reference token {t} pruned away"

    emb = sd["decoder.token_embedding.weight"][kept]  # [n, 384] f32
    amax = np.maximum(np.abs(emb).max(1), 1e-9)
    scl = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(emb / scl[:, None]), -127, 127).astype(np.int8)
    for i, t in enumerate(kept):
        if t in input_only:
            scl[i] = -1.0  # argmax skip marker (input embedding only)

    pieces = [tok.decode([t]) for t in kept]
    blob = b""
    offs = [0]
    for p in pieces:
        blob += p.encode()
        offs.append(len(blob))
    vocabtb = struct.pack("<I", len(kept)) \
        + b"".join(struct.pack("<I", o) for o in offs) + blob
    return {
        "embp": q.tobytes(),
        "embpscl": scl.astype("<f4").tobytes(),
        "embpids": ids.astype("<u4").tobytes(),
        "embf": emb.astype("<f4").tobytes(),
        "vocabtb": vocabtb,
    }, len(kept)


def filter_bytes(tflite_path):
    """The bytes of a submodel's one 384x384 int8 filter tensor."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    for d in interp.get_tensor_details():
        if d["dtype"] == np.int8 and int(np.prod(d["shape"])) == 384 * 384:
            try:
                return interp.get_tensor(d["index"]).tobytes()
            except ValueError:
                pass
    raise AssertionError(f"no filter tensor in {tflite_path}")


def pack_emb8(q, scl, ids):
    """int8 [n, 384] pruned embedding + f32 row scales + u32 ids -> "embc8"
    entry: 64-row chunks of [scales][ids][rows], 49 blocks each (firmware
    app.rs lm_head mirrors the layout). The same rows the 4-bit chunks
    are coded from, uncoded: twice the read per token, no nibble unpack.
    The firmware prefers this entry when present."""
    n = q.shape[0]
    rows = -(-n // 64) * 64
    qp = np.zeros((rows, 384), np.int8)
    qp[:n] = q
    sp = np.zeros(rows, "<f4")
    sp[:n] = scl
    ip = np.zeros(rows, "<u4")
    ip[:n] = ids
    out = bytearray()
    for c in range(rows // 64):
        sl = slice(c * 64, (c + 1) * 64)
        part = sp[sl].tobytes() + ip[sl].tobytes() + qp[sl].tobytes()
        assert len(part) == 49 * BLOCK
        out += part
    return bytes(out)


def encoder_assets(sd, scales, mq_enc, conv1, conv2):
    """LN params, GELU LUTs, tile-major positional encoding, mel tables."""
    a = {}
    for l in range(common.N_LAYERS):
        p = f"encoder.blocks.{l}."
        for stem, w in [("ln1", "attn_ln"), ("ln2", "mlp_ln")]:
            a[f"e{l}{stem}_gb"] = (sd[p + w + ".weight"].astype("<f4").tobytes()
                                   + sd[p + w + ".bias"].astype("<f4").tobytes())
        for j in range(4):
            m = mq_enc[l]
            a[f"e{l}lut{j}"] = tape.gelu_lut(
                m[f"fc1{'abcd'[j]}"].out_q, m[f"fc2p{j}"].in_q).tobytes()
    a["lnpost_gb"] = (sd["encoder.ln_post.weight"].astype("<f4").tobytes()
                      + sd["encoder.ln_post.bias"].astype("<f4").tobytes())
    a["g1lut"] = tape.gelu_lut(conv1.out_q, conv2[0].in_q).tobytes()
    g2q = scales["enc.gelu2"]
    for i in range(3):
        a[f"g2lut{i}"] = tape.gelu_lut(conv2[i].out_q, g2q[:2]).tobytes()

    # positional encoding, TILE-MAJOR f32 [384, 64] chunks (pad cols zero)
    pos = np.zeros((common.STATE, tape.PAD_W), np.float32)
    pos[:, :common.AUDIO_CTX] = \
        sd["encoder.positional_embedding"][:common.AUDIO_CTX].T
    tiles = [np.ascontiguousarray(pos[:, i * 64:(i + 1) * 64])
             for i in range(tape.PAD_W // 64)]
    a["posenc"] = b"".join(t.astype("<f4").tobytes() for t in tiles)

    hann, cos_tab, filt = tape.mel_tables()
    a["hann"] = hann.astype("<f4").tobytes()
    a["melcos"] = cos_tab.astype("<f4").tobytes()
    a["melfilt"] = np.ascontiguousarray(filt).astype("<f4").tobytes()
    a["posdec"] = sd["decoder.positional_embedding"].astype("<f4").tobytes()
    return a


def write_plan(scales, mq_enc, mq_dec, conv1, conv2, ref, scratch_lba,
               vocab_n):
    sot = [int(t) for t in ref["sot_sequence"]]
    blank = [int(t) for t in ref["blank"]]
    b = struct.pack("<II", PLAN_MAGIC, 1)
    b += struct.pack("<II", scratch_lba, vocab_n)
    b += struct.pack("<I4I", len(sot), *(sot + [0] * (4 - len(sot))))
    b += struct.pack("<I", int(ref["eot"]))
    b += struct.pack("<I4I", len(blank), *(blank + [0] * (4 - len(blank))))
    b += pq(conv1.in_q) + pq(conv2[0].in_q) + pq(scales["enc.gelu2"][:2])
    b += pq(scales["enc.x"][:2]) + pq(scales["enc.out"][:2])
    b += pq(mq_dec[0]["xk"].in_q) + pq(scales["dec.x"][:2])
    for l in range(common.N_LAYERS):
        m = mq_enc[l]
        b += pq(m["q"].in_q) + pq(m["q"].out_q) + pq(m["k"].out_q) \
            + pq(m["v"].out_q) + pq(m["out"].in_q) + pq(m["out"].out_q) \
            + pq(scales[f"enc.b{l}.res1"][:2]) + pq(m["fc1a"].in_q)
        for j in range(4):
            b += pq(m[f"fc2p{j}"].out_q)
        b += pq(scales[f"enc.b{l}.res2"][:2])
    for l in range(common.N_LAYERS):
        m = mq_dec[l]
        b += pq(m["q"].in_q) + pq(m["q"].out_q) + pq(m["k"].out_q) \
            + pq(m["v"].out_q) + pq(m["out"].in_q) + pq(m["out"].out_q) \
            + pq(scales[f"dec.b{l}.res1"][:2]) + pq(m["xq"].in_q) \
            + pq(m["xq"].out_q) + pq(m["xk"].out_q) + pq(m["xv"].out_q) \
            + pq(m["xout"].in_q) + pq(m["xout"].out_q) \
            + pq(scales[f"dec.b{l}.res2"][:2]) + pq(m["fc1a"].in_q)
        for j in range(4):
            b += pq(m[f"fc2p{j}"].out_q)
        b += pq(scales[f"dec.b{l}.res3"][:2])
    return b


def main():
    out = common.OUT
    _, sd = common.load_weights()
    with open(os.path.join(out, "scales.json")) as f:
        scales = {k: (v["scale"], v["zp"], v["bits"])
                  for k, v in json.load(f).items()}
    ref = np.load(os.path.join(out, "ref.npz"))
    suppress, tok = decode_model.build_suppress(ref)

    conv1 = Submodel("wconv1")
    conv2 = [Submodel(f"wconv2{p}") for p in "abc"]
    mq_enc = [{k: Submodel(v) for k, v in common.submodel_names(l).items()}
              for l in range(common.N_LAYERS)]
    mq_dec = [{k: Submodel(v)
               for k, v in common.decoder_submodel_names(l).items()}
              for l in range(common.N_LAYERS)]
    for l in range(common.N_LAYERS):
        assert mq_dec[l]["xk"].in_q == mq_dec[0]["xk"].in_q

    entries = []  # (name, bytes)
    for b in sorted(os.listdir(os.path.join(out, "blobs"))):
        if b.endswith(".bin"):
            with open(os.path.join(out, "blobs", b), "rb") as f:
                entries.append((b[:-4], f.read()))
    plan_dir = os.path.join(out, "decoder-plan")
    for name in sorted(os.listdir(plan_dir)):
        if name in ("plan.json", "vocab.json", "emb.f32.bin", "emb.i8.bin",
                    "emb_scales.f32.bin", "pos.f32.bin", "enc_out.i8.bin",
                    "suppress.u32.bin"):
            continue  # replaced by pruned/standalone equivalents
        with open(os.path.join(plan_dir, name), "rb") as f:
            entries.append((name, f.read()))

    # one entry per decoder layer with its seven small constants (ln1,
    # xln, ln2 gamma/beta, four GELU tables): the firmware reads it once
    # per layer per token instead of seven times
    for l in range(common.N_LAYERS):
        parts = [f"b{l}_ln1_gb.bin", f"b{l}_xln_gb.bin", f"b{l}_ln2_gb.bin"] + \
            [f"b{l}_lut{j}.bin" for j in range(4)]
        bundle = b""
        for name in parts:
            with open(os.path.join(plan_dir, name), "rb") as f:
                d = f.read()
            assert len(d) == (256 if "lut" in name else 3072), name
            bundle += d
        entries.append((f"dc{l}", bundle))

    vocab, vocab_n = build_vocab(sd, ref, suppress, tok)
    assert vocab_n <= 16384, "row scales must fit the interlayer buffer"
    entries += sorted(vocab.items())
    entries += sorted(encoder_assets(sd, scales, mq_enc, conv1, conv2).items())

    # The per-token decoder blobs ship 4-bit ("LAY5", quant4.py): the
    # level-coded weights are requantized in the tflite (quant4_gate.py ->
    # out/submodels-q4l) and the blob recompiled from that (compile_q4l.sh
    # -> out/blobs-q4l, linked at DEC_BASE), because the Axon compiler
    # folds -zp_in * sum(w) per output channel into the command stream's
    # bias words: patching the filter bytes of the int8 blob would leave
    # that term stale (the 2026-09-01 constant-token decode). The firmware
    # reads the packed entry into a landing zone and expands it at
    # DEC_BASE while the NPU runs the previous blob. Q4L=0 ships the int8
    # blobs instead (the firmware runs them unpipelined at the slot base).
    per_token = {}
    for l in range(common.N_LAYERS):
        for k, name in common.decoder_submodel_names(l).items():
            if k not in ("xk", "xv"):
                per_token[name] = mq_dec[l][k]
    packed_n = 0
    saved = 0
    pack_q4 = os.environ.get("Q4L", "1") != "0"
    for i, (name, data) in enumerate(entries):
        if name not in per_token or not pack_q4:
            continue
        with open(os.path.join(out, "blobs-q4l", f"{name}.bin"), "rb") as f:
            blob = f.read()
        w = filter_bytes(os.path.join(out, "submodels-q4l", f"{name}.tflite"))
        assert blob.count(w) == 1, f"{name}: level-coded filter not in the q4l blob"
        w_off = blob.find(w)
        assert w_off + len(w) == len(blob), f"{name}: weights not at tail"
        assert w_off % 4 == 0 and len(blob) <= DEC_BYTES, name
        assert data.count(w) == 0, f"{name}: int8 blob carries the q4l filter?"
        packed = quant4.pack_blob_lvl(blob, w_off, DEC_BASE)
        assert len(packed) <= LAND_BYTES, name
        packed_n += 1
        saved += len(blob) - len(packed)
        entries[i] = (name, packed)
    emb_q = np.frombuffer(vocab["embp"], np.int8).reshape(-1, 384)
    embc = quant4.pack_emb(emb_q,
                           np.frombuffer(vocab["embpscl"], "<f4"),
                           np.frombuffer(vocab["embpids"], "<u4"))
    entries.append(("embc4", embc))
    saved += len(vocab["embp"]) - len(embc)  # embp kept for host tooling
    # int8 rows alongside (EMB_INT8=0 leaves them out for a slow stick)
    if os.environ.get("EMB_INT8", "1") != "0":
        entries.append(("embc8", pack_emb8(
            emb_q, np.frombuffer(vocab["embpscl"], "<f4"),
            np.frombuffer(vocab["embpids"], "<u4"))))
    print(f"q4: {packed_n} decoder blobs (LAY5) + embc4 packed, "
          f"{saved / 1e6:.1f} MB less storage traffic per token cycle")

    # firmware match id: the buffer addresses of the ELF the blobs were
    # linked against; the firmware refuses a stale card at boot.
    elf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "firmware", "target", "thumbv8m.main-none-eabihf",
                       "release", "whisper-firmware")
    addrs = {}
    for line in subprocess.check_output(["arm-none-eabi-nm", elf]).decode().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in ("nrf_axon_interlayer_buffer",
                                            "nrf_axon_psum_buffer"):
            addrs[parts[2]] = int(parts[0], 16)
    entries.append(("fwid", struct.pack(
        "<II", addrs["nrf_axon_interlayer_buffer"],
        addrs["nrf_axon_psum_buffer"])))

    # per-entry integrity sums (u32 wrapping byte-sum, index order): SPI
    # mode runs with CRC off, so the firmware verifies each blob after
    # loading it into the slot and retries the read on mismatch.
    entries.append(("sums", b"\0" * (4 * (len(entries) + 2))))
    sums_idx = len(entries) - 1

    # place everything, then the plan (needs the final block count)
    entries.append(("plan", b""))  # reserve the entry slot
    assert len(entries) <= (HEADER_BLOCKS * BLOCK - 16) // 32, "index full"
    for name, _ in entries:
        assert len(name) < 24, f"name too long: {name}"

    def layout(entries):
        off = HEADER_BLOCKS
        placed = []
        for name, data in entries:
            placed.append((name, off, data))
            off += (len(data) + BLOCK - 1) // BLOCK
        return placed, off

    placed, total = layout(entries)
    scratch_lba = (total + 255) // 256 * 256  # aligned margin past the image
    plan = write_plan(scales, mq_enc, mq_dec, conv1, conv2, ref,
                      scratch_lba, vocab_n)
    entries[-1] = ("plan", plan)
    sums = [(sum(d) & 0xFFFFFFFF) if i != sums_idx else 0
            for i, (_, d) in enumerate(entries)]
    entries[sums_idx] = ("sums", struct.pack(f"<{len(sums)}I", *sums))
    placed, total = layout(entries)

    index = struct.pack("<8sII", b"WSPRIMG1", 2, len(placed))
    payload = bytearray()
    for name, off, data in placed:
        index += struct.pack("<24sII", name.encode(), off, len(data))
        payload += data
        payload += b"\0" * ((-len(data)) % BLOCK)
    img = bytearray(index)
    img += b"\0" * (HEADER_BLOCKS * BLOCK - len(img))
    img += payload
    path = os.path.join(out, "sd.img")
    with open(path, "wb") as f:
        f.write(img)
    with open(os.path.join(out, "sd-index.json"), "w") as f:
        json.dump({n: {"lba": o, "bytes": len(d)} for n, o, d in placed},
                  f, indent=1)
    print(f"{path}: {len(img) / 1e6:.1f} MB, {len(placed)} entries, "
          f"scratch at block {scratch_lba}, vocab {vocab_n} kept")

    # --- SD smoke-test tape --------------------------------------------------
    tdir = os.path.join(out, "tape-sdtest")
    os.makedirs(tdir, exist_ok=True)
    rng = np.random.default_rng(11)
    pattern = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    for fn, data in [("pattern.bin", pattern), ("zeros.bin", bytes(4096)),
                     ("magic.bin", bytes(img[:16]))]:
        with open(os.path.join(tdir, fn), "wb") as f:
            f.write(data)
    test_lba = scratch_lba + 20000  # past all scratch regions
    steps = [
        {"op": "cmd", "code": CMD_SD_INIT, "args": []},
        {"op": "write", "file": "pattern.bin", "addr": ARENA_BASE},
        {"op": "cmd", "code": CMD_SD_WRITE, "args": [test_lba, ARENA_BASE, 8]},
        {"op": "write", "file": "zeros.bin", "addr": ARENA_BASE},
        {"op": "cmd", "code": CMD_SD_READ, "args": [test_lba, ARENA_BASE, 8]},
        {"op": "check", "file": "pattern.bin", "addr": ARENA_BASE,
         "label": "sd-roundtrip", "tol": 0},
        {"op": "cmd", "code": CMD_SD_READ, "args": [0, ARENA_BASE, 1]},
        {"op": "check", "file": "magic.bin", "addr": ARENA_BASE,
         "label": "sd-image-magic", "tol": 0},
    ]
    with open(os.path.join(tdir, "tape.json"), "w") as f:
        json.dump({"steps": steps}, f, indent=1)
    print(f"sd test tape -> {tdir} (test lba {test_lba})")


if __name__ == "__main__":
    main()
