"""Tape generator: emit device schedules for the host tape player.

A tape is a JSON list of five generic ops (blob / write / cmd / check /
read) plus the data files they reference; the host (host/) plays it
against the firmware mailbox. All pipeline structure and every address
lives here.

Goldens are computed by the "device model": the emitted .tflite submodels
run in the TFLite interpreter (the Axon engine is bit-exact against it) and
the CPU glue is mirrored operation for operation from firmware kernels.rs.
Each stage is checked in isolation -- its inputs are (re)written from
golden files -- so a deviation pinpoints exactly one stage.

Tapes:
  ln1q    the original two-stage smoke tape (LN + q projection)
  block0  one full encoder block on a 64-frame sequence: LN1, q/k/v, six
          fused attention heads, out-projection, residual, LN2, 4 fc1
          tiles, per-tile GELU LUTs, 4 fc2 partials, fc2 recombination

Requires out/scales.json, out/submodels/*.tflite, out/blobs/*.bin
(regenerated for the CURRENT firmware ELF).
"""

import json
import os
import shutil
import struct
import sys

import numpy as np

import common
import layered

# Mirrors of firmware constants (memory.x / src/main.rs).
ARENA_BASE = 0x2003_2000
ARENA_BYTES = 100 * 1024
CMD_RUN_NPU = 2
CMD_LUT8 = 3
CMD_LN = 8
CMD_ADD16 = 9
CMD_ADDPOS = 10
CMD_ATTN_HEAD = 12
CMD_FC2SUM = 13

T_TILE = 64  # matches the compiled blobs' input width
C = common.STATE
HD = common.HEAD_DIM


def pack_quant(scale, zp):
    return struct.pack("<fi", float(scale), int(zp))


def q8(x, q):
    return common.quantize(x, q[0], q[1])


def dq(a, q):
    return (a.astype(np.float32) - q[1]) * np.float32(q[0])


class Tape:
    """Collects steps + data files; `alloc` bumps the arena, `reset` reuses
    it (stages are stateless: every input is written from a golden file)."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.steps = []
        self.off = 0

    def alloc(self, size, align=4):
        self.off = (self.off + align - 1) & ~(align - 1)
        addr = ARENA_BASE + self.off
        self.off += size
        # top 8 bytes are the firmware's crash breadcrumb
        assert self.off <= ARENA_BYTES - 8, "arena overflow"
        return addr

    def reset(self):
        self.off = 0

    def write(self, name, data, addr=None):
        addr = self.alloc(len(data)) if addr is None else addr
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)
        self.steps.append({"op": "write", "file": name, "addr": addr})
        return addr

    def blob(self, name):
        src = os.path.join(common.OUT, "blobs", f"{name}.bin")
        shutil.copy(src, self.dir)
        self.steps.append({"op": "blob", "file": f"{name}.bin"})

    def cmd(self, code, args):
        self.steps.append({"op": "cmd", "code": code,
                           "args": [int(x) for x in args]})

    def check(self, name, data, addr, label, tol=0, width=1):
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)
        self.steps.append({"op": "check", "file": name, "addr": int(addr),
                           "label": label, "tol": tol, "width": width})

    def save(self):
        with open(os.path.join(self.dir, "tape.json"), "w") as f:
            json.dump({"steps": self.steps}, f, indent=1)
        print(f"tape: {len(self.steps)} steps -> {self.dir}")


class Submodel:
    """An emitted tflite: interpreter + its authoritative quantization."""

    def __init__(self, name):
        import tensorflow as tf

        self.name = name
        self.interp = tf.lite.Interpreter(
            model_path=os.path.join(common.OUT, "submodels", f"{name}.tflite"))
        self.interp.allocate_tensors()
        self.ind = self.interp.get_input_details()[0]
        self.outd = self.interp.get_output_details()[0]
        self.in_q = self.ind["quantization"]
        self.out_q = self.outd["quantization"]

    def run(self, planar_in):
        """planar [C, W] int8 -> planar [C_out, W] int8."""
        self.interp.set_tensor(
            self.ind["index"], np.ascontiguousarray(planar_in.T)[None, None])
        self.interp.invoke()
        return self.interp.get_tensor(self.outd["index"])[0, 0].T.copy()


# --- kernel mirrors (must match firmware kernels.rs bit for bit in intent) --

def ln_golden(x16, sq, gamma, beta, out_q):
    x = dq(x16, sq)  # [C, W]
    mean = x.mean(0, dtype=np.float32)
    var = ((x - mean) ** 2).mean(0, dtype=np.float32)
    inv = (1.0 / np.sqrt(var + np.float32(1e-5))).astype(np.float32)
    y = (x - mean) * inv * gamma[:, None] + beta[:, None]
    return q8(y, out_q)


def attn_golden(qv, kv, vv, q_q, k_q, v_q, ctx_q, tk=None):
    """Fused per-head attention, planar [C, W] in/out. `tk` limits the keys
    (padding frames are queries but never keys)."""
    wq = qv.shape[1]
    tk = kv.shape[1] if tk is None else tk
    score_mult = np.float32(q_q[0] * k_q[0] / np.sqrt(HD))
    ctx = np.empty((C, wq), np.int8)
    for h in range(common.HEADS):
        r = slice(h * HD, (h + 1) * HD)
        qi = qv[r].astype(np.int64) - q_q[1]
        ki = kv[r, :tk].astype(np.int64) - k_q[1]
        vi = vv[r, :tk].astype(np.int64) - v_q[1]
        scores = (qi.T @ ki).astype(np.float32) * score_mult  # [wq, tk]
        m = scores.max(1, keepdims=True)
        e = np.exp(scores - m, dtype=np.float32)
        p = e * (np.float32(1.0) / e.sum(1, keepdims=True, dtype=np.float32))
        p8 = np.clip(np.round(p * 256) - 128, -128, 127).astype(np.int64)
        acc = vi @ (p8 + 128).T  # [HD, wq]
        ctx[r] = q8(acc.astype(np.float32) * np.float32(v_q[0] / 256.0), ctx_q)
    return ctx


def add16_golden(a16, qa, b8, qb, qd):
    return common.quantize(dq(a16, qa) + dq(b8, qb), qd[0], qd[1], bits=16)


def gelu_lut(in_q, out_q):
    grid = (np.arange(-128, 128, dtype=np.float32) - in_q[1]) * np.float32(in_q[0])
    return q8(layered.gelu(grid), out_q)


def fc2sum_golden(parts, pqs, a16, qa, qd):
    s = dq(a16, qa)
    for p, pq in zip(parts, pqs):
        s = s + dq(p, pq)
    return common.quantize(s, qd[0], qd[1], bits=16)


# --- tape generators --------------------------------------------------------

def capture(sd, scales, names, mode="int8"):
    """Record simulation site values (int8 device-semantics sim or float)."""
    ref = np.load(os.path.join(common.OUT, "ref.npz"))
    eng = layered.Engine(sd, mode, scales, record=set(names))
    layered.encoder(eng, ref["mel_chunk"], common.AUDIO_CTX)
    return eng.recorded


def capture_x16(sd, scales):
    """First-block input residual from the device-semantics simulation."""
    x = capture(sd, scales, ["enc.x"])["enc.x"][:T_TILE]
    s16 = scales["enc.x"]
    return common.quantize(x, s16[0], s16[1], bits=16).T.copy(), s16


def gen_block0(sd, scales):
    t = Tape(os.path.join(common.OUT, "tape-block0"))
    p = "encoder.blocks.0."
    mq = {n: Submodel(n) for n in
          ["wq0", "wk0", "wv0", "wout0", "wfc1a", "wfc1b", "wfc1c", "wfc1d",
           "wfc2p0", "wfc2p1", "wfc2p2", "wfc2p3"]}
    assert mq["wq0"].in_q == mq["wk0"].in_q == mq["wv0"].in_q
    assert len({mq[f"wfc1{c}"].in_q for c in "abcd"}) == 1

    x16, s16 = capture_x16(sd, scales)
    nbytes8 = C * T_TILE          # one planar int8 activation

    def ln_stage(label, src16, w_name, out_q):
        t.reset()
        a_src = t.write(f"{label}_in.bin", src16.tobytes())
        gamma = sd[w_name + ".weight"].astype("<f4")
        beta = sd[w_name + ".bias"].astype("<f4")
        a_gb = t.write(f"{label}_gb.bin", gamma.tobytes() + beta.tobytes())
        a_dst = t.alloc(nbytes8)
        params = struct.pack("<6I", a_src, a_dst, C, T_TILE,
                             a_gb, a_gb + 4 * C) \
            + pack_quant(*s16[:2]) + pack_quant(*out_q)
        a_p = t.write(f"{label}_p.bin", params)
        t.cmd(CMD_LN, [a_p])
        g = ln_golden(src16, s16, sd[w_name + ".weight"],
                      sd[w_name + ".bias"], out_q)
        t.check(f"{label}_exp.bin", g.tobytes(), a_dst, label, tol=1)
        return g

    def npu_stage(label, model, planar_in):
        t.reset()
        a_in = t.write(f"{label}_in.bin", planar_in.tobytes())
        a_out = t.alloc(planar_in.nbytes)  # 384 -> 384 throughout
        t.blob(model.name)
        t.cmd(CMD_RUN_NPU, [a_in, a_out])
        g = model.run(planar_in)
        # tol 1: the Axon's fixed-point requant occasionally rounds a
        # .5-boundary value one LSB away from the TFLite interpreter
        # (1-2 of 24576 values on some submodels). Stages are
        # golden-isolated, so this never cascades.
        t.check(f"{label}_exp.bin", g.tobytes(), a_out, label, tol=1)
        return g

    # LN1 + attention projections
    g_ln1 = ln_stage("b0_ln1", x16, p + "attn_ln", mq["wq0"].in_q)
    g_q = npu_stage("b0_q", mq["wq0"], g_ln1)
    g_k = npu_stage("b0_k", mq["wk0"], g_ln1)
    g_v = npu_stage("b0_v", mq["wv0"], g_ln1)

    # Fused attention, one command per head
    t.reset()
    a_q = t.write("b0_attn_q.bin", g_q.tobytes())
    a_k = t.write("b0_attn_k.bin", g_k.tobytes())
    a_v = t.write("b0_attn_v.bin", g_v.tobytes())
    a_ctx = t.alloc(nbytes8)
    q_q, k_q, v_q = mq["wq0"].out_q, mq["wk0"].out_q, mq["wv0"].out_q
    ctx_q = mq["wout0"].in_q
    score_mult = np.float32(q_q[0] * k_q[0] / np.sqrt(HD))
    for h in range(common.HEADS):
        off = h * HD * T_TILE
        params = struct.pack(
            "<9I3i", a_q + off, a_k + off, a_v + off, a_ctx + off,
            HD, T_TILE, T_TILE, T_TILE, T_TILE,
            int(q_q[1]), int(k_q[1]), int(v_q[1])) \
            + struct.pack("<ff", score_mult, v_q[0]) + pack_quant(*ctx_q)
        a_p = t.write(f"b0_attn_h{h}_p.bin", params)
        t.cmd(CMD_ATTN_HEAD, [a_p])
    g_ctx = attn_golden(g_q, g_k, g_v, q_q, k_q, v_q, ctx_q)
    t.check("b0_attn_exp.bin", g_ctx.tobytes(), a_ctx, "b0_attn", tol=1)

    # Out-projection + residual 1 (int16 + int8 + int16 out = 120 KB, so
    # the elementwise add is tiled to bound the arena)
    g_o = npu_stage("b0_out", mq["wout0"], g_ctx)
    r1_q = scales["enc.b0.res1"]
    g_res1 = add16_golden(x16, s16, g_o, mq["wout0"].out_q, r1_q)
    n = C * T_TILE
    for ci in range(2):
        sl = slice(ci * n // 2, (ci + 1) * n // 2)
        t.reset()
        a_a = t.write(f"b0_res1c{ci}_a.bin", x16.reshape(-1)[sl].tobytes())
        a_b = t.write(f"b0_res1c{ci}_b.bin", g_o.reshape(-1)[sl].tobytes())
        a_dst = t.alloc(n)  # n/2 int16 elements
        params = struct.pack("<4I", a_a, a_b, a_dst, n // 2) \
            + pack_quant(*s16[:2]) + pack_quant(*mq["wout0"].out_q) \
            + pack_quant(*r1_q[:2])
        a_p = t.write(f"b0_res1c{ci}_p.bin", params)
        t.cmd(CMD_ADD16, [a_p])
        t.check(f"b0_res1c{ci}_exp.bin", g_res1.reshape(-1)[sl].tobytes(),
                a_dst, f"b0_res1c{ci}", tol=1, width=2)

    # LN2 + MLP
    s16_saved = s16
    s16 = r1_q  # ln_stage quantizes its src with s16
    g_ln2 = ln_stage("b0_ln2", g_res1, p + "mlp_ln", mq["wfc1a"].in_q)
    s16 = s16_saved

    g_f1, g_gelu, g_p2 = [], [], []
    for j, part in enumerate("abcd"):
        f1 = npu_stage(f"b0_fc1{part}", mq[f"wfc1{part}"], g_ln2)
        g_f1.append(f1)
        lut = gelu_lut(mq[f"wfc1{part}"].out_q, mq[f"wfc2p{j}"].in_q)
        t.reset()
        a_lut = t.write(f"b0_gelu{j}_lut.bin", lut.tobytes())
        a_in = t.write(f"b0_gelu{j}_in.bin", f1.tobytes())
        a_out = t.alloc(nbytes8)
        t.cmd(CMD_LUT8, [a_lut, a_in, a_out, C * T_TILE])
        g = lut[(f1.astype(np.int32) + 128)]
        g_gelu.append(g)
        t.check(f"b0_gelu{j}_exp.bin", g.tobytes(), a_out, f"b0_gelu{j}", tol=0)
        g_p2.append(npu_stage(f"b0_fc2p{j}", mq[f"wfc2p{j}"], g))

    # fc2 recombination + residual 2, tiled to bound the arena
    r2_q = scales["enc.b0.res2"]
    pqs = [mq[f"wfc2p{j}"].out_q for j in range(4)]
    g_res2 = fc2sum_golden(g_p2, pqs, g_res1, r1_q, r2_q)
    n = C * T_TILE
    chunk = n // 4
    for ci in range(4):
        sl = slice(ci * chunk, (ci + 1) * chunk)
        t.reset()
        a_parts = [t.write(f"b0_fc2s{ci}_p{j}.bin",
                           g_p2[j].reshape(-1)[sl].tobytes())
                   for j in range(4)]
        a_a = t.write(f"b0_fc2s{ci}_a.bin", g_res1.reshape(-1)[sl].tobytes())
        a_dst = t.alloc(2 * chunk)
        params = struct.pack("<7I", *a_parts, a_a, a_dst, chunk) \
            + b"".join(pack_quant(*q) for q in pqs) \
            + pack_quant(*r1_q[:2]) + pack_quant(*r2_q[:2])
        a_p = t.write(f"b0_fc2s{ci}_p.bin", params)
        t.cmd(CMD_FC2SUM, [a_p])
        t.check(f"b0_fc2s{ci}_exp.bin", g_res2.reshape(-1)[sl].tobytes(),
                a_dst, f"b0_fc2sum{ci}", tol=1, width=2)

    t.save()


def gen_ln1q(sd, scales):
    t = Tape(os.path.join(common.OUT, "tape-ln1q"))
    m = Submodel("wq0")
    x16, s16 = capture_x16(sd, scales)
    p = "encoder.blocks.0."

    a_src = t.write("x16.bin", x16.tobytes())
    gamma = sd[p + "attn_ln.weight"].astype("<f4")
    beta = sd[p + "attn_ln.bias"].astype("<f4")
    a_gb = t.write("gamma_beta.bin", gamma.tobytes() + beta.tobytes())
    a_ln1 = t.alloc(C * T_TILE)
    a_q = t.alloc(C * T_TILE)
    params = struct.pack("<6I", a_src, a_ln1, C, T_TILE, a_gb, a_gb + 4 * C) \
        + pack_quant(*s16[:2]) + pack_quant(*m.in_q)
    a_p = t.write("lnparams.bin", params)
    t.cmd(CMD_LN, [a_p])
    g_ln1 = ln_golden(x16, s16, sd[p + "attn_ln.weight"],
                      sd[p + "attn_ln.bias"], m.in_q)
    t.check("expect_ln1.bin", g_ln1.tobytes(), a_ln1, "enc.b0.ln1", tol=1)
    t.steps.append({"op": "write", "file": "expect_ln1.bin", "addr": a_ln1})
    t.blob("wq0")
    t.cmd(CMD_RUN_NPU, [a_ln1, a_q])
    t.check("expect_q.bin", m.run(g_ln1).tobytes(), a_q, "enc.b0.q", tol=0)
    t.save()


# --- the full encoder at audio_ctx=600, frame-tiled --------------------------

CTX = common.AUDIO_CTX      # real encoder frames
PAD_W = 640                 # padded to a whole number of 64-frame tiles
MEL_W = 2 * PAD_W


def gen_encoder(sd, scales):
    t = Tape(os.path.join(common.OUT, "tape-encoder"))
    ref = np.load(os.path.join(common.OUT, "ref.npz"))

    conv1 = Submodel("wconv1")
    conv2 = [Submodel(f"wconv2{p}") for p in "abc"]
    assert conv2[0].in_q == conv2[1].in_q == conv2[2].in_q

    # Divergence tracking against the int8 simulation, per stage: the tape's
    # checks only prove device == device-model; this is the quality signal.
    sim_sites = ["enc.conv1", "enc.gelu1", "enc.conv2", "enc.gelu2", "enc.x",
                 "enc.out"]
    for l in range(common.N_LAYERS):
        sim_sites += [f"enc.b{l}.{s}"
                      for s in ("ln1", "q", "ctx", "res1", "res2")]
    which_ref = os.environ.get("TAPE_REF", "int8")
    sim = capture(sd, scales, sim_sites, mode=which_ref)
    print(f"stage divergence vs the {which_ref} pipeline:")

    def snr(name, planar, q):
        """planar int array [C, W]; sim site is [frames, C] fake-quant."""
        s = sim.get(name)
        if s is None:
            return
        dev = dq(planar[:, :s.shape[0]].T, q)
        err = float(np.sqrt(np.mean((dev - s) ** 2)))
        sig = float(np.sqrt(np.mean(s ** 2))) or 1e-9
        print(f"  {name}: snr {20 * np.log10(sig / max(err, 1e-9)):6.1f} dB")

    def npu_tiled(label, model, full, t_in, t_out, halo_pad, blob_stride=1):
        """Run `model` over frame tiles of a full planar array.
        full [C, W_in]; returns golden [C_out, n_tiles*t_out]."""
        cin = full.shape[0]
        n_tiles = (full.shape[1] * t_out) // (t_in - 2 * halo_pad) // t_out
        src = np.pad(full, ((0, 0), (halo_pad, halo_pad)),
                     constant_values=model.in_q[1]) if halo_pad else full
        t.reset()
        a_in = t.alloc(cin * t_in)
        a_out = t.alloc(0)  # placeholder; sized below on first tile
        outs = []
        t.blob(model.name)
        for i in range(n_tiles):
            lo = i * (t_in - 2 * halo_pad) * blob_stride // blob_stride
            tile = np.ascontiguousarray(src[:, lo:lo + t_in])
            g = model.run(tile)
            if i == 0:
                a_out = t.alloc(g.nbytes)
            t.write(f"{label}_t{i}_in.bin", tile.tobytes(), a_in)
            t.cmd(CMD_RUN_NPU, [a_in, a_out])
            t.check(f"{label}_t{i}_exp.bin", g.tobytes(), a_out,
                    f"{label}[{i}]", tol=1)
            outs.append(g)
        return np.concatenate(outs, axis=1)

    def lut_stage(label, lut, full):
        """Elementwise int8 LUT over a full planar array, flat-chunked."""
        g = lut[full.astype(np.int32) + 128]
        flat_g, flat_in = g.reshape(-1), full.reshape(-1)
        chunk = 24576
        t.reset()
        a_lut = t.write(f"{label}_lut.bin", lut.tobytes())
        a_in = t.alloc(chunk)
        a_out = t.alloc(chunk)
        for ci, lo in enumerate(range(0, flat_in.size, chunk)):
            n = min(chunk, flat_in.size - lo)
            t.write(f"{label}_c{ci}_in.bin", flat_in[lo:lo + n].tobytes(), a_in)
            t.cmd(CMD_LUT8, [a_lut, a_in, a_out, n])
            t.check(f"{label}_c{ci}_exp.bin", flat_g[lo:lo + n].tobytes(),
                    a_out, f"{label}[{ci}]", tol=0)
        return g

    def ln_tiled(label, src16, sq, w_name, out_q):
        gamma, beta = sd[w_name + ".weight"], sd[w_name + ".bias"]
        g = ln_golden(src16, sq, gamma, beta, out_q)
        t.reset()
        a_gb = t.write(f"{label}_gb.bin",
                       gamma.astype("<f4").tobytes()
                       + beta.astype("<f4").tobytes())
        a_in = t.alloc(2 * C * T_TILE)
        a_out = t.alloc(C * T_TILE)
        params = struct.pack("<6I", a_in, a_out, C, T_TILE, a_gb,
                             a_gb + 4 * C) \
            + pack_quant(*sq[:2]) + pack_quant(*out_q)
        a_p = t.write(f"{label}_p.bin", params)
        for i in range(PAD_W // T_TILE):
            sl = slice(i * T_TILE, (i + 1) * T_TILE)
            t.write(f"{label}_t{i}_in.bin",
                    np.ascontiguousarray(src16[:, sl]).tobytes(), a_in)
            t.cmd(CMD_LN, [a_p])
            t.check(f"{label}_t{i}_exp.bin",
                    np.ascontiguousarray(g[:, sl]).tobytes(), a_out,
                    f"{label}[{i}]", tol=1)
        return g

    def add16_chunked(label, a16, qa, b8, qb, qd):
        g = add16_golden(a16, qa, b8, qb, qd)
        fa, fb, fg = a16.reshape(-1), b8.reshape(-1), g.reshape(-1)
        chunk = 12288
        for ci, lo in enumerate(range(0, fa.size, chunk)):
            n = min(chunk, fa.size - lo)
            t.reset()
            a_a = t.write(f"{label}_c{ci}_a.bin", fa[lo:lo + n].tobytes())
            a_b = t.write(f"{label}_c{ci}_b.bin", fb[lo:lo + n].tobytes())
            a_d = t.alloc(2 * n)
            params = struct.pack("<4I", a_a, a_b, a_d, n) \
                + pack_quant(*qa[:2]) + pack_quant(*qb[:2]) \
                + pack_quant(*qd[:2])
            a_p = t.write(f"{label}_c{ci}_p.bin", params)
            t.cmd(CMD_ADD16, [a_p])
            t.check(f"{label}_c{ci}_exp.bin", fg[lo:lo + n].tobytes(), a_d,
                    f"{label}[{ci}]", tol=1, width=2)
        return g

    # --- stem: mel -> conv1 -> gelu -> conv2 -> gelu -> +pos -> x16 --------
    mel = ref["mel_chunk"]  # [80, 1200] float
    mel_q = q8(np.pad(mel, ((0, 0), (0, MEL_W - mel.shape[1]))), conv1.in_q)
    g_c1 = npu_tiled("conv1", conv1, mel_q, T_TILE + 2, T_TILE, 1)
    snr("enc.conv1", g_c1, conv1.out_q)
    lut1 = gelu_lut(conv1.out_q, conv2[0].in_q)
    g_g1 = lut_stage("gelu1", lut1, g_c1)
    snr("enc.gelu1", g_g1, conv2[0].in_q)

    gelu2_q = scales["enc.gelu2"]
    parts = []
    for p, m in zip("abc", conv2):
        g = npu_tiled(f"conv2{p}", m, g_g1, 2 * T_TILE + 2, T_TILE, 1,
                      blob_stride=2)
        lut = gelu_lut(m.out_q, gelu2_q[:2])
        parts.append(lut_stage(f"gelu2{p}", lut, g))
    g_g2 = np.concatenate(parts, axis=0)  # [384, 640] at gelu2_q
    snr("enc.gelu2", g_g2, gelu2_q)

    # positional embedding add -> int16 residual (chunked ADDPOS)
    s16 = scales["enc.x"]
    pos = np.zeros((C, PAD_W), np.float32)
    pos[:, :CTX] = sd["encoder.positional_embedding"][:CTX].T
    x16 = common.quantize(dq(g_g2, gelu2_q) + pos, s16[0], s16[1], bits=16)
    snr("enc.x", x16, s16)
    fa, fp, fg = g_g2.reshape(-1), pos.reshape(-1), x16.reshape(-1)
    chunk = 12288
    for ci, lo in enumerate(range(0, fa.size, chunk)):
        n = min(chunk, fa.size - lo)
        t.reset()
        a_a = t.write(f"pos_c{ci}_a.bin", fa[lo:lo + n].tobytes())
        a_b = t.write(f"pos_c{ci}_b.bin",
                      fp[lo:lo + n].astype("<f4").tobytes())
        a_d = t.alloc(2 * n)
        params = struct.pack("<4I", a_a, a_b, a_d, n) \
            + pack_quant(*gelu2_q[:2]) + pack_quant(*s16[:2])
        a_p = t.write(f"pos_c{ci}_p.bin", params)
        t.cmd(CMD_ADDPOS, [a_p])
        t.check(f"pos_c{ci}_exp.bin", fg[lo:lo + n].tobytes(), a_d,
                f"pos[{ci}]", tol=1, width=2)

    # --- transformer blocks ------------------------------------------------
    sq = s16
    for l in range(common.N_LAYERS):
        p = f"encoder.blocks.{l}."
        names = common.submodel_names(l)
        mq = {k: Submodel(v) for k, v in names.items()}
        assert mq["q"].in_q == mq["k"].in_q == mq["v"].in_q
        assert len({mq[f"fc1{c}"].in_q for c in "abcd"}) == 1
        r1_q = scales[f"enc.b{l}.res1"]
        r2_q = scales[f"enc.b{l}.res2"]

        g_ln1 = ln_tiled(f"b{l}_ln1", x16, sq, p + "attn_ln", mq["q"].in_q)
        snr(f"enc.b{l}.ln1", g_ln1, mq["q"].in_q)
        g_q = npu_tiled(f"b{l}_q", mq["q"], g_ln1, T_TILE, T_TILE, 0)
        snr(f"enc.b{l}.q", g_q, mq["q"].out_q)
        g_k = npu_tiled(f"b{l}_k", mq["k"], g_ln1, T_TILE, T_TILE, 0)
        g_v = npu_tiled(f"b{l}_v", mq["v"], g_ln1, T_TILE, T_TILE, 0)

        # attention: K/V resident per head, q/ctx tiles cycled
        q_q, k_q, v_q = mq["q"].out_q, mq["k"].out_q, mq["v"].out_q
        ctx_q = mq["out"].in_q
        g_ctx = attn_golden(g_q, g_k, g_v, q_q, k_q, v_q, ctx_q, tk=CTX)
        score_mult = np.float32(q_q[0] * k_q[0] / np.sqrt(HD))
        for h in range(common.HEADS):
            r = slice(h * HD, (h + 1) * HD)
            t.reset()
            a_k = t.write(f"b{l}_attn_h{h}_k.bin",
                          np.ascontiguousarray(g_k[r]).tobytes())
            a_v = t.write(f"b{l}_attn_h{h}_v.bin",
                          np.ascontiguousarray(g_v[r]).tobytes())
            a_q = t.alloc(HD * T_TILE)
            a_c = t.alloc(HD * T_TILE)
            params = struct.pack(
                "<9I3i", a_q, a_k, a_v, a_c, HD, T_TILE, T_TILE, CTX, PAD_W,
                int(q_q[1]), int(k_q[1]), int(v_q[1])) \
                + struct.pack("<ff", score_mult, v_q[0]) + pack_quant(*ctx_q)
            a_p = t.write(f"b{l}_attn_h{h}_p.bin", params)
            for i in range(PAD_W // T_TILE):
                sl = slice(i * T_TILE, (i + 1) * T_TILE)
                t.write(f"b{l}_attn_h{h}_t{i}_q.bin",
                        np.ascontiguousarray(g_q[r, sl]).tobytes(), a_q)
                t.cmd(CMD_ATTN_HEAD, [a_p])
                t.check(f"b{l}_attn_h{h}_t{i}_exp.bin",
                        np.ascontiguousarray(g_ctx[r, sl]).tobytes(), a_c,
                        f"b{l}_attn_h{h}[{i}]", tol=1)

        snr(f"enc.b{l}.ctx", g_ctx, ctx_q)
        g_o = npu_tiled(f"b{l}_out", mq["out"], g_ctx, T_TILE, T_TILE, 0)
        g_res1 = add16_chunked(f"b{l}_res1", x16, sq, g_o,
                               mq["out"].out_q, r1_q)
        snr(f"enc.b{l}.res1", g_res1, r1_q)

        g_ln2 = ln_tiled(f"b{l}_ln2", g_res1, r1_q, p + "mlp_ln",
                         mq["fc1a"].in_q)
        g_p2, pqs = [], []
        for j, part in enumerate("abcd"):
            f1 = npu_tiled(f"b{l}_fc1{part}", mq[f"fc1{part}"], g_ln2,
                           T_TILE, T_TILE, 0)
            lut = gelu_lut(mq[f"fc1{part}"].out_q, mq[f"fc2p{j}"].in_q)
            gg = lut_stage(f"b{l}_gelu{j}", lut, f1)
            g_p2.append(npu_tiled(f"b{l}_fc2p{j}", mq[f"fc2p{j}"], gg,
                                  T_TILE, T_TILE, 0))
            pqs.append(mq[f"fc2p{j}"].out_q)

        g_res2 = fc2sum_golden(g_p2, pqs, g_res1, r1_q, r2_q)
        fr1, fr2 = g_res1.reshape(-1), g_res2.reshape(-1)
        fp2 = [g.reshape(-1) for g in g_p2]
        chunk = 12288
        for ci, lo in enumerate(range(0, fr1.size, chunk)):
            n = min(chunk, fr1.size - lo)
            t.reset()
            a_parts = [t.write(f"b{l}_fc2s{ci}_p{j}.bin",
                               fp2[j][lo:lo + n].tobytes())
                       for j in range(4)]
            a_a = t.write(f"b{l}_fc2s{ci}_a.bin", fr1[lo:lo + n].tobytes())
            a_d = t.alloc(2 * n)
            params = struct.pack("<7I", *a_parts, a_a, a_d, n) \
                + b"".join(pack_quant(*q) for q in pqs) \
                + pack_quant(*r1_q[:2]) + pack_quant(*r2_q[:2])
            a_p = t.write(f"b{l}_fc2s{ci}_p.bin", params)
            t.cmd(CMD_FC2SUM, [a_p])
            t.check(f"b{l}_fc2s{ci}_exp.bin", fr2[lo:lo + n].tobytes(), a_d,
                    f"b{l}_fc2sum[{ci}]", tol=1, width=2)

        snr(f"enc.b{l}.res2", g_res2, r2_q)
        x16, sq = g_res2, r2_q

    # --- final layernorm ---------------------------------------------------
    out_q = scales["enc.out"]
    assert out_q[2] == 8
    g_out = ln_tiled("ln_post", x16, sq, "encoder.ln_post", out_q[:2])
    t.save()

    # offline: how far is the device pipeline from the int8 simulation?
    sim = capture(sd, scales, ["enc.out"])["enc.out"]  # [600, 384] f32
    dev = dq(g_out[:, :CTX].T, out_q)
    err = float(np.sqrt(np.mean((dev - sim) ** 2)))
    sig = float(np.sqrt(np.mean(sim ** 2)))
    print(f"device-model encoder vs int8 sim: rms err {err:.4f} "
          f"(signal {sig:.4f}, snr {20 * np.log10(sig / max(err, 1e-9)):.1f} dB)")


def main():
    _, sd = common.load_weights()
    with open(os.path.join(common.OUT, "scales.json")) as f:
        scales = {k: (v["scale"], v["zp"], v["bits"])
                  for k, v in json.load(f).items()}
    which = sys.argv[1] if len(sys.argv) > 1 else "block0"
    {"ln1q": gen_ln1q, "block0": gen_block0,
     "encoder": gen_encoder}[which](sd, scales)


if __name__ == "__main__":
    main()
