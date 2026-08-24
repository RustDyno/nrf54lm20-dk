"""Layered Whisper execution engine.

Splits the model into exactly the pieces the device will run:

  NPU submodels (int8, static weights): conv1, conv2, every attention
    projection (q/k/v/out, cross q/k/v/out), mlp fc1, mlp fc2 (input-split into
    FC2_SPLITS partial models), and the vocab-tiled logits projection.
  CPU glue ops (Cortex-M33): layernorm, gelu (int8 LUT), softmax, the two
    dynamic attention matmuls (QK^T, probs*V), residual adds, argmax.

Three backends share this one code path:
  float: pure f32 -- validates the decomposition against openai-whisper.
  calib: f32 + min/max recording at every quantization site.
  int8:  real int8 weights/activations with int32 accumulation at every NPU
         op (TFLite semantics), fake-quant value view between ops.

Quantization sites are named; the calibrated scales are the contract between
the exporter (which bakes them into the .tflite submodels) and the firmware
(which applies the same glue-op scales).
"""

import numpy as np

import common
from common import (act_qparams, quantize, dequantize, weight_qparams,
                    STATE, HEADS, HEAD_DIM, N_LAYERS, FC2_SPLITS)


def sym_qparams(lo, hi, bits=8):
    amax = max(abs(float(lo)), abs(float(hi)))
    if amax == 0.0:
        return 1.0, 0
    return amax / (2 ** (bits - 1) - 1), 0


class Act:
    """An activation with its quantization site parameters attached."""

    __slots__ = ("x", "scale", "zp")

    def __init__(self, x, scale=None, zp=0):
        self.x = x
        self.scale = scale
        self.zp = zp

    def q(self):
        """Exact int8 view (values are already on the quantization grid).
        Only NPU submodel inputs call this; those sites are always 8-bit."""
        return quantize(self.x, self.scale, self.zp)


def gelu(x):
    import torch
    return (torch.erf(torch.from_numpy(x / np.sqrt(2, dtype=np.float32)))
            .numpy() * 0.5 + 0.5) * x


class Engine:
    PROBS_SCALE = 1.0 / 256.0  # fixed softmax output quantization
    PROBS_ZP = -128

    def __init__(self, sd, mode="float", scales=None, record=None):
        assert mode in ("float", "calib", "int8")
        self.sd = sd
        self.mode = mode
        self.scales = scales or {}   # site -> (scale, zp, bits)
        self.ranges = {}             # calib: site -> [lo, hi]
        self.meta = {}               # calib: site -> (sym, bits)
        self._wq = {}                # weight name -> (q, per-channel scales)
        self.record = record         # set of site names to capture
        self.recorded = {}           # site -> value seen at that site

    # --- quantization sites ------------------------------------------------

    def _record(self, name, x):
        """Accumulate rows across calls (decoder sites fire once per step)."""
        if name in self.recorded:
            self.recorded[name] = np.concatenate([self.recorded[name], x])
        else:
            self.recorded[name] = x.copy()

    def site(self, name, x, sym=False, bits=8):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.mode == "float":
            if self.record is not None and name in self.record:
                self._record(name, x)
            return Act(x)
        if self.mode == "calib":
            lo, hi = float(x.min()), float(x.max())
            r = self.ranges.setdefault(name, [lo, hi])
            r[0], r[1] = min(r[0], lo), max(r[1], hi)
            self.meta[name] = (sym, bits)
            return Act(x)
        scale, zp, bits = self.scales[name]
        q = quantize(x, scale, zp, bits)
        out = Act(dequantize(q, scale, zp), scale, zp)
        if self.record is not None and name in self.record:
            self._record(name, out.x)
        return out

    def finish_calib(self):
        out = {}
        for name, (lo, hi) in self.ranges.items():
            sym, bits = self.meta[name]
            s, z = (sym_qparams(lo, hi, bits) if sym
                    else act_qparams(lo, hi, bits))
            out[name] = (s, z, bits)
        return out

    def weight(self, name, axis=0):
        if name not in self._wq:
            self._wq[name] = weight_qparams(self.sd[name], axis=axis)
        return self._wq[name]

    # --- NPU ops (int8 semantics per TFLite / Axon) ------------------------

    def _mm_int8(self, a, w_name, bias, out_site, sym_out=False):
        """int8 matmul core: a [N, K] against weights [M, K] -> site [N, M]."""
        w = self.sd[w_name]
        b = self.sd.get(bias) if bias else None
        if self.mode != "int8":
            y = a.x @ w.T + (b if b is not None else 0.0)
            return self.site(out_site, y, sym=sym_out)
        wq, ws = self.weight(w_name)
        # f64 matmul is exact for these integer ranges and uses BLAS (numpy
        # integer matmul does not); the device accumulates in int32.
        acc = (a.q().astype(np.float64) - a.zp) @ wq.T.astype(np.float64)
        mult = a.scale * ws  # per-output-channel effective scale
        if b is not None:
            acc += np.round(b / mult)
        return self.site(out_site, acc * mult, sym=sym_out)

    def npu_fc(self, out_site, a, w_name, bias=None, sym_out=False):
        return self._mm_int8(a, w_name, bias, out_site, sym_out)

    def npu_fc2_split(self, out_site, a, w_name, bias):
        """fc2 input-dim split: FC2_SPLITS partial submodels, int8 partial
        outputs, CPU accumulation of the dequantized partials."""
        w = self.sd[w_name]
        k = w.shape[1] // FC2_SPLITS
        total = None
        for i in range(FC2_SPLITS):
            part = Act(a.x[:, i * k:(i + 1) * k], a.scale, a.zp)
            if self.mode != "int8":
                y = part.x @ w[:, i * k:(i + 1) * k].T
                if i == 0:
                    y = y + self.sd[bias]
                p = self.site(f"{out_site}.p{i}", y)
            else:
                pname = f"{w_name}.p{i}"
                if pname not in self._wq:
                    self._wq[pname] = weight_qparams(w[:, i * k:(i + 1) * k])
                wq, ws = self._wq[pname]
                acc = ((part.q().astype(np.float64) - part.zp)
                       @ wq.T.astype(np.float64))
                mult = part.scale * ws
                if i == 0:
                    acc += np.round(self.sd[bias] / mult)
                p = self.site(f"{out_site}.p{i}", acc * mult)
            total = p.x if total is None else total + p.x
        return self.site(out_site, total)

    def npu_conv1d(self, out_site, a, w_name, bias, stride):
        """k=3, pad=1 conv over time; a [T, Cin] -> site [T/stride, Cout]."""
        w = self.sd[w_name]  # [Cout, Cin, 3]
        b = self.sd[bias]
        t_out = a.x.shape[0] // stride
        if self.mode != "int8":
            xp = np.pad(a.x, ((1, 1), (0, 0)))
            y = np.zeros((t_out, w.shape[0]), np.float32)
            for dt in range(3):
                y += xp[dt:dt + t_out * stride:stride] @ w[:, :, dt].T
            return self.site(out_site, y + b)
        wq, ws = self.weight(w_name)
        xq = a.q().astype(np.float64) - a.zp
        xp = np.pad(xq, ((1, 1), (0, 0)))  # zero after zp removal == real zero
        acc = np.zeros((t_out, w.shape[0]), np.float64)
        for dt in range(3):
            acc += xp[dt:dt + t_out * stride:stride] @ wq[:, :, dt].T.astype(np.float64)
        mult = a.scale * ws
        acc += np.round(b / mult)
        return self.site(out_site, acc * mult)

    # --- CPU glue ops ------------------------------------------------------

    def ln(self, out_site, a, w_name, sym_out=False):
        x = a.x
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + 1e-5)
        y = y * self.sd[w_name + ".weight"] + self.sd[w_name + ".bias"]
        return self.site(out_site, y, sym=sym_out)

    def gelu(self, out_site, a):
        # Device: 256-entry int8 LUT (exact for a scalar map).
        return self.site(out_site, gelu(a.x))

    def add(self, out_site, a, b_act):
        # Residual adds live on the CPU; the residual stream is kept at
        # 16-bit to stop quantization noise accumulating across blocks.
        return self.site(out_site, a.x + b_act.x, bits=16)

    def cpu_logits(self, a, w_name):
        """Final vocab projection on the CPU: int8 x int8 -> int32 -> f32.

        Streaming the 19 MB embedding matrix dominates the cost either way;
        doing the MACs on the M33 keeps the logits exact (no output
        quantization), which argmax stability requires.
        """
        w = self.sd[w_name]
        if self.mode != "int8":
            return Act(a.x @ w.T)
        wq, ws = self.weight(w_name)
        acc = (a.q().astype(np.float64) - a.zp) @ wq.T.astype(np.float64)
        return Act((acc * (a.scale * ws)).astype(np.float32))

    def attention(self, out_site, q, k, v):
        """q [Tq, 384], k/v [Tk, 384] -> context [Tq, 384] at out_site.

        Device: int8 SMLAD matmuls with f32 softmax. q/k/v sites are
        symmetric so QK^T needs no zero-point correction; probs*V needs one
        per-column correction vector (fixed probs zero-point).
        """
        tq = q.x.shape[0]
        sc = np.float32(HEAD_DIM ** -0.25)
        ctx = np.empty((tq, STATE), np.float32)
        for h in range(HEADS):
            s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            if self.mode != "int8":
                scores = (q.x[:, s] * sc) @ (k.x[:, s] * sc).T
                probs = _softmax(scores)
                ctx[:, s] = probs @ v.x[:, s]
            else:
                qi = q.q()[:, s].astype(np.float64)
                ki = k.q()[:, s].astype(np.float64)
                vi = v.q()[:, s].astype(np.float64)
                scores = ((qi @ ki.T) * (q.scale * k.scale / HEAD_DIM ** 0.5)
                          ).astype(np.float32)
                probs = _softmax(scores)
                pq = quantize(probs, self.PROBS_SCALE, self.PROBS_ZP)
                acc = (pq.astype(np.float64) - self.PROBS_ZP) @ vi
                ctx[:, s] = acc * (self.PROBS_SCALE * v.scale)
        return self.site(out_site, ctx)


def _softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


# --- model graphs ----------------------------------------------------------

def encoder(eng, mel, audio_ctx):
    """mel [80, 2*audio_ctx] float log-mel -> encoder output Act [T, 384]."""
    x = eng.site("enc.mel", mel.T)  # time-major [2T, 80]
    x = eng.npu_conv1d("enc.conv1", x, "encoder.conv1.weight",
                       "encoder.conv1.bias", stride=1)
    x = eng.gelu("enc.gelu1", x)
    x = eng.npu_conv1d("enc.conv2", x, "encoder.conv2.weight",
                       "encoder.conv2.bias", stride=2)
    x = eng.gelu("enc.gelu2", x)
    pos = eng.sd["encoder.positional_embedding"][:audio_ctx]
    x = eng.site("enc.x", x.x + pos, bits=16)
    for l in range(N_LAYERS):
        p = f"encoder.blocks.{l}."
        n = f"enc.b{l}."
        y = eng.ln(n + "ln1", x, p + "attn_ln")
        q = eng.npu_fc(n + "q", y, p + "attn.query.weight",
                       p + "attn.query.bias", sym_out=True)
        k = eng.npu_fc(n + "k", y, p + "attn.key.weight", sym_out=True)
        v = eng.npu_fc(n + "v", y, p + "attn.value.weight",
                       p + "attn.value.bias", sym_out=True)
        c = eng.attention(n + "ctx", q, k, v)
        o = eng.npu_fc(n + "attn_out", c, p + "attn.out.weight",
                       p + "attn.out.bias")
        x = eng.add(n + "res1", x, o)
        y = eng.ln(n + "ln2", x, p + "mlp_ln")
        y = eng.npu_fc(n + "fc1", y, p + "mlp.0.weight", p + "mlp.0.bias")
        y = eng.gelu(n + "gelu", y)
        y = eng.npu_fc2_split(n + "fc2", y, p + "mlp.2.weight", p + "mlp.2.bias")
        x = eng.add(n + "res2", x, y)
    return eng.ln("enc.out", x, "encoder.ln_post", sym_out=True)


def cross_kv(eng, enc_out):
    """Per decoder layer, K/V over the encoder output (once per chunk)."""
    kv = []
    for l in range(N_LAYERS):
        p = f"decoder.blocks.{l}.cross_attn."
        n = f"dec.b{l}."
        k = eng.npu_fc(n + "xk", enc_out, p + "key.weight", sym_out=True)
        v = eng.npu_fc(n + "xv", enc_out, p + "value.weight",
                       p + "value.bias", sym_out=True)
        kv.append((k, v))
    return kv


class DecoderState:
    def __init__(self):
        self.self_kv = [([], []) for _ in range(N_LAYERS)]  # int8-site rows


def decoder_step(eng, token, pos_idx, state, xkv):
    """One greedy step -> dequantized logits [vocab]."""
    emb = (eng.sd["decoder.token_embedding.weight"][token]
           + eng.sd["decoder.positional_embedding"][pos_idx])
    x = eng.site("dec.x", emb[None, :], bits=16)
    for l in range(N_LAYERS):
        p = f"decoder.blocks.{l}."
        n = f"dec.b{l}."
        y = eng.ln(n + "ln1", x, p + "attn_ln")
        q = eng.npu_fc(n + "q", y, p + "attn.query.weight",
                       p + "attn.query.bias", sym_out=True)
        k = eng.npu_fc(n + "k", y, p + "attn.key.weight", sym_out=True)
        v = eng.npu_fc(n + "v", y, p + "attn.value.weight",
                       p + "attn.value.bias", sym_out=True)
        ks, vs = state.self_kv[l]
        ks.append(k)
        vs.append(v)
        kcat = Act(np.concatenate([a.x for a in ks]), k.scale, k.zp)
        vcat = Act(np.concatenate([a.x for a in vs]), v.scale, v.zp)
        c = eng.attention(n + "ctx", q, kcat, vcat)
        o = eng.npu_fc(n + "attn_out", c, p + "attn.out.weight",
                       p + "attn.out.bias")
        x = eng.add(n + "res1", x, o)
        y = eng.ln(n + "xln", x, p + "cross_attn_ln")
        q = eng.npu_fc(n + "xq", y, p + "cross_attn.query.weight",
                       p + "cross_attn.query.bias", sym_out=True)
        xk, xv = xkv[l]
        c = eng.attention(n + "xctx", q, xk, xv)
        o = eng.npu_fc(n + "xout", c, p + "cross_attn.out.weight",
                       p + "cross_attn.out.bias")
        x = eng.add(n + "res2", x, o)
        y = eng.ln(n + "ln2", x, p + "mlp_ln")
        y = eng.npu_fc(n + "fc1", y, p + "mlp.0.weight", p + "mlp.0.bias")
        y = eng.gelu(n + "gelu", y)
        y = eng.npu_fc2_split(n + "fc2", y, p + "mlp.2.weight", p + "mlp.2.bias")
        x = eng.add(n + "res3", x, y)
    x = eng.ln("dec.out", x, "decoder.ln")
    logits = eng.cpu_logits(x, "decoder.token_embedding.weight")
    return logits.x[0]


def greedy_decode(eng, enc_out, sot_sequence, eot, suppress, blank,
                  max_tokens=common.MAX_TOKENS, log_gaps=None):
    """Greedy decode against an encoder output. Returns generated tokens."""
    xkv = cross_kv(eng, enc_out)
    state = DecoderState()
    tokens = list(sot_sequence)
    # Prime the self-attention KV cache with the SOT prompt.
    for i, t in enumerate(tokens[:-1]):
        decoder_step(eng, t, i, state, xkv)
    out = []
    for step in range(max_tokens):
        logits = decoder_step(eng, tokens[-1], len(tokens) - 1, state, xkv)
        logits = logits.copy()
        logits[suppress] = -np.inf
        if step == 0:
            logits[blank] = -np.inf
        t = int(np.argmax(logits))
        if log_gaps is not None:
            srt = np.sort(logits[np.isfinite(logits)])
            log_gaps.append(float(srt[-1] - srt[-2]))
        if t == eot:
            break
        out.append(t)
        tokens.append(t)
    return out
