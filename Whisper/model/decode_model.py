"""Device-model greedy decoder + plan export for the host decode driver.

Runs the decoder exactly as the DK will: interpreter submodels (quant
params authoritative), kernel-mirror glue, host-side logits over the int8
embedding matrix. The predicted transcript is the hardware pass bar.

Also emits out/decoder-plan.json plus asset files (embedding tables, LN
params, GELU LUTs, suppress list, vocabulary) for `whisper-host decode`.

Everything token-rate runs at width 4 (only column 0 meaningful; pointwise
stages are per-position, so the padding columns never contaminate col 0).

Requires: out/enc_out.npz (tape.py encoder), decoder submodels + blobs.
"""

import json
import os


import numpy as np

import common

import tape
from tape import Submodel, ln_golden, attn_golden, gelu_lut

MAX_TOKENS = 40
W = 4  # token tensor width


def load_all():
    _, sd = common.load_weights()
    with open(os.path.join(common.OUT, "scales.json")) as f:
        scales = {k: (v["scale"], v["zp"], v["bits"])
                  for k, v in json.load(f).items()}
    enc = np.load(os.path.join(common.OUT, "enc_out.npz"))
    ref = np.load(os.path.join(common.OUT, "ref.npz"))
    mods = {}
    for l in range(common.N_LAYERS):
        for k, name in common.decoder_submodel_names(l).items():
            mods[(l, k)] = Submodel(name)
    return sd, scales, enc, ref, mods


def build_suppress(ref):
    """Proper whisper default suppression (mirrors simulate.build_suppress)."""
    import whisper
    from whisper.tokenizer import get_tokenizer

    model = whisper.load_model(common.MODEL_NAME, device="cpu",
                               download_root=os.path.join(common.DATA, "ckpt"))
    tok = get_tokenizer(model.is_multilingual,
                        num_languages=model.num_languages,
                        language="en", task="transcribe")
    s = set(tok.non_speech_tokens)
    s.update([tok.transcribe, tok.translate, tok.sot, tok.sot_prev,
              tok.sot_lm])
    if tok.no_speech is not None:
        s.add(tok.no_speech)
    s.update(range(tok.timestamp_begin, tok.timestamp_begin + 1501))
    return np.array(sorted(t for t in s if t < 51864), dtype=np.int64), tok


def quantize_emb(sd):
    """Per-row symmetric int8 embedding for host-side logits."""
    emb = sd["decoder.token_embedding.weight"]  # [vocab, 384] f32
    amax = np.maximum(np.abs(emb).max(1), 1e-9)
    rs = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(emb / rs[:, None]), -127, 127).astype(np.int8)
    return q, rs


def requant8(a, from_q, to_q):
    """int8 -> int8 under a different (scale, zp). Every submodel boundary
    must either match by construction or pass through this."""
    f = (a.astype(np.float32) - from_q[1]) * np.float32(from_q[0])
    return common.quantize(f, to_q[0], to_q[1])


def cross_kv(enc, mods):
    """Cross K/V per layer over encoder frame tiles (as the device runs it).
    enc.out's quantization is NOT the xk/xv input quantization -- requantize
    (feeding the planar directly misreads every key by a scale factor and a
    zero-point shift; found as a systematic cross-attention bias)."""
    out = []
    for l in range(common.N_LAYERS):
        m = mods[(l, "xk")]
        ep = requant8(enc["planar"], (float(enc["scale"]), int(enc["zp"])),
                      m.in_q)
        assert mods[(l, "xv")].in_q == m.in_q
        ks, vs = [], []
        for i in range(ep.shape[1] // tape.T_TILE):
            tile = np.ascontiguousarray(
                ep[:, i * tape.T_TILE:(i + 1) * tape.T_TILE])
            ks.append(m.run(tile))
            vs.append(mods[(l, "xv")].run(tile))
        out.append((np.concatenate(ks, 1), np.concatenate(vs, 1)))
    return out


def pad4(col):
    """[C] int8/int16 -> planar [C, 4] with zero pad columns."""
    x = np.zeros((col.shape[0], W), col.dtype)
    x[:, 0] = col
    return x


class Decoder:
    """The device pipeline, one token at a time."""

    def __init__(self, sd, scales, mods, xkv):
        self.sd, self.scales, self.mods, self.xkv = sd, scales, mods, xkv
        self.kcache = [[] for _ in range(common.N_LAYERS)]
        self.vcache = [[] for _ in range(common.N_LAYERS)]
        self.x_q = scales["dec.x"]
        self.out_q = scales["dec.out"]

    def step(self, token, pos_idx):
        sd, scales, mods = self.sd, self.scales, self.mods
        emb = (sd["decoder.token_embedding.weight"][token]
               + sd["decoder.positional_embedding"][pos_idx])
        x16 = pad4(common.quantize(emb, self.x_q[0], self.x_q[1], bits=16))
        sq = self.x_q
        for l in range(common.N_LAYERS):
            p = f"decoder.blocks.{l}."
            m = {k: mods[(l, k)] for k in common.decoder_submodel_names(l)}
            r1 = scales[f"dec.b{l}.res1"]
            r2 = scales[f"dec.b{l}.res2"]
            r3 = scales[f"dec.b{l}.res3"]

            # self-attention over the growing KV cache
            ln1 = ln_golden(x16, sq, sd[p + "attn_ln.weight"],
                            sd[p + "attn_ln.bias"], m["q"].in_q)
            gq = m["q"].run(ln1)
            self.kcache[l].append(m["k"].run(ln1)[:, 0])
            self.vcache[l].append(m["v"].run(ln1)[:, 0])
            kc = np.stack(self.kcache[l], axis=1)  # planar [384, t]
            vc = np.stack(self.vcache[l], axis=1)
            ctx = attn_golden(gq[:, :1], kc, vc, m["q"].out_q, m["k"].out_q,
                              m["v"].out_q, m["out"].in_q)
            o = m["out"].run(pad4(ctx[:, 0]))
            x16 = tape.add16_golden(x16, sq, o, m["out"].out_q, r1)

            # cross-attention against the per-chunk K/V
            xk, xv = self.xkv[l]
            xln = ln_golden(x16, r1, sd[p + "cross_attn_ln.weight"],
                            sd[p + "cross_attn_ln.bias"], m["xq"].in_q)
            gxq = m["xq"].run(xln)
            xctx = attn_golden(gxq[:, :1], xk, xv, m["xq"].out_q,
                               m["xk"].out_q, m["xv"].out_q, m["xout"].in_q,
                               tk=common.AUDIO_CTX)
            xo = m["xout"].run(pad4(xctx[:, 0]))
            x16 = tape.add16_golden(x16, r1, xo, m["xout"].out_q, r2)

            # mlp
            ln2 = ln_golden(x16, r2, sd[p + "mlp_ln.weight"],
                            sd[p + "mlp_ln.bias"], m["fc1a"].in_q)
            parts, pqs = [], []
            for j, part in enumerate("abcd"):
                f1 = m[f"fc1{part}"].run(ln2)
                lut = gelu_lut(m[f"fc1{part}"].out_q, m[f"fc2p{j}"].in_q)
                g = lut[f1.astype(np.int32) + 128]
                parts.append(m[f"fc2p{j}"].run(g))
                pqs.append(m[f"fc2p{j}"].out_q)
            x16 = tape.fc2sum_golden(parts, pqs, x16, r2, r3)
            sq = r3

        # The LM head (final LN + vocab projection) runs on the HOST in f32
        # from the int16 residual: it decides tokens, and knife-edge logit
        # gaps (0.66 on this clip) do not survive an int8 hidden state.
        x = tape.dq(x16[:, 0], sq)
        mean, var = x.mean(), x.var()
        y = (x - mean) / np.sqrt(var + 1e-5)
        return (y * self.sd["decoder.ln.weight"]
                + self.sd["decoder.ln.bias"]).astype(np.float32)


def logits_f32(hid, emb):
    """Host-side logits: f32 hidden x f32 embedding."""
    return emb @ hid


def main():
    sd, scales, enc, ref, mods = load_all()
    suppress, tok = build_suppress(ref)
    emb = sd["decoder.token_embedding.weight"].astype(np.float32)
    print("computing cross K/V ...")
    xkv = cross_kv(enc, mods)

    dec = Decoder(sd, scales, mods, xkv)
    tokens = list(int(t) for t in ref["sot_sequence"])
    eot = int(ref["eot"])
    blank = ref["blank"]
    out = []
    for i, t in enumerate(tokens[:-1]):
        dec.step(t, i)
    for step in range(MAX_TOKENS):
        hid = dec.step(tokens[-1], len(tokens) - 1)
        logits = logits_f32(hid, emb)
        logits[suppress] = -np.inf
        if step == 0:
            logits[blank] = -np.inf
        nxt = int(np.argmax(logits))
        if nxt == eot:
            break
        out.append(nxt)
        tokens.append(nxt)
        print(f"  token {step}: {nxt} {tok.decode([nxt])!r}")

    text = tok.decode(out)
    print(f"\ndevice-model transcript: {text!r}")
    print(f"reference:               {str(ref['text'])!r}")
    export_plan(sd, scales, mods, out, suppress, tok)


def export_plan(sd, scales, mods, golden_tokens, suppress, tok):
    """Assets + plan for the Rust decode driver."""
    d = os.path.join(common.OUT, "decoder-plan")
    os.makedirs(d, exist_ok=True)

    sd["decoder.token_embedding.weight"].astype("<f4").tofile(
        os.path.join(d, "emb.f32.bin"))
    sd["decoder.positional_embedding"].astype("<f4").tofile(
        os.path.join(d, "pos.f32.bin"))
    emb_q, row_scales = quantize_emb(sd)
    emb_q.tofile(os.path.join(d, "emb.i8.bin"))
    row_scales.astype("<f4").tofile(os.path.join(d, "emb_scales.f32.bin"))
    suppress.astype("<u4").tofile(os.path.join(d, "suppress.u32.bin"))
    with open(os.path.join(d, "vocab.json"), "w") as f:
        json.dump([tok.decode([i]) for i in range(51864)], f)

    def qj(q):
        return {"scale": float(q[0]), "zp": int(q[1])}

    blocks = []
    for l in range(common.N_LAYERS):
        p = f"decoder.blocks.{l}."
        n = common.decoder_submodel_names(l)
        m = {k: mods[(l, k)] for k in n}
        for stem, w in [("ln1", "attn_ln"), ("xln", "cross_attn_ln"),
                        ("ln2", "mlp_ln")]:
            gb = (sd[p + w + ".weight"].astype("<f4").tobytes()
                  + sd[p + w + ".bias"].astype("<f4").tobytes())
            with open(os.path.join(d, f"b{l}_{stem}_gb.bin"), "wb") as f:
                f.write(gb)
        for j, part in enumerate("abcd"):
            gelu_lut(m[f"fc1{part}"].out_q, m[f"fc2p{j}"].in_q).tofile(
                os.path.join(d, f"b{l}_lut{j}.bin"))
        blocks.append({
            "blobs": {k: n[k] for k in n},
            "q": {k: {"in": qj(m[k].in_q), "out": qj(m[k].out_q)} for k in n},
            "res1": qj(scales[f"dec.b{l}.res1"]),
            "res2": qj(scales[f"dec.b{l}.res2"]),
            "res3": qj(scales[f"dec.b{l}.res3"]),
        })
    final_gb = (sd["decoder.ln.weight"].astype("<f4").tobytes()
                + sd["decoder.ln.bias"].astype("<f4").tobytes())
    with open(os.path.join(d, "final_gb.bin"), "wb") as f:
        f.write(final_gb)

    plan = {
        "heads": common.HEADS, "head_dim": common.HEAD_DIM,
        "state": common.STATE, "audio_ctx": common.AUDIO_CTX,
        "pad_w": tape.PAD_W, "t_tile": tape.T_TILE,
        "max_tokens": MAX_TOKENS,
        "sot_sequence": [int(t) for t in
                         np.load(os.path.join(common.OUT, "ref.npz"))
                         ["sot_sequence"]],
        "eot": int(np.load(os.path.join(common.OUT, "ref.npz"))["eot"]),
        "blank": [int(t) for t in
                  np.load(os.path.join(common.OUT, "ref.npz"))["blank"]],
        "dec_x": qj(scales["dec.x"]),
        "dec_out": qj(scales["dec.out"]),
        "enc_out": qj((np.load(os.path.join(common.OUT, "enc_out.npz"))
                       ["scale"],
                       np.load(os.path.join(common.OUT, "enc_out.npz"))
                       ["zp"])),
        "golden_tokens": golden_tokens,
        "blocks": blocks,
    }
    with open(os.path.join(d, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    # the encoder output itself, for the driver
    np.load(os.path.join(common.OUT, "enc_out.npz"))["planar"].tofile(
        os.path.join(d, "enc_out.i8.bin"))
    print(f"decoder plan + assets -> {d}")


if __name__ == "__main__":
    main()
