//! Plan-driven greedy decoder: the token loop lives here (a static tape
//! cannot express free-running decode, where KV-cache content depends on
//! sampled tokens). Every transformer computation runs ON THE DEVICE via
//! mailbox commands; the host pages weights/activations, keeps the KV
//! caches, and runs the LM head (final layernorm + vocab projection) in
//! f32 from the int16 residual it reads back.
//!
//!   whisper-host decode <firmware.elf> <plan_dir> <blobs_dir>
//!
//! The plan (model/decode_model.py export_plan) carries blob names and
//! every quantization parameter; addresses are laid out here.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use probe_rs::{Core, MemoryInterface};
use serde::Deserialize;

use crate::{setup, Mailbox, CMD_PING, CMD_RUN_NPU, SLOT_BASE};

const CMD_SD_INIT: u32 = 20;
const CMD_SD_READ: u32 = 21;

const CMD_LUT8: u32 = 3;
const CMD_LN: u32 = 8;
const CMD_ADD16: u32 = 9;
const CMD_ATTN_HEAD: u32 = 12;
const CMD_FC2SUM: u32 = 13;

#[derive(Deserialize, Clone, Copy)]
struct Q {
    scale: f32,
    zp: i32,
}

#[derive(Deserialize)]
struct BlockQ {
    #[serde(rename = "in")]
    q_in: Q,
    out: Q,
}

#[derive(Deserialize)]
struct Block {
    blobs: HashMap<String, String>,
    q: HashMap<String, BlockQ>,
    res1: Q,
    res2: Q,
    res3: Q,
}

#[derive(Deserialize)]
struct Plan {
    heads: usize,
    head_dim: usize,
    state: usize,
    audio_ctx: usize,
    pad_w: usize,
    t_tile: usize,
    max_tokens: usize,
    sot_sequence: Vec<u32>,
    eot: u32,
    blank: Vec<u32>,
    dec_x: Q,
    enc_out: Q,
    golden_tokens: Vec<u32>,
    blocks: Vec<Block>,
}

// Arena layout (mirrors nothing: the driver owns it; firmware memory.x
// fixes the region).
const A: u32 = 0x2003_2000;
const A_X16: u32 = A; // int16 [384,4]
const A_LN: u32 = A + 3072; // int8 [384,4]
const A_Q: u32 = A + 4608;
const A_K: u32 = A + 6144;
const A_V: u32 = A + 7680;
const A_CTX: u32 = A + 9216;
const A_O: u32 = A + 10752;
const A_P: u32 = A + 12288; // fc2 partials, 4 x 1536
const A_GB: u32 = A + 18432; // gamma+beta f32
const A_LUT: u32 = A + 21504;
const A_PAR: u32 = A + 21760;
const A_BIG: u32 = A + 22016; // K/V region (2 x 38400 cross, 2 x 18432 self)
const BIG_HALF: u32 = 38400;

const W: usize = 4; // token tensor width (pointwise conv minimum)
const C: usize = 384;

fn pack_q(v: &mut Vec<u8>, q: Q) {
    v.extend_from_slice(&q.scale.to_le_bytes());
    v.extend_from_slice(&q.zp.to_le_bytes());
}

struct Dev<'a> {
    core: Core<'a>,
    mb: Mailbox,
    blobs_dir: PathBuf,
    loaded: String,
    streamed: usize,
    /// blob name -> (lba, bytes) when weights come from the SD card
    sd_index: Option<HashMap<String, (u32, u32)>>,
}

impl<'a> Dev<'a> {
    fn blob(&mut self, name: &str) -> Result<()> {
        if self.loaded == name {
            return Ok(());
        }
        if let Some(ix) = &self.sd_index {
            let (lba, bytes) = *ix
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("{name} not on the card"))?;
            let blocks = bytes.div_ceil(512);
            let rc = self.mb.call(&mut self.core, CMD_SD_READ,
                                  &[lba, SLOT_BASE as u32, blocks])?;
            if rc != 0 {
                bail!("SD read of {name} failed with {rc}");
            }
        } else {
            let data = std::fs::read(self.blobs_dir.join(format!("{name}.bin")))
                .with_context(|| format!("blob {name}"))?;
            self.core.write(SLOT_BASE, &data)?;
            self.streamed += data.len();
        }
        self.loaded = name.to_string();
        Ok(())
    }

    fn write(&mut self, addr: u32, data: &[u8]) -> Result<()> {
        self.core.write(addr as u64, data)?;
        self.streamed += data.len();
        Ok(())
    }

    fn cmd(&mut self, code: u32, args: &[u32]) -> Result<()> {
        let rc = self.mb.call(&mut self.core, code, args)?;
        if rc != 0 {
            bail!("device cmd {code} failed with {rc}");
        }
        Ok(())
    }

    fn npu(&mut self, blob: &str, input: u32, output: u32) -> Result<()> {
        self.blob(blob)?;
        self.cmd(CMD_RUN_NPU, &[input, output])
    }

    fn ln(&mut self, gb: &[u8], src: u32, dst: u32, w: usize, sq: Q, dq: Q) -> Result<()> {
        self.write(A_GB, gb)?;
        let mut p = Vec::new();
        for v in [src, dst, C as u32, w as u32, A_GB, A_GB + 4 * C as u32] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        pack_q(&mut p, sq);
        pack_q(&mut p, dq);
        self.write(A_PAR, &p)?;
        self.cmd(CMD_LN, &[A_PAR])
    }

    fn add16(&mut self, a: u32, b: u32, dst: u32, len: usize, qa: Q, qb: Q, qd: Q) -> Result<()> {
        let mut p = Vec::new();
        for v in [a, b, dst, len as u32] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        pack_q(&mut p, qa);
        pack_q(&mut p, qb);
        pack_q(&mut p, qd);
        self.write(A_PAR, &p)?;
        self.cmd(CMD_ADD16, &[A_PAR])
    }

    #[allow(clippy::too_many_arguments)]
    fn attn(&mut self, q: u32, k: u32, v: u32, ctx: u32, tk: usize, kstride: usize,
            zq: i32, zk: i32, zv: i32, score_mult: f32, v_scale: f32, ctx_q: Q) -> Result<()> {
        let mut p = Vec::new();
        for x in [q, k, v, ctx, 64, 1, W as u32, tk as u32, kstride as u32] {
            p.extend_from_slice(&x.to_le_bytes());
        }
        for x in [zq, zk, zv] {
            p.extend_from_slice(&x.to_le_bytes());
        }
        p.extend_from_slice(&score_mult.to_le_bytes());
        p.extend_from_slice(&v_scale.to_le_bytes());
        pack_q(&mut p, ctx_q);
        self.write(A_PAR, &p)?;
        self.cmd(CMD_ATTN_HEAD, &[A_PAR])
    }

    fn fc2sum(&mut self, pq: [Q; 4], qa: Q, qd: Q) -> Result<()> {
        let mut p = Vec::new();
        for j in 0..4u32 {
            p.extend_from_slice(&(A_P + j * 1536).to_le_bytes());
        }
        for v in [A_X16, A_X16, (C * W) as u32] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        for q in pq {
            pack_q(&mut p, q);
        }
        pack_q(&mut p, qa);
        pack_q(&mut p, qd);
        self.write(A_PAR, &p)?;
        self.cmd(CMD_FC2SUM, &[A_PAR])
    }
}

fn load_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("{path:?}"))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn quant16(x: f32, q: Q) -> i16 {
    (((x / q.scale).round() as i32) + q.zp).clamp(-32768, 32767) as i16
}

fn requant8(v: i8, from: Q, to: Q) -> i8 {
    let f = (v as i32 - from.zp) as f32 * from.scale;
    ((f / to.scale).round() as i32 + to.zp).clamp(-128, 127) as i8
}

pub fn decode(elf: &str, plan_dir: &str, blobs_dir: &str, use_sd: bool) -> Result<()> {
    let dir = Path::new(plan_dir);
    let plan: Plan = serde_json::from_str(
        &std::fs::read_to_string(dir.join("plan.json")).context("plan.json")?)?;
    assert_eq!(plan.state, C);
    let hd = plan.head_dim;

    let pos = load_f32(&dir.join("pos.f32.bin"))?;
    let final_gb = load_f32(&dir.join("final_gb.bin"))?;
    let suppress: Vec<u32> = std::fs::read(dir.join("suppress.u32.bin"))?
        .chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let vocab: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("vocab.json"))?)?;
    let enc_out = std::fs::read(dir.join("enc_out.i8.bin"))?; // [384][pad_w]
    let mut emb_file = std::fs::File::open(dir.join("emb.f32.bin"))?;
    let mut gbs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut luts: HashMap<String, Vec<u8>> = HashMap::new();
    for l in 0..plan.blocks.len() {
        for stem in ["ln1", "xln", "ln2"] {
            gbs.insert(format!("{l}:{stem}"),
                       std::fs::read(dir.join(format!("b{l}_{stem}_gb.bin")))?);
        }
        for j in 0..4 {
            luts.insert(format!("{l}:{j}"),
                        std::fs::read(dir.join(format!("b{l}_lut{j}.bin")))?);
        }
    }

    let (mut session, mailbox_addr, _rtt) = setup(elf)?;
    let core = session.core(0)?;
    let sd_index = if use_sd {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(Path::new(plan_dir).join("../sd-index.json"))
                .context("sd-index.json")?)?;
        Some(raw.into_iter().map(|(k, v)| {
            (k, (v["lba"].as_u64().unwrap() as u32,
                 v["bytes"].as_u64().unwrap() as u32))
        }).collect())
    } else {
        None
    };
    let mut dev = Dev {
        mb: Mailbox { base: mailbox_addr, seq: 0 },
        core,
        blobs_dir: PathBuf::from(blobs_dir),
        loaded: String::new(),
        streamed: 0,
        sd_index,
    };
    dev.mb.seq = dev.core.read_word_32(mailbox_addr + 4)?;
    dev.mb.call(&mut dev.core, CMD_PING, &[])?;
    if dev.sd_index.is_some() {
        let rc = dev.mb.call(&mut dev.core, CMD_SD_INIT, &[])?;
        if rc != 0 {
            bail!("SD init failed with {rc}");
        }
        eprintln!("weights from the SD card");
    }

    // --- cross K/V, computed on the NPU once per chunk -------------------
    eprintln!("computing cross K/V on the device ...");
    let t0 = Instant::now();
    let n_tiles = plan.pad_w / plan.t_tile;
    let tile_bytes = C * plan.t_tile;
    // per layer: planar [384][audio_ctx] i8, pad columns cropped
    let mut cross: Vec<(Vec<i8>, Vec<i8>)> = Vec::new();
    for (l, blk) in plan.blocks.iter().enumerate() {
        let xin = &blk.q["xk"].q_in;
        assert_eq!(blk.q["xv"].q_in.scale, xin.scale);
        let re: Vec<u8> = enc_out.iter()
            .map(|&b| requant8(b as i8, plan.enc_out, *xin) as u8).collect();
        let mut kv = (vec![0i8; C * plan.audio_ctx], vec![0i8; C * plan.audio_ctx]);
        for (which, out) in [("xk", &mut kv.0), ("xv", &mut kv.1)] {
            for i in 0..n_tiles {
                let mut tile = vec![0u8; tile_bytes];
                for c in 0..C {
                    let s = c * plan.pad_w + i * plan.t_tile;
                    tile[c * plan.t_tile..(c + 1) * plan.t_tile]
                        .copy_from_slice(&re[s..s + plan.t_tile]);
                }
                dev.write(A_BIG, &tile)?;
                dev.npu(&plan.blocks[l].blobs[which], A_BIG, A_BIG + tile_bytes as u32)?;
                let mut o = vec![0u8; tile_bytes];
                dev.core.read_8((A_BIG + tile_bytes as u32) as u64, &mut o)?;
                for c in 0..C {
                    for t in 0..plan.t_tile {
                        let col = i * plan.t_tile + t;
                        if col < plan.audio_ctx {
                            out[c * plan.audio_ctx + col] = o[c * plan.t_tile + t] as i8;
                        }
                    }
                }
            }
        }
        cross.push(kv);
        eprintln!("  layer {l} done");
    }
    eprintln!("cross K/V in {:.0} s", t0.elapsed().as_secs_f64());

    // --- token loop ------------------------------------------------------
    let mut kcache: Vec<Vec<[i8; C]>> = vec![Vec::new(); plan.blocks.len()];
    let mut vcache: Vec<Vec<[i8; C]>> = vec![Vec::new(); plan.blocks.len()];
    let mut tokens: Vec<u32> = plan.sot_sequence.clone();
    let mut text = String::new();
    let mut mismatches = 0usize;
    let t_all = Instant::now();

    for step in 0..(plan.sot_sequence.len() - 1 + plan.max_tokens) {
        let t_tok = Instant::now();
        let pos_idx = step;
        let token = tokens[step] as usize;

        // x16 = quantize(emb[token] + pos[step])
        let mut row = vec![0u8; C * 4];
        emb_file.seek(SeekFrom::Start((token * C * 4) as u64))?;
        emb_file.read_exact(&mut row)?;
        let mut x16 = vec![0u8; C * W * 2];
        for c in 0..C {
            let e = f32::from_le_bytes(row[c * 4..c * 4 + 4].try_into().unwrap());
            let v = quant16(e + pos[pos_idx * C + c], plan.dec_x);
            x16[c * W * 2..c * W * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        dev.write(A_X16, &x16)?;

        let mut sq = plan.dec_x;
        for (l, blk) in plan.blocks.iter().enumerate() {
            let q = |k: &str| &blk.q[k];
            // self-attention
            dev.ln(&gbs[&format!("{l}:ln1")], A_X16, A_LN, W, sq, q("q").q_in)?;
            dev.npu(&blk.blobs["q"], A_LN, A_Q)?;
            dev.npu(&blk.blobs["k"], A_LN, A_K)?;
            dev.npu(&blk.blobs["v"], A_LN, A_V)?;
            for (addr, cache) in [(A_K, &mut kcache[l]), (A_V, &mut vcache[l])] {
                let mut buf = vec![0u8; C * W];
                dev.core.read_8(addr as u64, &mut buf)?;
                let mut col = [0i8; C];
                for c in 0..C {
                    col[c] = buf[c * W] as i8;
                }
                cache.push(col);
            }
            let t = kcache[l].len();
            for (base, cache) in [(A_BIG, &kcache[l]), (A_BIG + BIG_HALF, &vcache[l])] {
                let mut buf = vec![0u8; C * t];
                for (ti, col) in cache.iter().enumerate() {
                    for c in 0..C {
                        buf[c * t + ti] = col[c] as u8;
                    }
                }
                dev.write(base, &buf)?;
            }
            let sm = q("q").out.scale * q("k").out.scale / (hd as f32).sqrt();
            for h in 0..plan.heads {
                dev.attn(A_Q + (h * hd * W) as u32,
                         A_BIG + (h * hd * t) as u32,
                         A_BIG + BIG_HALF + (h * hd * t) as u32,
                         A_CTX + (h * hd * W) as u32,
                         t, t,
                         q("q").out.zp, q("k").out.zp, q("v").out.zp,
                         sm, q("v").out.scale, q("out").q_in)?;
            }
            dev.npu(&blk.blobs["out"], A_CTX, A_O)?;
            dev.add16(A_X16, A_O, A_X16, C * W, sq, q("out").out, blk.res1)?;

            // cross-attention
            dev.ln(&gbs[&format!("{l}:xln")], A_X16, A_LN, W, blk.res1, q("xq").q_in)?;
            dev.npu(&blk.blobs["xq"], A_LN, A_Q)?;
            let smx = q("xq").out.scale * q("xk").out.scale / (hd as f32).sqrt();
            for h in 0..plan.heads {
                let (ck, cv) = &cross[l];
                let s = h * hd * plan.audio_ctx;
                let kb: Vec<u8> = ck[s..s + hd * plan.audio_ctx].iter().map(|&x| x as u8).collect();
                let vb: Vec<u8> = cv[s..s + hd * plan.audio_ctx].iter().map(|&x| x as u8).collect();
                dev.write(A_BIG, &kb)?;
                dev.write(A_BIG + BIG_HALF, &vb)?;
                dev.attn(A_Q + (h * hd * W) as u32, A_BIG, A_BIG + BIG_HALF,
                         A_CTX + (h * hd * W) as u32,
                         plan.audio_ctx, plan.audio_ctx,
                         q("xq").out.zp, q("xk").out.zp, q("xv").out.zp,
                         smx, q("xv").out.scale, q("xout").q_in)?;
            }
            dev.npu(&blk.blobs["xout"], A_CTX, A_O)?;
            dev.add16(A_X16, A_O, A_X16, C * W, blk.res1, q("xout").out, blk.res2)?;

            // mlp
            dev.ln(&gbs[&format!("{l}:ln2")], A_X16, A_LN, W, blk.res2, q("fc1a").q_in)?;
            for j in 0..4 {
                let part = ["a", "b", "c", "d"][j];
                dev.npu(&blk.blobs[&format!("fc1{part}")], A_LN, A_Q)?;
                dev.write(A_LUT, &luts[&format!("{l}:{j}")])?;
                dev.cmd(CMD_LUT8, &[A_LUT, A_Q, A_K, (C * W) as u32])?;
                dev.npu(&blk.blobs[&format!("fc2p{j}")], A_K, A_P + (j * C * W) as u32)?;
            }
            let pq = [blk.q["fc2p0"].out, blk.q["fc2p1"].out,
                      blk.q["fc2p2"].out, blk.q["fc2p3"].out];
            dev.fc2sum(pq, blk.res2, blk.res3)?;
            sq = blk.res3;
        }

        // LM head on the host: f32 LN from the int16 residual + vocab argmax
        let mut buf = vec![0u8; C * W * 2];
        dev.core.read_8(A_X16 as u64, &mut buf)?;
        let mut x = [0f32; C];
        for c in 0..C {
            let v = i16::from_le_bytes([buf[c * W * 2], buf[c * W * 2 + 1]]);
            x[c] = (v as i32 - sq.zp) as f32 * sq.scale;
        }
        let mean = x.iter().sum::<f32>() / C as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / C as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let hid: Vec<f32> = (0..C)
            .map(|c| (x[c] - mean) * inv * final_gb[c] + final_gb[C + c])
            .collect();

        if step < plan.sot_sequence.len() - 1 {
            continue; // prompt token: no sampling
        }
        let out_idx = step - (plan.sot_sequence.len() - 1);
        let mut best = f32::MIN;
        let mut best_id = 0u32;
        emb_file.seek(SeekFrom::Start(0))?;
        let mut rd = std::io::BufReader::with_capacity(1 << 20, &emb_file);
        let mut row = vec![0u8; C * 4];
        for id in 0..vocab.len() as u32 {
            rd.read_exact(&mut row)?;
            let banned = suppress.binary_search(&id).is_ok()
                || (out_idx == 0 && plan.blank.contains(&id));
            if !banned {
                let mut s = 0f32;
                for (c, ch) in row.chunks_exact(4).enumerate() {
                    s += f32::from_le_bytes(ch.try_into().unwrap()) * hid[c];
                }
                if s > best {
                    best = s;
                    best_id = id;
                }
            }
        }
        let golden = plan.golden_tokens.get(out_idx).copied();
        let mark = match golden {
            Some(g) if g != best_id && best_id != plan.eot =>
                { mismatches += 1; "  <-- device-model predicted different" }
            None if best_id != plan.eot => { mismatches += 1; "  (beyond golden)" }
            _ => "",
        };
        eprintln!("token {out_idx:2}: {best_id:6} {:?} ({:.0} s){mark}",
                  vocab[best_id as usize], t_tok.elapsed().as_secs_f64());
        if best_id == plan.eot {
            break;
        }
        text.push_str(&vocab[best_id as usize]);
        tokens.push(best_id);
    }

    eprintln!("\nTRANSCRIPT: {text:?}");
    eprintln!("{} tokens, {} MB streamed, {:.0} s total, {} deviations vs device-model",
              tokens.len() - plan.sot_sequence.len(),
              dev.streamed / (1024 * 1024),
              t_all.elapsed().as_secs_f64(), mismatches);
    Ok(())
}
