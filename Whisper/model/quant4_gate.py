"""Multi-clip gate for the level-coded 4-bit decoder (speed pass 8).

For every clip: device-style log-mel (whisper's mel plus the firmware's
level lift), the int8 encoder simulation (layered.Engine on
out/scales.json), cross K/V and a greedy decode through the interpreter
submodels twice -- the int8 decoder blobs as shipped, then copies whose
384x384 filters are requantized with quant4.requant_lvl (written to
out/submodels-q4l/, which is also the compile input for the 4-bit blobs,
so they are kept). Reports the float whisper transcript for context, the
teacher-forced agreement of the 4-bit decoder against the int8 one, and
both free-running transcripts. Both decodes use the device LM head (int8
rows x f32 hidden x row scale).

Run: PYTHONPATH=. pixi run python quant4_gate.py [clip.wav ...]
Default clips: the JFK reference, three perturbations of it (slower,
faster, 20 dB white noise) and the microphone captures under out/.
"""

import os
import sys
import wave

import numpy as np

import common
import decode_model
import layered
import quant4
import tape
from quant4_check import SM, PER_TOKEN_KINDS, run_decode

MEL_TARGET_MAX = 1.845  # firmware app.rs: peak log10-mel of the JFK clip
MEL_MIN_LIFT, MEL_MAX_LIFT = -2.0, 4.0


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == common.SAMPLE_RATE and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def perturb(audio, kind):
    rng = np.random.default_rng(3)
    if kind.startswith("speed"):
        f = float(kind[5:])
        n = int(len(audio) / f)
        return np.interp(np.arange(n) * f, np.arange(len(audio)), audio) \
            .astype(np.float32)
    if kind.startswith("noise"):
        snr_db = float(kind[5:])
        sig = np.sqrt(np.mean(audio ** 2))
        noise = rng.standard_normal(len(audio)).astype(np.float32)
        noise *= sig / (10 ** (snr_db / 20))
        return np.clip(audio + noise, -1, 1)
    raise ValueError(kind)


def device_mel(audio):
    """[80, 1200] normalized log-mel as the firmware feeds the encoder:
    whisper's mel of the 12 s chunk, lifted so the peak sits where the
    calibration clip's did (a constant shift of the normalized mel by
    lift/4, since the clamp at max-8 is relative)."""
    import whisper

    chunk = whisper.pad_or_trim(audio, common.AUDIO_CTX * 2 * common.HOP)
    mel = whisper.log_mel_spectrogram(chunk).numpy().astype(np.float32)
    peak = float(mel.max()) * 4.0 - 4.0
    lift = float(np.clip(MEL_TARGET_MAX - peak, MEL_MIN_LIFT, MEL_MAX_LIFT))
    return mel + np.float32(lift / 4.0), lift


def float_transcript(model, audio):
    import whisper
    from whisper.decoding import DecodingOptions

    mel = whisper.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)
    mel = mel[:, : whisper.audio.N_FRAMES]
    opts = DecodingOptions(language="en", task="transcribe",
                           without_timestamps=True, temperature=0.0,
                           fp16=False)
    return whisper.decode(model, mel, opts).text


def patched_mods(mods0):
    """Level-coded copies of the per-token submodels (xk/xv stay int8)."""
    import tensorflow as tf

    pdir = os.path.join(common.OUT, "submodels-q4l"
                        + ("" if quant4.GL == 16 else str(quant4.GL)))
    os.makedirs(pdir, exist_ok=True)
    mods, stats = {}, []
    for l in range(common.N_LAYERS):
        for k, name in common.decoder_submodel_names(l).items():
            if k not in PER_TOKEN_KINDS:
                mods[(l, k)] = mods0[(l, k)]
                continue
            src = os.path.join(common.OUT, "submodels", f"{name}.tflite")
            dst = os.path.join(pdir, f"{name}.tflite")
            interp = tf.lite.Interpreter(model_path=src)
            interp.allocate_tensors()
            w = None
            for d in interp.get_tensor_details():
                if d["dtype"] == np.int8 and int(np.prod(d["shape"])) == 384 * 384:
                    try:
                        w = interp.get_tensor(d["index"])
                    except ValueError:
                        pass
            assert w is not None, name
            w2 = quant4.requant_lvl(w)
            raw = open(src, "rb").read()
            b = w.tobytes()
            assert raw.count(b) == 1, name
            if not os.path.exists(dst) or open(dst, "rb").read().count(w2.tobytes()) != 1:
                open(dst, "wb").write(raw.replace(b, w2.tobytes()))
            d = w2.astype(np.float32) - w.astype(np.float32)
            stats.append((name, float(np.sqrt((d * d).mean())), int(np.abs(d).max())))
            mods[(l, k)] = SM(dst)
    return mods, stats


def main():
    import whisper

    sd, scales, _enc, ref, mods0 = decode_model.load_all()
    suppress, tok = decode_model.build_suppress(ref)
    model = whisper.load_model(common.MODEL_NAME, device="cpu",
                               download_root=os.path.join(common.DATA, "ckpt"))
    emb_q, emb_rs = decode_model.quantize_emb(sd)
    embf = emb_q.astype(np.float32)

    def lm8(hid):
        return (embf @ hid) * emb_rs

    mods4, stats = patched_mods(mods0)
    rms = np.mean([r for _, r, _ in stats])
    worst = sorted(stats, key=lambda s: -s[1])[:3]
    print(f"level-coded weights (G={quant4.GL}): avg rms err {rms:.2f} LSB, "
          "worst " + ", ".join(f"{n} {r:.2f}/max{m}" for n, r, m in worst))

    clips = sys.argv[1:]
    if not clips:
        jfk = os.path.join(common.OUT, "jfk16k.wav")
        clips = [jfk, jfk + "#speed0.92", jfk + "#speed1.08", jfk + "#noise20"]
        clips += sorted(os.path.join(common.OUT, f) for f in os.listdir(common.OUT)
                        if f.startswith("mic-") and f.endswith(".wav"))

    same = 0
    for clip in clips:
        path, _, kind = clip.partition("#")
        audio = read_wav(path)
        if kind:
            audio = perturb(audio, kind)
        label = os.path.basename(path) + (f" [{kind}]" if kind else "")
        mel, lift = device_mel(audio)
        eng = layered.Engine(sd, "int8", scales)
        enc = layered.encoder(eng, mel, common.AUDIO_CTX)
        planar = np.full((common.STATE, tape.PAD_W), enc.zp, np.int8)
        q = enc.q().T  # [384, 600]
        planar[:, :q.shape[1]] = q
        xkv = decode_model.cross_kv({"planar": planar, "scale": enc.scale,
                                     "zp": enc.zp}, mods0)
        base, _, _ = run_decode(sd, scales, mods0, xkv, ref, suppress, lm8)
        _, _, agree = run_decode(sd, scales, mods4, xkv, ref, suppress, lm8,
                                 force=base)
        free, _, _ = run_decode(sd, scales, mods4, xkv, ref, suppress, lm8)
        eq = free == base
        same += eq
        print(f"\n== {label}  (mel lift {lift:+.2f})")
        print(f"  float whisper: {float_transcript(model, audio)!r}")
        print(f"  int8 decoder:  {tok.decode(base)!r}")
        print(f"  4-bit decoder: {tok.decode(free)!r}"
              f"  [{'== int8' if eq else 'DIFFERS'}], teacher-forced "
              f"{sum(agree)}/{len(agree)}")
        sys.stdout.flush()
    print(f"\n{same}/{len(clips)} clips: 4-bit transcript identical to int8")


if __name__ == "__main__":
    main()
