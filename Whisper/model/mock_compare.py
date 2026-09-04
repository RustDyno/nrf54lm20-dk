"""Compare what the device actually computed against the host mirror.

The mock-usb rig (tools/mockusb) serves the model image out of a work file,
so every scratch block the device writes lands back in that file: the mel
it derived, the encoder output, the cross K/V, and the PCM it recorded.
Nothing extra has to be dumped off the board -- this just reads them.

    pixi run python mock_compare.py [out/mock-work.img]

With injected audio the mel is directly comparable to the reference clip's
own mel; with a live microphone run the PCM section is the interesting one
(it is written out as a wav so it can be listened to).
"""

import struct
import sys
import wave

import numpy as np

BLOCK = 512
HEADER_BLOCKS = 16
PLAN_MAGIC = 0x4E4C5057

# Scratch layout, mirrored from firmware/src/app.rs.
S_PCM, S_INJECT, S_MELF, S_MEL, S_EO = 0, 750, 768, 1600, 8960
N_SAMPLES = 192_000
T = 64            # frames per mel tile
N_TILES = 20
N_MELS = 80
N_FRAMES = 1200   # real frames at audio_ctx 600


def read_at(f, lba, nbytes):
    f.seek(lba * BLOCK)
    return f.read(nbytes)


def load_plan(f):
    """scratch_lba and the mel (conv1 input) quantization, from the image."""
    hdr = read_at(f, 0, HEADER_BLOCKS * BLOCK)
    if hdr[:8] != b"WSPRIMG1":
        raise SystemExit("not a Whisper model image")
    n = struct.unpack_from("<I", hdr, 12)[0]
    for i in range(n):
        e = 16 + i * 32
        name = hdr[e:e + 24].rstrip(b"\0").decode()
        if name == "plan":
            lba = struct.unpack_from("<I", hdr, e + 24)[0]
            p = read_at(f, lba, 128)
            if struct.unpack_from("<I", p, 0)[0] != PLAN_MAGIC:
                raise SystemExit("bad plan magic")
            scratch = struct.unpack_from("<I", p, 8)[0]
            # field order per make_sd_image.write_plan; conv1.in_q sits
            # after magic/version/scratch/vocab, the sot block, eot and
            # the blank block.
            scale, zp = struct.unpack_from("<fi", p, 60)
            return scratch, scale, zp
    raise SystemExit("no plan entry")


def device_mel(f, scratch, scale, zp):
    """The int8 mel the device fed to conv1, as float."""
    raw = read_at(f, scratch + S_MEL, N_TILES * N_MELS * T)
    q = np.frombuffer(raw, dtype=np.int8).reshape(N_TILES, N_MELS, T)
    # tiles are channel-planar [C][W]; concatenate along frames
    mel = np.concatenate([q[i] for i in range(N_TILES)], axis=1)
    return (mel[:, :N_FRAMES].astype(np.float32) - zp) * scale, mel[:, :N_FRAMES]


def device_pcm(f, scratch):
    raw = read_at(f, scratch + S_PCM, N_SAMPLES * 2)
    return np.frombuffer(raw, dtype="<i2")


def injected(f, scratch):
    return read_at(f, scratch + S_INJECT, 16)[:8] == b"WMOCKAU1"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/mock-work.img"
    f = open(path, "rb")
    scratch, scale, zp = load_plan(f)
    inj = injected(f, scratch)
    print(f"work image {path}")
    print(f"  scratch at block {scratch}, mel quant scale {scale:.8f} zp {zp}")
    print(f"  audio source: {'injected clip' if inj else 'microphone'}")
    print()

    pcm = device_pcm(f, scratch)
    nz = int(np.count_nonzero(pcm))
    peak = int(np.max(np.abs(pcm))) if nz else 0
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    print("PCM in S_PCM (what the pipeline actually consumed):")
    print(f"  {len(pcm)} samples, {nz} nonzero, peak {peak} "
          f"({20 * np.log10(max(peak, 1) / 32768):.1f} dBFS), rms {rms:.1f}")
    out_wav = path.rsplit(".", 1)[0] + "-pcm.wav"
    w = wave.open(out_wav, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(pcm.tobytes())
    w.close()
    print(f"  written to {out_wav}")
    print()

    mel, q = device_mel(f, scratch, scale, zp)
    print("mel the device fed to conv1:")
    print(f"  int8 range [{q.min()}, {q.max()}]  "
          f"dequantized [{mel.min():.3f}, {mel.max():.3f}]")
    if q.min() == -128 and q.max() < 0:
        print("  WARNING: pinned against the int8 floor -- the mel is far")
        print("  below the range the quantization was calibrated for.")
    print()

    try:
        ref = np.load("out/ref.npz", allow_pickle=True)
    except FileNotFoundError:
        print("(out/ref.npz not present; skipping the reference comparison)")
        return
    rm = ref["mel_chunk"][:, :N_FRAMES]
    print("reference mel (out/ref.npz mel_chunk):")
    print(f"  range [{rm.min():.3f}, {rm.max():.3f}]")
    if inj:
        d = mel - rm
        print()
        print("device vs reference (same clip, so these should agree):")
        print(f"  max |diff| {np.abs(d).max():.4f}   rms {np.sqrt((d ** 2).mean()):.4f}")
        print(f"  correlation {np.corrcoef(mel.ravel(), rm.ravel())[0, 1]:.6f}")
        # one int8 step is the floor of what the device can represent
        print(f"  (one int8 step at this scale = {scale:.4f})")
    else:
        print()
        print("live-mic run: the reference mel is a different signal, so")
        print("compare the RANGES above rather than sample by sample.")


if __name__ == "__main__":
    main()
