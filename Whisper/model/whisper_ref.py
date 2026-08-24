"""Float reference: canonical openai-whisper greedy decode of the test clip.

Produces out/ref.npz with the reference tokens/text plus the exact mel input,
so the layered pipeline can be compared stage by stage against ground truth.
"""

import os
import sys

import numpy as np

import common


def main():
    import whisper
    from whisper.decoding import DecodingOptions
    from whisper.tokenizer import get_tokenizer

    clip = sys.argv[1] if len(sys.argv) > 1 else os.path.join(common.DATA, "jfk.flac")
    os.makedirs(common.OUT, exist_ok=True)

    model, _sd = common.load_weights()
    audio = whisper.load_audio(clip)
    print(f"clip: {clip} ({len(audio) / common.SAMPLE_RATE:.2f} s)")

    # Canonical path: 30 s padded window, full 1500-frame audio context.
    mel_full = whisper.log_mel_spectrogram(audio, padding=whisper.audio.N_SAMPLES)
    mel_full = mel_full[:, : whisper.audio.N_FRAMES]

    opts = DecodingOptions(language="en", task="transcribe",
                           without_timestamps=True, temperature=0.0, fp16=False)
    result = whisper.decode(model, mel_full, opts)
    print(f"reference transcript: {result.text!r}")
    print(f"reference tokens: {result.tokens}")

    # Truncated-context mel for the device-sized pipeline (AUDIO_CTX frames).
    chunk = whisper.pad_or_trim(audio, common.AUDIO_CTX * 2 * common.HOP)
    mel_chunk = whisper.log_mel_spectrogram(chunk)

    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        language="en", task="transcribe")
    np.savez(
        os.path.join(common.OUT, "ref.npz"),
        text=np.array(result.text),
        tokens=np.array(result.tokens, dtype=np.int64),
        mel_full=mel_full.numpy().astype(np.float32),
        mel_chunk=mel_chunk.numpy().astype(np.float32),
        audio=audio.astype(np.float32),
        sot_sequence=np.array(tok.sot_sequence_including_notimestamps, dtype=np.int64),
        eot=np.array(tok.eot, dtype=np.int64),
        non_speech=np.array(sorted(tok.non_speech_tokens), dtype=np.int64),
        blank=np.array(tok.encode(" ") + [tok.eot], dtype=np.int64),
        timestamp_begin=np.array(tok.timestamp_begin, dtype=np.int64),
    )
    print(f"saved {os.path.join(common.OUT, 'ref.npz')}")


if __name__ == "__main__":
    main()
