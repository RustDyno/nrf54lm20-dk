"""Shared configuration, quantization helpers, and weight loading.

The layered pipeline splits Whisper into NPU-sized submodels (conv / fully
connected, int8, static weights) plus CPU glue ops (layernorm, gelu, softmax,
attention matmuls, residual adds). Everything here is backend-neutral; the
execution engines live in layered.py.
"""

import os

import numpy as np

MODEL_NAME = "tiny.en"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")

SAMPLE_RATE = 16_000
N_MELS = 80
HOP = 160

# Whisper's native encoder context: 30 s of audio -> 3000 mel frames -> 1500
# frames after conv2 (stride 2). The device build runs a truncated context
# (whisper.cpp's audio_ctx trick): positional embeddings are sliced to the
# first AUDIO_CTX rows and audio is chunked to AUDIO_CTX * 2 mel frames.
FULL_AUDIO_CTX = 1500
AUDIO_CTX = 600  # 12 s chunks targeted for the device pipeline

MAX_TOKENS = 48  # greedy decode budget per chunk (device KV cache size)

# MLP fc2 (1536 -> 384) weights exceed the device weight slot; the input dim is
# split into this many submodels whose int8 partial outputs are accumulated on
# the CPU. This is the one tiling that changes numerics; the simulator models
# it explicitly.
FC2_SPLITS = 4

STATE = 384
HEADS = 6
HEAD_DIM = STATE // HEADS
N_LAYERS = 4
MLP_DIM = 4 * STATE


# --- int8 quantization (TFLite conventions) --------------------------------
# Activations: per-tensor, asymmetric, int8. Weights: per-channel, symmetric.

def act_qparams(lo, hi, bits=8):
    """Scale/zero-point covering [lo, hi], always including 0."""
    lo = min(float(lo), 0.0)
    hi = max(float(hi), 0.0)
    lvl = 2 ** bits
    scale = (hi - lo) / (lvl - 1)
    if scale == 0.0:
        return 1.0, 0
    zp = int(np.clip(round(-lvl // 2 - lo / scale), -lvl // 2, lvl // 2 - 1))
    return scale, zp


def quantize(x, scale, zp, bits=8):
    lvl = 2 ** bits
    q = np.clip(np.round(x / scale) + zp, -lvl // 2, lvl // 2 - 1)
    return q.astype(np.int8 if bits <= 8 else np.int16)


def dequantize(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale


def weight_qparams(w, axis=0):
    """Per-channel symmetric int8: returns (q, scales); zero-point is 0."""
    w = np.asarray(w, dtype=np.float32)
    red = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.max(np.abs(w), axis=red, keepdims=True)
    scales = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(w / scales), -127, 127).astype(np.int8)
    return q, np.squeeze(scales)


def submodel_names(l):
    """Per-block Axon submodel names. Block 0 keeps the names it was first
    compiled under; later blocks use a uniform w{l}<kind> scheme."""
    if l == 0:
        n = {"q": "wq0", "k": "wk0", "v": "wv0", "out": "wout0"}
        n.update({f"fc1{p}": f"wfc1{p}" for p in "abcd"})
        n.update({f"fc2p{j}": f"wfc2p{j}" for j in range(4)})
        return n
    n = {"q": f"w{l}q", "k": f"w{l}k", "v": f"w{l}v", "out": f"w{l}out"}
    n.update({f"fc1{p}": f"w{l}fc1{p}" for p in "abcd"})
    n.update({f"fc2p{j}": f"w{l}fc2p{j}" for j in range(4)})
    return n


def decoder_submodel_names(l):
    """Per-block decoder Axon submodel names."""
    n = {"q": f"d{l}q", "k": f"d{l}k", "v": f"d{l}v", "out": f"d{l}out",
         "xq": f"d{l}xq", "xout": f"d{l}xout",
         "xk": f"d{l}xk", "xv": f"d{l}xv"}
    n.update({f"fc1{p}": f"d{l}fc1{p}" for p in "abcd"})
    n.update({f"fc2p{j}": f"d{l}fc2p{j}" for j in range(4)})
    return n


# --- weight loading --------------------------------------------------------

def load_weights():
    """Load the Whisper checkpoint into a flat dict of numpy arrays."""
    import whisper

    model = whisper.load_model(MODEL_NAME, device="cpu",
                               download_root=os.path.join(DATA, "ckpt"))
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    dims = model.dims
    assert dims.n_audio_state == STATE and dims.n_audio_head == HEADS
    assert dims.n_audio_layer == N_LAYERS and dims.n_text_layer == N_LAYERS
    return model, sd
