"""Board-side ASR core: fbank + AX650 NPU encoder + greedy search on CPU.

Used by both the CLI test and the Flask web service. Only depends on numpy,
onnxruntime and `ax_engine.py` (ctypes binding, no extra pip package).
"""

from __future__ import annotations

import re
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ax_engine import AxEngine

FRAMES = 1000          # 10 s window
SUB = 4                # zipformer2 subsampling
BLANK = 0
CTX = 2
LOG_FLOOR = 1.1920929e-07
SILENCE = float(np.log(LOG_FLOOR))

# ---------------------------------------------------------------- fbank


def _povey(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1)), 0.85)


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_banks(nb: int, sr: int, nfft: int, low: float, high: float) -> np.ndarray:
    if high <= 0:
        high = sr / 2 + high
    w = sr / nfft
    pts = np.linspace(_mel(low), _mel(high), nb + 2)
    bins = (700.0 * (np.exp(pts / 1127.0) - 1.0)) / w
    banks = np.zeros((nb, nfft // 2 + 1))
    for b in range(nb):
        left, center, right = bins[b], bins[b + 1], bins[b + 2]
        for k in range(int(np.floor(left)), int(np.ceil(right)) + 1):
            if k < 0 or k > nfft // 2:
                continue
            if left <= k <= center:
                banks[b, k] = (k - left) / (center - left)
            elif center < k <= right:
                banks[b, k] = (right - k) / (right - center)
    return banks


_POVEY = _povey(400)
_BANKS = _mel_banks(80, 16000, 512, 20.0, -400.0)

try:  # 板端优先用 kaldi-native-fbank（C++ 实现，比纯 numpy 快两个数量级）
    import kaldi_native_fbank as _knf
except Exception:  # pragma: no cover
    _knf = None


def _make_fbank_knf():
    opts = _knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.window_type = "povey"
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    return opts


_KNF_OPTS = _make_fbank_knf() if _knf is not None else None


def fbank80(x: np.ndarray) -> np.ndarray:
    if _knf is not None:
        fbank = _knf.OnlineFbank(_KNF_OPTS)
        fbank.accept_waveform(16000, np.ascontiguousarray(x, dtype=np.float32))
        fbank.input_finished()
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, 80), dtype=np.float32)
        return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)

    x = np.asarray(x, dtype=np.float64)
    nframes = (len(x) + 80) // 160
    if nframes == 0:
        return np.zeros((0, 80), dtype=np.float32)
    out = np.empty((nframes, 80), dtype=np.float64)
    for f in range(nframes):
        start = f * 160 + 80 - 200
        if start >= 0 and start + 400 <= len(x):
            fr = x[start:start + 400].copy()
        else:
            idx = np.arange(start, start + 400)
            period = 2 * (len(x) - 1) if len(x) > 1 else 2
            m = np.mod(idx, period)
            idx = np.where(m >= len(x), period - m, m)
            fr = x[np.clip(idx, 0, len(x) - 1)].copy()
        fr -= fr.mean()
        fr[1:] -= 0.97 * fr[:-1]
        fr[0] -= 0.97 * fr[0]
        fr *= _POVEY
        spec = np.fft.rfft(fr, n=512)
        power = spec.real ** 2 + spec.imag ** 2
        out[f] = np.log(np.maximum(_BANKS @ power, LOG_FLOOR))
    return out.astype(np.float32)


def trim_silence(pcm: np.ndarray, frame: int = 160, rel_thr: float = 0.05,
                 pad_ms: int = 500) -> np.ndarray:
    """去掉首尾静音（能量门限），返回裁剪后的波形。"""
    n = len(pcm) // frame
    if n < 10:
        return pcm
    e = np.array([float(np.mean(pcm[i * frame:(i + 1) * frame] ** 2)) for i in range(n)])
    thr = max(e.max() * rel_thr, 1e-7)
    idx = np.where(e > thr)[0]
    if len(idx) == 0:
        return pcm
    pad = int(pad_ms / 10)   # 单位：帧（10 ms）
    a = max(0, idx[0] - pad)
    b = min(n, idx[-1] + 1 + pad)
    return pcm[a * frame:b * frame]


def valid_encoder_frames(num_input_frames: int, out_frames: int = 248) -> int:
    """估算需要解码的 encoder 输出帧数。

    模型是「流式训练 + 非流式导出」，短音频的有效输出帧比例比 1/4 高（有右上下文），
    实测比例在 0.24–0.31 之间。这里取 0.30 * T + 20 的安全上界，
    可以在不丢字的前提下少解码一部分尾部静音帧（解码是 CPU 上最慢的一步）。
    """
    est = int(0.30 * num_input_frames) + 20
    return max(1, min(out_frames, est))


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("only 16-bit PCM wav is supported")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        a = a[::2]
    if sr != 16000:
        raise ValueError(f"expect 16 kHz audio, got {sr}")
    return a


_CJK = r"\u4e00-\u9fff"


def clean_text(text: str) -> str:
    """把 BPE 输出整理成可读文本：去掉 CJK 字符之间的空格，保留英文词间空格。"""
    text = text.replace("▁", " ").strip()
    text = re.sub(r"\s+", " ", text)
    # 中文字符之间的空格去掉
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    # 中文与标点之间、标点与中文之间的空格
    text = re.sub(rf"\s+(?=[，。！？、；：])", "", text)
    text = re.sub(rf"(?<=[，。！？、；：])\s+", "", text)
    return text.strip()


# ---------------------------------------------------------------- ASR


class Asr:
    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        self.enc = AxEngine(str(d / "encoder.axmodel"))
        self.io = self.enc.describe()
        in_shape = self.io["inputs"][0]["shape"]
        self.frames = in_shape[2]
        self.out_shape = self.io["outputs"][0]["shape"]

        # 小模型在 AX650 的弱核上单线程吃力，joiner 用 4 线程能快约 2 倍
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.dec = ort.InferenceSession(str(d / "decoder.onnx"), so,
                                        providers=["CPUExecutionProvider"])
        self.joi = ort.InferenceSession(str(d / "joiner.onnx"), so,
                                        providers=["CPUExecutionProvider"])
        self.tokens = [l.split()[0] for l in
                       (d / "tokens.txt").read_text(encoding="utf-8").splitlines() if l]

    # ---- pieces
    def encode(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        if feats.shape[0] < self.frames:
            pad = np.full((self.frames - feats.shape[0], 80), SILENCE, dtype=np.float32)
            feats = np.concatenate([feats, pad])
        x = feats[: self.frames][None, None, ...].astype(np.float32)
        t0 = time.perf_counter()
        shape, data = self.enc.run([x])[0]
        dt = (time.perf_counter() - t0) * 1000
        raw = np.frombuffer(data, dtype=np.float32).reshape(shape)[0]
        return raw, dt

    def greedy(self, enc: np.ndarray) -> str:
        y = np.zeros((1, CTX), dtype=np.int64)
        dec_in = self.dec.get_inputs()[0].name
        dec = self.dec.run(None, {dec_in: y})[0]
        j_in = [i.name for i in self.joi.get_inputs()]
        hyp = []
        for t in range(enc.shape[0]):
            logit = self.joi.run(None, {j_in[0]: enc[t:t + 1].astype(np.float32),
                                        j_in[1]: dec})[0]
            k = int(logit.reshape(-1).argmax())
            if k != BLANK:
                hyp.append(k)
                y = np.array([[y[0, 1], k]], dtype=np.int64)
                dec = self.dec.run(None, {dec_in: y})[0]
        text = "".join(self.tokens[i] for i in hyp)
        return clean_text(text)


    def transcribe(self, pcm: np.ndarray, hop_frames: int = 800,
                   min_tail: int = 500, trim: bool = False) -> dict:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        t0 = time.perf_counter()
        if trim:
            pcm = trim_silence(pcm)
        feats = fbank80(pcm)
        t_feat = (time.perf_counter() - t0) * 1000

        parts, t_enc, t_dec = [], 0.0, 0.0
        start = 0
        while start < feats.shape[0]:
            if feats.shape[0] - start < min_tail and parts:
                break
            win = feats[start:start + self.frames]
            enc, dt = self.encode(win)
            t_enc += dt
            # 只解码有效帧（右侧补齐的静音帧不解码），按两级 subsampling 估算
            n_valid = valid_encoder_frames(win.shape[0], enc.shape[0])
            t1 = time.perf_counter()
            parts.append(self.greedy(enc[:n_valid]))
            t_dec += (time.perf_counter() - t1) * 1000
            if win.shape[0] < self.frames:
                break
            start += hop_frames

        text = " ".join(p for p in parts if p)
        total = (time.perf_counter() - t0) * 1000
        audio_ms = len(pcm) / 16000 * 1000
        return {
            "text": text,
            "audio_ms": round(audio_ms, 1),
            "feat_ms": round(t_feat, 1),
            "encoder_ms": round(t_enc, 1),
            "decoder_joiner_ms": round(t_dec, 1),
            "total_ms": round(total, 1),
            "rtf": round(total / audio_ms, 3) if audio_ms else None,
            "windows": len(parts),
        }
