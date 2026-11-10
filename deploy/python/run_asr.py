#!/usr/bin/env python3
"""X-ASR AXMODEL 板端推理（离线 10 s 窗口，贪心搜索）。

依赖（板端）：
    pip install axengine numpy

用法：
    python run_asr.py --models-dir ./models --audio test.wav
    python run_asr.py --models-dir ./models --audio-dir test_wavs

模型目录约定（从 HuggingFace 下载后解压）：
    models/
    ├── encoder.axmodel       # 必需，NPU
    ├── decoder.onnx          # 二选一：CPU（onnxruntime）
    ├── joiner.onnx
    └── tokens.txt
  如果同时存在 decoder.axmodel / joiner.axmodel，则优先用 NPU 版本。

实现要点：
- fbank 用纯 numpy 实现（scripts/host/fbank_kaldi.py 的独立副本），与训练口径一致：
  80 mel、25 ms/10 ms、povey 窗、preemph 0.97、snip_edges=False、high_freq=-400。
- encoder 输入 shape 固定 [1,1,1000,80]（10 s）；更长的音频按 8 s 步长切窗后拼接结果。
- 贪心搜索：context_size=2、blank_id=0、每帧最多出 1 个 token。
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

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
        l, c, r = bins[b], bins[b + 1], bins[b + 2]
        for k in range(int(np.floor(l)), int(np.ceil(r)) + 1):
            if k < 0 or k > nfft // 2:
                continue
            if l <= k <= c:
                banks[b, k] = (k - l) / (c - l)
            elif c < k <= r:
                banks[b, k] = (r - k) / (r - c)
    return banks


_POVEY = _povey(400)
_BANKS = _mel_banks(80, 16000, 512, 20.0, -400.0)


def fbank80(x: np.ndarray) -> np.ndarray:
    """[-1,1] float, 16 kHz → [T,80] log-mel（snip_edges=False）。"""
    x = np.asarray(x, dtype=np.float64)
    nframes = (len(x) + 80) // 160
    out = np.empty((nframes, 80), dtype=np.float64)
    for f in range(nframes):
        start = f * 160 + 80 - 200
        idx = np.arange(start, start + 400)
        if start >= 0 and start + 400 <= len(x):
            fr = x[start:start + 400].copy()
        else:
            period = 2 * (len(x) - 1)
            m = np.mod(idx, period)
            idx2 = np.where(m >= len(x), period - m, m)
            fr = x[np.clip(idx2, 0, len(x) - 1)].copy()
        fr -= fr.mean()
        fr[1:] -= 0.97 * fr[:-1]
        fr[0] -= 0.97 * fr[0]
        fr *= _POVEY
        spec = np.fft.rfft(fr, n=512)
        power = spec.real ** 2 + spec.imag ** 2
        out[f] = np.log(np.maximum(_BANKS @ power, 1.1920929e-07))
    return out.astype(np.float32)


# ---------------------------------------------------------------- audio


def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    if sw != 2:
        raise ValueError(f"{path}: 只支持 16-bit PCM")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        a = a[::2]
    if sr != 16000:
        raise ValueError(f"{path}: 需要 16 kHz，实际 {sr}")
    return a


# ---------------------------------------------------------------- runtime


class Recognizer:
    FRAMES = 1000          # 10 s
    SUB = 4                # zipformer2 subsampling
    BLANK = 0
    CTX = 2

    def __init__(self, models_dir: Path):
        self.dir = models_dir
        axmodel = models_dir / "encoder.axmodel"
        onnx_model = models_dir / "encoder.onnx"
        if axmodel.exists():
            from axengine import InferenceSession
            self.enc = InferenceSession(str(axmodel))
            self.enc_backend = "axmodel"
        elif onnx_model.exists():
            # 没有板子时（x86/macOS 上）用 fp32 ONNX 验证同一套推理逻辑
            import onnxruntime as ort
            self.enc = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
            self.enc_backend = "onnx"
        else:
            raise SystemExit(f"{models_dir} 下既没有 encoder.axmodel 也没有 encoder.onnx")
        self.enc_in = self.enc.get_inputs()[0].name
        self.enc_out = self.enc.get_outputs()[0].name
        # axmodel 的输入是 [1,1,T,80]（4 维包装），原始 fp32 ONNX 是 [1,T,80]
        try:
            self.enc_rank = len(self.enc.get_inputs()[0].shape)
        except Exception:
            self.enc_rank = 4

        # decoder / joiner：NPU 版优先，否则回退到 CPU 的 onnxruntime
        dec_ax = models_dir / "decoder.axmodel"
        joi_ax = models_dir / "joiner.axmodel"
        if dec_ax.exists() and joi_ax.exists():
            self.dec = InferenceSession(str(dec_ax))
            self.joi = InferenceSession(str(joi_ax))
            self.backend = "npu"
        else:
            import onnxruntime as ort
            self.dec = ort.InferenceSession(str(models_dir / "decoder.onnx"),
                                            providers=["CPUExecutionProvider"])
            self.joi = ort.InferenceSession(str(models_dir / "joiner.onnx"),
                                            providers=["CPUExecutionProvider"])
            self.backend = "cpu-decoder"

        self.tokens = [l.split()[0] for l in
                       (models_dir / "tokens.txt").read_text(encoding="utf-8").splitlines() if l]

    def _run(self, sess, feed):
        if isinstance(sess, object) and hasattr(sess, "get_inputs") and sess.__class__.__module__.startswith("axengine"):
            return sess.run([sess.get_outputs()[0].name], feed)
        return sess.run(None, feed)

    def encode(self, feats: np.ndarray) -> np.ndarray:
        """feats [T,80] → encoder_out [T/4,512]"""
        frames = self.FRAMES
        if feats.shape[0] < frames:
            pad = np.full((frames - feats.shape[0], 80), np.log(1.1920929e-07), dtype=np.float32)
            feats = np.concatenate([feats, pad])
        x = feats[:frames][None, None, ...].astype(np.float32) if self.enc_rank == 4 \
            else feats[:frames][None, ...].astype(np.float32)
        if self.enc_backend == "axmodel":
            out = self.enc.run([self.enc_out], {self.enc_in: x})[0]
        else:
            out = self.enc.run(None, {self.enc_in: x})[0]
        return np.asarray(out).reshape(-1, out.shape[-1])

    def greedy(self, enc: np.ndarray) -> str:
        y = np.zeros((1, self.CTX), dtype=np.int64)
        dec_in = self.dec.get_inputs()[0].name
        dec = np.asarray(self._run(self.dec, {dec_in: y})[0])
        joi_names = [i.name for i in self.joi.get_inputs()]
        hyp = []
        for t in range(enc.shape[0]):
            logit = np.asarray(self._run(self.joi, {
                joi_names[0]: enc[t:t + 1].astype(np.float32),
                joi_names[1]: dec,
            })[0])
            k = int(logit.reshape(-1).argmax())
            if k != self.BLANK:
                hyp.append(k)
                y = np.array([[y[0, 1], k]], dtype=np.int64)
                dec = np.asarray(self._run(self.dec, {dec_in: y})[0])
        text = "".join(self.tokens[i] for i in hyp)
        return text.replace("▁", " ").strip()

    def transcribe(self, samples: np.ndarray, hop_frames: int = 800,
                   min_tail_frames: int = 500) -> str:
        """按 10 s 窗口切分；尾部不足 min_tail_frames（5 s）的残段并入上一窗，避免重复出字。"""
        feats = fbank80(samples)
        parts = []
        start = 0
        while start < feats.shape[0]:
            remaining = feats.shape[0] - start
            if remaining < min_tail_frames and parts:
                break
            win = feats[start:start + self.FRAMES]
            parts.append(self.greedy(self.encode(win)))
            if win.shape[0] < self.FRAMES:
                break
            start += hop_frames
        return " ".join(p for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--audio", default=None, help="单个 wav")
    ap.add_argument("--audio-dir", default=None, help="目录下所有 wav")
    args = ap.parse_args()

    rec = Recognizer(Path(args.models_dir))
    print(f"[x-asr-axmodel] encoder backend: {rec.enc_backend}; decoder backend: {rec.backend}")

    files = []
    if args.audio:
        files = [Path(args.audio)]
    elif args.audio_dir:
        files = sorted(Path(args.audio_dir).glob("*.wav"))
    if not files:
        sys.exit("需要 --audio 或 --audio-dir")

    for f in files:
        samples = read_wav(f)
        text = rec.transcribe(samples)
        print(f"{f.name}\t{len(samples)/16000:.2f}s\t{text}")


if __name__ == "__main__":
    main()
