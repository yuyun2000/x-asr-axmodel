"""Kaldi 风格的 80 维 log-Mel fbank（纯 numpy 实现，与 kaldifeat 参数对齐）。

为什么自己写：目标机上 `kaldifeat` 没有 py3.10 的可用 wheel（源码编译失败），
macOS 上也装不上；而这条导出/推理链路只需要一种与 kaldifeat 数值等价的 fbank。
本文件实现 kaldi `Fbank` 的核心部分，参数固定为 X-ASR（icefall zipformer 配方）：

    dither=0, snip_edges=False, sample_rate=16000,
    num_bins=80, low_freq=20, high_freq=-400 (=7600 Hz),
    frame_length=25ms, frame_shift=10ms, window=povey,
    preemph=0.97, remove_dc_offset=True, use_power=True, use_log=True

自检：`scripts/local/03_check_fbank.py` 用本文件的特征跑 sherpa-onnx 的 int8
ONNX，识别文本应与 sherpa-onnx 自带 fbank 的结果一致。
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = -400.0
PREEMPH = 0.97
LOG_FLOOR = 1.0e-10
EPSILON = 1.1920929e-07  # float32 的 std::numeric_limits<float>::epsilon()


def _povey_window(size: int) -> np.ndarray:
    n = np.arange(size)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1)), 0.85)


def _mel(freq):
    """Kaldi 的 mel 刻度（1127*ln(1+f/700)，不是 HTK 那套）。"""
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_banks(num_bins: int, sample_rate: int, padded_size: int,
               low_freq: float, high_freq: float) -> np.ndarray:
    if high_freq <= 0:
        high_freq = sample_rate / 2 + high_freq
    fft_bin_width = sample_rate / padded_size
    mel_points = np.linspace(_mel(low_freq), _mel(high_freq), num_bins + 2)
    hz_points = 700.0 * (np.exp(mel_points / 1127.0) - 1.0)
    bins = hz_points / fft_bin_width
    banks = np.zeros((num_bins, padded_size // 2 + 1), dtype=np.float64)
    for b in range(num_bins):
        left, center, right = bins[b], bins[b + 1], bins[b + 2]
        for k in range(int(np.floor(left)), int(np.ceil(right)) + 1):
            if k < 0 or k > padded_size // 2:
                continue
            if left <= k <= center:
                banks[b, k] = (k - left) / (center - left)
            elif center < k <= right:
                banks[b, k] = (right - k) / (right - center)
    return banks


def _first_sample_of_frame(frame: int, frame_length: int, frame_shift: int,
                           snip_edges: bool) -> int:
    if snip_edges:
        return frame * frame_shift
    return frame * frame_shift + frame_shift // 2 - frame_length // 2


def _num_frames(num_samples: int, frame_length: int, frame_shift: int,
                snip_edges: bool) -> int:
    if snip_edges:
        if num_samples < frame_length:
            return 0
        return 1 + (num_samples - frame_length) // frame_shift
    return (num_samples + frame_shift // 2) // frame_shift


def _reflect_index(idx: np.ndarray, n: int) -> np.ndarray:
    """kaldi 的边界延拓：先镜像，越界再用镜像的镜像。"""
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    m = np.mod(idx, period)
    return np.where(m >= n, period - m, m)


def _extract_window(wave: np.ndarray, start: int, size: int) -> np.ndarray:
    idx = np.arange(start, start + size)
    if start >= 0 and start + size <= len(wave):
        return wave[start:start + size].astype(np.float64, copy=True)
    return wave[_reflect_index(idx, len(wave))].astype(np.float64, copy=True)


def _preemphasize(frame: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    """与 kaldi cu::Preemphasize 一致：首样本减自身，其余减前一采样。"""
    out = frame.copy()
    out[1:] -= coeff * frame[:-1]
    out[0] -= coeff * frame[0]
    return out


def fbank80(samples: np.ndarray,
            sample_rate: int = SAMPLE_RATE,
            snip_edges: bool = False,
            dither: float = 0.0,
            seed: int = 0) -> np.ndarray:
    """[-1,1] 的 float 波形 → [T, 80] 的 log-Mel fbank（float32）。"""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expect {SAMPLE_RATE} Hz audio, got {sample_rate}")
    wave = np.asarray(samples, dtype=np.float64).reshape(-1)
    if dither != 0.0:
        wave = wave + np.random.default_rng(seed).normal(0.0, dither, size=wave.shape)

    frame_length = int(round(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0))  # 400
    frame_shift = int(round(SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0))    # 160
    padded_size = 1 << (frame_length - 1).bit_length()                  # 512
    window = _povey_window(frame_length)
    banks = _mel_banks(NUM_BINS, SAMPLE_RATE, padded_size, LOW_FREQ, HIGH_FREQ)

    nframes = _num_frames(len(wave), frame_length, frame_shift, snip_edges)
    feats = np.empty((nframes, NUM_BINS), dtype=np.float64)
    for f in range(nframes):
        start = _first_sample_of_frame(f, frame_length, frame_shift, snip_edges)
        frame = _extract_window(wave, start, frame_length)
        frame -= frame.mean()                       # remove_dc_offset
        frame = _preemphasize(frame)                # preemph_coeff
        frame *= window                             # povey window
        spectrum = np.fft.rfft(frame, n=padded_size)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        feats[f] = np.log(np.maximum(banks @ power, EPSILON))

    return feats.astype(np.float32)


if __name__ == "__main__":
    import sys
    import wave as _wave

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/tmp/asrtest/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav")
    w = _wave.open(path)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    f = fbank80(a)
    print(f"{path}\n  samples={a.shape[0]} ({a.shape[0]/16000:.2f}s)  fbank={f.shape}  "
          f"range=[{f.min():.3f}, {f.max():.3f}]")
