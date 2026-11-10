#!/usr/bin/env python3
"""生成 Pulsar2 校准集（Numpy 格式 tar.gz）。

比 20_prep_task.py 里的"补静音"版本更贴近真实分布：
- 只取**语音窗口**：对每个 wav 以 5 s 步长滑 10 s 窗，跳过静音占比过高的窗口；
- 长音频（15–50 s）能出多个满窗；短音频只贡献一个窗；
- 目标样本数默认 24（Pulsar2 的 calibration_size 默认 32）。

用法：
    python 21_make_calib.py --wav-dir wavs --frames 1000 --out task/export \
        --name x4d --num 24
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbank_kaldi import EPSILON, fbank80  # noqa: E402

FRAME = 16000  # 10 ms
WINDOW = 1000  # 10 s


def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    if sw != 2 or sr != 16000:
        raise ValueError(f"{path}: 需要 16 kHz / 16-bit")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a[::2] if ch == 2 else a


def speech_ratio(x: np.ndarray) -> float:
    """简单能量门限估计：帧能量高于整体的部分视为语音。"""
    n = len(x) // FRAME
    if n == 0:
        return 0.0
    e = np.array([float(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2)) for i in range(n)])
    thr = max(e.max() * 0.05, 1e-6)
    return float((e > thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--frames", type=int, default=WINDOW)
    ap.add_argument("--out", required=True, help="含 calib_data/ 的输出目录")
    ap.add_argument("--name", default="x4d")
    ap.add_argument("--num", type=int, default=24)
    ap.add_argument("--min-speech", type=float, default=0.5)
    args = ap.parse_args()

    win = args.frames * FRAME
    hop = win // 2
    cands = []
    for wav in sorted(Path(args.wav_dir).glob("*.wav")):
        x = read_wav(wav)
        if len(x) < win:
            # 短音频：只用一次，补静音，标记为低优先级
            pad = np.full(win - len(x), 0.0, dtype=np.float32)
            seg = np.concatenate([x, pad])
            cands.append((speech_ratio(seg[: len(x)]), seg, wav.name, 0))
            continue
        for start in range(0, len(x) - win + 1, hop):
            seg = x[start:start + win]
            cands.append((speech_ratio(seg), seg, wav.name, start))
        # 末尾补齐一个窗，避免丢弃尾部
        tail = x[-win:]
        cands.append((speech_ratio(tail), tail, wav.name, len(x) - win))

    # 语音占比高的优先，同时保证文件名多样性
    cands.sort(key=lambda c: -c[0])
    picked, seen_name = [], {}
    for ratio, seg, name, start in cands:
        if ratio < args.min_speech and len(picked) >= args.num // 2:
            continue
        if seen_name.get(name, 0) >= 4:
            continue
        seen_name[name] = seen_name.get(name, 0) + 1
        picked.append((ratio, seg, name, start))
        if len(picked) >= args.num:
            break

    calib_dir = Path(args.out) / "calib_data" / args.name
    calib_dir.mkdir(parents=True, exist_ok=True)
    for old in calib_dir.glob("*.npy"):
        old.unlink()
    for i, (ratio, seg, name, start) in enumerate(picked):
        feat = fbank80(seg)[: args.frames]
        if feat.shape[0] < args.frames:
            pad = np.full((args.frames - feat.shape[0], 80),
                          float(np.log(EPSILON)), dtype=np.float32)
            feat = np.concatenate([feat, pad])
        np.save(calib_dir / f"{i:04d}.npy", feat[None, None, ...].astype(np.float32))
        print(f"[calib] {i:02d} {name}@{start/16000:.1f}s speech={ratio:.2f}")

    tar_path = Path(args.out) / "calib_data" / f"{args.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(calib_dir.glob("*.npy")):
            tar.add(f, arcname=f.name)
    print(f"[calib] {len(picked)} 组 -> {tar_path}")


if __name__ == "__main__":
    main()
