#!/usr/bin/env python3
"""三方对比：fp32 ONNX / sherpa-onnx 发布的 int8 ONNX / 我们的 axmodel。

目的：证明 axmodel 的精度至少不差于 sherpa-onnx 官方发布的 int8 版本（后者的
encoder/decoder/joiner ONNX 是社区默认部署形态）。

前置：
- `--sherpa-dir` 指向 sherpa-onnx 的 X-ASR int8 模型目录（含 encoder*.int8.onnx /
  decoder*.onnx / joiner*.int8.onnx / tokens.txt）
- `task/simulate/multi/<wav>/output/encoder_out.bin` 由 51_simulate_multi.py 生成

用法：
    python 52_compare_with_sherpa_int8.py --task-dir task --wav-dir wavs \
        --sherpa-dir /path/to/sherpa-x-asr-int8 --onnx-dir onnx \
        --tokens tokens/tokens_nopunct.txt --out reports/three_way.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbank_kaldi import EPSILON, fbank80  # noqa: E402

FRAMES = 1000
SILENCE = float(np.log(EPSILON))
BLANK, CTX = 0, 2


def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    if sw != 2 or sr != 16000:
        raise ValueError(f"{path}: need 16 kHz 16-bit audio")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a[::2] if ch == 2 else a


def features(path: Path) -> np.ndarray:
    f = fbank80(read_wav(path))[:FRAMES]
    if f.shape[0] < FRAMES:
        f = np.concatenate([f, np.full((FRAMES - f.shape[0], 80), SILENCE, np.float32)])
    return f


def encode_onnx(sess, feat):
    name = sess.get_inputs()[0].name
    rank = len(sess.get_inputs()[0].shape)
    x = feat[None, None, ...] if rank == 4 else feat[None, ...]
    return sess.run(None, {name: x.astype(np.float32)})[0]


def greedy(enc, dec_path, joi_path, tokens):
    sd = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    sj = ort.InferenceSession(joi_path, providers=["CPUExecutionProvider"])
    y = np.zeros((1, CTX), dtype=np.int64)
    d = sd.run(None, {sd.get_inputs()[0].name: y})[0]
    en, dn = sj.get_inputs()[0].name, sj.get_inputs()[1].name
    hyp = []
    for t in range(enc.shape[1]):
        k = int(sj.run(None, {en: enc[:, t, :], dn: d})[0][0].argmax())
        if k != BLANK:
            hyp.append(k)
            y = np.array([[y[0, 1], k]], dtype=np.int64)
            d = sd.run(None, {sd.get_inputs()[0].name: y})[0]
    return "".join(tokens[i] for i in hyp).replace("▁", " ").strip()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--sherpa-dir", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", default="reports/three_way.json")
    args = ap.parse_args()

    task = Path(args.task_dir).resolve()
    tokens = [l.split()[0] for l in Path(args.tokens).read_text(encoding="utf-8").splitlines() if l]

    fp32 = ort.InferenceSession(str(task / "export" / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    sherpa_enc = ort.InferenceSession(
        str(next(Path(args.sherpa_dir).glob("encoder*.int8.onnx"))),
        providers=["CPUExecutionProvider"])
    sherpa_dec = str(next(Path(args.sherpa_dir).glob("decoder*.onnx")))
    sherpa_joi = str(next(Path(args.sherpa_dir).glob("joiner*.int8.onnx")))

    dec = glob.glob(str(Path(args.onnx_dir) / "decoder*.onnx"))[0]
    joi = glob.glob(str(Path(args.onnx_dir) / "joiner*.onnx"))[0]

    rows = []
    for wav in sorted(Path(args.wav_dir).glob("*.wav")):
        feat = features(wav)
        # 1) fp32 ONNX（参考）
        t_fp32 = greedy(encode_onnx(fp32, feat), dec, joi, tokens)
        # 2) sherpa-onnx 的 int8 ONNX
        t_sherpa = greedy(encode_onnx(sherpa_enc, feat), sherpa_dec, sherpa_joi, tokens)
        # 3) 我们的 axmodel（仿真输出）
        sim = task / "simulate" / "multi" / wav.stem / "output" / "encoder_out.bin"
        t_ax = None
        if sim.exists():
            ref = encode_onnx(fp32, feat)
            ax = np.fromfile(sim, dtype=np.float32).reshape(ref.shape)
            t_ax = greedy(ax, dec, joi, tokens)

        row = {
            "wav": wav.name,
            "edit_fp32_vs_sherpa": edit(t_fp32, t_sherpa),
            "edit_fp32_vs_axmodel": edit(t_fp32, t_ax) if t_ax is not None else None,
            "edit_sherpa_vs_axmodel": edit(t_sherpa, t_ax) if t_ax is not None else None,
            "text_fp32": t_fp32,
            "text_sherpa": t_sherpa,
            "text_axmodel": t_ax,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("text_")},
                         ensure_ascii=False), flush=True)

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
