#!/usr/bin/env python3
"""用仿真输出做「字级一致性」校验：ONNX encoder 输出 vs AXMODEL 仿真输出。

cosine 只能说明数值接近，但 ASR 真正关心的是**解出来的字一样**。做法：

1. 取同一帧 fbank（`task/export/sample_input.npy`）；
2. 分别用 ONNX encoder 和 `pulsar2 run` 仿真输出作为 encoder_out；
3. 用同一套 fp32 decoder + joiner 贪心解码（与 sherpa-onnx 的 greedy_search 一致）；
4. 比较两段文本（编辑距离 / 是否完全一致）。

用法：
    python 50_text_consistency.py --task-dir task --onnx-dir onnx --tokens tokens/tokens_nopunct.txt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort

BLANK_ID = 0
CONTEXT_SIZE = 2


def greedy(encoder_out: np.ndarray, decoder_path: str, joiner_path: str,
           tokens: list[str]) -> str:
    ses_d = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    ses_j = ort.InferenceSession(joiner_path, providers=["CPUExecutionProvider"])
    y = np.array([[BLANK_ID] * CONTEXT_SIZE], dtype=np.int64)
    dec_out = ses_d.run(None, {ses_d.get_inputs()[0].name: y})[0]
    enc_name = ses_j.get_inputs()[0].name
    dec_name = ses_j.get_inputs()[1].name

    hyp = []
    for t in range(encoder_out.shape[1]):
        logit = ses_j.run(None, {enc_name: encoder_out[:, t, :], dec_name: dec_out})[0]
        k = int(logit[0].argmax())
        if k != BLANK_ID:
            hyp.append(k)
            y = np.array([[y[0, 1], k]], dtype=np.int64)
            dec_out = ses_d.run(None, {ses_d.get_inputs()[0].name: y})[0]
    return "".join(tokens[i] for i in hyp).replace("▁", " ").strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--onnx-dir", required=True, help="含 decoder.fp32.onnx / joiner.fp32.onnx")
    ap.add_argument("--tokens", required=True)
    args = ap.parse_args()

    task = Path(args.task_dir)
    sample = np.load(task / "export" / "sample_input.npy")
    ref = np.load(task / "export" / "source_output.npy")

    sim_bin = task / "simulate" / "output" / "encoder_out.bin"
    if not sim_bin.exists():
        raise SystemExit(f"找不到仿真输出 {sim_bin}，先跑 40_build.py")
    ax_out = np.fromfile(sim_bin, dtype=np.float32).reshape(ref.shape)

    tokens = [l.split()[0] for l in Path(args.tokens).read_text(encoding="utf-8").splitlines() if l]
    dec = glob.glob(str(Path(args.onnx_dir) / "decoder*.onnx"))[0]
    joi = glob.glob(str(Path(args.onnx_dir) / "joiner*.onnx"))[0]

    t_ref = greedy(ref, dec, joi, tokens)
    t_ax = greedy(ax_out, dec, joi, tokens)
    dist = edit_distance(t_ref, t_ax)
    print(f"ONNX  : {t_ref}")
    print(f"AXMODEL: {t_ax}")
    print(f"一致: {t_ref == t_ax}   token 级编辑距离: {dist} / {len(t_ref)}")


if __name__ == "__main__":
    main()
