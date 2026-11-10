#!/usr/bin/env python3
"""多样本仿真验证：对每条测试音频跑 ONNX 参考 + Pulsar2 仿真，报 cosine 和字级一致性。

用法（目标机，需先有 task/compile/model.axmodel）：
    PYTHONPATH=$HOME/AX-model/Magnetar:$PYTHONPATH python 51_simulate_multi.py \
        --task-dir task --wav-dir wavs --onnx-dir onnx \
        --tokens tokens/tokens_nopunct.txt --out reports/multi_sim.json
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbank_kaldi import EPSILON, fbank80  # noqa: E402

FRAMES = 1000
SILENCE = float(np.log(EPSILON))
BLANK = 0
CTX = 2


def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    if sw != 2 or sr != 16000:
        raise ValueError(f"{path}: 需要 16 kHz / 16-bit")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a[::2] if ch == 2 else a


def feats_for(path: Path) -> np.ndarray:
    f = fbank80(read_wav(path))[:FRAMES]
    if f.shape[0] < FRAMES:
        f = np.concatenate([f, np.full((FRAMES - f.shape[0], 80), SILENCE, np.float32)])
    return f[None, None, ...].astype(np.float32)


def greedy(enc: np.ndarray, dec_path: str, joi_path: str, tokens: list[str]) -> str:
    sd = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    sj = ort.InferenceSession(joi_path, providers=["CPUExecutionProvider"])
    y = np.zeros((1, CTX), dtype=np.int64)
    d = sd.run(None, {sd.get_inputs()[0].name: y})[0]
    en, dn = sj.get_inputs()[0].name, sj.get_inputs()[1].name
    hyp = []
    for t in range(enc.shape[0]):
        k = int(sj.run(None, {en: enc[:, t, :], dn: d})[0][0].argmax())
        if k != BLANK:
            hyp.append(k)
            y = np.array([[y[0, 1], k]], dtype=np.int64)
            d = sd.run(None, {sd.get_inputs()[0].name: y})[0]
    return "".join(tokens[i] for i in hyp).replace("▁", " ").strip()


def edit(a: str, b: str) -> int:
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
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", default="reports/multi_sim.json")
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()

    from magnetar.pulsar2_util import resolve_backend
    from magnetar.stages.simulate import run as simulate_run

    task = Path(args.task_dir).resolve()
    backend = resolve_backend()
    tokens = [l.split()[0] for l in Path(args.tokens).read_text(encoding="utf-8").splitlines() if l]
    dec = glob.glob(str(Path(args.onnx_dir) / "decoder*.onnx"))[0]
    joi = glob.glob(str(Path(args.onnx_dir) / "joiner*.onnx"))[0]

    sess = ort.InferenceSession(str(task / "export" / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))[: args.limit]
    rows = []
    for wav in wavs:
        x = feats_for(wav)
        ref = sess.run(None, {"x4d": x})[0]
        out_dir = task / "simulate" / "output"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        m = simulate_run(task, x, backend, input_name="x4d", output_name="encoder_out")
        ax = np.fromfile(out_dir / "encoder_out.bin", dtype=np.float32).reshape(ref.shape)
        t_ref = greedy(ref, dec, joi, tokens)
        t_ax = greedy(ax, dec, joi, tokens)
        rows.append({
            "wav": wav.name,
            "cosine": round(float(m["cosine_similarity"]), 6),
            "mae": round(float(m["mae"]), 4),
            "text_same": t_ref == t_ax,
            "token_edit_distance": edit(t_ref, t_ax),
            "onnx_text": t_ref,
            "axmodel_text": t_ax,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cosines = [r["cosine"] for r in rows]
    print(f"mean cosine = {np.mean(cosines):.6f}  min = {min(cosines):.6f}  "
          f"text match = {sum(r['text_same'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
