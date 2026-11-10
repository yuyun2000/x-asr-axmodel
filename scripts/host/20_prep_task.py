#!/usr/bin/env python3
"""把 icefall 导出的 fp32 离线 encoder 变成 Magnetar/Pulsar2 可直接吃的任务目录。

做四件事：
1. 把 encoder 的 3 维输入 `x[1,T,80]` 包一层 4 维输入 `x4d[1,1,T,80]`
   （Pulsar2 的 tensor_layout 枚举只有 NCHW/NHWC，4 维更好走通用路径）；
2. 用真实音频算 fbank，生成 Pulsar2 的校准集（Numpy 格式 tar.gz，≥3 组）；
3. 写 `model_meta.json`（Magnetar 的 COMPILE/SIMULATE 都读它）；
4. 留一份 sample/source 输出给 SIMULATE 对分。

用法：
    python 20_prep_task.py --onnx onnx/encoder.fp32.onnx \
        --wav-dir test_wavs --frames 1000 --out task
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbank_kaldi import EPSILON, fbank80  # noqa: E402

SILENCE = float(np.log(EPSILON))  # 与 fbank 的对数下限一致，用来补齐到固定帧数


def read_wav(path: Path) -> np.ndarray:
    w = wave.open(str(path), "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    if sw != 2:
        raise ValueError(f"{path}: expect 16-bit PCM")
    if sr != 16000:
        raise ValueError(f"{path}: expect 16 kHz, got {sr}")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        a = a[::2]
    return a


def wrap_to_4d(src: Path, dst: Path, frames: int) -> tuple[str, int]:
    """返回 (输入名, 输出维度)。"""
    m = onnx.load(str(src))
    g = m.graph
    if len(g.input) != 1:
        raise SystemExit(f"expect exactly 1 graph input, got {[i.name for i in g.input]}")
    orig = g.input[0]
    dims = [d.dim_value or d.dim_param for d in orig.type.tensor_type.shape.dim]
    if dims != [1, frames, 80]:
        raise SystemExit(f"unexpected input shape {dims}, expect [1, {frames}, 80]")

    new_in = "x4d"
    shape_name = "x4d_reshape_shape"
    g.initializer.append(
        numpy_helper.from_array(np.array([1, frames, 80], dtype=np.int64), name=shape_name))
    g.node.insert(0, helper.make_node(
        "Reshape", inputs=[new_in, shape_name], outputs=[orig.name]))
    g.input.remove(orig)
    g.input.insert(0, helper.make_tensor_value_info(
        new_in, TensorProto.FLOAT, [1, 1, frames, 80]))

    onnx.checker.check_model(m)
    onnx.save(m, str(dst))

    out_dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    return new_in, out_dims[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="icefall 导出的 encoder fp32 ONNX")
    ap.add_argument("--wav-dir", required=True, help="真实音频目录（16 kHz 单声道 wav）")
    ap.add_argument("--frames", type=int, default=1000, help="固定帧数（1000 = 10 s）")
    ap.add_argument("--out", required=True, help="任务目录（会生成 export/ 子目录）")
    ap.add_argument("--calib-size", type=int, default=8)
    args = ap.parse_args()

    frames = args.frames
    exp = Path(args.out) / "export"
    calib_dir = exp / "calib_data"
    calib_dir.mkdir(parents=True, exist_ok=True)

    in_name, out_dim = wrap_to_4d(Path(args.onnx), exp / "model.onnx", frames)
    print(f"[onnx] 4D 包装完成：{in_name}[1,1,{frames},80] -> encoder_out[1,{frames//2},{out_dim}]")

    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"{args.wav_dir} 下没有 wav")

    calib = []
    for wav in wavs[: args.calib_size]:
        feat = fbank80(read_wav(wav))
        if feat.shape[0] >= frames:
            feat = feat[:frames]
        else:
            pad = np.full((frames - feat.shape[0], 80), SILENCE, dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        calib.append(feat[None, None, ...].astype(np.float32))
        print(f"[calib] {wav.name}: {feat.shape[0]} 帧")

    npy_dir = calib_dir / f"{in_name}"
    npy_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(calib):
        np.save(npy_dir / f"{i:04d}.npy", arr)
    tar_path = calib_dir / f"{in_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(npy_dir.glob("*.npy")):
            tar.add(f, arcname=f.name)
    print(f"[calib] {len(calib)} 组 -> {tar_path}")

    sample = calib[0]
    np.save(exp / "sample_input.npy", sample)
    sess = ort.InferenceSession(str(exp / "model.onnx"), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {in_name: sample})[0].astype(np.float32)
    np.save(exp / "source_output.npy", ref)
    print(f"[onnx] 参考输出 encoder_out {ref.shape}, mean={ref.mean():.4f}, std={ref.std():.4f}")

    meta = {
        "model_name": "x_asr_encoder",
        "framework": "icefall-zipformer2",
        "inputs": [{
            "name": in_name,
            "shape": [1, 1, frames, 80],
            "dtype": "float32",
            "layout": "NCHW",
        }],
        "outputs": [{
            "name": "encoder_out",
            "shape": list(ref.shape),
            "dtype": "float32",
        }],
        "opset": 13,
        "notes": "X-ASR-zh-en offline encoder, static 10 s window (1000 fbank frames)",
    }
    (exp / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {exp / 'model_meta.json'}")


if __name__ == "__main__":
    main()
