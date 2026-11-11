#!/usr/bin/env python3
"""把 decoder / joiner 也转成 axmodel（放到 NPU 上）。

两件事：
1. decoder 的输入是 int64 token 对，Pulsar2 不吃 int64 → 在图里插一个 Cast，
   对外暴露 int32 输入；
2. 用真实音频跑一遍 fp32 的 encoder / decoder / joiner，收集真实分布作为
   两个小模型的校准数据（joiner 需要 encoder_out + decoder_out 两路输入，
   decoder 需要 token 对）。

产物：onnx/decoder.int32.onnx、onnx/joiner.ax.onnx、
      task/aux_calib/{decoder,encoder_out,decoder_out}/*.npy 及其 tar.gz
"""

from __future__ import annotations

import argparse
import glob
import io
import tarfile
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbank_kaldi import EPSILON, fbank80  # noqa: E402

FRAMES = 1000
BLANK, CTX = 0, 2


def rewrite_decoder(src: Path, dst: Path) -> None:
    m = onnx.load(str(src))
    inp = m.graph.input[0]
    if inp.type.tensor_type.elem_type != TensorProto.INT64:
        raise SystemExit(f"{src} input is not int64")
    cast_out = inp.name + "_i64"
    cast = helper.make_node("Cast", [inp.name], [cast_out],
                            to=TensorProto.INT64, name="input_to_int64")
    for node in m.graph.node:
        for i, name in enumerate(node.input):
            if name == inp.name:
                node.input[i] = cast_out
    m.graph.node.insert(0, cast)
    inp.type.tensor_type.elem_type = TensorProto.INT32
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    print(f"[decoder] {dst.name}: in={[(i.name, i.type, i.shape) for i in s.get_inputs()]}")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--export-dir", default="task/export")
    ap.add_argument("--wav-dir", default="wavs")
    ap.add_argument("--out-dir", default="task/aux_calib")
    ap.add_argument("--num-samples", type=int, default=128)
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rewrite_decoder(onnx_dir / "decoder.fp32.onnx", onnx_dir / "decoder.int32.onnx")

    enc = ort.InferenceSession(str(Path(args.export_dir) / "model.onnx"),
                               providers=["CPUExecutionProvider"])
    dec_i32 = ort.InferenceSession(str(onnx_dir / "decoder.int32.onnx"),
                                   providers=["CPUExecutionProvider"])
    dec_fp32 = ort.InferenceSession(str(onnx_dir / "decoder.fp32.onnx"),
                                    providers=["CPUExecutionProvider"])
    joi = ort.InferenceSession(str(onnx_dir / "joiner.fp32.onnx"),
                               providers=["CPUExecutionProvider"])
    joi_in = [i.name for i in joi.get_inputs()]

    wavs = sorted(glob.glob(str(Path(args.wav_dir) / "*.wav")))
    enc_samples, dec_samples, tok_samples = [], [], []
    for wav in wavs:
        pcm = read_wav(Path(wav))[: 16000 * 10]
        feats = fbank80(pcm)[:FRAMES]
        if feats.shape[0] < FRAMES:
            feats = np.concatenate([feats, np.full((FRAMES - feats.shape[0], 80),
                                                   np.log(EPSILON), np.float32)])
        x = feats[None, None, ...].astype(np.float32)
        enc_out = enc.run(None, {enc.get_inputs()[0].name: x})[0][0]   # [T,512]

        y = np.zeros((1, CTX), dtype=np.int64)
        dec_state = dec_fp32.run(None, {dec_fp32.get_inputs()[0].name: y})[0]
        for t in range(enc_out.shape[0]):
            enc_samples.append(enc_out[t:t + 1].copy())
            dec_samples.append(dec_state.copy())
            tok_samples.append(y.copy())
            logit = joi.run(None, {joi_in[0]: enc_out[t:t + 1], joi_in[1]: dec_state})[0]
            k = int(logit.reshape(-1).argmax())
            if k != BLANK:
                y = np.array([[y[0, 1], k]], dtype=np.int64)
                dec_state = dec_fp32.run(None, {dec_fp32.get_inputs()[0].name: y})[0]

    rng = np.random.default_rng(0)
    pick = rng.choice(len(enc_samples), size=min(args.num_samples, len(enc_samples)),
                      replace=False)

    def dump(name: str, arrays: list[np.ndarray], dtype) -> None:
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.npy"):
            f.unlink()
        with tarfile.open(out_dir / f"{name}.tar.gz", "w:gz") as tar:
            for i, idx in enumerate(pick):
                arr = np.ascontiguousarray(arrays[idx], dtype=dtype)
                buf = io.BytesIO()
                np.save(buf, arr)
                info = tarfile.TarInfo(f"{i:04d}.npy")
                info.size = buf.tell()
                buf.seek(0)
                tar.addfile(info, buf)

    # joiner 需要两路输入，按同样的索引对齐；decoder 的 token 对以 float32 喂入
    # （图内 Cast 回 int64，小整数在 fp32 里精确可表示）
    dump("encoder_out", enc_samples, np.float32)
    dump("decoder_out", dec_samples, np.float32)
    dump("y", tok_samples, np.float32)

    print(f"[calib] {len(pick)} samples -> {out_dir}")
    print("[calib] joiner inputs:", joi_in, "| decoder input dtype: int32")


if __name__ == "__main__":
    main()
