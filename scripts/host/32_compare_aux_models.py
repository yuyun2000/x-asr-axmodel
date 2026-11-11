#!/usr/bin/env python3
"""对比 decoder / joiner 的 axmodel 与 fp32 ONNX（Pulsar2 仿真）。

用法（在 Linux 宿主上，需 Pulsar2）：
    python3 32_compare_aux_models.py --samples 32
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

PULSAR2 = "/home/m5stack/.cache/magnetar/pulsar2/8.0"


def load_tar(path: Path) -> list[np.ndarray]:
    out = []
    with tarfile.open(path) as t:
        for m in sorted(t.getmembers(), key=lambda x: x.name):
            out.append(np.load(io.BytesIO(t.extractfile(m).read())))
    return out


def run_axmodel(model: Path, inputs: dict[str, np.ndarray], work: Path) -> dict[str, np.ndarray]:
    """用 pulsar2 run 跑一条样本，返回 {输出名: 数组}。"""
    import shutil
    if work.exists():
        shutil.rmtree(work)
    (work / "in").mkdir(parents=True)
    (work / "out").mkdir(parents=True)
    for name, arr in inputs.items():
        arr.astype(np.float32).tofile(work / "in" / f"{name}.bin")
    ld = f"{PULSAR2}/lib/ld-linux-x86-64.so.2"
    py = f"{PULSAR2}/python3/bin/python3"
    main = f"{PULSAR2}/pulsar2/axnn/yamain/main.py"
    cmd = [ld, py, main, "run", "--model", str(model),
           "--input_dir", str(work / "in"), "--output_dir", str(work / "out")]
    env = {
        "LD_LIBRARY_PATH": f"{PULSAR2}/lib",
        "PYTHONHOME": f"{PULSAR2}/python3",
        "PYTHONPATH": f"{PULSAR2}/pulsar2/axnn:{PULSAR2}/pulsar2/axnn/axnn/tools:{PULSAR2}/pulsar2",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    out = {}
    for f in (work / "out").glob("*.bin"):
        out[f.stem] = np.fromfile(f, dtype=np.float32)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--aux-dir", default="task/compile_aux")
    ap.add_argument("--calib-dir", default="task/aux_calib")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--work", default="/tmp/aux_sim")
    args = ap.parse_args()

    calib = Path(args.calib_dir)
    enc_s = load_tar(calib / "encoder_out.tar.gz")
    dec_s = load_tar(calib / "decoder_out.tar.gz")
    tok_s = load_tar(calib / "y.tar.gz")
    n = min(args.samples, len(enc_s))

    joi = ort.InferenceSession(f"{args.onnx_dir}/joiner.fp32.onnx",
                               providers=["CPUExecutionProvider"])
    j_in = [i.name for i in joi.get_inputs()]
    dec = ort.InferenceSession(f"{args.onnx_dir}/decoder.fp32.onnx",
                               providers=["CPUExecutionProvider"])

    joiner_ref = [joi.run(None, {j_in[0]: enc_s[i], j_in[1]: dec_s[i]})[0].reshape(-1)
                  for i in range(n)]
    dec_ref = [dec.run(None, {dec.get_inputs()[0].name: tok_s[i].astype(np.int64)})[0].reshape(-1)
               for i in range(n)]

    for variant in ("joiner.axmodel", "joiner.u16.axmodel"):
        model = Path(args.aux_dir) / variant
        if not model.exists():
            continue
        cos_l, arg_ag = [], 0
        for i in range(n):
            ref = joiner_ref[i]
            got = run_axmodel(model, {"encoder_out": enc_s[i], "decoder_out": dec_s[i]},
                              Path(args.work) / f"joi_{variant}_{i}")["logit"]
            cos_l.append(cosine(ref, got))
            arg_ag += int(ref.argmax() == got.argmax())
        print(f"[{variant}] n={n} cosine mean={np.mean(cos_l):.7f} min={np.min(cos_l):.7f} "
              f"argmax-agree={arg_ag}/{n}")

    for variant in ("decoder.axmodel", "decoder.u16.axmodel"):
        model = Path(args.aux_dir) / variant
        if not model.exists():
            continue
        cos_d = []
        for i in range(n):
            got = run_axmodel(model, {"y": tok_s[i]},
                              Path(args.work) / f"dec_{variant}_{i}")["decoder_out"]
            cos_d.append(cosine(dec_ref[i], got))
        print(f"[{variant}] n={n} cosine mean={np.mean(cos_d):.7f} min={np.min(cos_d):.7f}")


if __name__ == "__main__":
    main()
