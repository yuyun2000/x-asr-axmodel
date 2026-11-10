#!/usr/bin/env python3
"""驱动 Magnetar 的 TOOLCHAIN → COMPILE → SIMULATE 三个阶段（跳过 EXPORT，因为
ONNX 已由 icefall 导出并经过 20_prep_task.py 处理）。

用法（在目标机、已 source ~/AX-model/env.sh 的情况下）：
    PYTHONPATH=$HOME/AX-model/Magnetar:$PYTHONPATH python 40_build.py \
        --task-dir task --target AX650
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--target", default="AX650")
    ap.add_argument("--npu-mode", default="NPU1")
    ap.add_argument("--export-src", default=None,
                    help="外部 export 目录（含 model.onnx / model_meta.json / calib_data）")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-toolchain", action="store_true",
                    help="跳过 Magnetar TOOLCHAIN（arm.com 交叉编译器常被限速卡住；"
                         "编译/仿真只需要 Pulsar2 独立包）")
    ap.add_argument("--backend", default=None, help="显式指定 Pulsar2 后端，如 pkg:/path/8.0")
    args = ap.parse_args()

    from magnetar.stages.init import run as init_run
    from magnetar.stages.toolchain import run as toolchain_run
    from magnetar.stages.compile import run as compile_run
    from magnetar.stages.simulate import run as simulate_run

    task_dir = Path(args.task_dir).resolve()
    if not (task_dir / "config.json").exists():
        init_run({
            "TASK_DIR": str(task_dir),
            "MODEL_NAME": "x_asr_encoder",
            "TARGET_HARDWARE": args.target,
            "NPU_MODE": args.npu_mode,
            "SOURCE": "local:icefall-x-asr",
        })
        print(f"[init] {task_dir}")

    if args.export_src:
        src = Path(args.export_src).resolve()
        dst = task_dir / "export"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_dir():
                shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dst / p.name)
        print(f"[export] 从 {src} 同步到 {dst}")

    if args.skip_toolchain:
        from magnetar.pulsar2_util import resolve_backend
        backend = args.backend or resolve_backend()
        cfg_path = task_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        bsp_out = Path.home() / ".cache" / "magnetar" / "bsp" / "ax650" / "msp" / "out"
        cfg.update({
            "PULSAR2_BACKEND": backend,
            "BSP_ROOT": str(bsp_out.parent.parent) if bsp_out.exists() else "",
            "AX_RUNTIME_ROOT": str(bsp_out) if bsp_out.exists() else "",
            "CXX_TOOLCHAIN": "",
        })
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[toolchain] skipped, backend={backend}")
    else:
        backend = toolchain_run(None, task_dir)
        print(f"[toolchain] backend={backend}")

    if not args.skip_compile:
        compile_run(task_dir, args.target, backend)
        ax = task_dir / "compile" / "model.axmodel"
        print(f"[compile] {ax} {ax.stat().st_size/1e6:.2f} MB")

    sample = np.load(task_dir / "export" / "sample_input.npy")
    metrics = simulate_run(task_dir, sample, backend,
                           input_name="x4d", output_name="encoder_out")
    print("[simulate]", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
