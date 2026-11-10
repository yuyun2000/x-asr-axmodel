#!/usr/bin/env bash
# 一键复现：导出 → 组装 → If 展开 → 形状冻结 → 编译 → 仿真 → 字级一致性
# 前提：已跑过 00_setup.sh，且 weight/pretrained.pt、tokens/ 已就位。
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
source "$HOME/AX-model/env.sh"
ROOT="${XASR_ROOT:-$HOME/work/x-asr-axmodel}"
PY="$MAGNETAR_BASE_VENV/bin/python"
cd "$ROOT"

echo "==== 1/6 导出 fp32 静态 ONNX"
bash 10_export_onnx.sh

echo "==== 2/6 组装任务目录（4D 包装 + 校准集 + meta）"
"$PY" 20_prep_task.py --onnx onnx/encoder.fp32.onnx --wav-dir wavs --frames 1000 --out task

echo "==== 3/6 展开 If 节点"
"$PY" 25_fold_if.py --src task/export/model.onnx \
  --dst task/export/model_folded.onnx --input x4d:1,1,1000,80

echo "==== 4/7 冻结 shape 计算链"
"$PY" 26_freeze_shapes.py --src task/export/model_folded.onnx \
  --dst task/export/model_static.onnx --input x4d:1,1,1000,80

cp -n task/export/model.onnx task/export/model.orig.onnx   # 留存原始图备查
cp task/export/model_static.onnx task/export/model.onnx

echo "==== 5/7 把 Swoosh 的 Log 改写成 Softplus（Pulsar2 不支持 Log）"
"$PY" 27_fold_swoosh.py --src task/export/model.onnx \
  --dst task/export/model_softplus.onnx --input x4d:1,1,1000,80
cp task/export/model_softplus.onnx task/export/model.onnx

echo "==== 6/7 编译 + 仿真"
PYTHONPATH="$HOME/AX-model/Magnetar:$PYTHONPATH" \
  "$PY" 40_build.py --task-dir task --target AX650 --skip-toolchain

echo "==== 7/7 字级一致性 + 多样本仿真"
"$PY" 50_text_consistency.py --task-dir task --onnx-dir onnx --tokens tokens/tokens_nopunct.txt
PYTHONPATH="$HOME/AX-model/Magnetar:$PYTHONPATH" "$PY" 51_simulate_multi.py \
  --task-dir task --wav-dir wavs --onnx-dir onnx --tokens tokens/tokens_nopunct.txt \
  --out reports/multi_sim.json --limit 4

echo "==== 完成，报告在 task/simulate/simulate_report.md"
