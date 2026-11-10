#!/usr/bin/env bash
# 一键复现：导出 → 组装 → 校准 → If 展开 → 形状冻结 → Log→Softplus → 编译 → 仿真 → 字级一致性
# 前提：已跑过 00_setup.sh，且 weight/pretrained.pt、tokens/ 已就位。
# 注意：已发布的 axmodel 用的是 configs/pulsar2-released.json（8 组校准），
#       这里走 Magnetar 的 COMPILE 会自动取 min(30, 样本数)，产物与发布版略有差异。
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
source "$HOME/AX-model/env.sh"
ROOT="${XASR_ROOT:-$HOME/work/x-asr-axmodel}"
PY="$MAGNETAR_BASE_VENV/bin/python"
cd "$ROOT"

step=1
say() { echo "==== $((step++))/8 $*"; }

say "导出 fp32 静态 ONNX"
bash 10_export_onnx.sh

say "组装任务目录（4D 包装 + meta）"
"$PY" 20_prep_task.py --onnx onnx/encoder.fp32.onnx --wav-dir wavs --frames 1000 --out task

say "生成真实 fbank 校准集（16 组滑窗）"
"$PY" 21_make_calib.py --wav-dir wavs --frames 1000 --out task/export --name x4d --num 16

say "展开 If 节点"
"$PY" 25_fold_if.py --src task/export/model.onnx \
  --dst task/export/model_folded.onnx --input x4d:1,1,1000,80

say "冻结 shape 计算链"
"$PY" 26_freeze_shapes.py --src task/export/model_folded.onnx \
  --dst task/export/model_static.onnx --input x4d:1,1,1000,80

say "Log(1+Exp(z)) → Softplus（Pulsar2 不支持 Log）"
cp -n task/export/model.onnx task/export/model.orig.onnx   # 留存原始图备查
cp task/export/model_static.onnx task/export/model.onnx
"$PY" 27_fold_swoosh.py --src task/export/model.onnx \
  --dst task/export/model_softplus.onnx --input x4d:1,1,1000,80
cp task/export/model_softplus.onnx task/export/model.onnx

say "编译 + 仿真"
# pulsar2 run 会把输出目录里已存在的 bin 当参考做 bit-true 比较，先清干净
rm -rf task/simulate/input task/simulate/output task/simulate/metrics.json task/simulate/pulsar2_run.log
PYTHONPATH="$HOME/AX-model/Magnetar:$PYTHONPATH" \
  "$PY" 40_build.py --task-dir task --target AX650 --skip-toolchain

say "字级一致性 + 多样本仿真"
"$PY" 50_text_consistency.py --task-dir task --onnx-dir onnx --tokens tokens/tokens_nopunct.txt
PYTHONPATH="$HOME/AX-model/Magnetar:$PYTHONPATH" "$PY" 51_simulate_multi.py \
  --task-dir task --wav-dir wavs --onnx-dir onnx --tokens tokens/tokens_nopunct.txt \
  --out reports/multi_sim.json --limit 4

echo "==== 完成，报告在 task/simulate/simulate_report.md"
