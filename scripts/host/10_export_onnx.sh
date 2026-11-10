#!/usr/bin/env bash
# 用 icefall 的官方导出脚本，把 X-ASR 的 streaming checkpoint 导出成
# 「非流式 + 静态 shape + 无 int8」的 fp32 ONNX（Pulsar2 友好）。
#
# 与原 CI（sherpa-onnx/scripts/zipformer-transducer/x-asr/export-non-streaming.sh）
# 的差异只有三处：
#   --dynamic-axes 0        T 固定，不留动态轴（Pulsar2 需要静态 shape）
#   --max-len 1000          固定 10 s = 1000 帧 @16 kHz / 10 ms
#   --keep-x-lens 0         去掉 x_lens 输入，只留一个 fbank 张量
# 以及不打开 --enable-int8-quantization（Pulsar2 自己量化）。
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}"
source "$HOME/AX-model/env.sh"
ROOT="${XASR_ROOT:-$HOME/work/x-asr-axmodel}"
PY="$MAGNETAR_BASE_VENV/bin/python"
MAX_LEN="${MAX_LEN:-1000}"

CKPT="${XASR_CKPT:-$ROOT/weight/pretrained.pt}"
TOKENS="${XASR_TOKENS:-$ROOT/tokens/tokens_nopunct.txt}"
EXP="$ROOT/exp/nopunct"

[ -s "$CKPT" ] || { echo "checkpoint 不存在：$CKPT"; exit 1; }
[ -s "$TOKENS" ] || { echo "tokens 不存在：$TOKENS"; exit 1; }

mkdir -p "$EXP"
ln -sf "$CKPT" "$EXP/epoch-99.pt"
cp "$TOKENS" "$EXP/tokens.txt"

cd "$ROOT/icefall/egs/librispeech/ASR"

PYTHONPATH="$ROOT/k2stub:$ROOT/pydeps:$ROOT/icefall:$ROOT/icefall/egs/librispeech/ASR/zipformer" \
"$PY" ./zipformer/export-streaming-as-non-streaming-onnx.py \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir "$EXP" \
  --tokens "$EXP/tokens.txt" \
  --max-len "$MAX_LEN" \
  --dynamic-axes 0 \
  --keep-x-lens 0 \
  --use-int32-inputs 0 \
  --enable-int8-quantization 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 \
  --chunk-size "-1" \
  --left-context-frames "-1"

mkdir -p "$ROOT/onnx"
for f in encoder decoder joiner; do
  src="$EXP/$f-epoch-99-avg-1.onnx"
  [ -s "$src" ] && cp -v "$src" "$ROOT/onnx/$f.fp32.onnx"
done
ls -lh "$ROOT/onnx"
