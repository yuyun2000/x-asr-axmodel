#!/usr/bin/env bash
# 目标机（Linux x86_64）一次性环境准备：目录、依赖、icefall、tokens。
# 用法：bash 00_setup.sh
set -euo pipefail

# env.sh 里用了 $PYTHONPATH 且 set -u 会报未定义，先给默认值
export PYTHONPATH="${PYTHONPATH:-}"
source "$HOME/AX-model/env.sh"
export PATH="$HOME/.local/bin:$PATH"
ROOT="${XASR_ROOT:-$HOME/work/x-asr-axmodel}"
PY="$MAGNETAR_BASE_VENV/bin/python"
GH_PROXY="${GH_PROXY:-https://gh-proxy.com}"

cd "$ROOT"
mkdir -p weight tokens onnx axmodel reports logs calib

# --- 1. tokens / bpe（小文件，走 gh-proxy） -----------------------------------
if [ ! -s tokens/tokens.txt ]; then
  curl -sL -o tokens/tokens.txt \
    "$GH_PROXY/https://raw.githubusercontent.com/Gilgamesh-J/X-ASR/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/tokens.txt"
fi
if [ ! -s tokens/bpe.model ]; then
  curl -sL -o tokens/bpe.model \
    "$GH_PROXY/https://raw.githubusercontent.com/Gilgamesh-J/X-ASR/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/bpe_punc.model"
fi

# --- 2. icefall（只要包本身 + librispeech 的 zipformer 目录） ------------------
if [ ! -d icefall ]; then
  git clone --depth 1 "$GH_PROXY/https://github.com/k2-fsa/icefall.git" icefall
fi

# --- 3. 依赖（--no-deps 装进 pydeps，避免拉进 torch 1.13 覆盖 base venv） -----
install_pkg() {
  uv pip install --python "$PY" --target pydeps --no-deps "$@" >/dev/null 2>&1 || true
}
if [ ! -d pydeps ]; then
  mkdir -p pydeps
  install_pkg lhotse sentencepiece tabulate toolz click pyyaml soundfile tqdm \
              packaging filelock cffi pycparser
fi

PYTHONPATH_REAL="$ROOT/k2stub:$ROOT/pydeps:$ROOT/icefall:$ROOT/icefall/egs/librispeech/ASR/zipformer"
export PYTHONPATH="$PYTHONPATH_REAL"

# --- 4. 迭代补齐缺失的纯 python 依赖 -----------------------------------------
for _ in $(seq 1 20); do
  out=$(PYTHONPATH="$PYTHONPATH_REAL" "$PY" -c "import train; print('IMPORT_OK')" 2>&1 | tail -3 || true)
  if echo "$out" | grep -q IMPORT_OK; then
    echo "[ok] train.py 可导入，依赖齐全"
    break
  fi
  mod=$(echo "$out" | sed -n "s/.*No module named '\([^'.]*\).*/\1/p" | head -1)
  if [ -z "$mod" ]; then
    echo "[fail] 非缺包错误，原始输出："
    echo "$out"
    exit 1
  fi
  case "$mod" in
    absl) pkg=absl-py ;;
    *)    pkg="$mod" ;;
  esac
  echo "[dep] 安装 $pkg"
  install_pkg "$pkg"
done

"$PY" - <<'PY'
import sys, torch
sys.path.insert(0, "k2stub")
import k2
print("torch", torch.__version__)
print("k2 stub", k2.__version__)
PY
