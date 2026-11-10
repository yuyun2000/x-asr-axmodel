#!/usr/bin/env python3
"""把转换产物上传到 HuggingFace 模型仓库（供板端一键下载）。

用法：
    export HF_TOKEN=hf_xxx
    python 60_upload_hf.py --task-dir task --repo-id yunyu1258/x-asr-axmodel

会创建/更新仓库并上传：
    encoder.axmodel      （仅在上传前做了精度验收时加 --accept）
    decoder.onnx / joiner.onnx
    tokens.txt
    test_wavs/*
    README.md（模型卡，含精度/体积/限制）
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

MODEL_CARD = """---
license: apache-2.0
language:
- zh
- en
library_name: axengine
tags:
- automatic-speech-recognition
- axera
- ax650
- axmodel
- zipformer
- sherpa-onnx
---

# X-ASR → AX650 (AXMODEL)

[X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en)（0.16B Zipformer2 transducer，
中英）离线 encoder 转换到 AX650 NPU 的结果。

## 文件

| 文件 | 说明 |
|---|---|
| `encoder.axmodel` | NPU 模型，输入 `x4d [1,1,1000,80]`（10 s fbank），输出 `encoder_out [1,248,512]` |
| `decoder.onnx` / `joiner.onnx` | CPU 侧贪心搜索用（onnxruntime） |
| `tokens.txt` | 5000 token，无标点版 |
| `test_wavs/` | 16 kHz 单声道测试音频 |

## 推理

见 <https://github.com/yuyun2000/x-asr-axmodel> 的 `deploy/README.md`：

```bash
python run_asr.py --models-dir models --audio models/test_wavs/0.wav
```

## 精度与体积

| 项 | 值 |
|---|---|
| MACs | 17.12 GMAC / 10 s |
| encoder.axmodel | 172.6 MB |
| 仿真 cosine（ONNX vs AXMODEL） | 见仓库 `reports/03-simulate.md` |

## 限制

- 固定 10 s 窗口；更长的音频按 8 s 步长切窗；
- 无标点输出（配 `tokens.txt` 的无标点版本）；
- 需要 AX650 / AX630C 系列 NPU 与 AXERA `axengine` 运行时。

## 来源与许可

模型权重来自 X-ASR（Apache-2.0），转换脚本与推理代码见
<https://github.com/yuyun2000/x-asr-axmodel>。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--repo-id", default="yunyu1258/x-asr-axmodel")
    ap.add_argument("--encoder", default=None, help="默认取 <task-dir>/compile/model.axmodel")
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--tokens", default="tokens/tokens_nopunct.txt")
    ap.add_argument("--wav-dir", default="wavs")
    ap.add_argument("--metrics", default=None, help="simulate/metrics.json，写进模型卡")
    ap.add_argument("--accept", action="store_true",
                    help="确认精度已验收（cosine >= 0.99）后才上传 axmodel")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    task = Path(args.task_dir)
    enc = Path(args.encoder) if args.encoder else task / "compile" / "model.axmodel"
    if not enc.exists():
        raise SystemExit(f"找不到 axmodel: {enc}")

    metrics = {}
    mpath = Path(args.metrics) if args.metrics else task / "simulate" / "metrics.json"
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))

    cos = metrics.get("cosine_similarity")
    if not args.accept:
        raise SystemExit(
            f"仿真 cosine = {cos}；验收线 0.99。确认可发布请加 --accept（或先提升精度）")
    if cos is not None and cos < 0.99:
        raise SystemExit(f"cosine {cos:.4f} < 0.99，拒绝上传（加 --accept 可强制，但不建议）")

    card = MODEL_CARD
    if metrics:
        card += f"\n| 仿真 cosine | {cos:.4f} |\n| MAE | {metrics.get('mae', 0):.4f} |\n"
    card = card.replace("| 仿真 cosine（ONNX vs AXMODEL） | 见仓库 `reports/03-simulate.md` |",
                        f"| 仿真 cosine（ONNX vs AXMODEL） | **{cos:.4f}** |" if cos else
                        "| 仿真 cosine（ONNX vs AXMODEL） | 见仓库 `reports/03-simulate.md` |")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "README.md").write_text(card, encoding="utf-8")
        shutil.copy2(enc, td / "encoder.axmodel")
        for name in ("decoder.fp32.onnx", "joiner.fp32.onnx"):
            src = Path(args.onnx_dir) / name
            if src.exists():
                dst = td / name.replace(".fp32", "")
                shutil.copy2(src, dst)
        shutil.copy2(args.tokens, td / "tokens.txt")
        wavs = td / "test_wavs"
        wavs.mkdir()
        for w in sorted(Path(args.wav_dir).glob("*.wav"))[:4]:
            shutil.copy2(w, wavs / w.name)
        api.upload_folder(folder_path=str(td), repo_id=args.repo_id, repo_type="model")
    print(f"上传完成：https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
