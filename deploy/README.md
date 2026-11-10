# 板端部署指南

目标：把 X-ASR（中英，0.16B）的 encoder 跑在 AX650 的 NPU 上，decoder/joiner 留在 CPU，
10 秒窗口做离线识别。**本仓库的 axmodel 尚未通过精度验收**（见 `reports/03-simulate.md`），
所以下面这条链路目前只到「结构跑通」，精度达标后再把模型替换掉即可——脚本不用改。

## 0. 你会拿到什么

HuggingFace 模型仓库（`yunyu1258/x-asr-axmodel`）：

```
encoder.axmodel      # NPU，10 s 窗口输入 [1,1,1000,80]
decoder.onnx         # CPU 回退用（onnxruntime）
joiner.onnx          # CPU 回退用
tokens.txt           # 5000 token（无标点版）
test_wavs/           # 4 条 16 kHz 单声道测试音频
```

模型仓库地址（跑通精度后上传）：<https://huggingface.co/yunyu1258/x-asr-axmodel>

## 1. 板端环境

| 组件 | 说明 |
|---|---|
| AX650 板卡 + 驱动 | 板上直接跑 axengine；或者用 AXCL（USB/PCIe 加速卡）从 x86 主机跑 |
| `axengine` | `pip install axengine`（AXERA 官方 Python 包，随板端 SDK 提供） |
| `numpy` | `pip install numpy` |
| （可选）`onnxruntime` | `pip install onnxruntime`，只在 decoder/joiner 没有 axmodel 时需要 |

## 2. 下载模型

```bash
# 方式 A：huggingface-cli
pip install -U huggingface_hub
huggingface-cli download yunyu1258/x-asr-axmodel --local-dir models

# 方式 B：国内镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download yunyu1258/x-asr-axmodel --local-dir models
```

## 3. 跑起来

```bash
python run_asr.py --models-dir models --audio models/test_wavs/0.wav
# 批量
python run_asr.py --models-dir models --audio-dir models/test_wavs
```

期望输出形如：

```
[x-asr-axmodel] decoder backend: cpu-decoder
0.wav   10.05s  昨天是 MONDAY TODAY IS 礼拜二 THE DAY AFTER TOMORROW 是星期三
```

（这条参考文本来自同一条音频在 fp32 ONNX 上的输出，见 `reports/01-export.md`。）

## 4. 没有板子怎么验证

- **验证转换链路**：`scripts/host/40_build.py` 走 Pulsar2 仿真（不需要板子），
  对比 ONNX 与 axmodel 输出的 cosine。
- **验证推理逻辑**：把 `run_asr.py` 里的 `InferenceSession` 换成 onnxruntime 就能在
  x86/macOS 上跑同一套 fbank + 贪心搜索（本仓库的 `scripts/local/03_check_fbank.py`
  就是这么验的）。

## 5. 性能预期（待板端实测填表）

| 项 | 值 |
|---|---|
| encoder MACs | 17.12 GMAC / 10 s 音频 |
| axmodel 体积 | 172.6 MB |
| NPU 模式 | NPU1（单核） |
| 端到端延迟 | 待测 |
| 内存占用 | 待测 |

## 6. 已知限制

- 输入是**固定 10 秒**窗口：更长的音频按 8 秒步长切窗（`run_asr.py` 已实现），
  短音频补静音；
- 模型是**无标点**版本（`tokens_nopunct`），需要标点得再接一个标点模型；
- 中文输出的繁简取决于模型本身，本模型为简体（Qwen3-ASR 才有繁体问题，X-ASR 没有）。
