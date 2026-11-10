# X-ASR → AXMODEL (AX650)

把 [X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en)（0.16B，Zipformer2 transducer，中英）转换成 AXERA AXMODEL，跑在 AX650 / AX630C / AX620Q 上。

这是独立实验仓库，目标是先在本仓库把「转换 → 编译 → 仿真验证」跑通；跑通且精度验收后再考虑向上游
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 提 PR（其 Axera 后端目前只实现了 Offline SenseVoice）。

## 现状

| 阶段 | 状态 | 产物 | 备注 |
|---|---|---|---|
| 0. 模型来源确认 | ✅ | — | 官方 checkpoint + icefall 导出脚本 |
| 1. fp32 离线 ONNX 导出 | ✅ 2026-11-10 | `onnx/encoder.fp32.onnx`（571 MB） | 官方 CI 同款脚本，关掉 int8；导出文本与 sherpa int8 完全一致 |
| 2. 静态 shape（NPU 友好） | ✅ 2026-11-10 | `task/export/model.onnx` | `--max-len 1000` / `--keep-x-lens 0` + 4D 包装，输出 `[1,248,512]` |
| 3. Pulsar2 编译 | 🚧 | `task/compile/model.axmodel` | AX650 INT8，7 组真实 fbank 校准 |
| 4. 仿真精度验证 | ⬜ | `reports/simulate.md` | cosine ≥ 0.99 |
| 5. 端到端出字验证 | ⬜ | `reports/e2e.md` | 需要 AXCL/开发板 |
| 6. 上游 PR | ⬜ | — | 见 `docs/03-upstream-plan.md` |

## 关键决策

**为什么转「离线 encoder」而不是流式 encoder**

- 流式 encoder 有 ~100 个 cache state 输入/输出，跨 chunk 管理状态，AXERA 上要写状态机；
- 离线 encoder 是**无状态单次前向**（`x[1,T,80] → encoder_out[1,T/2,512]`），与 sherpa-onnx 里
  SenseVoice 的 Axera 后端形态一致，最容易先跑通；
- 精度上离线档也是 X-ASR 最强的一档（WenetSpeech net 5.83 / meeting 7.06）。

**为什么用官方 checkpoint 重新导出，而不是直接转 sherpa 的 int8 ONNX**

Pulsar2 需要 fp32 ONNX 自己量化；sherpa releases 里只放了 `MatMulInteger` 的 int8 图。
官方 CI 用的就是 icefall 的 `export-streaming-as-non-streaming-onnx.py`，只把
`--enable-int8-quantization` 关掉即可得到同结构 fp32 图，且该脚本 docstring 里就有
「suitable for NPU」的参数组合。

**固定输入长度 10 s（1000 帧）**

`--max-len 1000` 把 T 固定成 1000 帧 = 10 s @16kHz/10ms hop，编码器输出 500 帧。
超过 10 s 的音频按 sherpa-onnx 的常规做法先用 VAD 切段。

## 目录

```
scripts/local/   本机（macOS）脚本：取参考模型、测试音频、跑 sherpa 基线
scripts/host/    目标机（Linux x86_64）脚本：导出 ONNX、静态化、编译、仿真
configs/         .magnetarrc 模板
docs/            设计文档与踩坑记录
reports/         转换报告（生成）
assets/          tokens.txt、测试音频等小文件
```

## 环境

- 转换主机：`m5stack@192.168.20.38`（Ubuntu 24.04, x86_64, 16 核, 62 GB）
  - Magnetar `~/AX-model/Magnetar`，Pulsar2 8.0 `~/.cache/magnetar/pulsar2/8.0`
  - base venv Python 3.10 + torch 2.14.1+cpu + onnx 1.22 + ort 1.23.2
- 目标芯片：AX650（Pulsar2 仿真验证；板端验证需要 AXCL 或开发板）

## 快速复现

```bash
# 0) 目标机准备（一次性）
bash scripts/host/00_setup.sh

# 1) 导出 fp32 离线 ONNX（动态长度）
bash scripts/host/10_export_onnx.sh

# 2) 静态 shape + 自检
~/work/x-asr-axmodel/pydeps/../bin/python scripts/host/20_prepare_static.py

# 3) 编译 + 仿真
bash scripts/host/50_compile.sh
```

详细步骤见 `docs/01-pipeline.md`。
