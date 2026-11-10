# X-ASR → AX650 AXMODEL

把 [X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en)（0.16B，Zipformer2 transducer，
中英双语，Apache-2.0）的离线 encoder 转换成 AXERA AX650 的 axmodel，并给出板端推理与复现流程。

转换参考了 AXERA 官方的 [zipformer.axera](https://github.com/AXERA-TECH/zipformer.axera) 工程
（U16 混合量化 + SmoothQuant + `conv_bias FP32` 那一套）。

## 已发布的产物

| 产物 | 位置 |
|---|---|
| sherpa-onnx PR（转换流水线） | <https://github.com/k2-fsa/sherpa-onnx/pull/4185> |
| 板端模型（encoder.axmodel + decoder/joiner.onnx + tokens + 测试音频） | <https://huggingface.co/yunyu1258/x-asr-axmodel> |

## 板端用（模型下载后一行跑）

```bash
pip install axengine numpy
huggingface-cli download yunyu1258/x-asr-axmodel --local-dir models
python deploy/python/run_asr.py --models-dir models --audio models/test_wavs/0.wav
```

详细步骤见 [deploy/README.md](deploy/README.md)。模型仓库：<https://huggingface.co/yunyu1258/x-asr-axmodel>

## 现状

| 阶段 | 状态 | 产物 / 报告 |
|---|---|---|
| 0. 模型来源与导出方案 | ✅ | `docs/02-decisions.md` |
| 1. fp32 静态 ONNX 导出 | ✅ | `onnx/encoder.fp32.onnx`（571 MB），导出文本与 sherpa int8 逐字一致 |
| 2. If 展开（43→0） | ✅ | `scripts/host/25_fold_if.py`，cosine 1.0 / max diff 0 |
| 3. shape 计算链冻结（2477 个） | ✅ | `scripts/host/26_freeze_shapes.py`，节点 10775→4819 |
| 4. Pulsar2 编译 | ✅ | `model.axmodel` 165 MB，**17.12 GMAC / 10 s**，单 NPU 子图 |
| 5. 精度验收（cosine ≥ 0.99） | ✅ | **cosine 0.999012**，端到端文本与 ONNX 逐 token 一致，见 `reports/03-simulate.md` |
| 6. 板端推理 + 部署文档 | 🚧 | `deploy/`（脚本已就绪，待板端实测） |
| 7. HF 模型发布 | ✅ | <https://huggingface.co/yunyu1258/x-asr-axmodel> |
| 8. 上游 sherpa-onnx PR | ✅ | <https://github.com/k2-fsa/sherpa-onnx/pull/4185> |

## 为什么是「离线 10 秒窗口」

- 离线 encoder 是**无状态单次前向**（`x[1,1000,80] → encoder_out[1,248,512]`），最容易在 NPU 上
  跑通，部署也最简单（不需要管理几十个 cache 状态张量）；
- 精度上离线档也是 X-ASR 最强的一档（WenetSpeech net/meeting 5.83/7.06）；
- 10 s 覆盖绝大多数一句话场景，更长的音频按 8 s 步长切窗。

流式版本（160/480/960/1920 ms）留作后续：需要状态张量管理 + 与
[zipformer.axera](https://github.com/AXERA-TECH/zipformer.axera) 类似的 segment/shift 调度。

## 目录

```
scripts/host/     转换流水线（00_setup → 10_export → 20/21 校准 → 25_fold → 26_freeze
                  → 40_build → 50_text_consistency → 60_upload_hf）
scripts/host/k2stub/   k2 wheel 与 torch 2.x 不兼容，用 stub 顶掉
scripts/local/    本机验证（fbank 数值等价）
deploy/           板端部署（Python 推理 + 说明）
docs/             流水线、决策记录、上游 PR 计划
reports/          转换/仿真/踩坑记录
assets/           tokens.txt（Apache-2.0，来源见 NOTICE）
```

## 环境

- 转换主机：`m5stack@192.168.20.38`（Ubuntu 24.04 / x86_64 / 16 核 / 62 GB）
  - Magnetar `~/AX-model/Magnetar`，Pulsar2 8.0 `~/.cache/magnetar/pulsar2/8.0`
  - base venv：Python 3.10 + torch 2.14.1+cpu + onnx 1.22 + ort 1.23.2
- 目标芯片：AX650（Pulsar2 仿真验证；板端验证需要 AXCL 或开发板）

## 复现

```bash
# 目标机上（一次性）
bash scripts/host/00_setup.sh
# 全流程（导出 → 校准 → 编译 → 仿真 → 字级一致性）
bash scripts/host/run_all.sh
```

细节见 [docs/01-pipeline.md](docs/01-pipeline.md)。

## 许可

- 转换脚本与文档：Apache-2.0
- 模型权重来自 X-ASR（Apache-2.0），见 `assets/NOTICE.md`
