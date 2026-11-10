# 上游化路径（先跑通再 PR）

## 已完成（2026-11-10）

| 项 | 状态 |
|---|---|
| 精度验收 | ✅ Pulsar2 仿真 mean cosine **0.9981**（min 0.9967），ONNX↔AXMODEL 文本逐 token 一致（0.wav） |
| 模型发布 | ✅ <https://huggingface.co/yunyu1258/x-asr-axmodel>（encoder.axmodel + decoder/joiner.onnx + tokens + run_asr.py） |
| 转换仓库 | ✅ <https://github.com/yuyun2000/x-asr-axmodel> |
| sherpa-onnx PR | ✅ <https://github.com/k2-fsa/sherpa-onnx/pull/4185>（转换流水线，2 commit / +679 行） |
| 运行时（C++） | ⬜ 未做：本机没有 aarch64 交叉编译器与 AX 开发板，无法编译/测试 |

PR 内容与官方 zipformer 教程对齐：`scripts/zipformer-transducer/x-asr/axera/` 下放
`prepare-encoder.py`（If 展开 / 形状冻结 / Swoosh→Softplus）、`make-calibration-set.py`、
`pulsar2-encoder.json`、`verify-with-simulator.py` 与 README。

## 下一步（需要资源）

1. **板端验证**：拿到 AX650/AXCL 后跑 `pulsar2 run` + 实机推理，补齐延迟/内存数字；
2. **运行时 PR**：给 sherpa-onnx 加 `provider=axera` 的离线 transducer 支持
   （模板化 `OfflineRecognizerTransducerImpl` + `OfflineTransducerModelAxera`），
   需要交叉编译环境；可参考 PR #3271 的流式实现；
3. **精度再提升**：U16 混合量化（Pulsar2 cmodel 在 U16 上会崩，需真板）、
   用真实业务音频做校准；
4. **跟进 PR #4185 review**。

---


前提：本仓库的 axmodel 跑通且仿真 cosine ≥ 0.99，并且至少有一个真实音频的「字级一致」证据。
在拿到这些之前不提 PR——上游对「只产出二进制模型」的改动接受度低，需要的是可复现的代码路径。

## 现状（2026-11 调研）

| 项 | 现状 |
|---|---|
| Axera 后端 | `sherpa-onnx/csrc/axera/` + `csrc/axcl/`，**只有 Offline SenseVoice** 一条路径 |
| 构建 | `build-axera-linux-aarch64.sh ax650\|ax630c\|ax620q`（需 AXERA BSP SDK 1.45.0_p39）+ `build-axcl-linux-aarch64.sh` |
| 开关 | `-DSHERPA_ONNX_ENABLE_AXERA=ON` |
| 预转换模型 | Releases 里**没有** axmodel 包（对比 RKNN/Ascend/QNN 都有） |
| 文档 | 侧边栏没有 Axera 章节，README 只链到 axera-tech.com |

## PR 需要的四块工作

### 1. C++ 运行时（核心）

新增 `sherpa-onnx/csrc/axera/offline-transducer-model-axera.{h,cc}`，形态参照
`offline-sense-voice-model-axera.cc`：

- encoder：加载 `encoder.axmodel`，`AX_ENGINE_RunSync`，输入 `[1,1,T,80]`（或直接 3D，取决于
  Pulsar2 是否接受 3D 的 DefaultLayout）→ 输出 `encoder_out[1,T/2,512]`；
- decoder / joiner：继续走 ONNX Runtime（`OfflineTransducerModel` 已有实现），不重复造轮子；
- 形状管理：固定 T（如 1000 帧），超过 T 的音频由 VAD 切段；不足补静音。

> 与 SenseVoice 的差别：SenseVoice 是单个非自回归模型，整图放 NPU 就完事；X-ASR 是
> transducer，NPU 只承接 encoder，解码仍是 CPU 上的贪心搜索。这正是需要新类而不是
> 复用 `offline-sense-voice-model-axera` 的原因。

### 2. 模型发布

照 RKNN 的做法，在 Releases 加一个 tag（如 `asr-models-axera`），内含：

```
sherpa-onnx-ax650-x-asr-zh-en-2026-xx/
├── encoder.axmodel        # 本仓库产物
├── decoder.onnx           # fp32，CPU
├── joiner.onnx            # fp32，CPU
├── tokens.txt
└── test_wavs/
```

**禁止**把 axmodel 打进 git；走 Releases / HF / ModelScope。

### 3. 文档

新增 `docs/source/onnx/axera/index.rst`（对应侧边栏），内容：芯片支持矩阵、构建命令、
模型下载、C++/Python 示例、已知限制（只支持 10 s 窗口、encoder-only on NPU）。

### 4. CI

- 复用现成的 `.github/workflows/axera-linux-aarch64.yaml`（已存在，跑 build + sense-voice 测试）；
- 增加一条：下载 `asr-models-axera` 包 → 跑 `test-offline-transducer-axera` 用例；
- 无板 CI 只能做「加载 + 形状检查」，精度验证需要在真板或 AXCL 主机上做，这一点要在 PR
  描述里写清楚（上游对无板验证的接受度取决于已有的 SenseVoice 先例）。

## 提交顺序建议

1. **PR-1（低风险）**：`docs/source/onnx/axera/` 文档 + README 支持矩阵补 X-ASR 说明；
2. **PR-2（主菜）**：C++ 运行时 + CMake 开关 + 单测（加载 / 形状 / 与 ONNX 数值一致）；
3. **PR-3**：Releases 模型包 + 测试脚本 + CI 接入。

这样即使 PR-2 讨论周期长，PR-1 也能先落地，且不会挡住后续。

## 风险点

| 风险 | 说明 | 应对 |
|---|---|---|
| 无 AX 开发板 | 只能给仿真 cosine，无法给板端延迟/内存 | 用 AXCL（主机+加速卡）补一档，或在 PR 里声明待补 |
| 模型许可证 | X-ASR 是 Apache-2.0，可再分发 | 在模型包 README 标注来源与许可证 |
| 上游对「新后端」的维护成本顾虑 | Axera 后端 2025-12 才进主线，维护者是人少的 k2-fsa | PR 里附完整的构建/测试脚本 + CI，降低维护成本 |
| 精度口径 | 10 s 窗口 + VAD 切段，与官方 WenetSpeech 全句 CER 不可直接比 | 报告里写清「同窗口、同 VAD」的对比口径 |
