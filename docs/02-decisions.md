# 关键决策与取舍

## D1：先转「离线 encoder」，不碰流式 encoder

| | 离线 encoder | 流式 encoder（160/480/960/1920 ms） |
|---|---|---|
| 图结构 | 单次前向，无状态 | 每次 chunk 带 ~100 个 cache 张量进出 |
| 图规模 | 输入 `x[1,T,80]`，输出 `encoder_out[1,T/2,512]` | 输入 `x + 多个 cached_*`，输出同数量状态 |
| Pulsar2 友好度 | 高（单输入单输出） | 低（要保证状态张量顺序/形状严格一致） |
| 精度（WenetSpeech net/meeting） | 5.83 / 7.06 | 480 ms：7.38 / 9.31；160 ms：9.45 / 12.04 |
| 风险 | 无状态，跑通即用 | 状态管理写错会静默出错 |

结论：先用离线档打通「转换 → 编译 → 仿真 → 出字」全链路，再考虑流式（流式对产品更有价值，
但状态张量需要单独的 runtime 设计）。

## D2：从官方 checkpoint 重新导出 fp32，而不是直接转 sherpa 的 int8 ONNX

- sherpa-onnx Releases 里 X-ASR 只提供 `MatMulInteger` 的 int8 ONNX（ORT dynamic quantization 产物）；
- Pulsar2 的输入必须是 fp32 ONNX，由它自己做 INT8 量化（MinMax），二次量化会掉精度；
- 官方 CI 的导出脚本 `export-streaming-as-non-streaming-onnx.py` 自带
  `--enable-int8-quantization` 开关，关掉即可得到同结构 fp32 图；
- 该脚本 docstring 里甚至直接给了「suitable for NPU」的参数组合
  （`--keep-x-lens 0 --max-len 1000 --dynamic-axes 0 --enable-int8-quantization 0`）。

## D3：`k2` 用 stub 顶替

PyPI 的 `k2` wheel 编译时绑定 `torch==1.13.1+cu117`，在 base venv 的 torch 2.14 下 import 直接报错。
导出链路只用到 `k2.SymbolTable.from_file()`（读 tokens.txt）和 `k2.version`，因此用
`scripts/host/k2stub/k2/` 顶替；真正用到 FSA 的路径会显式抛 `NotImplementedError`，不会静默出错。

依赖安装用 `uv pip install --target pydeps --no-deps`，避免把 torch 1.13（约 800 MB）拖进
`pydeps` 覆盖 base venv 的 torch 2.14。

## D4：fbank 自己实现

`kaldifeat` 在目标机（py3.10 / torch 2.14）和本机（macOS arm64 / py3.12）都没有可用 wheel，
源码编译均失败。`scripts/host/fbank_kaldi.py` 用 numpy 复刻 kaldi `Fbank`：

| 参数 | 值 | 依据 |
|---|---|---|
| snip_edges | `False` | sherpa-onnx `features.h` 默认 `snip_edges = false`（transducer 路径不改它） |
| num_bins / low / high | 80 / 20 / -400 | icefall 各 zipformer 配方 + `pretrained.py` 示例 |
| dither / preemph / window | 0 / 0.97 / povey | kaldi 默认 |
| 对数下限 | `log(float32 eps) ≈ -15.94` | 与 sherpa-onnx 的静音帧一致 |

验证方式见 `scripts/local/03_check_fbank.py`：本仓库 fbank + sherpa 发布的 int8 ONNX 跑贪心解码，
文本与 sherpa-onnx 自己的 fbank 结果一致（zh/en 两条样本均验证通过）。

## D5：固定 10 秒窗口（1000 帧）

- NPU 需要静态 shape，Pulsar2 对动态轴支持有限；
- 10 s 覆盖绝大多数语音指令/一句话客服；
- 更长的音频按 sherpa-onnx 的常规做法用 VAD 切段；
- 短音频用 `log(eps)`（静音值）补齐，与真实静音帧分布一致，不会污染校准统计。

## D6：输入包成 4 维 `[1,1,T,80]`

Pulsar2 的 `Layout` 枚举只有 `NHWC / NCHW`，3 维输入没有对应语义；Magnetar 的
`_build_config` 也按 4 维来推导 channel 相关配置。包一层 Reshape 的成本可忽略。

## D7：decoder / joiner 不上 NPU（第一阶段）

- 参数量极小（decoder ≈ 11 MB、joiner ≈ 10 MB），per-frame 调用，CPU 上开销可忽略；
- 上 NPU 会产生大量小 kernel 调用，反而更慢；
- 这也和 sherpa-onnx 现有的 Axera 后端形态一致：只把重计算部分（encoder）放 NPU。
