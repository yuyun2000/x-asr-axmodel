# 转换流水线

目标：`X-ASR-zh-en`（0.16B，Zipformer2 transducer）的**离线 encoder** → AX650 的 INT8 AXMODEL。

```
官方 checkpoint (pretrained.pt)
   │  10_export_onnx.sh  （icefall export-streaming-as-non-streaming-onnx.py）
   ▼
onnx/encoder.fp32.onnx      x[1,1000,80] → encoder_out[1,500,512]   （静态、fp32、无 x_lens）
   │  20_prep_task.py      （4D 包装 + 真实 fbank 校准集 + model_meta.json）
   ▼
task/export/model.onnx      x4d[1,1,1000,80] → Reshape → x[1,1000,80] → …
   │  25_fold_if.py        （把 43 个条件恒定的 If 展开成直线代码，Pulsar2 不支持 If）
   ▼
task/export/model_folded.onnx
   │  40_build.py          （Magnetar COMPILE → SIMULATE）
   ▼
task/compile/model.axmodel + task/simulate/simulate_report.md（cosine / MAE）
```

## 0. 环境（目标机 192.168.20.38，一次性）

```bash
bash 00_setup.sh
```

做的事：下载 tokens / bpe、clone icefall（shallow）、把 lhotse / k2 依赖装进
`pydeps/`（`--no-deps`，避免把 torch 1.13 拖进来覆盖 base venv），并用 `k2stub`
顶掉按 torch 1.13 编译的 k2 wheel。

> `k2` wheel 是 `torch==1.13.1+cu117` 编译的，在 base venv 的 torch 2.14 下直接
> ImportError。导出链路只用 `k2.SymbolTable`（读 tokens.txt），所以用 stub 替代，
> 真正用到 FSA 的路径会显式报错。

## 1. 导出 fp32 静态 ONNX

```bash
bash 10_export_onnx.sh
```

关键参数（其余与原 CI 的 `export-non-streaming.sh` 完全一致）：

| 参数 | 值 | 原因 |
|---|---|---|
| `--dynamic-axes` | `0` | Pulsar2 只吃静态 shape |
| `--max-len` | `1000` | 固定 10 s 窗口（1000 帧 @16 kHz / 10 ms） |
| `--keep-x-lens` | `0` | 去掉 `x_lens`，只留一个 fbank 张量 |
| `--enable-int8-quantization` | `0` | fp32 交给 Pulsar2 量化，避免二次量化 |

模型超参来自 sherpa-onnx 的 CI 脚本（`num_encoder_layers 2,2,4,5,4,2` 等），
必须和训练配置一致，否则 `load_state_dict` 会报 key 不匹配。

## 2. 组装 Magnetar 任务目录

```bash
$MAGNETAR_BASE_VENV/bin/python 20_prep_task.py \
  --onnx onnx/encoder.fp32.onnx --wav-dir wavs --frames 1000 --out task
```

产物：

| 文件 | 说明 |
|---|---|
| `task/export/model.onnx` | 4 维输入包装后的图（`x4d[1,1,1000,80]`） |
| `task/export/calib_data/x4d.tar.gz` | 7 组真实 fbank 校准样本（Numpy 格式） |
| `task/export/sample_input.npy` | 仿真对分用的输入 |
| `task/export/source_output.npy` | ONNX 参考输出 |
| `task/export/model_meta.json` | Magnetar COMPILE/SIMULATE 读的元数据 |

为什么包成 4 维：Pulsar2 的 `Layout` 枚举只有 `NHWC / NCHW`，3 维输入没有对应的
布局语义；包一层 `[1,1,T,80]` 再在图内 Reshape 回去，代价可以忽略。

## 2.5 展开 If（Pulsar2 必需）

```bash
$MAGNETAR_BASE_VENV/bin/python 25_fold_if.py \
  --src task/export/model.onnx --dst task/export/model_folded.onnx --input x4d:1,1,1000,80
```

icefall 导出的图里有 43 个 `If` 节点（Zipformer 的 `ConvModule` / `downsample` 按输入长度
分支，被 TorchScript 保留成控制流）。条件只依赖静态 shape，在 T=1000 下是常量。
Pulsar2 的量化执行器不支持 `If`，会报 `Op Execution Error: .../If`。

本脚本把 condition 加进图输出让 onnxruntime 求值，再把选中的分支内联进主图。
验证结果：`cosine(orig, folded) = 1.00000000`，`max|diff| = 0.000000`，节点 10405 → 10775。

> 试过 `onnxsim`（0.7.3），在 598 MB / 10405 节点的图上跑 10 分钟以上未收敛，已放弃；
> Pulsar2 自带的 `enable_onnxsim` 同理。

然后把 `task/export/model.onnx` 换成 folded 版本（原图留 `model.orig.onnx` 备查）。

## 3. 编译 + 仿真

```bash
source ~/AX-model/env.sh
PYTHONPATH=$HOME/AX-model/Magnetar:$PYTHONPATH \
  $MAGNETAR_BASE_VENV/bin/python 40_build.py --task-dir task --target AX650 --skip-toolchain
```

产物：

- `task/compile/model.axmodel`（AX650 INT8）
- `task/compile/pulsar2_config.json`（可复现的编译配置）
- `task/simulate/simulate_report.md`（cosine / MAE / max_abs_diff）
- `task/simulate/pulsar2_run.log`

> `--skip-toolchain`：Magnetar 的 TOOLCHAIN 阶段会去 arm.com 下载 aarch64 交叉编译器，
> 该站点限速会卡住（AGENTS.md 已记录）；编译/仿真并不需要它，所以跳过。

验收线：`cosine_similarity ≥ 0.99`（Magnetar 的 accuracy gate）。
