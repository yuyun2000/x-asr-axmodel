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
   │  40_build.py          （Magnetar TOOLCHAIN → COMPILE → SIMULATE）
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

## 3. 编译 + 仿真

```bash
source ~/AX-model/env.sh
PYTHONPATH=$HOME/AX-model/Magnetar:$PYTHONPATH \
  $MAGNETAR_BASE_VENV/bin/python 40_build.py --task-dir task --target AX650
```

产物：

- `task/compile/model.axmodel`（AX650 INT8）
- `task/compile/pulsar2_config.json`（可复现的编译配置）
- `task/simulate/simulate_report.md`（cosine / MAE / max_abs_diff）
- `task/simulate/pulsar2_run.log`

验收线：`cosine_similarity ≥ 0.99`（Magnetar 的 accuracy gate）。
