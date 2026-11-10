# 踩坑记录（转换 X-ASR 到 AX650 的真实障碍）

按遇到顺序记录，每条都写「现象 → 根因 → 处理」。

## 1. `k2` wheel 与 torch 2.x 不兼容

```
ImportError: k2 was built using PyTorch 1.13.1+cu117
But you are using PyTorch 2.14.1+cpu to run it
```

- 根因：PyPI 的 `k2` 只有按 torch 1.13 编译的 wheel；
- 处理：`scripts/host/k2stub/` 顶替，只实现 `SymbolTable.from_file()` / `k2.version`，
  icefall 导出链路只用到这两个。真的用到 FSA 时会显式抛错。

## 2. `uv pip install --target` 会把 torch 拖进来

第一次装依赖时 `pydeps/` 里出现 `torch-1.13.1`（3.4 GB），会覆盖 base venv 的 torch 2.14。

- 处理：全部依赖用 `--no-deps` 安装，缺什么补什么（`00_setup.sh` 里有自动迭代补齐的循环）。

## 3. `icefall/env.py` 读 `k2.version.__build_type__`

```
AttributeError: module 'k2.version' has no attribute '__build_type__'
```

- 处理：stub 补齐 `__version__ / __build_type__ / __git_sha1__ / __git_date__`，
  `k2/__init__.py` 补 `with_cuda = False`。

## 4. Magnetar TOOLCHAIN 卡在 arm.com 交叉编译器

```
aria2c ... gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
```

- 现象：`40_build.py` 跑起来后 10 分钟没有任何输出（stdout 被重定向还会有缓冲，更难发现）；
- 根因：`toolchain.run()` → `ensure_bsp()` 会去 developer.arm.com 下交叉编译器，该站点限速；
- 处理：编译/仿真根本不需要交叉编译器（只有 C++ SDK 生成需要），
  给 `40_build.py` 加 `--skip-toolchain`，直接把 `PULSAR2_BACKEND` 写进任务配置。

## 5. Pulsar2 先用自己编译的 ORT 跑优化，遇到超大 ONNX 报 external data

```
!utils::HasExternalDataInMemory(tensor_proto) was false
```

- 现象：`onnx_opt` 阶段 ORT session 初始化直接失败（`OnnxOptimizationError`）；
- 处理：`disable_onnx_optimization: true` 绕过该步骤（代价是少了一层图优化）。

## 6. **Pulsar2 不支持 `If` 节点**（关键）

```
CodeException: (ErrorCode.QuantError, RuntimeError("Op Execution Error:
/encoder/0/layers.0/conv_module2/depthwise_conv/If ..."))
```

- 根因：icefall 的 `export-streaming-as-non-streaming-onnx.py` 用 `torch.jit.trace` 导出，
  Zipformer 的 `ConvModule` / `downsample` 里按输入长度分支的写法被保留成 43 个 `If`；
- 条件只依赖静态 shape，在 T=1000 下是常量；
- 处理：`25_fold_if.py` 用 onnxruntime 求 condition，再把选中的分支内联进主图；
- 结果：`cosine(orig, folded) = 1.00000000`，`max|diff| = 0`。

> 试过 `onnxsim 0.7.3`（以及 Pulsar2 的 `enable_onnxsim`）：598 MB / 10405 节点的图上
> 跑 >10 分钟仍未收敛，放弃。

## 7. **Pulsar2 要求 `Reshape` 的形状是静态常量**（关键）

```
op name: /encoder/3/encoder/1/nonlin_attention/Reshape_1, Reshape, shapefn failed.
input: {'x': Tensor(FP32, shape=(62, 1, 576)), 'shape': Tensor(S64, name=.../Concat_1_output_0)}
```

- 根因：`NonlinAttention` 里 `x.reshape(x.shape[0], -1, d)` 导出后 shape 是一条
  Shape→Gather→Concat 计算链；
- 处理：`26_freeze_shapes.py` 把 2477 个形状张量求值成 initializer，再剪掉死节点；
- 结果：节点 10775 → 4819，`cosine = 1.00000000`，`max|diff| = 0`。

## 8. 校准数据必须真实

`CALIBRATION_DIR` 缺失时 Magnetar 会用扰动兜底，精度虚高。本仓库用 7 条真实音频
（中/英/日、5–21 s）算 fbank，按固定 1000 帧补齐，打包成 Numpy 格式 tar.gz。

## 9. 输出形状不是 1/2 而是 1/4

Zipformer2 的 subsample 是 **4 倍**：输入 1000 帧 → `encoder_out[1, 248, 512]`。
一开始按 1/2 估算（500）是错的，以实际导出为准。
