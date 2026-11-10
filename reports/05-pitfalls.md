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

## 10. **`Log` 节点会让图被切成 33 个 NPU/CPU 子图**（关键，参考 AX 官方教程后解决）

```
subgraph [25], type: GraphType.ONNX, /encoder/4/encoder/3/conv_module2/out_proj/Log(AxLog)
subgraph [26], type: GraphType.NPU
```

- 根因：icefall 的 Zipformer 用 `SwooshL/SwooshR` 激活，导出后是
  `Log(1 + Exp(-|x|))` 的数值稳定写法。**Pulsar2 不支持 `Log`**，于是每个 Log 单独被切成
  ONNX(CPU) 子图 —— 实测 17 个 NPU 子图与 16 个 CPU 子图相互穿插，
  编译时间爆炸（>60 min 不结束），端侧还要在 CPU/NPU 之间来回拷贝。
- AXERA 官方 [zipformer.axera](https://github.com/AXERA-TECH/zipformer.axera) 的
  `cut_reciprocal_encoder.py` 就是同一类做法（把 NPU 不支持的 `Reciprocal` 换成 `Div`）。
- 处理：`27_fold_swoosh.py` 把 `Log(1 + Exp(z))` 改写成 `Softplus(z)`
  （先验证 Pulsar2 支持 `Softplus`：`x → Softplus → y` 最小图能编译通过）。
  99 处改写后：**1 个 NPU 子图、0 个 CPU 子图**，数值 `cosine = 1.00000000`（max diff 3.5e-5）。

## 11. 单子图的大模型会让 Pulsar2 的 cmodel 仿真崩溃

```
AX650CModelError: assert(wdma[0].size == 0) error in .../cmodel/sys/sys_cv.cpp run() at line 60
```

- 现象：合并成单个 NPU 子图后（`max_cycle = 97,949,776`），`pulsar2 run` 仿真在
  `subgraph_npu_0` 上崩；而拆成 33 个子图（v1）时同一份 ONNX 能正常仿真（cosine 0.955）。
- 判断：cmodel（软件仿真器）对超长单子图有 bug/限制，不是模型本身的问题。
- 对策见 `reports/03-simulate.md` 的处理记录。

## 12. 混精路径的 `Pow(x, -0.5)`

```
TileFailException("AxPow, AxPow only supports power with 0.25, 0.5, 2 and 4,
                   howerver got [-0.5]") @ /encoder_embed/out_norm/Pow_1
```

- 现象：纯 INT8 能编过，但一开 `highest_mix_precision: true` 就编译失败；
- 根因：LayerNorm 里的 20 处 `rsqrt` 被导出成 `Pow(x, -0.5)`，纯 INT8 路径下会被
  融合进 `LayerNormalization`，混精打断融合后 `AxPow` 就遇到不支持的指数；
- 处理：`28_fold_pow.py` 改写成 `1 / Sqrt(x)`（与 AX 官方 `Reciprocal → Div` 同理），
  20 处替换后 `cosine = 1.00000000`（max diff 3.1e-5）。

## 13. 混精 + Softplus 的 LUT 限制（最终放弃混精）

```
TileFailException: AxSoftplus, 'dont support lut_float opr in AXOPS/ONNXOPS/CUSTOM_OPS'
op: /encoder/0/layers.0/feed_forward1/out_proj/Log
```

- 把 `Pow(-0.5)` 改写掉之后，混精路径继续推进，但卡在了第 10 条里的 `Softplus`：
  量化路径下 `AxSoftplus` 用**定点 LUT** 实现，而混精会让该层变成 FP32，FP32 的 LUT
  （`lut_float`）不被支持；
- 也就是说 **`Log→Softplus` 改写与混合精度互斥**：要混精就得保留 `Log`，而 `Log` 会把图
  切成 33 个子图（编译超时 + 精度崩），死循环；
- 结论：**放弃混精，交付纯 INT8 + Softplus** 这一条唯一验证通过的路径。

### 三条精度改进路线全部关闭（合起来看）

| 路线 | 阻塞点 |
|---|---|
| U16 逐层量化（AX 官方默认） | Pulsar2 cmodel 仿真崩溃，无法验证 |
| `highest_mix_precision` | `AxSoftplus` 不支持 FP32 LUT |
| SmoothQuant | 编译被 SIGTERM（内存峰值） |
