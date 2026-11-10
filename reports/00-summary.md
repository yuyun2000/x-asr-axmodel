# 交付总结（2026-11-10 ~ 11-11）

## 一句话

把 X-ASR（0.16B Zipformer2 transducer，中英）的离线 encoder 转成 AX650 的 axmodel，
仿真 cosine **0.9981（min 0.9967）**，模型已发布到 HuggingFace，转换流水线提了 sherpa-onnx PR。

## 三个交付物

| 交付物 | 地址 | 状态 |
|---|---|---|
| 板端模型 | <https://huggingface.co/yunyu1258/x-asr-axmodel> | encoder.axmodel（165 MB）+ decoder/joiner.onnx + tokens + run_asr.py + 测试音频 |
| 转换仓库 | <https://github.com/yuyun2000/x-asr-axmodel> | 流水线脚本、决策记录、踩坑、精度报告 |
| 上游 PR | <https://github.com/k2-fsa/sherpa-onnx/pull/4185> | 6 个文件 +700 行左右；CodeRabbit 两轮 review 已处理 |

## 精度

| 指标 | 值 |
|---|---|
| 仿真 cosine（ONNX vs AXMODEL，4 条音频） | mean **0.9981**，min **0.9967** |
| 文本（相对 fp32 ONNX 的 token 编辑距离） | 0 / 10 / 2 / 2，合计 14 |
| 对照 sherpa-onnx 官方 int8 模型（同条件） | 0 / 0 / 3 / 5，合计 8 |
| 结论 | 与官方 int8 模型同量级：2 条更准、1 条持平、1 条（英文重复词 `ALWAYS ALWAYS`）更差 |

## 关键技术点（复现别踩的坑）

1. **`Log` 是精度杀手**：Zipformer 的 `SwooshL/SwooshR` 导出成
   `Log(1+Exp(-abs(x)))`，Pulsar2 不支持 `Log` → 图被切成 17 个 NPU + 16 个 CPU 子图，
   编译 1 小时不结束、cosine 只有 0.955、解码全空。改成 `Softplus` 后：
   **1 个 NPU 子图、cosine 0.999、文本逐 token 一致**。
   这与 AX 官方 `cut_reciprocal_encoder.py`（Reciprocal→Div）是同一类处理。
2. **`If`（43 个）和动态 shape（2477 个）** 必须展开/冻结，否则 Pulsar2 直接报错。
3. **`compiler.check` 设为 0**，否则 "confused op refine" 阶段会卡很久。
4. **`pulsar2 run` 会把输出目录里已存在的 bin 当参考做 bit-true 比较** ——
   多次仿真同一目录会报 `output differs ... (bit true)`，每条样本用独立目录。
5. **K2 wheel 绑定 torch 1.13**，在 torch 2.x 环境用 stub 顶替。
6. **Magnetar 的 TOOLCHAIN 阶段卡在 arm.com 交叉编译器**，编译/仿真用
   `--skip-toolchain` 绕过（只有 C++ SDK 生成需要它）。

## 没能做的（外部依赖）

| 项 | 原因 | 需要 |
|---|---|---|
| 板端实测延迟/内存 | 没有 AX650 板或 AXCL | 一块板 / AXCL 主机 |
| C++ 运行时 PR（`provider=axera` + 离线 transducer） | 无 aarch64 交叉编译器与板卡，写出来也无法验证 | 交叉工具链 + 板 |
| U16 混合量化提精度 | Pulsar2 的 cmodel 在 U16 上崩，无法验证 | 真板验证 |
| `highest_mix_precision` | `AxPow only supports 0.25/0.5/2/4, got [-0.5]` | 再做一次 Pow(-0.5)→Sqrt+Div 图改写 |
| SmoothQuant | 编译被 SIGTERM（疑似内存峰值） | 更大内存的机器 |
