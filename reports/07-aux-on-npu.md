# decoder / joiner 上 NPU（实测）

日期：2026-11-11 · 板卡：AX8850（兼容 AX650）`192.168.10.144`

## 结论

| 指标 | CPU（ORT 基线） | **默认：joiner NPU + decoder CPU** | 全 NPU |
|---|---|---|---|
| joiner 单次 | 1.359 ms | **0.284 ms（4.8×）** | 0.284 ms |
| decoder 单次 | 0.653 ms | 0.653 ms（保持 fp32） | **0.176 ms（3.7×）** |
| 10.05 s 音频端到端 | 636 ms | **360 ms** | 333 ms |
| RTF（10 s） | 0.063 | **0.036** | 0.033 |
| 域内 6 条（中/英）相对 fp32 的编辑距离 | 8 | **8（完全一致）** | 4 |
| 16 条混合集（含粤语/音乐/长英文）编辑距离 | 187 | 193（+3%） | 253（+35%） |

- **默认选择「decoder 留 CPU + joiner 上 NPU(U16)」**：域内文本与 fp32 基线**逐字一致**，
  速度接近翻倍（1148 → 591 ms/条，混合集平均）；只有域外长文本（粤语、唱歌）会轻微漂移。
- 全 NPU 模式（`aux_backend="npu"`）再快 0.5×，但长句/域外偏差明显（+35%），不设为默认。

## 怎么转的

| 步骤 | 说明 |
|---|---|
| 1. decoder 输入改 float32 | ONNX 输入是 `int64` token 对，Pulsar2 不吃 int64；在输入后插 `Cast(to=INT64)`、把图输入声明的类型改成 float32。token id 是小整数，fp32 精确可表示，实测输出 `max\|diff\| = 0` |
| 2. 校准数据 | 用真实音频跑一遍 fp32 的 encoder/decoder/joiner，收集 128 组 `(encoder_out[512], decoder_out[512])` 和对应的 token 对；joiner 两路输入各一份 tar，按文件名对齐 |
| 3. joiner 量化 | 8-bit (U8) 与 **U16** 各编译一版 |
| 4. 验证 | Pulsar2 仿真对比 logits：cosine / argmax 一致率；再上板做端到端文本对比 |

产物（`task/compile_aux/`）：

| 文件 | 体积 | 说明 |
|---|---|---|
| `decoder.axmodel` | 11 MB | U8，输入 float32 `[1,2]`，输出 `[1,512]` |
| `decoder.u16.axmodel` | 5.2 MB | U16 |
| `joiner.axmodel` | 2.7 MB | U8，输入 `[1,512]×2`，输出 `[1,5000]` |
| `joiner.u16.axmodel` | 2.7 MB | U16 |

## 仿真精度（32 组真实中间张量 vs fp32）

| 模型 | cosine 均值 | cosine 最低 | argmax 一致 |
|---|---|---|---|
| joiner U8 | 0.9999836 | 0.9999557 | **30/32** |
| **joiner U16** | **0.9999993** | **0.9999974** | **32/32** |
| decoder U8 | 0.9991173 | 0.9983981 | — |
| decoder U16 | 0.9997588 | 0.9993374 | — |

joiner 输出是 5000 维 logits，U8 下会有 2/32 的 argmax 翻转；U16 完全没有翻转 —— 这是
**默认用 U16 joiner** 的依据。

## 板端矩阵（16 条测试音频）

| 配置 | 编辑距离 | 平均耗时 |
|---|---|---|
| decoder CPU + joiner CPU | 187 | 1148 ms |
| decoder CPU + **joiner NPU(U16)** | **193** | **591 ms** |
| decoder CPU + joiner NPU(U8) | 206 | 549 ms |
| decoder NPU(U8) + joiner NPU(U16) | 253 | 542 ms |
| decoder NPU(U16) + joiner NPU(U8) | 242 | 494 ms |

域内 6 条（0/1/2/3/en/zh）逐条编辑距离：

| 配置 | 0 | 1 | 2 | 3 | en | zh | 合计 |
|---|---|---|---|---|---|---|---|
| CPU 基线 | 0 | 5 | 0 | 2 | 0 | 1 | 8 |
| **joiner NPU(U16) + decoder CPU** | 0 | 5 | 0 | 2 | 0 | 1 | **8** |
| 全 NPU | 0 | 0 | 1 | 2 | 0 | 1 | 4 |

## 切换方式

```python
from asr_engine import Asr
Asr("/root/x-asr/models", aux_backend="auto")  # 默认：joiner NPU + decoder CPU（精度优先）
Asr("/root/x-asr/models", aux_backend="npu")   # 全 NPU（最快）
Asr("/root/x-asr/models", aux_backend="ort")   # 全 CPU（基线）
```

## 为什么不把 decoder 也默认放 NPU

decoder 是自回归里的**回馈环节**：它的输出会直接作为 joiner 的输入并参与下一步决策，
量化误差会沿时间累积。实测长句（粤语 30 s、88 s 英文）受影响最明显；
而 decoder 在 CPU 上只有 0.653 ms/次、一句话通常 20–60 次（约 15–40 ms），
换成 NPU 只省 10–30 ms，却要付出长句精度代价 —— 不划算。

反观 joiner 是**每帧一次**（10 s 窗口 248 次），CPU 上 1.36 ms/次 ≈ 340 ms，
是解码耗时的绝对大头；上 NPU 后降到 0.284 ms/次，收益最大。
