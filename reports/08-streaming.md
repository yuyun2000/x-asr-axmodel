# 流式模型转换与板端验证（960 ms chunk）

日期：2026-11-12 · 板卡：`192.168.10.105`（AX650N_M5stack_8G，6 GB CMM）

## 结论

| 项 | 结果 |
|---|---|
| 流式 encoder axmodel | **157 MB**，单个 NPU 子图，116 输入 / 116 输出 |
| 仿真精度（encoder_out） | **0.9985**（零状态校准）/ 0.9748（真实状态校准，见下） |
| 板端增量识别 | 每 960 ms 一块，**单块往返 ~80 ms** |
| 板端整句 | 10.56 s 音频总耗时 ~760 ms（enc 304 + dec 456），**RTF ≈ 0.072** |
| 文本（0.wav） | 与离线版**完全一致**："昨天是 MONDAY TODAY IS 礼拜二 THE DAY AFTER TOMORROW 是星期三" |
| 文本（3.wav） | 离线 "…一般现在时我们把它时态写上"，流式 "…一般现代时对我把它时态写上去"（3 字差异） |
| 常驻内存 | web_server 87 MB + stream_server 126 MB ≈ **213 MB RSS** |

## 模型结构（960 ms）

- 输入 **116** 个：`x [1,109,80]` + 115 个 cache（`cached_key_0 [256,1,128]`、`cached_conv1_0 [1,192,15]`、…）
- 输出 **116** 个：`encoder_out [1,24,512]` + 115 个 `new_cached_*`（回写同名状态）
- 每次前进 **96 帧 = 960 ms**，输入窗口 **109 = 96 + 13**（icefall 导出定义）
- 单次 IO 约 4.2 MB

## 转换过程（三个坑）

| 坑 | 现象 | 解法 |
|---|---|---|
| 官方 HF 里的流式 ONNX 是 `dynamic-batch 1` | Pulsar2 报 `Slice, shapefn failed`（含 BOOL/S32 的 mask 链） | 用 icefall `export-onnx-streaming.py --dynamic-batch 0 --chunk-size 48 --left-context-frames 256` 自己重导，节点 10891 → 6676 |
| `processed_lens` 是 int64 输入，Pulsar2 不支持 | 无法量化 | 冻结成常量（256），并在图里常量折叠 344 个 mask 节点；实测 `processed_lens` 动态 vs 冻结 256 的流式文本基本一致（0.wav/30 s 粤语完全相同，3.wav 差 1 字） |
| 常量折叠后 `new_processed_lens` 变成纯常量输出 | 量化报 `KeyError('new_processed_lens')` | 从 `graph.output` 去掉（运行时自己维护） |

最终静态图 4216 节点，与 fp32 原图数值一致（max diff 6e-5）。

## 校准集

- 第一版：cache 全零（只覆盖第一块）→ encoder_out cosine **0.9985**，但 cache 输出均值只有 0.968
- 第二版：跑 fp32 流式，按 chunk 记录 `(x, 所有 cache)`，取 16 组真实状态 → cache 输出均值 0.984（min 0.944），但 encoder_out cosine 降到 0.9748

板端实测**两版文本都对得上**（0.wav 完全一致），当前板端部署用的是**第二版（真实状态校准）**，长句更稳。

## 板端运行时

- `deploy/board/stream_asr.py`：Python ctypes 引擎常驻，115 个 state buffer，`accept(pcm)` 边喂边出字
- **AX 运行时不允许同一进程加载第二个 axmodel**（实测 segfault）→ 流式服务单独进程 `stream_server.py`（127.0.0.1:8082），主服务 `web_server.py` 把 `/api/stream/*` 代理过去
- encoder 引擎跨会话共享，每个浏览器会话只分配自己的状态（4 MB）

## 板端接口

```
POST /api/stream/start                      -> {sid, backend, chunk_ms}
POST /api/stream/pcm?sid=&rate=16000        -> {text, audio_ms, encoder_ms, decoder_joiner_ms, roundtrip_ms}
POST /api/stream/end?sid=                   -> {text, final}
GET  /api/stream/health                     -> {loaded, sessions}
```

网页新增「流式实时」标签页：浏览器麦克风采集 → 每 500 ms 送一块 → 每 960 ms 出一次增量文本。

## 已知问题

- CLI 脚本正常退出时偶发 `munmap_chunk(): invalid pointer`（AX 运行时退出清理与 ctypes 缓冲冲突）；常驻服务不受影响，CLI 里用 `os._exit(0)` 规避
- 冻结 `processed_lens=256` 会让左上下文 mask 永远视为「全部有效」，长音频第 2 块起与真实动态略有差异；如需完全一致，可把 mask 张量提升为图输入（CPU 侧计算后喂入）
