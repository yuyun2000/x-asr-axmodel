# 板端实机部署与性能（AX650）

日期：2026-11-11 · 设备：AX650 开发板 `192.168.10.144`（Ubuntu 22.04 aarch64 / 8 核 / 3.8 GB）

## 结论

| 项 | 结果 |
|---|---|
| 模型能否在板上跑 | ✅ encoder 通过 `/soc/lib/libax_engine.so` + 自写 ctypes 绑定加载并推理 |
| encoder 延迟 | **80 ms / 10 s 音频** |
| 端到端延迟 | 402–629 ms（4.7–10.05 s 音频） |
| RTF | **0.063–0.086** |
| 服务内存 | ~293 MB RSS |
| 识别质量 | 4 条测试音频中 **2 条与 fp32 ONNX 完全一致**，另 2 条仅英文词/近音差异 |
| 网页服务 | ✅ `https://192.168.10.144:8443/`（上传 + 麦克风）与 `http://192.168.10.144:8081/`（上传） |

## 网页服务

- 页面：M5Stack 设计语言（`deploy/board/static/tokens.css` + `components.css`，
  `templates/index.html`），上传 / 示例音频 / 麦克风三个入口；
- 麦克风走 `AudioContext` + `ScriptProcessor` 采集 16 kHz PCM，每 1.2 s POST 最近 10 s 音频，
  边录边出字；浏览器上下文不是 16 kHz 时服务端 FFT 重采样；
- 桌面 1440 px 与移动端 390 px 均无横向滚动（Chrome headless 实测 `scrollWidth == innerWidth`）。

## 关键实现

板卡上没有 PyPI 的 `axengine`，AX 示例也都是 C/C++。为了不让部署依赖交叉编译，
新增 `deploy/board/ax_engine.py`：用 **ctypes** 直接绑 `libax_engine.so`：

```
AX_SYS_Init → AX_ENGINE_Init(attr) → AX_ENGINE_CreateHandle(model, len)
→ AX_ENGINE_CreateContext → AX_ENGINE_GetIOInfo
→ AX_SYS_MemAlloc / MemAllocCached → AX_ENGINE_RunSync → AX_SYS_MinvalidateCache
```

踩到的两个坑：

1. `AX_ENGINE_CreateHandle` 之前必须调用 **`AX_ENGINE_Init`**，否则返回 `0x80060087`；
2. 模型缓冲区要 **128 字节对齐**（用 `create_string_buffer(len+128)` + 手动对齐）。

## 性能优化（均在板上实测）

| 改动 | 结果 |
|---|---|
| 初始版本（全帧解码 + 单线程 ORT + 自写 numpy fbank） | 10 s 音频 1146 ms（RTF 0.114） |
| 只解码有效帧（`0.30·T + 20` 上界） | 5 s 音频 760→273 ms |
| joiner 的 ORT session 用 4 线程 | 解码 670→400 ms（joiner 2.86→1.40 ms/帧） |
| fbank 换 `kaldi-native-fbank` | 347→145 ms |
| **最终** | 10 s 音频 **629 ms（RTF 0.063）** |

## 精度对照（板端 vs 之前桌面端的 fp32 ONNX 参考）

| wav | 板端 | 与 fp32 的差异 |
|---|---|---|
| 0.wav | 昨天是 MONDAY TODAY IS 礼拜二 THE DAY AFTER TOMORROW 是星期三 | 一致 |
| 1.wav | 这是第一种第二种叫呃与 ALWAYS OS 什么意思啊 | `ALWAYS ALWAYS`→`ALWAYS OS` |
| 2.wav | 就是频繁的啊不认识记下来 FREQUENTLY 频繁的 | 一致 |
| 3.wav | 第一句是个什么时态加了 E S 是一般现在时我们把它时态写上 | `YES`→`E S`、`现在时`→`现在时` 等 2 处 |

板端结果比 `reports/03-simulate.md` 的仿真结果**更好**：板端用 `kaldi-native-fbank`，
与 sherpa-onnx 部署口径一致；仿真是用自写 numpy fbank 算的特征。

## 复现

见 `deploy/board/README.md`（含 systemd 自启、HTTP API、常见问题）。
