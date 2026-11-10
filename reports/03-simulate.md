# 编译与仿真结果

目标机：`m5stack@192.168.20.38`（16 核 / 62 GB / Ubuntu 24.04），Pulsar2 **8.0**（`e9a1aa15`），
AX650 / NPU1，无开发板 → 精度验证走 `pulsar2 run` 软件仿真。

## 模型规模

| 项 | 值 |
|---|---|
| ONNX 输入 | `x4d [1,1,1000,80]`（10 s 窗口，80 维 fbank） |
| ONNX 输出 | `encoder_out [1,248,512]`（Zipformer2 subsample = 4×） |
| **MACs** | **17,120,005,440（17.12 GMAC / 10 s）** |
| axmodel 体积 | 172.6 MB（168,593.6 KB） |
| 编译耗时 | 07:56 → 08:34，约 **38 分钟**（含 EndToEnd 精度分析） |

## 第 1 次（INT8 / MinMax / 7 组"补静音"校准）

| 指标 | 值 | 判定 |
|---|---|---|
| cosine_similarity | **0.9549** | ❌ 低于验收线 0.99 |
| mae | 0.546 | — |
| max_abs_diff | 8.65 | — |
| 逐帧 cosine | median 0.965 / p10 0.920 / min 0.797 | — |

**更关键的是逐帧 cosine 之外的失效**：拿仿真输出跑贪心解码，**整段输出为空**
（ONNX 输出同一段音频是「昨天是 MONDAY TODAY IS 礼拜二 THE DAY AFTER TOMORROW 是星期三」）。
也就是说 INT8 后的 encoder 输出虽然"看起来还像"，但对 transducer 的 joiner 已经不可用了。

输出统计对比（说明不是量纲问题）：

```
ref  mean 0.1108 std 2.4039 min -7.274 max 8.304
ax   mean 0.1032 std 2.3963 min -6.373 max 6.186
diff mean 0.0077 rms 0.7213      最佳缩放比 0.958
```

## 第 2 次（改量化配置，进行中）

第 1 次的问题定位在两处：

1. **校准集质量差**：7 组样本里 6 组是把 3–9 s 音频补静音到 1000 帧，静音占比 30–70%，
   MinMax 的激活范围被静音帧主导；
2. **MinMax 对 transformer 的离群值敏感**。

对应改动（`21_make_calib.py` + 新配置）：

| 项 | 第 1 次 | 第 2 次 |
|---|---|---|
| 校准样本 | 7 组（多为补静音） | **16 组**，滑窗取满 10 s 语音（speech ratio 0.62–1.00） |
| calibration_method | MinMax | **Percentile** |
| enable_smooth_quant | false | **true**（threshold 0.5 / strength 0.5） |
| highest_mix_precision | false | false（保持纯 INT8，遵守工作区量化口径） |

待第 2 次编译完成后回填本表。
