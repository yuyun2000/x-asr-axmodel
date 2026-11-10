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

## 第 2/3 次尝试（都没跑完）

| 尝试 | 配置 | 结果 |
|---|---|---|
| v2 | 16 组满窗校准 + **Percentile** + **SmoothQuant** | 编译 79 分钟仍在 compiler 阶段，判定异常（v1 同样规模只需 30 分钟），终止 |
| v3 | 16 组满窗校准 + **Percentile**（去掉 SmoothQuant） | 编译 63 分钟仍在 compiler 阶段，同样异常，终止 |

结论：在这张 17.12 GMAC 的图上，**Percentile 校准方法 + SmoothQuant 会让 Pulsar2 的
compiler 阶段耗时失控**（MinMax 是 30 分钟量级，Percentile 63 分钟未完成）。
时间成本上不可接受，暂时回到 MinMax。

## 待验证的改进方向（优先级从高到低）

第 1 次的问题定位在两处：

1. **校准集质量差**：7 组样本里 6 组是把 3–9 s 音频补静音到 1000 帧，静音占比 30–70%，
   MinMax 的激活范围被静音帧主导；
2. **MinMax 对 transformer 的离群值敏感**。

对应改动（`21_make_calib.py` + 新配置）：

| 方向 | 说明 | 成本 |
|---|---|---|
| ① 只换校准集（MinMax + 16 组满窗） | 最接近 v1，编译时间可控；静音帧污染是 v1 的已知缺陷 | 编译 ~40 min |
| ② 混精（`highest_mix_precision: true` 或对敏感层用 `layer_configs.data_type=FP32`） | Pulsar2 的标准精度补救；但违反「默认 INT8、highest_mix_precision=false」的口径，**需要你点头** | 编译 ~40 min，模型变大、变慢 |
| ③ 更多真实业务音频做校准（32 组以上） | 语音分布更贴近线上（方言、噪声、远场） | 需要素材 |
| ④ QAT / AdaRound / LSQ | Pulsar2 支持 `enable_adaround` / `enable_lsq`，代价是编译时间与调参 | 小时级 |
| ⑤ 换更小的子模型 | 例如只转前 3 个 encoder stack 验证通路（精度不可用，仅验证编译链路） | 30 min |

第 1 次的结论已经能确定一件事：**「能编译、能跑仿真」是通的，「INT8 精度达标」还没通**。
下一步建议先走 ①（低成本验证校准集的影响），不行再讨论 ②。
