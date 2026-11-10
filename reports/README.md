# 报告

| 文件 | 内容 |
|---|---|
| `01-export.md` | fp32 静态 ONNX 导出（导出文本与 sherpa int8 一致） |
| `02-compile.md` | Pulsar2 编译（模型大小、MACs、耗时） |
| `03-simulate.md` | 仿真对分（cosine / MAE / max_abs_diff） |
| `04-text-consistency.md` | ONNX vs AXMODEL 仿真输出的字级一致性 |
| `05-pitfalls.md` | 踩坑记录（If 节点、形状冻结、toolchain 卡住等） |

原始产物在目标机 `~/work/x-asr-axmodel/task/`：

- `task/compile/model.axmodel`、`task/compile/pulsar2_config.json`、`task/compile/compile.log`
- `task/simulate/simulate_report.md`、`task/simulate/metrics.json`、`task/simulate/pulsar2_run.log`
