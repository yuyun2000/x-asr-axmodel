# CI：在 GitHub Actions 上导出 / 编译 axmodel

这里的脚本是 `sherpa-onnx` PR
<https://github.com/k2-fsa/sherpa-onnx/pull/4185> 里新增文件的副本，用来在本仓库
直接跑 CI（上游那份受 `repository_owner` 限制，只能在 k2-fsa / csukuangfj 账号下跑）。

触发方式：

```bash
gh workflow run export-x-asr-axera.yaml --repo yuyun2000/x-asr-axmodel
gh run watch --repo yuyun2000/x-asr-axmodel
```

流程（`.github/workflows/export-x-asr-axera.yaml`）：

1. 在 icefall CPU 容器里导出静态 fp32 ONNX（10 s 窗口）+ 图改写 + 校准集；
2. 宿主机下载 Pulsar2 8.0 lite 包（830 MB），修掉包里两个启动脚本路径 bug；
3. `pulsar2 build` 编译 encoder（实测 34 min 26 s，峰值内存 9.27 GB，产物 165 MB）；
4. `pulsar2 run` 仿真验证（cosine / MAE）；
5. 上传 `encoder.axmodel` 作为 artifact。

与上游的差异只有脚本路径（`ci/axera/`）和去掉了 `repository_owner` 限制。
