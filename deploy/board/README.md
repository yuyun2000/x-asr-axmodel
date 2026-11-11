# AX8850 板端部署（实测通过）

在 **AX8850 开发板（兼容 AX650）**（Ubuntu 22.04 aarch64，8 核，3.8 GB 内存）上跑 X-ASR encoder 的
`axmodel`，并提供一个局域网页：上传音频 / 麦克风实时识别。

## 板端环境（实测）

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04 LTS，kernel 5.15.73，aarch64 |
| CPU / 内存 | 8 核 / 3.8 GB |
| Python | 3.10.12（系统自带） |
| AX 运行时 | `/soc/lib/libax_engine.so` + `/soc/include/ax_engine_api.h`（板卡 SDK 自带） |
| 额外依赖 | `numpy`、`flask`（已有）、`onnxruntime`（pip 装）、`kaldi-native-fbank`（pip 装）、`ffmpeg`（已有） |

```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime kaldi-native-fbank
```

> 板卡上没有 PyPI 的 `axengine` 包，所以本目录的 `ax_engine.py` 用 **ctypes 直接绑
> `libax_engine.so`**（`AX_SYS_Init` → `AX_ENGINE_Init` → `CreateHandle` → `CreateContext`
> → `GetIOInfo` → `RunSync`），不引入额外依赖，也不需要交叉编译。

## 部署步骤

```bash
# 0. 端口：443（HTTPS）+ 80（HTTP）+ 8443/8081（兼容旧链接），均为 0.0.0.0
#    如果 80/443 被别的服务占用，服务会跳过并在日志里打印 "skip http :80"

# 1. 传模型（约 190 MB）
mkdir -p /root/x-asr/models /root/x-asr/test_wavs
cd /root/x-asr/models
#   encoder.axmodel / decoder.onnx / joiner.onnx / tokens.txt
#   从 HuggingFace 下载：huggingface-cli download yunyu1258/x-asr-axmodel --local-dir .
#   或者从开发机 scp / curl 过来

# 2. 传本目录脚本（web 页面需要 templates/ 与 static/ 一起传）
#   ax_engine.py  asr_engine.py  web_server.py  test_board.py
#   templates/index.html
#   static/tokens.css  static/components.css

# 目录结构（部署后）
#   /root/x-asr/{ax_engine.py,asr_engine.py,web_server.py,templates/,static/,models/,test_wavs/}

# 3. 先跑命令行自检
cd /root/x-asr && python3 test_board.py

# 4. 起网页服务（HTTPS 8443 + HTTP 8081）
python3 web_server.py --models-dir /root/x-asr/models --port 8443 --http-port 8081
```

浏览器打开（局域网任意设备）：

| 入口 | 说明 |
|---|---|
| **https://192.168.10.144/** | 完整功能（上传 + 示例试听 + 麦克风），默认监听 443（同时保留 8443） |
| **http://192.168.10.144/** | 上传 / 试听（`http` 下浏览器禁用麦克风），监听 80（同时保留 8081） |
| **http://192.168.10.144/setup** | HTTPS 证书安装说明：下载本地 CA，按系统装一次就无警告 |

首次打开 HTTPS 会提示"您的连接不是私密连接"，点 **高级 → 继续前往** 即可使用（麦克风可用）；
装过 CA 之后不再提示。证书由板端首次启动时用 openssl 生成的本地 CA 签出，
SAN 含 `IP:192.168.10.144`、`DNS:localhost`、`DNS:ax8850.local`，有效期 825 天。

页面用 **M5Stack 官网设计语言**（`static/tokens.css` + `static/components.css`，与前缀
`m5-` 的组件类），模板在 `templates/index.html`：

| 上传 | 麦克风 |
|---|---|
| ![上传](../docs/images/web-upload.png) | ![麦克风](../docs/images/web-mic.png) |

移动端（390 px）自适应：![移动端](../docs/images/web-mobile.png)

### 开机自启（可选）

```bash
cp x-asr.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now x-asr
systemctl status x-asr --no-pager
```

## 实测性能（板端，10.05 s 音频）

| 项 | 值 |
|---|---|
| encoder（NPU，10 s 窗口） | **80 ms** |
| decoder + joiner（CPU，4 线程） | 250–400 ms |
| fbank（kaldi-native-fbank） | 70–145 ms |
| **端到端** | **629 ms**（10.05 s）/ 402 ms（4.7 s）/ 598 ms（8.8 s） |
| **RTF** | **0.063 – 0.086** |
| 服务进程内存 | ~293 MB RSS |
| 编码器体积 | 172 MB（INT8） |

优化过程（都是实测）：

| 改动 | 效果 |
|---|---|
| 只用有效帧解码（`valid_encoder_frames`，0.30·T+20 上界） | 10 s 音频不变，5 s 音频 760→273 ms |
| joiner ONNX 用 4 线程（`intra_op_num_threads=4`） | 解码 670→400 ms（joiner 2.86→1.40 ms/帧） |
| fbank 换 `kaldi-native-fbank` | 347→145 ms（纯 numpy 实现的 1/2.4） |

## 识别结果（板端 vs fp32 ONNX）

| wav | 板端输出 | 与 fp32 ONNX |
|---|---|---|
| 0.wav | 昨天是 MONDAY TODAY IS 礼拜二 THE DAY AFTER TOMORROW 是星期三 | **完全一致** |
| 1.wav | 这是第一种第二种叫呃与 ALWAYS OS 什么意思啊 | 1 处（`ALWAYS ALWAYS`→`ALWAYS OS`） |
| 2.wav | 就是频繁的啊不认识记下来 FREQUENTLY 频繁的 | **完全一致** |
| 3.wav | 第一句是个什么时态加了 E S 是一般现在时我们把它时态写上 | 2 处（`YES`→`E S` 等） |

**注意**：板端跑出来比仿真（`reports/03-simulate.md`）还要好一些 —— 因为板端用的是
`kaldi-native-fbank`（与 sherpa-onnx 部署口径一致），仿真时用的是自写 numpy fbank。

## HTTP API

| 接口 | 说明 |
|---|---|
| `GET /` | 网页 |
| `GET /api/health` | 设备、模型 IO、累计请求/音频量 |
| `POST /api/asr` | multipart `audio=@file.wav`（支持 wav/mp3/m4a/webm/ogg，服务端 ffmpeg 转 PCM） |
| `POST /api/asr_pcm?rate=16000` | raw PCM int16 小端，麦克风流式用；`rate != 16000` 时服务端 FFT 重采样 |
| `GET /api/samples` | 板端自带测试音频列表 |
| `GET /api/sample_audio/<name>` | 示例音频流（网页里的播放器直接用这个） |
| `GET /asr-ca.crt` | 下载本地 CA 证书 |
| `GET /setup` | 证书安装说明页 |
| `POST /api/asr_sample` | `{"name": "0.wav"}` 直接跑自带音频 |

```bash
curl -F "audio=@test.wav" http://192.168.10.144/api/asr
curl -X POST -H 'Content-Type: application/json' \
     -d '{"name":"0.wav"}' http://192.168.10.144/api/asr_sample
```

## 已知限制

- 输入固定 10 s 窗口；超过 10 s 的音频按 8 s 步长切窗（网页端麦克风只保留最近 10 s）；
- 无标点输出（模型本身是无标点版本）；
- 麦克风必须走 HTTPS（浏览器安全上下文限制），自签名证书需要手动信任；
- 并发：模型实例加锁串行，多个浏览器同时请求会排队（单次 ~0.6 s）。
