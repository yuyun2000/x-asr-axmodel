#!/usr/bin/env python3
"""X-ASR on AX650 — 板端网页推理服务。

功能：
  1. 上传音频文件（wav/mp3/m4a/webm/ogg，服务端用 ffmpeg 转成 16 kHz 单声道 PCM）
  2. 浏览器麦克风实时识别（PCM 流式发送，边录边出字）
  3. 同一个模型实例被加锁串行调用，多客户端安全

页面用 M5Stack 设计语言（tokens.css + components.css 放在 static/ 下，模板在 templates/）。

启动：
  python3 web_server.py --models-dir /root/x-asr/models --port 8443            # HTTPS（麦克风需要）
  python3 web_server.py --models-dir /root/x-asr/models --port 8080 --no-ssl   # 只要上传

麦克风需要"安全上下文"：http://<局域网IP> 下浏览器会禁用 getUserMedia，
所以默认用自签名证书跑 HTTPS（浏览器会提示不安全，点"继续访问"即可）。
"""

from __future__ import annotations

import argparse
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from asr_engine import Asr, read_wav

app = Flask(__name__)
ASR: Asr | None = None
SAMPLES_DIR = "/root/x-asr/test_wavs"
LOCK = threading.Lock()
STATS = {"requests": 0, "audio_ms": 0.0, "infer_ms": 0.0}
START_TIME = time.time()


def pcm_from_any(path: Path) -> np.ndarray:
    """把任意音频文件转成 16 kHz 单声道 float32 PCM（用板上的 ffmpeg）。"""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace")[:300])
    raw = np.frombuffer(out.stdout, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def resample_to_16k(pcm: np.ndarray, src_rate: int) -> np.ndarray:
    """FFT 重采样到 16 kHz（浏览器 AudioContext 不保证是 16 kHz）。"""
    if src_rate == 16000 or len(pcm) == 0:
        return pcm
    n_out = int(round(len(pcm) * 16000 / src_rate))
    spec = np.fft.rfft(pcm)
    spec = spec[: n_out // 2 + 1]
    out = np.fft.irfft(spec, n=n_out) * (n_out / len(pcm))
    return out.astype(np.float32)


def _run(pcm: np.ndarray) -> dict:
    t0 = time.perf_counter()
    with LOCK:
        r = ASR.transcribe(pcm)
    r["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    STATS["requests"] += 1
    STATS["audio_ms"] += r["audio_ms"]
    STATS["infer_ms"] += r["total_ms"]
    return r


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "device": "AX650 NPU",
        "model": ASR.io,
        "uptime_s": round(time.time() - START_TIME, 1),
        **STATS,
    })


@app.get("/api/samples")
def api_samples():
    d = Path(SAMPLES_DIR)
    return jsonify(sorted(p.name for p in d.glob("*.wav")) if d.exists() else [])


@app.post("/api/asr_sample")
def api_asr_sample():
    name = (request.json or {}).get("name", "")
    path = Path(SAMPLES_DIR) / Path(name).name
    if not path.is_file():
        return jsonify({"error": f"no such sample: {name}"}), 400
    return jsonify(_run(read_wav(path)))


@app.post("/api/asr")
def api_asr():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "no audio file"}), 400
    suffix = Path(f.filename or "a.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        path = Path(tmp.name)
    try:
        if suffix == ".wav":
            try:
                pcm = read_wav(path)
            except Exception:
                pcm = pcm_from_any(path)
        else:
            pcm = pcm_from_any(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        path.unlink(missing_ok=True)
    if len(pcm) == 0:
        return jsonify({"error": "empty audio"}), 400
    return jsonify(_run(pcm))


@app.post("/api/asr_pcm")
def api_asr_pcm():
    raw = request.get_data(cache=False)
    if len(raw) < 2:
        return jsonify({"error": "empty pcm"}), 400
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    rate = int(request.args.get("rate", 16000))
    pcm = resample_to_16k(pcm, rate)
    return jsonify(_run(pcm))


def make_cert(cert_dir: Path) -> tuple[str, str]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    key, crt = cert_dir / "key.pem", cert_dir / "cert.pem"
    if not (key.exists() and crt.exists()):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(key), "-out", str(crt), "-days", "825",
                        "-subj", "/CN=x-asr-ax650"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(crt), str(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/root/x-asr/models")
    ap.add_argument("--samples-dir", default="/root/x-asr/test_wavs")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--cert-dir", default="/root/x-asr/certs")
    ap.add_argument("--http-port", type=int, default=8081,
                    help="同时开一个 HTTP 端口（0 = 关闭）")
    args = ap.parse_args()

    global ASR, SAMPLES_DIR
    SAMPLES_DIR = args.samples_dir
    ASR = Asr(args.models_dir)
    print(f"[x-asr] model loaded: frames={ASR.frames} out={ASR.out_shape}", flush=True)

    ssl_ctx = None
    if not args.no_ssl:
        crt, key = make_cert(Path(args.cert_dir))
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(crt, key)

    from werkzeug.serving import make_server

    servers = []
    srv = make_server("0.0.0.0", args.port, app, threaded=True, ssl_context=ssl_ctx)
    servers.append((f"https://0.0.0.0:{args.port}", threading.Thread(
        target=srv.serve_forever, daemon=True)))
    if args.http_port:
        srv2 = make_server("0.0.0.0", args.http_port, app, threaded=True)
        servers.append((f"http://0.0.0.0:{args.http_port}", threading.Thread(
            target=srv2.serve_forever, daemon=True)))

    for addr, t in servers:
        t.start()
        print(f"[x-asr] serving on {addr}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
