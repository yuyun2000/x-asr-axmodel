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
CA_DIR = "/root/x-asr/certs"


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


@app.get("/setup")
def setup_page():
    host = request.host.split(":")[0]
    return render_template("setup.html", host=host)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "device": "AX8850 NPU",
        "model": ASR.io,
        "aux_backend": ASR.aux_backend,
        "uptime_s": round(time.time() - START_TIME, 1),
        **STATS,
    })


@app.get("/asr-ca.crt")
def ca_cert():
    from flask import send_file
    return send_file(str(Path(CA_DIR) / "ca.crt"), mimetype="application/x-x509-ca-cert",
                     as_attachment=True, download_name="x-asr-local-ca.crt")


@app.get("/api/sample_audio/<name>")
def sample_audio(name):
    from flask import send_file
    path = Path(SAMPLES_DIR) / Path(name).name
    if not path.is_file():
        return jsonify({"error": f"no such sample: {name}"}), 404
    return send_file(str(path), mimetype="audio/wav", conditional=True)


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


def make_cert(cert_dir: Path, ip: str) -> tuple[str, str]:
    """生成自签 CA + 带 SAN 的服务器证书。

    浏览器对没有 SAN 的旧式自签证书只给报错页；带 SAN（IP/域名）之后会走标准的
    "高级 → 继续前往" 流程。把 CA 证书下载安装到系统后则完全无警告。
    """
    import subprocess

    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_key, ca_crt = cert_dir / "ca.key", cert_dir / "ca.crt"
    srv_key, srv_crt = cert_dir / "key.pem", cert_dir / "cert.pem"
    ext = cert_dir / "san.cnf"

    if ca_crt.exists() and srv_crt.exists():
        return str(srv_crt), str(srv_key)

    def run(cmd):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(ca_key), "-out", str(ca_crt), "-days", "3650",
         "-subj", "/CN=X-ASR Local CA", "-addext",
         "basicConstraints=critical,CA:TRUE",
         "-addext", "keyUsage=critical,keyCertSign,cRLSign"])

    ext.write_text(
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost,DNS:ax8850.local,DNS:x-asr.local\n")

    run(["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(srv_key), "-out", str(cert_dir / "server.csr"),
         "-subj", "/CN=AX8850 X-ASR"])
    run(["openssl", "x509", "-req", "-in", str(cert_dir / "server.csr"),
         "-CA", str(ca_crt), "-CAkey", str(ca_key), "-CAcreateserial",
         "-out", str(srv_crt), "-days", "825", "-sha256", "-extfile", str(ext)])
    return str(srv_crt), str(srv_key)


def local_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/root/x-asr/models")
    ap.add_argument("--samples-dir", default="/root/x-asr/test_wavs")
    ap.add_argument("--https-ports", default="443,8443",
                    help="HTTPS 端口列表，默认 443（不带端口即可访问）和 8443（兼容旧链接）")
    ap.add_argument("--http-ports", default="80,8081",
                    help="HTTP 端口列表（上传用；浏览器在 http 下禁用麦克风）")
    ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--cert-dir", default="/root/x-asr/certs")
    ap.add_argument("--ip", default="", help="证书 SAN 里的 IP，默认自动探测")
    args = ap.parse_args()

    global ASR, SAMPLES_DIR
    SAMPLES_DIR = args.samples_dir
    ASR = Asr(args.models_dir)
    print(f"[x-asr] model loaded: frames={ASR.frames} out={ASR.out_shape}", flush=True)

    ssl_ctx = None
    if not args.no_ssl:
        global CA_DIR
        CA_DIR = args.cert_dir
        ip = args.ip or local_ip()
        crt, key = make_cert(Path(args.cert_dir), ip)
        print(f"[x-asr] https certificate for IP {ip}, CA at /asr-ca.crt", flush=True)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(crt, key)

    from werkzeug.serving import make_server

    def parse(spec: str) -> list[int]:
        return [int(x) for x in spec.replace(" ", "").split(",") if x]

    servers = []
    for port in ([] if args.no_ssl else parse(args.https_ports)):
        try:
            srv = make_server("0.0.0.0", port, app, threaded=True, ssl_context=ssl_ctx)
        except OSError as e:
            print(f"[x-asr] skip https :{port} ({e})", flush=True)
            continue
        servers.append((f"https://0.0.0.0:{port}", threading.Thread(
            target=srv.serve_forever, daemon=True)))
    for port in parse(args.http_ports):
        try:
            srv = make_server("0.0.0.0", port, app, threaded=True)
        except OSError as e:
            print(f"[x-asr] skip http :{port} ({e})", flush=True)
            continue
        servers.append((f"http://0.0.0.0:{port}", threading.Thread(
            target=srv.serve_forever, daemon=True)))

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
