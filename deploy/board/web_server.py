#!/usr/bin/env python3
"""X-ASR on AX650 — 板端网页推理服务。

功能：
  1. 上传音频文件（wav/mp3/m4a/webm/ogg，服务端用 ffmpeg 转成 16 kHz 单声道 PCM）
  2. 浏览器麦克风实时识别（PCM 流式发送，边录边出字）
  3. 同一个模型实例被加锁串行调用，多客户端安全

启动：
  python3 web_server.py --models-dir /root/x-asr/models --port 8443            # HTTPS（麦克风需要）
  python3 web_server.py --models-dir /root/x-asr/models --port 8080 --no-ssl   # 只要上传

麦克风需要"安全上下文"：http://<局域网IP> 下浏览器会禁用 getUserMedia，
所以默认用自签名证书跑 HTTPS（浏览器会提示不安全，点"继续访问"即可）。
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from asr_engine import Asr, read_wav

app = Flask(__name__)
ASR: Asr | None = None
SAMPLES_DIR = "/root/x-asr/test_wavs"
LOCK = threading.Lock()
STATS = {"requests": 0, "audio_ms": 0.0, "infer_ms": 0.0}
START_TIME = time.time()

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>X-ASR · AX650 NPU</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%8E%99%EF%B8%8F%3C/text%3E%3C/svg%3E">
<style>
  :root { --bg:#f6f7f9; --card:#fff; --fg:#1c1f23; --muted:#6b7280; --accent:#1f6feb;
          --ok:#0f9d58; --warn:#b45309; --line:#e5e7eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:20px 24px 12px; }
  h1 { margin:0; font-size:20px; font-weight:650; }
  .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  .wrap { max-width:900px; margin:0 auto; padding:0 16px 40px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:18px; margin-top:16px; }
  .tabs { display:flex; gap:8px; margin-top:16px; }
  .tab { flex:1; text-align:center; padding:10px; border:1px solid var(--line); border-radius:10px;
         background:#fff; cursor:pointer; font-size:14px; }
  .tab.active { border-color:var(--accent); color:var(--accent); font-weight:600;
                box-shadow:0 0 0 3px rgba(31,111,235,.08); }
  .pane { display:none; } .pane.active { display:block; }
  .drop { border:2px dashed var(--line); border-radius:12px; padding:28px; text-align:center;
          color:var(--muted); cursor:pointer; }
  .drop.hover { border-color:var(--accent); color:var(--accent); }
  button { font-size:14px; padding:9px 16px; border-radius:9px; border:1px solid var(--line);
           background:#fff; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button.rec { background:#dc2626; color:#fff; border-color:#dc2626; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .out { margin-top:14px; padding:14px; border-radius:10px; background:#f8fafc;
         border:1px solid var(--line); min-height:64px; white-space:pre-wrap; line-height:1.7;
         font-size:16px; }
  .meta { margin-top:10px; font-size:12.5px; color:var(--muted); font-family:ui-monospace,Menlo,monospace;
          white-space:pre-wrap; }
  .pill { display:inline-block; font-size:12px; padding:2px 8px; border-radius:999px;
          background:#eef2ff; color:#3730a3; margin-right:6px; }
  .rec-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#dc2626;
             margin-right:6px; animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:.25; } }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .hint { font-size:12.5px; color:var(--muted); margin-top:8px; }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:24px 0 40px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>X-ASR · AX650 NPU 语音识别</h1>
    <div class="sub">
      <span class="pill" id="dev">AX650</span>
      <span class="pill">encoder 10 s 窗口</span>
      <span class="pill" id="stat">–</span>
    </div>
  </header>

  <div class="tabs">
    <div class="tab active" data-tab="file">上传音频</div>
    <div class="tab" data-tab="mic">麦克风实时</div>
  </div>

  <div class="card pane active" id="pane-file">
    <div class="drop" id="drop">点击或拖拽音频文件到此处（wav / mp3 / m4a / webm / ogg）</div>
    <input type="file" id="file" accept="audio/*" style="display:none">
    <div class="hint">10 s 以内的音频一次推理；更长的音频按 8 s 步长切窗。也可以直接点下面的示例：</div>
    <div class="row" id="samples" style="margin-top:8px"></div>
    <div class="out" id="file-out">（等待上传）</div>
    <div class="meta" id="file-meta"></div>
  </div>

  <div class="card pane" id="pane-mic">
    <div class="row">
      <button class="primary" id="rec">开始录音</button>
      <button id="stop" disabled>停止</button>
      <span id="rec-state" class="hint"></span>
    </div>
    <div class="hint">边录边识别：每 1.2 s 发送一次累积音频，后面显示的是最新结果。</div>
    <div class="out" id="mic-out">（等待录音）</div>
    <div class="meta" id="mic-meta"></div>
  </div>
</div>

<footer>sherpa-onnx 生态 · X-ASR Zipformer2 transducer · AX650 INT8</footer>

<script>
const $ = s => document.querySelector(s);
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('#pane-' + t.dataset.tab).classList.add('active');
});

// ---------------- 上传 ----------------
const drop = $('#drop'), fileInput = $('#file');
drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = () => drop.classList.remove('hover');
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
};
fileInput.onchange = () => { if (fileInput.files.length) upload(fileInput.files[0]); };

async function upload(f) {
  $('#file-out').textContent = '识别中…';
  $('#file-meta').textContent = '';
  const fd = new FormData();
  fd.append('audio', f);
  try {
    const r = await fetch('/api/asr', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) { $('#file-out').textContent = '失败：' + j.error; return; }
    $('#file-out').textContent = j.text || '（无识别结果）';
    $('#file-meta').textContent =
      `文件 ${f.name} · 音频 ${(j.audio_ms/1000).toFixed(2)}s · 推理 ${j.total_ms.toFixed(0)}ms ` +
      `(encoder ${j.encoder_ms.toFixed(0)}ms, 解码 ${j.decoder_joiner_ms.toFixed(0)}ms) · RTF ${j.rtf}`;
    refreshStat();
  } catch (e) { $('#file-out').textContent = '请求失败：' + e; }
}

// ---------------- 麦克风 ----------------
let ctx, stream, processor, source, chunks = [], recording = false, timer = null, lastRate = 16000;

$('#rec').onclick = startRec;
$('#stop').onclick = stopRec;

async function startRec() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
  } catch (e) {
    $('#rec-state').textContent = '拿不到麦克风：' + e.message +
      '（http 下浏览器会禁用麦克风，请用 https 打开）';
    return;
  }
  ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  lastRate = ctx.sampleRate;
  source = ctx.createMediaStreamSource(stream);
  processor = ctx.createScriptProcessor(4096, 1, 1);
  chunks = [];
  processor.onaudioprocess = e => {
    if (!recording) return;
    const d = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(d));   // 攒原始 PCM
  };
  source.connect(processor);
  processor.connect(ctx.destination);

  recording = true;
  $('#rec').disabled = true; $('#stop').disabled = false;
  $('#rec-state').innerHTML = '<span class="rec-dot"></span>录音中';
  $('#mic-out').textContent = '';
  timer = setInterval(sendChunk, 1200);
  await ctx.resume();
}

function concatChunks(maxSec = 10) {
  const total = chunks.reduce((a, c) => a + c.length, 0);
  const take = Math.min(total, Math.round(maxSec * 16000));
  const out = new Float32Array(take);
  let off = total - take;              // 只保留最近的 maxSec 秒
  let w = 0;
  for (const c of chunks) {
    if (off >= c.length) { off -= c.length; continue; }
    const piece = c.subarray(off);
    off = 0;
    out.set(piece.subarray(0, take - w), w);
    w += piece.length;
    if (w >= take) break;
  }
  return out;
}

function toPCM16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    let s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

async function sendChunk() {
  if (!recording || chunks.length === 0) return;
  const pcm = toPCM16(concatChunks(10));
  const t0 = performance.now();
  try {
    const r = await fetch('/api/asr_pcm?rate=' + (ctx ? ctx.sampleRate : 16000), {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
      body: pcm.buffer
    });
    const j = await r.json();
    if (j.error) { $('#mic-meta').textContent = '失败：' + j.error; return; }
    $('#mic-out').textContent = j.text || '…';
    $('#mic-meta').textContent =
      `累积 ${(j.audio_ms/1000).toFixed(1)}s · 推理 ${j.total_ms.toFixed(0)}ms · ` +
      `往返 ${(performance.now()-t0).toFixed(0)}ms · RTF ${j.rtf}`;
    refreshStat();
  } catch (e) { $('#mic-meta').textContent = '请求失败：' + e; }
}

function stopRec() {
  recording = false;
  clearInterval(timer);
  if (processor) { source.disconnect(); processor.disconnect(); }
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (ctx) ctx.close();
  $('#rec').disabled = false; $('#stop').disabled = true;
  $('#rec-state').textContent = '已停止';
  sendFinal();
}

async function sendFinal() {
  if (chunks.length === 0) { $('#mic-out').textContent = '（没有录到声音）'; return; }
  const pcm = toPCM16(concatChunks(10));
  try {
    const r = await fetch('/api/asr_pcm?rate=' + (lastRate || 16000), {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
      body: pcm.buffer
    });
    const j = await r.json();
    if (!j.error) {
      $('#mic-out').textContent = j.text || '（无识别结果）';
      $('#mic-meta').textContent =
        `最终 ${(j.audio_ms/1000).toFixed(1)}s · 推理 ${j.total_ms.toFixed(0)}ms · RTF ${j.rtf}`;
      refreshStat();
    }
  } catch (e) { /* ignore */ }
}

async function loadSamples() {
  try {
    const list = await (await fetch('/api/samples')).json();
    const box = document.querySelector('#samples');
    box.innerHTML = '';
    for (const name of list) {
      const b = document.createElement('button');
      b.textContent = '▶ ' + name;
      b.onclick = async () => {
        $('#file-out').textContent = '识别中…';
        const r = await fetch('/api/asr_sample', {method:'POST',
          headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
        const j = await r.json();
        if (j.error) { $('#file-out').textContent = '失败：' + j.error; return; }
        $('#file-out').textContent = j.text;
        $('#file-meta').textContent =
          `示例 ${name} · 音频 ${(j.audio_ms/1000).toFixed(2)}s · 推理 ${j.total_ms.toFixed(0)}ms · RTF ${j.rtf}`;
        refreshStat();
      };
      box.appendChild(b);
    }
  } catch (e) { /* ignore */ }
}
loadSamples();

async function refreshStat() {
  try {
    const j = await (await fetch('/api/health')).json();
    $('#dev').textContent = j.device;
    $('#stat').textContent = `请求 ${j.requests} · 累计音频 ${(j.audio_ms/1000).toFixed(1)}s`;
  } catch (e) { /* ignore */ }
}
refreshStat();
</script>
</body>
</html>
"""


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
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "device": "AX650 NPU",
        "model": ASR.io,
        "uptime_s": round(time.time() - START_TIME, 1),
        **STATS,
    })


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
    # HTTPS：麦克风（getUserMedia）需要安全上下文
    srv = make_server("0.0.0.0", args.port, app, threaded=True, ssl_context=ssl_ctx)
    servers.append((f"https://0.0.0.0:{args.port}", threading.Thread(target=srv.serve_forever, daemon=True)))
    # 可选：并发的 HTTP 端口，方便 curl / 只上传的场景
    if args.http_port:
        srv2 = make_server("0.0.0.0", args.http_port, app, threaded=True)
        servers.append((f"http://0.0.0.0:{args.http_port}", threading.Thread(target=srv2.serve_forever, daemon=True)))

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
