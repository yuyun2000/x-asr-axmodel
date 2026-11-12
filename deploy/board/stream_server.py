"""流式识别独立服务（单独进程）。

为什么要单独进程：AX 运行时在**同一个进程里加载第二个 axmodel 会段错误**（实测），
所以离线服务和流式服务分开跑，主服务（web_server.py）用 HTTP 把 /api/stream/* 转发过来。

端口只监听 127.0.0.1，不对外暴露。
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request

from stream_asr import StreamingAsr

app = Flask(__name__)
ENGINE: StreamingAsr | None = None
SESSIONS: dict[str, StreamingAsr] = {}
LOCK = threading.Lock()
MODELS = "/root/x-asr/models/streaming"   # 可在 main() 里用 --models-dir 覆盖


def resample_to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000 or pcm.size == 0:
        return pcm
    n_out = int(round(len(pcm) * 16000 / rate))
    spec = np.fft.rfft(pcm)
    spec = spec[: n_out // 2 + 1]
    return (np.fft.irfft(spec, n=n_out) * (n_out / len(pcm))).astype(np.float32)


def engine() -> StreamingAsr:
    global ENGINE
    if ENGINE is None:
        with LOCK:
            if ENGINE is None:
                ENGINE = StreamingAsr(MODELS)
    return ENGINE


@app.get("/health")
def health():
    return jsonify({"ok": True, "loaded": ENGINE is not None, "sessions": len(SESSIONS)})


@app.post("/start")
def start():
    try:
        base = engine()
    except Exception as e:  # pragma: no cover
        return jsonify({"error": f"streaming model unavailable: {e}"}), 500
    sid = uuid.uuid4().hex[:12]
    with LOCK:
        SESSIONS[sid] = StreamingAsr(MODELS, enc=base.enc)
    return jsonify({"sid": sid, "backend": base.backend, "chunk_ms": 960})


@app.post("/pcm")
def pcm():
    sid = request.args.get("sid", "")
    eng = SESSIONS.get(sid)
    if eng is None:
        return jsonify({"error": "unknown sid"}), 400
    raw = request.get_data(cache=False)
    arr = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    rate = int(request.args.get("rate", 16000))
    arr = resample_to_16k(arr, rate)
    t0 = time.perf_counter()
    with LOCK:
        text = eng.accept(arr)
    st = eng.stats()
    st["text"] = text
    st["roundtrip_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    st["sid"] = sid
    return jsonify(st)


@app.post("/end")
def end():
    sid = request.args.get("sid", "")
    eng = SESSIONS.pop(sid, None)
    if eng is None:
        return jsonify({"error": "unknown sid"}), 400
    with LOCK:
        text = eng.finish()
    st = eng.stats()
    st["text"] = text
    st["sid"] = sid
    st["final"] = True
    return jsonify(st)


def main() -> None:
    global MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=MODELS)
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()

    MODELS = args.models_dir
    engine()
    print(f"[x-asr-stream] ready on 127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
