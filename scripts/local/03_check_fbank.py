#!/usr/bin/env python3
"""验证 numpy fbank 与 sherpa-onnx 内部 fbank 数值等价。

做法：用本仓库的 fbank 算出特征 → 直接跑 sherpa-onnx 发布的 X-ASR int8 ONNX
（encoder + decoder + joiner 贪心解码）→ 与 sherpa-onnx Python API 的识别结果对比。

只要两边的输出文本一致，就说明特征足够接近（fbank 的几个开关：snip_edges、
preemphasis、mel 下限/上限都已被这条文本链路覆盖）。

用法：
    python 03_check_fbank.py <model_dir> <wav>
示例：
    python 03_check_fbank.py /tmp/asrtest/sherpa-onnx-x-asr-zipformer-transducer-zh-en-int8-2026-06-03 \
        /tmp/asrtest/sherpa-onnx-x-asr-zipformer-transducer-zh-en-int8-2026-06-03/test_wavs/0.wav
"""

from __future__ import annotations

import glob
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from fbank_kaldi import fbank80  # noqa: E402

BLANK_ID = 0
CONTEXT_SIZE = 2


def read_wav(path: str) -> tuple[np.ndarray, int]:
    w = wave.open(path, "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n)
    w.close()
    assert sw == 2, "expect 16-bit PCM"
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        a = a[::2]
    if sr != 16000:
        raise SystemExit(f"expect 16 kHz, got {sr}")
    return a, sr


def greedy_decode(model_dir: Path, feats: np.ndarray):
    sess_enc = ort.InferenceSession(str(next(model_dir.glob("encoder*.int8.onnx"))),
                                    providers=["CPUExecutionProvider"])
    sess_dec = ort.InferenceSession(str(next(model_dir.glob("decoder*.onnx"))),
                                    providers=["CPUExecutionProvider"])
    sess_join = ort.InferenceSession(str(next(model_dir.glob("joiner*.int8.onnx"))),
                                     providers=["CPUExecutionProvider"])

    x = feats[None, ...].astype(np.float32)
    enc_inputs = sess_enc.get_inputs()
    feeds = {enc_inputs[0].name: x}
    if len(enc_inputs) > 1:
        lens_dtype = np.int32 if enc_inputs[1].type == "tensor(int32)" else np.int64
        feeds[enc_inputs[1].name] = np.array([feats.shape[0]], dtype=lens_dtype)
    encoder_out = sess_enc.run(None, feeds)[0]           # [1, T//2, 512]

    tokens = [l.split()[0] for l in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines() if l]
    y = np.array([[BLANK_ID] * CONTEXT_SIZE], dtype=np.int64)
    dec_out = sess_dec.run(None, {sess_dec.get_inputs()[0].name: y})[0]
    enc_name = sess_join.get_inputs()[0].name
    dec_name = sess_join.get_inputs()[1].name

    hyp = []
    for t in range(encoder_out.shape[1]):
        logit = sess_join.run(None, {enc_name: encoder_out[:, t, :], dec_name: dec_out})[0]
        k = int(logit[0].argmax())
        if k != BLANK_ID:
            hyp.append(k)
            y = np.array([[y[0, 1], k]], dtype=np.int64)
            dec_out = sess_dec.run(None, {sess_dec.get_inputs()[0].name: y})[0]

    text = "".join(tokens[i] for i in hyp).replace("▁", " ").strip()
    return text


def main() -> None:
    model_dir = Path(sys.argv[1])
    wav = sys.argv[2]
    samples, _ = read_wav(wav)

    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(next(model_dir.glob("encoder*.int8.onnx"))),
        decoder=str(next(model_dir.glob("decoder*.onnx"))),
        joiner=str(next(model_dir.glob("joiner*.int8.onnx"))),
        tokens=str(model_dir / "tokens.txt"), model_type="zipformer2", num_threads=4)
    st = rec.create_stream()
    st.accept_waveform(16000, samples)
    rec.decode_stream(st)
    ref = st.result.text

    for snip in (False, True):
        feats = fbank80(samples, snip_edges=snip)
        got = greedy_decode(model_dir, feats)
        same = got.replace(" ", "") == ref.replace(" ", "")
        print(f"[snip_edges={snip}] frames={feats.shape[0]} match={same}")
        print(f"    ours:  {got[:120]}")
        print(f"    sherpa:{ref[:120]}")


if __name__ == "__main__":
    main()
