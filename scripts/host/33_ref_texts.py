"""在宿主机上用 fp32 ONNX 生成所有 wav 的参考文本。"""
import glob, json, wave
import numpy as np, onnxruntime as ort, sys
from pathlib import Path
sys.path.insert(0, "scripts/host")
from fbank_kaldi import EPSILON, fbank80

FRAMES, BLANK, CTX = 1000, 0, 2
enc = ort.InferenceSession("task/export/model.onnx", providers=["CPUExecutionProvider"])
dec = ort.InferenceSession("onnx/decoder.fp32.onnx", providers=["CPUExecutionProvider"])
joi = ort.InferenceSession("onnx/joiner.fp32.onnx", providers=["CPUExecutionProvider"])
tokens = [l.split()[0] for l in open("tokens/tokens_nopunct.txt", encoding="utf-8") if l.strip()]
j_in = [i.name for i in joi.get_inputs()]
d_in = dec.get_inputs()[0].name

def greedy(e):
    y = np.zeros((1, CTX), dtype=np.int64)
    d = dec.run(None, {d_in: y})[0]
    hyp = []
    for t in range(e.shape[0]):
        k = int(joi.run(None, {j_in[0]: e[t:t+1], j_in[1]: d})[0].reshape(-1).argmax())
        if k != BLANK:
            hyp.append(k)
            y = np.array([[y[0, 1], k]], dtype=np.int64)
            d = dec.run(None, {d_in: y})[0]
    return "".join(tokens[i] for i in hyp).replace("▁", " ").strip()

def transcribe(pcm):
    """和板端一致的切窗：10 s 窗口，8 s 步长，尾部 <5 s 丢弃。"""
    f = fbank80(pcm)
    parts, start = [], 0
    while start < f.shape[0]:
        if f.shape[0] - start < 500 and parts:
            break
        win = f[start:start + FRAMES]
        if win.shape[0] < FRAMES:
            win = np.concatenate([win, np.full((FRAMES - win.shape[0], 80),
                                               np.log(EPSILON), np.float32)])
        e = enc.run(None, {enc.get_inputs()[0].name: win[None, None, ...].astype(np.float32)})[0][0]
        n_valid = min(e.shape[0], int(0.30 * min(FRAMES, f.shape[0] - start)) + 20)
        parts.append(greedy(e[:n_valid]))
        if win.shape[0] < FRAMES:
            break
        start += 800
    return " ".join(p for p in parts if p)

out = {}
for wav in sorted(glob.glob("wavs/*.wav")):
    with wave.open(wav, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    out[Path(wav).name] = transcribe(a)
    print(Path(wav).name, "|", out[Path(wav).name], flush=True)
json.dump(out, open("reports/fp32_ref_texts.json", "w"), ensure_ascii=False, indent=1)
print("saved reports/fp32_ref_texts.json")
