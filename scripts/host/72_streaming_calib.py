"""流式 encoder 的真实状态校准集：按 chunk 跑 fp32 encoder，记录 (x, 所有 cache)。"""
import glob, io, json, tarfile, wave
from pathlib import Path
import numpy as np, onnxruntime as ort
import kaldi_native_fbank as knf

ROOT = Path("/home/m5stack/work/x-asr-axmodel")
ONNX = Path("/tmp/ci_test/stream960_fz.onnx")      # 已冻结 processed_lens=256 并在图里
OUT = Path("/tmp/ci_test/stream960_fz_build")
NS = 16

enc = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
in_names = [i.name for i in enc.get_inputs()]
out_names = [o.name for o in enc.get_outputs()]

opts = knf.FbankOptions(); opts.frame_opts.dither = 0; opts.frame_opts.snip_edges = False
opts.frame_opts.samp_freq = 16000; opts.mel_opts.num_bins = 80
opts.mel_opts.low_freq = 20; opts.mel_opts.high_freq = -400
def fbank(path):
    with wave.open(str(path), "rb") as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    fb = knf.OnlineFbank(opts); fb.accept_waveform(16000, pcm); fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)

def init_states():
    st = {}
    for i in enc.get_inputs():
        dims = [d if isinstance(d, int) else 1 for d in i.shape]
        st[i.name] = np.zeros(dims, dtype=np.float32)
    return st

HOP, T = 96, 109
samples = []   # list of dict(name->array)
for wav in sorted(glob.glob(str(ROOT / "wavs/*.wav")))[:6]:
    feat = fbank(Path(wav))
    st = init_states()
    start = 0
    while start < feat.shape[0] and len(samples) < NS * 4:
        win = feat[start:start + T]
        if win.shape[0] < T:
            win = np.concatenate([win, np.full((T - win.shape[0], 80), float(np.log(1.1920929e-07)), np.float32)])
        feed = dict(st); feed["x"] = win[None, ...]
        samples.append({k: np.array(v, copy=True) for k, v in feed.items()})
        res = enc.run(None, feed)
        for name, val in zip(out_names, res):
            if name.startswith("new_"):
                st[name[4:]] = val
        start += HOP

# 均匀挑 NS 个样本
idx = np.linspace(0, len(samples) - 1, NS).astype(int)
pick = [samples[i] for i in idx]
print("collected chunks:", len(samples), "-> picked", len(pick))

(OUT / "calib").mkdir(parents=True, exist_ok=True)
input_configs = []
with tarfile.open(OUT / "all_inputs.tar.gz", "w:gz") as _unused:
    pass
for name in in_names:
    d = OUT / "calib" / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.npy"):
        f.unlink()
    tar_path = OUT / "calib" / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for i, s in enumerate(pick):
            arr = np.ascontiguousarray(s[name])
            np.save(d / f"{i:04d}.npy", arr)
            buf = io.BytesIO(); np.save(buf, arr)
            info = tarfile.TarInfo(f"{i:04d}.npy"); info.size = buf.tell(); buf.seek(0)
            tar.addfile(info, buf)
    input_configs.append({"tensor_name": name, "calibration_dataset": str(tar_path),
                          "calibration_format": "Numpy", "calibration_size": NS,
                          "calibration_mean": [], "calibration_std": []})

cfg = json.loads((OUT / "config.json").read_text())
cfg["quant"]["input_configs"] = input_configs
(OUT / "config.json").write_text(json.dumps(cfg, indent=1))
print("config updated with", len(input_configs), "inputs")
