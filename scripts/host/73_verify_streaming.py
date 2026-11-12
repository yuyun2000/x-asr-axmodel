"""用 pulsar2 run 验证流式 encoder axmodel：116 输入 / 116 输出逐一对 fp32 ONNX 求 cosine。"""
import subprocess, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

P = "/home/m5stack/.cache/magnetar/pulsar2/8.0"
ONNX = Path("/tmp/ci_test/stream960_fz.onnx")
AX = Path("/tmp/ci_test/stream960_fz_build/out/encoder_stream960.axmodel")
CAL = Path("/tmp/ci_test/stream960_fz_build/calib")
WORK = Path("/tmp/verify_stream_work")

sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
feeds = []
for i in sess.get_inputs():
    dims = [d for d in i.shape]
    f = CAL / i.name / "0000.npy"
    arr = np.load(f)
    feeds.append((i.name, arr))

feed = {n: a for n, a in feeds}
ref = sess.run(None, feed)
names = [o.name for o in sess.get_outputs()]

subprocess.run(["rm", "-rf", str(WORK)])
(WORK / "in").mkdir(parents=True); (WORK / "out").mkdir(parents=True)
for n, a in feeds:
    np.ascontiguousarray(a).tofile(WORK / "in" / f"{n}.bin")
env = {"LD_LIBRARY_PATH": f"{P}/lib", "PYTHONHOME": f"{P}/python3",
       "PYTHONPATH": f"{P}/pulsar2/axnn:{P}/pulsar2/axnn/axnn/tools:{P}/pulsar2",
       "PATH": "/usr/local/bin:/usr/bin:/bin"}
subprocess.run([f"{P}/lib/ld-linux-x86-64.so.2", f"{P}/python3/bin/python3",
                f"{P}/pulsar2/axnn/yamain/main.py", "run", "--model", str(AX),
                "--input_dir", str(WORK / "in"), "--output_dir", str(WORK / "out")],
               check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

cos = []
worst = None
for name, r in zip(names, ref):
    f = WORK / "out" / f"{name}.bin"
    got = np.fromfile(f, dtype=np.float32)
    r = np.asarray(r, np.float64).reshape(-1)
    got = got.astype(np.float64)
    if got.size != r.size:
        print(f"  !! {name}: size mismatch {got.size} vs {r.size}")
        continue
    c = float(r @ got / (np.linalg.norm(r) * np.linalg.norm(got) + 1e-12))
    cos.append(c)
    if worst is None or c < worst[1]:
        worst = (name, c)
print(f"outputs: {len(cos)}/{len(names)} | cosine mean={np.mean(cos):.6f} "
      f"min={np.min(cos):.6f} (worst: {worst[0]})")
