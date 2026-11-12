"""流式 X-ASR（960 ms chunk）在 AX NPU 上的运行时。

和离线的区别：流式 encoder 带 115 个 cache 状态（输入 + 输出同名带 `new_` 前缀），
每次只前进 96 帧（960 ms），输入窗口 109 帧（`chunk*2 + 13`，icefall 导出的定义）。

用法（板端）：

    from stream_asr import StreamingAsr
    asr = StreamingAsr("/root/x-asr/models/streaming")
    asr.start()
    for pcm_chunk in mic:                     # float32 / int16 16 kHz
        text = asr.accept(pcm)                # 返回当前累计文本（每次可能有新增）
    text = asr.finish()
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ax_engine import AxEngine
from asr_engine import fbank80, clean_text

FRAMES = 109        # 每次喂给 encoder 的特征帧数 = chunk_size*2 + 13
HOP = 96            # 每次前进的帧数（960 ms）
BLANK = 0
CTX = 2
LOG_FLOOR = 1.1920929e-07


class StreamingAsr:
    def __init__(self, models_dir: str | Path, aux_backend: str = "npu",
                 num_threads: int = 4, enc=None):
        d = Path(models_dir)
        # encoder 是无状态的（状态都在 Python 侧），多会话可以共用同一个引擎实例，
        # 避免每个浏览器会话都重新加载 157 MB 的 axmodel。
        self.enc = enc if enc is not None else AxEngine(str(d / "encoder_stream.axmodel"))
        self.io = self.enc.describe()
        self.in_meta = self.io["inputs"]
        self.out_meta = self.io["outputs"]
        # encoder_out 是主输出，其余 new_cached_* 回写到同名 state
        self.state_names = [m["name"] for m in self.in_meta if m["name"] != "x"]
        self.out_to_state = {}
        for m in self.out_meta:
            n = m["name"]
            if n.startswith("new_"):
                self.out_to_state[n] = n[4:]
        self.tokens = [l.split()[0] for l in
                       (d / "tokens.txt").read_text(encoding="utf-8").splitlines() if l]

        # decoder / joiner：优先 NPU（和离线版一致），没有就 CPU
        self.dec_ax = self.joi_ax = None
        if aux_backend != "ort":
            for cand in (d / "decoder.axmodel",):
                if cand.exists():
                    self.dec_ax = AxEngine(str(cand))
            for cand in (d / "joiner.u16.axmodel", d / "joiner.axmodel"):
                if cand.exists():
                    self.joi_ax = AxEngine(str(cand))
                    break
        if self.dec_ax is None or self.joi_ax is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = 1
            if self.dec_ax is None:
                self.dec = ort.InferenceSession(str(d / "decoder.onnx"), so,
                                                providers=["CPUExecutionProvider"])
            if self.joi_ax is None:
                self.joi = ort.InferenceSession(str(d / "joiner.onnx"), so,
                                                providers=["CPUExecutionProvider"])

        self.backend = ("npu" if self.dec_ax else "ort") + "+" + (
            "npu" if self.joi_ax else "ort")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.states = {}
        for m in self.in_meta:
            if m["name"] == "x":
                continue
            shape = m["shape"]
            self.states[m["name"]] = np.zeros(shape, dtype=np.float32)
        self.feat_buf = np.zeros((0, 80), dtype=np.float32)
        self.pcm_buf = np.zeros(0, dtype=np.float32)
        self.feat_pos = 0              # 已经喂到第几帧
        self.dec_y = np.zeros((1, CTX), dtype=np.int64)
        self.dec_state = self._decoder(self.dec_y)
        self.hyp = []
        self.text = ""
        self.frames_done = 0
        self.enc_ms = 0.0
        self.dec_ms = 0.0

    # --------------------------------------------------------------- decoding
    def _decoder(self, y: np.ndarray) -> np.ndarray:
        if self.dec_ax is not None:
            shape, data = self.dec_ax.run([y.astype(np.float32)])[0]
            return np.frombuffer(data, dtype=np.float32).reshape(shape)
        return self.dec.run(None, {self.dec.get_inputs()[0].name: y})[0]

    def _greedy(self, enc_out: np.ndarray) -> str:
        added = []
        j_in = None if self.joi_ax is not None else [i.name for i in self.joi.get_inputs()]
        for t in range(enc_out.shape[1]):
            frame = enc_out[:, t, :].astype(np.float32)
            if self.joi_ax is not None:
                shape, data = self.joi_ax.run([frame, self.dec_state])[0]
                logit = np.frombuffer(data, dtype=np.float32).reshape(shape)
            else:
                logit = self.joi.run(None, {j_in[0]: frame, j_in[1]: self.dec_state})[0]
            k = int(logit.reshape(-1).argmax())
            if k != BLANK:
                added.append(k)
                self.dec_y = np.array([[self.dec_y[0, 1], k]], dtype=np.int64)
                self.dec_state = self._decoder(self.dec_y)
        if added:
            self.hyp.extend(added)
            self.text = clean_text("".join(self.tokens[i] for i in self.hyp))
        return self.text

    # ------------------------------------------------------------------ audio
    def accept(self, pcm: np.ndarray) -> str:
        """追加 PCM（float32 [-1,1] 或 int16），处理所有凑够 109 帧的 chunk。"""
        pcm = np.asarray(pcm)
        if pcm.dtype == np.int16:
            pcm = pcm.astype(np.float32) / 32768.0
        pcm = pcm.reshape(-1).astype(np.float32)
        if pcm.size:
            self.feat_buf = np.concatenate([self.feat_buf, fbank80(pcm)], axis=0)
        while self.feat_buf.shape[0] >= FRAMES:
            win = self.feat_buf[:FRAMES]
            self._run_chunk(win)
            self.feat_buf = self.feat_buf[HOP:]
        return self.text

    def finish(self, tail_pad: bool = True) -> str:
        """把剩余不足 109 帧的部分补齐一次，输出最终文本。"""
        if self.feat_buf.shape[0] > 0 and tail_pad:
            win = self.feat_buf
            if win.shape[0] < FRAMES:
                win = np.concatenate(
                    [win, np.full((FRAMES - win.shape[0], 80), float(np.log(LOG_FLOOR)),
                                  np.float32)])
            self._run_chunk(win[:FRAMES])
            self.feat_buf = np.zeros((0, 80), dtype=np.float32)
        return self.text

    def _run_chunk(self, win: np.ndarray):
        x = win[None, None, ...].astype(np.float32) if len(self.in_meta[0]["shape"]) == 4 \
            else win[None, ...].astype(np.float32)
        inputs = [x] + [self.states[n] for n in self.state_names]
        t0 = time.perf_counter()
        outs = self.enc.run(inputs)
        self.enc_ms += (time.perf_counter() - t0) * 1000
        enc_out = None
        for (name, _size), (shape, data) in zip(
                [(m["name"], m["bytes"]) for m in self.out_meta], outs):
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
            if name == "encoder_out":
                enc_out = arr
            else:
                state = self.out_to_state.get(name)
                if state is not None:
                    self.states[state] = arr.copy()
        t1 = time.perf_counter()
        self._greedy(enc_out)
        self.dec_ms += (time.perf_counter() - t1) * 1000
        self.frames_done += enc_out.shape[1]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "chunks": self.frames_done,
            "audio_ms": round(self.frames_done * 40.0, 1),   # 24 帧/块，每帧 40 ms
            "encoder_ms": round(self.enc_ms, 1),
            "decoder_joiner_ms": round(self.dec_ms, 1),
            "text": self.text,
        }
