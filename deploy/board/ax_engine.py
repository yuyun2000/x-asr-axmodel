"""Minimal ctypes binding for the AXERA AX engine (board-side, no pip dependency).

The board SDK ships ``/soc/lib/libax_engine.so`` + ``/soc/include/ax_engine_api.h``
but no Python package (``axengine`` is not on PyPI for on-board use), so this file
binds the handful of C APIs needed to run an ``.axmodel`` from Python.

Only what this project needs is wrapped:

    AX_SYS_Init / AX_SYS_MemAlloc(Cached) / AX_SYS_MemFree
    AX_ENGINE_CreateHandle / CreateContext / GetIOInfo / RunSync / DestroyHandle

Struct layouts follow ``/soc/include/ax_engine_type.h`` for aarch64.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (POINTER, Structure, byref, c_char_p, c_int32, c_size_t,
                    c_uint8, c_uint32, c_uint64, c_void_p)

LIB_SYS = "/soc/lib/libax_sys.so"
LIB_ENGINE = "/soc/lib/libax_engine.so"
ALIGN = 128
SESSION_NAME = b"x-asr-axmodel"


class NpuAttr(Structure):
    _fields_ = [("eHardMode", c_int32), ("reserve", c_uint32 * 8)]


class IOMeta(Structure):
    _fields_ = [
        ("pName", c_char_p),
        ("pShape", POINTER(c_int32)),
        ("nShapeSize", c_uint8),
        ("eLayout", c_int32),
        ("eMemoryType", c_int32),
        ("eDataType", c_int32),
        ("pExtraMeta", c_void_p),
        ("nSize", c_uint32),
        ("nQuantizationValue", c_uint32),
        ("pStride", POINTER(c_int32)),
        ("u64Reserved", c_uint64 * 9),
    ]


class IOInfo(Structure):
    _fields_ = [
        ("pInputs", POINTER(IOMeta)),
        ("nInputSize", c_uint32),
        ("pOutputs", POINTER(IOMeta)),
        ("nOutputSize", c_uint32),
        ("nMaxBatchSize", c_uint32),
        ("bDynamicBatchSize", c_int32),
        ("u64Reserved", c_uint64 * 11),
    ]


class IOBuffer(Structure):
    _fields_ = [
        ("phyAddr", c_uint64),
        ("pVirAddr", c_void_p),
        ("nSize", c_uint32),
        ("pStride", POINTER(c_int32)),
        ("nStrideSize", c_uint8),
        ("u64Reserved", c_uint64 * 11),
    ]


class IOSetting(Structure):
    _fields_ = [("nWbtIndex", c_uint32), ("u64Reserved", c_uint64 * 7)]


class IO(Structure):
    _fields_ = [
        ("pInputs", POINTER(IOBuffer)),
        ("nInputSize", c_uint32),
        ("pOutputs", POINTER(IOBuffer)),
        ("nOutputSize", c_uint32),
        ("nBatchSize", c_uint32),
        ("pIoSetting", POINTER(IOSetting)),
        ("nParallelRun", c_uint32),
        ("u64Reserved", c_uint64 * 10),
    ]


class AxEngine:
    """Load one .axmodel and run it synchronously."""

    _sys = None
    _eng = None

    @classmethod
    def _libs(cls):
        if cls._sys is None:
            cls._sys = ctypes.CDLL(LIB_SYS, mode=ctypes.RTLD_GLOBAL)
            cls._eng = ctypes.CDLL(LIB_ENGINE, mode=ctypes.RTLD_GLOBAL)

            cls._sys.AX_SYS_Init.restype = c_int32
            cls._sys.AX_SYS_MemAlloc.restype = c_int32
            cls._sys.AX_SYS_MemAlloc.argtypes = [POINTER(c_uint64), POINTER(c_void_p),
                                                 c_uint32, c_uint32, c_char_p]
            cls._sys.AX_SYS_MemAllocCached.restype = c_int32
            cls._sys.AX_SYS_MemAllocCached.argtypes = [POINTER(c_uint64), POINTER(c_void_p),
                                                       c_uint32, c_uint32, c_char_p]
            cls._sys.AX_SYS_MemFree.argtypes = [c_uint64, c_void_p]
            cls._sys.AX_SYS_MinvalidateCache.argtypes = [c_uint64, c_void_p, c_uint32]

            cls._eng.AX_ENGINE_Init.restype = c_int32
            cls._eng.AX_ENGINE_Init.argtypes = [POINTER(NpuAttr)]
            cls._eng.AX_ENGINE_CreateHandle.restype = c_int32
            cls._eng.AX_ENGINE_CreateHandle.argtypes = [POINTER(c_void_p), c_void_p, c_uint32]
            cls._eng.AX_ENGINE_CreateContext.argtypes = [c_void_p]
            cls._eng.AX_ENGINE_GetIOInfo.restype = c_int32
            cls._eng.AX_ENGINE_GetIOInfo.argtypes = [c_void_p, POINTER(POINTER(IOInfo))]
            cls._eng.AX_ENGINE_RunSync.argtypes = [c_void_p, POINTER(IO)]
            cls._eng.AX_ENGINE_DestroyHandle.argtypes = [c_void_p]

            if cls._sys.AX_SYS_Init() != 0:
                raise RuntimeError("AX_SYS_Init failed")
            attr = NpuAttr()
            attr.eHardMode = 0
            if cls._eng.AX_ENGINE_Init(byref(attr)) != 0:
                raise RuntimeError("AX_ENGINE_Init failed")
        return cls._sys, cls._eng

    def __init__(self, model_path: str, release_model_buffer: bool = True):
        sys_lib, eng = self._libs()
        self._sys, self._eng = sys_lib, eng

        data = open(model_path, "rb").read()
        # SDK 要求模型 buffer 128 字节对齐
        self._model_buf = ctypes.create_string_buffer(len(data) + 128)
        base = ctypes.addressof(self._model_buf)
        aligned = (base + 127) & ~127
        ctypes.memmove(aligned, data, len(data))
        self._model_len = len(data)

        self.handle = c_void_p()
        ret = eng.AX_ENGINE_CreateHandle(byref(self.handle), c_void_p(aligned),
                                         self._model_len)
        if ret != 0:
            raise RuntimeError(f"AX_ENGINE_CreateHandle failed: {ret}")
        ret = eng.AX_ENGINE_CreateContext(self.handle)
        if ret != 0:
            raise RuntimeError(f"AX_ENGINE_CreateContext failed: {ret}")

        self.io_info = POINTER(IOInfo)()
        ret = eng.AX_ENGINE_GetIOInfo(self.handle, byref(self.io_info))
        if ret != 0:
            raise RuntimeError(f"AX_ENGINE_GetIOInfo failed: {ret}")

        # CreateHandle 之后 AX 已经把模型拷进 CMM，Python 侧的大 buffer 可以放掉；
        # 板子只有 2 GB 系统内存，165 MB 的副本必须省。
        if release_model_buffer:
            self._model_buf = None

        info = self.io_info.contents
        self.input_meta = [info.pInputs[i] for i in range(info.nInputSize)]
        self.output_meta = [info.pOutputs[i] for i in range(info.nOutputSize)]
        self.io = self.describe()

        # allocate device buffers once
        self._in_bufs, self._out_bufs = [], []
        io = IO()
        io.nInputSize = info.nInputSize
        io.nOutputSize = info.nOutputSize
        io.pInputs = (IOBuffer * info.nInputSize)()
        io.pOutputs = (IOBuffer * info.nOutputSize)()

        for i, meta in enumerate(self.input_meta):
            buf = IOBuffer()
            buf.nSize = meta.nSize
            phy, vir = c_uint64(), c_void_p()
            if self._sys.AX_SYS_MemAlloc(byref(phy), byref(vir), meta.nSize, ALIGN,
                                         SESSION_NAME) != 0:
                raise RuntimeError(f"AX_SYS_MemAlloc(input {i}) failed")
            buf.phyAddr, buf.pVirAddr = phy.value, vir
            io.pInputs[i] = buf
            self._in_bufs.append((buf.phyAddr, buf.pVirAddr))

        for i, meta in enumerate(self.output_meta):
            buf = IOBuffer()
            buf.nSize = meta.nSize
            phy, vir = c_uint64(), c_void_p()
            if self._sys.AX_SYS_MemAllocCached(byref(phy), byref(vir), meta.nSize, ALIGN,
                                               SESSION_NAME) != 0:
                raise RuntimeError(f"AX_SYS_MemAllocCached(output {i}) failed")
            buf.phyAddr, buf.pVirAddr = phy.value, vir
            io.pOutputs[i] = buf
            self._out_bufs.append((buf.phyAddr, buf.pVirAddr))

        self._io = io
        self._io_ptr = ctypes.pointer(io)

    def describe(self):
        def shape(meta):
            return [meta.pShape[i] for i in range(meta.nShapeSize)]

        return {
            "inputs": [{"name": m.pName.decode(), "shape": shape(m), "bytes": m.nSize}
                       for m in self.input_meta],
            "outputs": [{"name": m.pName.decode(), "shape": shape(m), "bytes": m.nSize}
                        for m in self.output_meta],
        }

    def run(self, inputs):
        """inputs: list of numpy arrays (float32) matching the model inputs."""
        for i, arr in enumerate(inputs):
            meta = self.input_meta[i]
            nbytes = arr.nbytes
            if nbytes != meta.nSize:
                raise ValueError(f"input {i} expects {meta.nSize} bytes, got {nbytes}")
            ctypes.memmove(self._io.pInputs[i].pVirAddr, arr.ctypes.data, nbytes)

        ret = self._eng.AX_ENGINE_RunSync(self.handle, self._io_ptr)
        if ret != 0:
            raise RuntimeError(f"AX_ENGINE_RunSync failed: {ret}")

        outputs = []
        for i, meta in enumerate(self.output_meta):
            buf = self._io.pOutputs[i]
            # invalidate cache so the CPU sees what the NPU wrote
            self._sys.AX_SYS_MinvalidateCache(buf.phyAddr, buf.pVirAddr, meta.nSize)
            data = ctypes.string_at(buf.pVirAddr, meta.nSize)
            shape = [meta.pShape[j] for j in range(meta.nShapeSize)]
            outputs.append((shape, data))
        return outputs

    def __del__(self):
        # 不在析构里显式 free：单进程生命周期内引擎常驻，AX 运行时随进程退出清理。
        # 早期版本在 __del__ 里 AX_ENGINE_DestroyHandle + AX_SYS_MemFree，
        # 实测会在解释器退出时 "munmap_chunk(): invalid pointer" 崩掉。
        pass
