"""Minimal `k2` stub used only for ONNX export.

icefall 的导出脚本只需要 `k2.SymbolTable`（读 tokens.txt）和 `k2.version`，
但 PyPI 上的 `k2` wheel 是按 PyTorch 1.13.1 编译的，在 torch 2.x 环境里会直接
ImportError。本仓库的导出流程不需要 FSA 训练/解码，因此用这个 stub 顶替。

放在 PYTHONPATH 最前面即可覆盖真正的 k2 包：

    PYTHONPATH=$PWD/scripts/host/k2stub:$PWD/pydeps ...

任何真正用到 FSA / RaggedTensor 的代码路径都会显式报错，不会被静默忽略。
"""

__version__ = "0.0.0-k2stub"
with_cuda = False

from . import version  # noqa: F401,E402


class SymbolTable:
    """k2.SymbolTable 的最小可用实现（仅 from_file / __getitem__ / __len__）。"""

    def __init__(self, symbols=None):
        self.symbols = list(symbols) if symbols is not None else []

    @classmethod
    def from_file(cls, filename):
        pairs = []
        max_id = -1
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line in {filename}: {line!r}. "
                        "Expecting '<symbol> <id>'"
                    )
                sym, idx = parts[0], int(parts[1])
                pairs.append((sym, idx))
                max_id = max(max_id, idx)
        symbols = [""] * (max_id + 1)
        for sym, idx in pairs:
            symbols[idx] = sym
        return cls(symbols)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.symbols[key]
        return self.symbols.index(key)

    def __len__(self):
        return len(self.symbols)

    def get(self, key, default=None):
        try:
            return self[key]
        except (ValueError, IndexError):
            return default

    def num_symbols(self):
        return len(self.symbols)


class _Unsupported:
    """k2 里本流程用不到的类型的占位符，被调用时立即报错。"""

    _name = "?"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"k2 stub: {self._name} is only needed for k2-based training/decoding, "
            "not for ONNX export."
        )


class Fsa(_Unsupported):
    _name = "k2.Fsa"


class RaggedTensor(_Unsupported):
    _name = "k2.RaggedTensor"


class RaggedShape(_Unsupported):
    _name = "k2.RaggedShape"
