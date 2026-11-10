#!/usr/bin/env python3
"""把 `Log(1 + Exp(-|x|))` 这种数值稳定的 softplus 改写成 ONNX 原生 `Softplus`。

为什么需要：icefall 的 Zipformer 用 `SwooshL/SwooshR` 激活，导出后是
`Log(Add(1, Exp(Neg(Abs(x)))))` 四五个节点。AX650 的 Pulsar2 **不支持 Log**，
会把它单独切成一个 ONNX(CPU) 子图 —— 实测 17 个 NPU 子图 + 16 个 CPU 子图相互穿插，
编译要一个多小时，端侧还要在 CPU/NPU 之间来回拷贝。

数学上 `log(1 + exp(-|x|)) == softplus(-|x|)`，而 Pulsar2 支持 `Softplus`
（用最小图验证过：`x → Softplus → y` 能编译成单个 NPU 子图）。

用法：
    python 27_fold_swoosh.py --src task/export/model.onnx \
        --dst task/export/model_softplus.onnx --input x4d:1,1,1000,80
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def get_const(node_by_out, init, name):
    if name in init:
        return init[name]
    n = node_by_out.get(name)
    if n is not None and n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    return None


def rewrite(model: onnx.ModelProto) -> int:
    g = model.graph
    node_by_out = {o: n for n in g.node for o in n.output}
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    killed, patched = set(), 0

    for node in g.node:
        if node.op_type != "Log":
            continue
        add = node_by_out.get(node.input[0])
        if add is None or add.op_type != "Add":
            continue
        exp_node, konst = None, None
        for i in add.input:
            p = node_by_out.get(i)
            if p is not None and p.op_type == "Exp":
                exp_node = p
            else:
                c = get_const(node_by_out, init, i)
                if c is not None:
                    konst = c
        if exp_node is None or konst is None:
            continue
        if np.asarray(konst).reshape(-1).size != 1 or \
                not np.isclose(float(np.asarray(konst).reshape(-1)[0]), 1.0):
            continue

        # Log(1 + Exp(z)) == Softplus(z)
        node.op_type = "Softplus"
        node.input[:] = [exp_node.input[0]]
        killed.add(id(exp_node))
        killed.add(id(add))
        patched += 1

    if killed:
        kept = [n for n in g.node if id(n) not in killed]
        del g.node[:]
        g.node.extend(kept)
        # 剪掉不再被引用的 initializer
        used = {i for n in g.node for i in n.input}
        inits = [i for i in g.initializer if i.name in used]
        del g.initializer[:]
        g.initializer.extend(inits)
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input", required=True, help="input_name:d0,d1,d2,d3")
    args = ap.parse_args()

    name, dims = args.input.split(":")
    feed = {name: np.zeros([int(x) for x in dims.split(",")], dtype=np.float32)}

    model = onnx.load(args.src)
    n = rewrite(model)
    print(f"[swoosh] 改写 {n} 处 Log → Softplus，剩余 Log = "
          f"{sum(1 for x in model.graph.node if x.op_type == 'Log')}")
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)

    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[swoosh] cosine(orig, softplus) = {cos:.8f}  max|diff| = {np.abs(a - b).max():.6f}")


if __name__ == "__main__":
    main()
