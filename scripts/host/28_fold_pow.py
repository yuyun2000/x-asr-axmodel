#!/usr/bin/env python3
"""把 `Pow(x, -0.5)` 改写成 `1 / Sqrt(x)`（AX650 的 NPU 不支持 -0.5 次幂）。

背景：Pulsar2 的 `AxPow` 只支持指数 0.25 / 0.5 / 2 / 4。Zipformer 的 LayerNorm
实现里有 20 处 `Pow(x, -0.5)`（即 rsqrt）。纯 INT8 路径下这些会被融合进
LayerNormalization，但一旦开启混精/逐层精度（`highest_mix_precision`），融合被打断，
编译就会报：

    TileFailException("AxPow, AxPow only supports power with 0.25, 0.5, 2 and 4,
    howerver got [-0.5]") @ /encoder_embed/out_norm/Pow_1

AXERA 官方 zipformer 工程是用 `cut_reciprocal_encoder.py` 把 `Reciprocal` 换成 `Div`，
这里是同一类处理：`x ** -0.5 = 1 / sqrt(x)`。

用法：
    python 28_fold_pow.py --src task/export/model.onnx \
        --dst task/export/model_powsqrt.onnx --input x4d:1,1,1000,80
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input", required=True, help="input_name:d0,d1,d2,d3")
    args = ap.parse_args()

    name, dims = args.input.split(":")
    feed = {name: np.random.default_rng(0).normal(
        0, 1, size=[int(x) for x in dims.split(",")]).astype(np.float32)}

    model = onnx.load(args.src)
    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    by_output = {o: n for n in graph.node for o in n.output}

    one_name = "pow_neg_half_one"
    if one_name not in init:
        graph.initializer.append(
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name))

    new_nodes, patched = [], 0
    for node in graph.node:
        if node.op_type != "Pow":
            new_nodes.append(node)
            continue
        exp_name = node.input[1]
        value = None
        if exp_name in init:
            value = float(np.asarray(init[exp_name]).reshape(-1)[0])
        else:
            producer = by_output.get(exp_name)
            if producer is not None and producer.op_type == "Constant":
                for attr in producer.attribute:
                    if attr.name == "value":
                        value = float(np.asarray(numpy_helper.to_array(attr.t)).reshape(-1)[0])
        if value is None or not np.isclose(value, -0.5):
            new_nodes.append(node)
            continue

        sqrt_out = node.output[0] + "_sqrt"
        new_nodes.append(helper.make_node(
            "Sqrt", [node.input[0]], [sqrt_out], name=node.name + "_sqrt"))
        new_nodes.append(helper.make_node(
            "Div", [one_name, sqrt_out], list(node.output), name=node.name + "_recip"))
        patched += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f"[pow] rewrote {patched} Pow(x, -0.5) nodes")

    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[pow] cosine(orig, rewritten) = {cos:.8f}  max|diff| = {np.abs(a - b).max():.6f}")


if __name__ == "__main__":
    main()
