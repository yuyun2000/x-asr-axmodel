#!/usr/bin/env python3
"""把「由 shape 计算出来的动态张量」冻结成常量 initializer。

为什么需要：Zipformer 的 `NonlinAttention` 等模块里有
`x.reshape(x.shape[0], -1, d)` 这类写法，导出后 `Reshape` 的 shape 输入是一条
Shape→Gather→Concat 的计算链。Pulsar2 的 shape function 要求 Reshape 的目标形状
是静态常量，否则报 `shapefn failed ... Reshape_xxx`。

在输入 shape 固定的前提下，这些链的结果都是常量。本脚本：
1. 找出所有「形状类输入」（Reshape/Expand/ConstantOfShape/... 的 shape 参数）中不是
   initializer 的张量；
2. 把它们临时加成图输出，用 onnxruntime 求值；
3. 写回同名张量（重命名为 `<name>__frozen` 的 initializer 并改写消费者）；
4. 剪掉不再被引用的节点；
5. 与输入模型做数值对分。

用法：
    python 26_freeze_shapes.py --src task/export/model_folded.onnx \
        --dst task/export/model_static.onnx --input x4d:1,1,1000,80
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

SHAPE_CONSUMERS = {
    "Reshape": [1],
    "Expand": [1],
    "ConstantOfShape": [0],
    "Tile": [1],
    "Slice": [1, 2, 3, 4],
    "Squeeze": [1],
    "Unsqueeze": [1],
    "ReduceMean": [1],
    "Pad": [1],
    "Split": [1],
    "TopK": [1],
}


def shape_inputs(model: onnx.ModelProto) -> set[str]:
    init = {i.name for i in model.graph.initializer}
    out = set()
    for n in model.graph.node:
        for idx in SHAPE_CONSUMERS.get(n.op_type, []):
            if idx < len(n.input) and n.input[idx] and n.input[idx] not in init:
                out.add(n.input[idx])
    return out


def prune(model: onnx.ModelProto) -> onnx.ModelProto:
    """删除对图输出没有贡献的节点与 initializer。"""
    producers = {o: n for n in model.graph.node for o in n.output}
    keep_nodes, need, stack = set(), set(), [o.name for o in model.graph.output]
    while stack:
        name = stack.pop()
        if name in need:
            continue
        need.add(name)
        node = producers.get(name)
        if node is not None:
            keep_nodes.add(id(node))
            stack.extend(i for i in node.input if i)
    kept = [n for n in model.graph.node if id(n) in keep_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used = {i for n in model.graph.node for i in n.input}
    inits = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    return model


def freeze(model: onnx.ModelProto, feed: dict) -> tuple[onnx.ModelProto, int]:
    targets = sorted(shape_inputs(model))
    if not targets:
        return model, 0

    probe = copy.deepcopy(model)
    producers = {o: n for n in probe.graph.node for o in n.output}
    for name in targets:
        node = producers.get(name)
        dtype = None
        if node is not None and node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to":
                    dtype = a.i
        vi = helper.make_tensor_value_info(
            name, dtype if dtype else onnx.TensorProto.INT64, None)
        probe.graph.output.append(vi)
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    values = sess.run(targets, feed)

    frozen = {name: np.asarray(v) for name, v in zip(targets, values)}
    existing = {i.name for i in model.graph.initializer}
    for name, arr in frozen.items():
        new_name = name + "__frozen"
        if new_name not in existing:
            model.graph.initializer.append(numpy_helper.from_array(arr, name=new_name))
            existing.add(new_name)
        for n in model.graph.node:
            n.input[:] = [new_name if i == name else i for i in n.input]
    return model, len(frozen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input", required=True, help="input_name:d0,d1,d2,d3")
    args = ap.parse_args()

    name, dims = args.input.split(":")
    feed = {name: np.zeros([int(x) for x in dims.split(",")], dtype=np.float32)}

    model = onnx.load(args.src)
    total = 0
    for it in range(6):
        model, n = freeze(model, feed)
        total += n
        print(f"[freeze] 第 {it+1} 轮冻结 {n} 个形状张量", flush=True)
        if n == 0:
            break

    model = prune(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f"[freeze] 完成，共冻结 {total} 个，节点 {len(model.graph.node)}", flush=True)

    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[freeze] cosine(orig, static) = {cos:.8f}  max|diff| = {np.abs(a-b).max():.6f}")


if __name__ == "__main__":
    main()
