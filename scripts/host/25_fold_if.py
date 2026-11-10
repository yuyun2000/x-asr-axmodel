#!/usr/bin/env python3
"""把 ONNX 里「条件恒定的 If 节点」展开成直线代码。

为什么需要：icefall 导出的 X-ASR encoder 里有 43 个 `If` 节点（来自 Zipformer 的
`ConvModule` / `downsample` 里按输入长度分支的写法，被 TorchScript 保留成控制流）。
条件只依赖静态 shape，在 T=1000 固定输入下是常量，但 Pulsar2 的量化执行器遇到
`If` 会直接报 `Op Execution Error`。

两步走（都基于**原始**模型，避免改动后互相影响）：
1. 对每个 `If`，从图输入反向收集它 condition 的依赖，拼成临时模型用 ORT 求值；
2. 按布尔值把对应分支内联进主图：分支里自己产生的名字加前缀，引用外层作用域的
   名字原样保留，分支输出替换成 `If` 的输出。

用法：
    python 25_fold_if.py --src task/export/model.onnx --dst task/export/model_folded.onnx \
        --input x4d:1,1,1000,80
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def eval_conditions(model: onnx.ModelProto, feed: dict) -> dict:
    """把所有 If 的 condition 临时加成图输出，用 ORT 一次跑出布尔值。

    直接用完整模型跑，意味着 ORT 会自己按控制流执行，嵌套/串联的 If 也能正确求值。
    """
    probe = copy.deepcopy(model)
    names = [n.input[0] for n in probe.graph.node if n.op_type == "If"]
    for name in names:
        probe.graph.output.append(
            helper.make_tensor_value_info(name, onnx.TensorProto.BOOL, None))
    sess = ort.InferenceSession(probe.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    values = sess.run(names, feed)
    return {n: bool(np.asarray(v).reshape(-1)[0]) for n, v in zip(names, values)}


def fold_all(model: onnx.ModelProto, feed: dict) -> onnx.ModelProto:
    graph = model.graph
    cond_values = eval_conditions(model, feed)  # key: condition 张量名
    print(f"[fold] {len(cond_values)} 个 If 的 condition 已求值", flush=True)

    new_nodes = []
    for node in list(graph.node):
        if node.op_type != "If":
            new_nodes.append(node)
            continue

        cond = cond_values[node.input[0]]
        branch = next(a.g for a in node.attribute
                      if a.name == ("then_branch" if cond else "else_branch"))
        prefix = (node.name or node.output[0]) + "_"

        # 分支 graph input 依次对应 If 的 inputs[1:]
        mapped = {bi.name: node.input[i + 1] for i, bi in enumerate(branch.input)}
        # 分支自己定义的名字需要加前缀；其它名字是外层作用域引用，原样保留
        inner = set(bi.name for bi in branch.input)
        inner |= {init.name for init in branch.initializer}
        inner |= {o for n in branch.node for o in n.output}

        def rn(name: str) -> str:
            if name in mapped:
                return mapped[name]
            if name in inner:
                return prefix + name
            return name

        for init in branch.initializer:
            new_init = copy.deepcopy(init)
            new_init.name = prefix + init.name
            graph.initializer.append(new_init)

        out_map = {prefix + bo.name: io for bo, io in zip(branch.output, node.output)}
        for n in branch.node:
            nn = copy.deepcopy(n)
            nn.name = prefix + (n.name or n.output[0])
            nn.input[:] = [rn(i) if i else i for i in nn.input]
            nn.output[:] = [out_map.get(rn(o), rn(o)) for o in nn.output]
            new_nodes.append(nn)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input", required=True, help="input_name:d0,d1,d2,d3")
    args = ap.parse_args()

    name, dims = args.input.split(":")
    shape = [int(x) for x in dims.split(",")]
    feed = {name: np.zeros(shape, dtype=np.float32)}

    model = onnx.load(args.src)
    fold_all(model, feed)

    remaining = sum(1 for n in model.graph.node if n.op_type == "If")
    print(f"[fold] 剩余 If: {remaining}，节点总数 {len(model.graph.node)}", flush=True)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)

    a = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[fold] cosine(orig, folded) = {cos:.8f}  max|diff| = {np.abs(a - b).max():.6f}")


if __name__ == "__main__":
    main()
