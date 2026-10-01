# -*- coding: utf-8 -*-
# round5-3 端侧链第 2 步：best.onnx -> best-raw.onnx（切三裸输出）+ ORT 裸输出对拍
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

SRC = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.onnx"
DST = "/home/ubuntu/isef/runs/pose-round5-3/weights/best-raw.onnx"
CUT = [
    "/model.23/Concat_output_0",    # boxes raw [1,64,8400]
    "/model.23/Concat_1_output_0",  # cls raw [1,7,8400]
    "/model.23/Concat_2_output_0",  # kpts raw [1,51,8400]
]
CUT_SHAPES = [[1, 64, 8400], [1, 7, 8400], [1, 51, 8400]]
V = "/home/ubuntu/isef/deploy/ncnn/pose-round5-3/verify"

m = onnx.load(SRC)
g = m.graph
by_output = {}
for n in g.node:
    for o in n.output:
        by_output[o] = n

keep = set()
stack = list(CUT)
while stack:
    name = stack.pop()
    if name in keep:
        continue
    keep.add(name)
    p = by_output.get(name)
    if p is not None:
        stack.extend(p.input)

kept_nodes = [n for n in g.node if any(o in keep for o in n.output)]
input_names = {x.name for x in g.input}
used_init = set()
for n in kept_nodes:
    used_init.update(i for i in n.input if i not in by_output and i not in input_names)
kept_init = [i for i in g.initializer if i.name in used_init]
print("nodes kept:", len(kept_nodes), "/", len(g.node), "inits kept:", len(kept_init), "/", len(g.initializer))

new_graph = helper.make_graph(
    kept_nodes,
    "raw",
    g.input,
    [helper.make_tensor_value_info(c, TensorProto.FLOAT, s) for c, s in zip(CUT, CUT_SHAPES)],
    kept_init,
)
new_model = helper.make_model(new_graph, opset_imports=m.opset_import)
new_model.ir_version = m.ir_version
onnx.checker.check_model(new_model)
onnx.save(new_model, DST)
print("saved", DST)

inp = np.fromfile(f"{V}/in_0.bin", dtype=np.float32).reshape(1, 3, 640, 640)
sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
outs = sess.run(None, {"images": inp})
names = ["boxraw", "clsraw", "kptraw"]
for nm, o in zip(names, outs):
    o.astype(np.float32).tofile(f"{V}/ort_{nm}.bin")
    np.save(f"{V}/ort_{nm}_shape.npy", np.array(o.shape))
    print(nm, o.shape)
print("DONE")
