import numpy as np
import onnx
import onnxruntime as ort
import sys

V = "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify"
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

CHECKS = [
    "/model.0/conv/Conv_output_0",
    "/model.0/act/Mul_output_0",
    "/model.1/conv/Conv_output_0",
    "/model.1/act/Mul_output_0",
    "/model.2/cv1/act/Mul_output_0",
    "/model.2/Split_output_1",
    "/model.2/m.0/cv2/act/Mul_output_0",
    "/model.2/m.0/Add_output_0",
    "/model.2/Concat_output_0",
]

m = onnx.load("/home/ubuntu/isef/runs/pose-round3/weights/best-raw.onnx")
node_outputs = set()
for n in m.graph.node:
    node_outputs.update(n.output)

for name in CHECKS:
    if name not in node_outputs:
        print(f"WARN: {name} not a node output in graph")
        continue
    vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    m.graph.output.append(vi)
try:
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
except Exception:
    pass

sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
inp = np.fromfile(f"{V}/in_{idx}.bin", dtype=np.float32).reshape(1, 3, 640, 640)
outs = sess.run(CHECKS, {"images": inp})

for k, (name, arr) in enumerate(zip(CHECKS, outs)):
    arr = np.asarray(arr).reshape(-1).astype(np.float32)
    arr.tofile(f"{V}/ort_diag_{idx}_{k}.bin")
    print(f"k={k} {name} {arr.shape[0]} floats")
print("DONE")
