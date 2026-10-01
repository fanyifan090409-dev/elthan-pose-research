import onnx
import sys

m = onnx.load("/home/ubuntu/isef/runs/pose-round3/weights/best.onnx")
g = m.graph
nodes = {n.name: n for n in g.node}
by_output = {}
for n in g.node:
    for o in n.output:
        by_output[o] = n

final = next(n for n in g.node if g.output[0].name in n.output)
print("final:", final.name, final.op_type, "inputs:", final.input)


def trace_back(name, depth=0, maxdepth=12):
    out = []
    cur = name
    for _ in range(maxdepth):
        p = by_output.get(cur)
        if p is None:
            out.append(f"{cur} <- (initializer/input)")
            break
        out.append(f"{cur} <- {p.op_type}:{p.name}")
        cur = p.input[0]
        if p.op_type == "Concat" and len(p.input) == 3:
            break
    return out


for i, inp in enumerate(final.input):
    print(f"--- input[{i}] {inp}")
    for line in trace_back(inp):
        print("   ", line)
