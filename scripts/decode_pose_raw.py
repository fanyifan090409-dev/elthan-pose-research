import numpy as np
import sys

V = "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify"
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def make_anchors(hw=[(80, 80), (40, 40), (20, 20)], strides=[8, 16, 32], offset=0.5):
    anchors, strd = [], []
    for (h, w), s in zip(hw, strides):
        sx = np.arange(w, dtype=np.float32) + offset
        sy = np.arange(h, dtype=np.float32) + offset
        syy, sxx = np.meshgrid(sy, sx, indexing="ij")
        anchors.append(np.stack([sxx.ravel(), syy.ravel()], 1))
        strd.append(np.full((h * w, 1), s, dtype=np.float32))
    return np.concatenate(anchors), np.concatenate(strd)


anchors, strides = make_anchors()
anchors_T = anchors.T[None]        # [1,2,8400]
strides_T = strides.T[None]        # [1,1,8400]

box = np.fromfile(f"{V}/out_{idx}_0.bin", dtype=np.float32).reshape(1, 64, 8400)
cls = np.fromfile(f"{V}/out_{idx}_1.bin", dtype=np.float32).reshape(1, 7, 8400)
kpt = np.fromfile(f"{V}/out_{idx}_2.bin", dtype=np.float32).reshape(1, 51, 8400)

# DFL
d = box.reshape(1, 4, 16, 8400)
e = np.exp(d - d.max(axis=2, keepdims=True))
d = (e / e.sum(axis=2, keepdims=True) * np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)).sum(2)
lt, rb = d[:, 0:2], d[:, 2:4]
xy = (anchors_T - lt + anchors_T + rb) / 2.0
wh = (anchors_T + rb) - (anchors_T - lt)
box_dec = np.concatenate([xy, wh], 1) * strides_T

cls_dec = 1.0 / (1.0 + np.exp(-cls))

k = kpt.reshape(1, 17, 3, 8400)
kxy = (k[:, :, :2] * 2.0 + (anchors_T.reshape(1, 1, 2, 8400) - 0.5)) * strides_T.reshape(1, 1, 1, 8400)
kv = 1.0 / (1.0 + np.exp(-k[:, :, 2:3]))
kpt_dec = np.concatenate([kxy, kv], 2).reshape(1, 51, 8400)

out = np.concatenate([box_dec, cls_dec, kpt_dec], 1).astype(np.float32)

ref = np.fromfile(f"{V}/ref_{idx}.bin", dtype=np.float32).reshape(1, 62, 8400)
d_abs = np.abs(ref - out)
print(f"img{idx} decoded-vs-pytorch: max={d_abs.max():.4f} mean={d_abs.mean():.6f}")
for c, nm in [(0, "box"), (4, "cls"), (11, "kpt")]:
    print(f"  {nm}: max={d_abs[0, c].max():.4f} mean={d_abs[0, c].mean():.6f}")

out.tofile(f"{V}/decoded_{idx}.bin")
print("DONE")
