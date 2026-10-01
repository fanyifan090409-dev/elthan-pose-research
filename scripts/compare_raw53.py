# -*- coding: utf-8 -*-
# round5-3 三档裸输出 vs ORT 参考 误差对比
import numpy as np

V = "/home/ubuntu/isef/deploy/ncnn/pose-round5-3/verify"
NAMES = ["boxraw", "clsraw", "kptraw"]
SHAPES = {"boxraw": (64, 8400), "clsraw": (7, 8400), "kptraw": (51, 8400)}
PREC = ["fp32", "fp16", "int8"]

for k, nm in enumerate(NAMES):
    ref = np.fromfile(f"{V}/ort_{nm}.bin", dtype=np.float32).reshape(SHAPES[nm])
    print(f"--- {nm} shape={SHAPES[nm]}")
    for p in PREC:
        d = np.fromfile(f"{V}/{p}_0_{k}.bin", dtype=np.float32).reshape(SHAPES[nm])
        if ref.shape == d.shape and ref.size == d.size:
            diff = np.abs(ref - d)
            rel = np.abs(ref - d) / (np.abs(ref) + 1e-6)
            print(f"  {p}: max_abs={diff.max():.5f} mean_abs={diff.mean():.6f} "
                  f"max_rel={rel.max():.4f} mean_rel={rel.mean():.6f}")
        else:
            print(f"  {p}: SHAPE MISMATCH ref={ref.shape} got={d.shape}")
