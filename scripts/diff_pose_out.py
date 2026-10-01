import numpy as np

V = "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify"
tag = __import__("sys").argv[1]

for i in range(2):
    ref = np.fromfile(f"{V}/ref_{i}.bin", dtype=np.float32).reshape(1, 62, 8400)
    out = np.fromfile(f"{V}/out_{i}.bin", dtype=np.float32).reshape(1, 62, 8400)
    d = np.abs(ref - out)
    print(f"{tag} img{i}: max_abs_diff={d.max():.6f} mean_abs={d.mean():.6f} cos={np.dot(ref.flatten(), out.flatten()) / (np.linalg.norm(ref) * np.linalg.norm(out)):.8f}")
