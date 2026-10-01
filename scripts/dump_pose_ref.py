import os, sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

OUT = "/home/ubuntu/isef/deploy/ncnn/pose-round3/verify"
os.makedirs(OUT, exist_ok=True)

imgs = sys.argv[1:] if len(sys.argv) > 1 else []

model = YOLO("/home/ubuntu/isef/runs/pose-round3/weights/best.pt")
torch_model = model.model
torch_model.eval()
torch_model.float()

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas, r, top, left

for i, p in enumerate(imgs):
    img = cv2.imread(p)
    if img is None:
        print("skip", p); continue
    canvas, r, top, left = letterbox(img)
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    blob.tofile(f"{OUT}/in_{i}.bin")
    np.save(f"{OUT}/meta_{i}.npy", np.array([img.shape[0], img.shape[1], r, top, left]))
    with torch.no_grad():
        out = torch_model(torch.from_numpy(blob.copy()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = out.cpu().numpy()
    print(i, os.path.basename(p), out.shape)
    out.tofile(f"{OUT}/ref_{i}.bin")
    np.save(f"{OUT}/refshape_{i}.npy", np.array(out.shape))
print("DONE")
