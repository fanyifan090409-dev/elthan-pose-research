#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/isef/data/raw/web-batch1")
FPS_OUT = 5.0
MAX_FRAMES = 100

videos = sorted((BASE / "videos").rglob("*.mp4"))
print(f"videos: {len(videos)}")
for v in videos:
    action = v.parent.name
    out_dir = BASE / "images" / action / v.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")):
        continue
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / FPS_OUT))
    n = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(out_dir / f"f{idx:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
            if n >= MAX_FRAMES:
                break
        idx += 1
    cap.release()
    print(f"{v.stem}: {n} frames")
