#!/usr/bin/env python3
"""OpenThermalPose v1+v2 → YOLO-pose 筛选（task #107，口径同 CrowdPose/OCHuman）

- 入训 = v1 train + v2 train；val/test 不入训，留作热成像评估集
- 筛选：bbox 高 ≥96px；已标注点(v>0) ≥8；不达标实例丢弃，全丢的图不保留
- 输出 yolo/images/train（软链）+ yolo/labels/train；文件名加 v1_/v2_ 前缀防重名
- --cap N：控比例采样（round5 热成像占比控制，seed 42）
- v1 是 L 模式灰度 PNG，ultralytics 训练集默认 IMREAD_COLOR 会自动复制三通道，无需转码
"""
import os
import random
import sys
from pathlib import Path

from PIL import Image

BASE = Path("/home/ubuntu/isef/datasets/openthermalpose")
SRC_DIRS = [("v1", BASE / "OpenThermalPose" / "train"), ("v2", BASE / "otp2_dataset" / "train")]
OUT = BASE / "yolo"
MIN_BBOX_H = 96
MIN_LABELED = 8


def main():
    cap = None
    if "--cap" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--cap") + 1])
    (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    stats = {"img": 0, "img_keep": 0, "inst": 0, "inst_keep": 0, "drop_h": 0, "drop_k": 0}
    kept = []  # (src_img, out_name, filtered_lines)
    for tag, src in SRC_DIRS:
        for lb in sorted((src / "labels").glob("*.txt")):
            stats["img"] += 1
            img_fn = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                if (src / "images" / (lb.stem + ext)).exists():
                    img_fn = lb.stem + ext
                    break
            if img_fn is None:
                continue
            with Image.open(src / "images" / img_fn) as im:
                img_h = im.size[1]
            lines = []
            for line in lb.read_text().strip().splitlines():
                t = line.split()
                if len(t) < 5 + 17 * 3:
                    continue
                try:
                    nums = [float(x) for x in t]
                except ValueError:
                    continue
                bbox = nums[1:5]
                kps = nums[5:]
                h_px = bbox[3] * img_h
                vis = sum(1 for i in range(2, len(kps), 3) if kps[i] > 0)
                stats["inst"] += 1
                if h_px < MIN_BBOX_H:
                    stats["drop_h"] += 1
                    continue
                if vis < MIN_LABELED:
                    stats["drop_k"] += 1
                    continue
                stats["inst_keep"] += 1
                lines.append(line)
            if lines:
                stats["img_keep"] += 1
                kept.append((src / "images" / img_fn, f"{tag}_{lb.stem}", lines))
    if cap and len(kept) > cap:
        random.seed(42)
        random.shuffle(kept)
        kept = kept[:cap]
        stats["img_keep"] = len(kept)
    for src_img, out_name, lines in kept:
        (OUT / "labels" / "train" / f"{out_name}.txt").write_text("\n".join(lines) + "\n")
        os.symlink(src_img, OUT / "images" / "train" / f"{out_name}{Path(src_img).suffix}")
    print(stats)


if __name__ == "__main__":
    main()
