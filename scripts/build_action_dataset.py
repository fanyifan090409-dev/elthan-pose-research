#!/usr/bin/env python3
"""build_action_dataset.py — 动作检测数据（Roboflow，6 类仅 bbox）并入 pose-round1

虚拟关键点 v=0 不参与 kpt loss；类 id 整体 +1（0 留给 person）。
筛选：无标注/空标注/损坏图片丢弃。输出 action_summary.json
"""
import json
import os
import shutil

from PIL import Image

COCO_KPT = 17
OUT = "/home/ubuntu/isef/datasets/pose-round1"
SRC = "/home/ubuntu/isef/datasets/action-det"


def link_image(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(split):
    src_img = os.path.join(SRC, split, "images")
    src_lbl = os.path.join(SRC, split, "labels")
    out_img = os.path.join(OUT, "images", split)
    out_lbl = os.path.join(OUT, "labels", split)
    n_img = n_box = 0
    per_class = {}
    for f in sorted(os.listdir(src_lbl)):
        if not f.endswith(".txt"):
            continue
        stem = f[:-4]
        img_f = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = stem + ext
            if os.path.exists(os.path.join(src_img, cand)):
                img_f = cand
                break
        if img_f is None:
            continue
        lines = []
        for ln in open(os.path.join(src_lbl, f)):
            p = ln.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = p
            c = int(c) + 1
            lines.append(f"{c} {x} {y} {w} {h} " + " ".join(["0.0 0.0 0"] * COCO_KPT))
            per_class[c] = per_class.get(c, 0) + 1
        if not lines:
            continue
        try:
            with Image.open(os.path.join(src_img, img_f)) as im:
                im.verify()
        except Exception:
            continue
        link_image(os.path.join(src_img, img_f), os.path.join(out_img, img_f))
        with open(os.path.join(out_lbl, f), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(lines)
    return n_img, n_box, per_class


if __name__ == "__main__":
    summary = {}
    for split in ("train", "val"):
        n_img, n_box, per_class = build(split)
        print(f"action {split}: {n_img} images, {n_box} boxes, per-class {per_class}")
        summary[split] = {"images": n_img, "boxes": n_box, "per_class": per_class}
    with open(os.path.join(OUT, "action_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
