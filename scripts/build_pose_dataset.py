#!/usr/bin/env python3
"""build_pose_dataset.py — 融合三源构建 YOLO pose 训练集（task #21 round-1）

源：
  1. COCO-Pose 2017（person 类，17 关键点）→ 抽样 + 筛选
  2. MPII（MMPose 已转 COCO json，16 关节）→ 16→17 映射
  3. 动作检测（Roboflow，6 动作类仅 bbox）→ 由 build_action_dataset.py 单独处理

合并类别：0=person, 1=sit down, 2=sit up, 3=push down, 4=push up, 5=weights down, 6=weights up
筛选规则：丢弃损坏/无标注图片；COCO 每图 person 实例上限 MAX_PERSONS；MPII 只保留 ≥1 可见关节的实例；
          val 集与 train 集图片去重（MPII 按 person 切分可能同图）。
输出：pose-round1/{images,labels}/{train,val} + data.yaml + summary.json
"""
import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

random.seed(42)

COCO_KPT = 17
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
MPII_TO_COCO = {9: 0, 13: 5, 12: 6, 14: 7, 11: 8, 15: 9, 10: 10, 3: 11, 2: 12, 4: 13, 1: 14, 5: 15, 0: 16}
NAMES = ["person", "sit down", "sit up", "push down", "push up", "weights down", "weights up"]

from PIL import Image


def link_image(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_coco(ann_path, img_dir, out_img, out_lbl, sample=None, max_persons=20, exclude_images=None):
    data = json.load(open(ann_path))
    imgmap = {i["id"]: i for i in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    valid = []
    for iid, anns in anns_by_img.items():
        info = imgmap[iid]
        if exclude_images and info["file_name"] in exclude_images:
            continue
        if any(a["num_keypoints"] >= 5 for a in anns):
            valid.append(iid)
    if sample:
        valid = random.sample(valid, min(sample, len(valid)))
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    n_img = n_person = 0
    for iid in valid:
        info = imgmap[iid]
        fname = info["file_name"]
        W, H = info["width"], info["height"]
        anns = sorted(anns_by_img[iid], key=lambda a: -a["num_keypoints"])[:max_persons]
        lines = []
        for a in anns:
            if a["num_keypoints"] < 1:
                continue
            x, y, w, h = a["bbox"]
            if w < 2 or h < 2:
                continue
            k = a["keypoints"]
            kstr = " ".join(f"{k[3*i]/W:.6f} {k[3*i+1]/H:.6f} {k[3*i+2]}" for i in range(COCO_KPT))
            lines.append(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f} {kstr}")
        if not lines:
            continue
        src = os.path.join(img_dir, fname)
        if not os.path.exists(src):
            continue
        link_image(src, os.path.join(out_img, fname))
        with open(os.path.join(out_lbl, fname.rsplit(".", 1)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_img += 1
        n_person += len(lines)
    return n_img, n_person


def build_mpii(json_path, img_dir, out_img, out_lbl, exclude_images=None, max_persons=20):
    data = json.load(open(json_path))
    by_img = {}
    for e in data:
        by_img.setdefault(e["image"], []).append(e)
    jobs = [(fn, ps) for fn, ps in by_img.items() if not (exclude_images and fn in exclude_images)]

    def process(item):
        fname, persons = item
        src = os.path.join(img_dir, fname)
        if not os.path.exists(src):
            return None
        try:
            with Image.open(src) as im:
                W, H = im.size
        except Exception:
            return None
        lines = []
        for e in persons:
            joints = e["joints"]
            vis = e["joints_vis"]
            coords = [[0.0, 0.0, 0] for _ in range(COCO_KPT)]
            for mi, ci in MPII_TO_COCO.items():
                x, y = joints[mi]
                if vis[mi] and x >= 1 and y >= 1:
                    coords[ci] = [x / W, y / H, 2]
            pts = [(joints[mi][0], joints[mi][1]) for mi in range(16) if vis[mi] and joints[mi][0] >= 1 and joints[mi][1] >= 1]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = bx1 - bx0, by1 - by0
            if bw < 1 or bh < 1:
                continue
            pad = 0.02
            bx0 -= bw * pad
            by0 -= bh * pad
            bw *= 1 + 2 * pad
            bh *= 1 + 2 * pad
            bx0 = max(0.0, bx0)
            by0 = max(0.0, by0)
            bw = min(W, bx0 + bw) - bx0
            bh = min(H, by0 + bh) - by0
            if bw < 1 or bh < 1:
                continue
            kstr = " ".join(f"{c[0]:.6f} {c[1]:.6f} {c[2]}" for c in coords)
            lines.append(f"0 {(bx0+bw/2)/W:.6f} {(by0+bh/2)/H:.6f} {bw/W:.6f} {bh/H:.6f} {kstr}")
        if not lines:
            return None
        return fname, lines, src

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process, jobs))
    n_img = n_person = 0
    for r in results:
        if r is None:
            continue
        fname, lines, src = r
        link_image(src, os.path.join(out_img, fname))
        with open(os.path.join(out_lbl, fname.rsplit(".", 1)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_img += 1
        n_person += len(lines)
    return n_img, n_person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ubuntu/isef/datasets/pose-round1")
    ap.add_argument("--coco-sample", type=int, default=15000)
    ap.add_argument("--max-persons", type=int, default=20)
    args = ap.parse_args()
    out = args.out
    ds = "/home/ubuntu/isef/datasets"

    summary = {}

    print("=== COCO train (sample=%d) ===" % args.coco_sample)
    n_img, n_person = build_coco(
        f"{ds}/coco2017/annotations/person_keypoints_train2017.json",
        f"{ds}/coco2017/train2017",
        f"{out}/images/train", f"{out}/labels/train",
        sample=args.coco_sample, max_persons=args.max_persons)
    print(f"coco train: {n_img} images, {n_person} persons")
    summary["coco_train"] = {"images": n_img, "persons": n_person}

    print("=== MPII train ===")
    n_img, n_person = build_mpii(
        f"{ds}/mpii/mpii_train.json",
        f"{ds}/mpii/images",
        f"{out}/images/train", f"{out}/labels/train",
        max_persons=args.max_persons)
    print(f"mpii train: {n_img} images, {n_person} persons")
    summary["mpii_train"] = {"images": n_img, "persons": n_person}

    train_images = set(os.listdir(f"{out}/images/train"))

    print("=== COCO val ===")
    n_img, n_person = build_coco(
        f"{ds}/coco2017/annotations/person_keypoints_val2017.json",
        f"{ds}/coco2017/val2017",
        f"{out}/images/val", f"{out}/labels/val",
        max_persons=args.max_persons, exclude_images=train_images)
    print(f"coco val: {n_img} images, {n_person} persons")
    summary["coco_val"] = {"images": n_img, "persons": n_person}

    print("=== MPII val ===")
    n_img, n_person = build_mpii(
        f"{ds}/mpii/mpii_val.json",
        f"{ds}/mpii/images",
        f"{out}/images/val", f"{out}/labels/val",
        exclude_images=train_images, max_persons=args.max_persons)
    print(f"mpii val: {n_img} images, {n_person} persons")
    summary["mpii_val"] = {"images": n_img, "persons": n_person}

    summary["train_images"] = len(os.listdir(f"{out}/images/train"))
    summary["val_images"] = len(os.listdir(f"{out}/images/val"))

    yaml = f"""path: {out}
train: images/train
val: images/val
names:
""" + "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES)) + f"""
kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}
"""
    with open(f"{out}/data.yaml", "w") as f:
        f.write(yaml)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
