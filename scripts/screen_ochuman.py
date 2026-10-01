#!/usr/bin/env python3
"""OCHuman → YOLO-pose 转换 + 筛选（task #104）

数据口径（与 CrowdPose 同款筛选，质量>数量）：
- 官方划分：train=350（无 mask 子集，ochuman.json 原始 19 点格式）、
  val=2500、test=2231（COCO 17 点格式）。COCO json 图像 id 与原始 image_id
  不通用，划分用 file_name 对齐。
- 入训集 = train(350) + val(2500)，test(2231) 留作干净遮挡基准（论文可报）。
- 筛选规则（入训集）：bbox 高 ≥ 96px；每实例已标注点(v>0) ≥ 8；出界点降 v=0；
  无 keypoints 的 bbox-only 实例跳过。
- 19→17 映射：head/neck 丢弃；vis 映射 0(missing)→0、1(visible)→2、
  2(self_occluded)→1、3(others_occluded)→1。COCO json 的 vis 0/1/2 直通。
- eval 集（test）不筛选，全量转换，与 CrowdPose val 代理同口径。
- 输出：yolo/labels/train（入训）、yolo-test/labels（基准）；
  图片软链待分卷到齐后由 --link-images 补。
"""
import json
import os
import sys
from pathlib import Path

BASE = Path("/home/ubuntu/isef/datasets/ochuman")
OUT = BASE / "yolo"
OUT_TEST = BASE / "yolo-test"
MIN_BBOX_H = 96
MIN_LABELED = 8

# OCHuman 19 点序（0..18）→ COCO 17 点索引；head(12)/neck(13) 丢弃
MAP19_17 = {16: 0, 18: 1, 17: 2, 15: 3, 14: 4, 3: 5, 0: 6, 4: 7, 1: 8,
            5: 9, 2: 10, 9: 11, 6: 12, 10: 13, 7: 14, 11: 15, 8: 16}
# 0=missing, 1=visible, 2=self_occluded, 3=others_occluded
VIS_MAP = {0: 0, 1: 2, 2: 1, 3: 1}


def load_json(p):
    with open(p) as f:
        return json.load(f)


def coco_fn_set(coco):
    return {img["file_name"] for img in coco["images"]}


def convert_train_ann(ann, w, h):
    """19 点原始标注 → 17 点 YOLO 行；不通过筛选返回 None。"""
    kps = ann.get("keypoints")
    if not kps or len(kps) < 19 * 3:
        return None
    x, y, bw, bh = ann["bbox"]
    if bh < MIN_BBOX_H:
        return None
    k17 = [0.0] * (17 * 3)
    labeled = 0
    for i19 in range(19):
        px, py, v = kps[3 * i19:3 * i19 + 3]
        idx = MAP19_17.get(i19)
        if idx is None:
            continue
        if v > 0:
            if px < 0 or py < 0 or px >= w or py >= h:
                v = 0  # 出界点不可用
            else:
                labeled += 1
        k17[3 * idx] = px / w
        k17[3 * idx + 1] = py / h
        k17[3 * idx + 2] = float(VIS_MAP.get(int(v), 0))
    if labeled < MIN_LABELED:
        return None
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw, nh = bw / w, bh / h
    vals = [0, cx, cy, nw, nh] + k17
    return " ".join(f"{t:.6f}" if isinstance(t, float) else str(t) for t in vals)


def convert_coco_ann(ann, w, h):
    """官方 COCO 17 点标注 → YOLO 行（vis 0/1/2 直通）。"""
    kps = ann.get("keypoints")
    if not kps or len(kps) < 17 * 3:
        return None
    x, y, bw, bh = ann["bbox"]
    if bh < MIN_BBOX_H:
        return None
    labeled = 0
    out_k = []
    for i in range(17):
        px, py, v = kps[3 * i:3 * i + 3]
        if v > 0:
            if px < 0 or py < 0 or px >= w or py >= h:
                v = 0
            else:
                labeled += 1
        out_k += [px / w, py / h, float(int(v))]
    if labeled < MIN_LABELED:
        return None
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    vals = [0, cx, cy, bw / w, bh / h] + out_k
    return " ".join(f"{t:.6f}" if isinstance(t, float) else str(t) for t in vals)


def convert_eval_ann(ann, w, h):
    """eval 集不筛选，仅出界点降 v=0。"""
    kps = ann.get("keypoints")
    if not kps or len(kps) < 17 * 3:
        return None
    x, y, bw, bh = ann["bbox"]
    out_k = []
    for i in range(17):
        px, py, v = kps[3 * i:3 * i + 3]
        if v > 0 and (px < 0 or py < 0 or px >= w or py >= h):
            v = 0
        out_k += [px / w, py / h, float(int(v))]
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    vals = [0, cx, cy, bw / w, bh / h] + out_k
    return " ".join(f"{t:.6f}" if isinstance(t, float) else str(t) for t in vals)


def main():
    link_images = "--link-images" in sys.argv
    raw = load_json(BASE / "ochuman.json")
    val_coco = load_json(BASE / "ochuman_coco_format_val_range_0.00_1.00.json")
    test_coco = load_json(BASE / "ochuman_coco_format_test_range_0.00_1.00.json")

    val_fn = coco_fn_set(val_coco)
    test_fn = coco_fn_set(test_coco)
    print(f"val_fn={len(val_fn)} test_fn={len(test_fn)} 交集={len(val_fn & test_fn)}")
    assert not (val_fn & test_fn), "val/test file_name 有交集，划分异常"

    all_fn = {img["file_name"] for img in raw["images"]}
    train_fn = all_fn - val_fn - test_fn
    print(f"ochuman 总图像={len(all_fn)}  train={len(train_fn)}")

    # 尺寸索引：file_name → (w, h)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in raw["images"]}
    for coco, tag in ((val_coco, "val"), (test_coco, "test")):
        for img in coco["images"]:
            dims.setdefault(img["file_name"], (img["width"], img["height"]))

    (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_TEST / "labels").mkdir(parents=True, exist_ok=True)
    (OUT_TEST / "images").mkdir(parents=True, exist_ok=True)

    src_images = BASE / "images"  # 分卷解压合并后

    stats = {"train_img": 0, "train_inst": 0, "train_drop_h": 0,
             "train_drop_k": 0, "train_drop_none": 0, "test_img": 0, "test_inst": 0}

    def write_lines(tag, fn, lines):
        stem = os.path.splitext(fn)[0]
        (OUT / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        if link_images:
            src = src_images / fn
            if src.exists():
                os.symlink(src, OUT / "images" / "train" / fn)

    # ---- train（350 原始格式）----
    for img in raw["images"]:
        fn = img["file_name"]
        if fn not in train_fn:
            continue
        w, h = img["width"], img["height"]
        lines = []
        for ann in img["annotations"]:
            line = convert_train_ann(ann, w, h)
            if line is None:
                kps = ann.get("keypoints")
                if not kps or len(kps) < 19 * 3:
                    stats["train_drop_none"] += 1
                elif ann["bbox"][3] < MIN_BBOX_H:
                    stats["train_drop_h"] += 1
                else:
                    stats["train_drop_k"] += 1
                continue
            lines.append(line)
        if lines:
            stats["train_img"] += 1
            stats["train_inst"] += len(lines)
            write_lines("train", fn, lines)

    # ---- val（2500 官方 COCO，入训）----
    by_fn = {}
    for ann in val_coco["annotations"]:
        by_fn.setdefault(ann["image_id"], []).append(ann)
    id2fn = {img["id"]: img["file_name"] for img in val_coco["images"]}
    for img in val_coco["images"]:
        fn = img["file_name"]
        w, h = img["width"], img["height"]
        lines = []
        for ann in by_fn.get(img["id"], []):
            line = convert_coco_ann(ann, w, h)
            if line is None:
                if ann["bbox"][3] < MIN_BBOX_H:
                    stats["train_drop_h"] += 1
                else:
                    stats["train_drop_k"] += 1
                continue
            lines.append(line)
        if lines:
            stats["train_img"] += 1
            stats["train_inst"] += len(lines)
            write_lines("train", fn, lines)

    # ---- test（2231，干净基准，不筛选）----
    by_fn_t = {}
    for ann in test_coco["annotations"]:
        by_fn_t.setdefault(ann["image_id"], []).append(ann)
    for img in test_coco["images"]:
        fn = img["file_name"]
        w, h = img["width"], img["height"]
        lines = []
        for ann in by_fn_t.get(img["id"], []):
            line = convert_eval_ann(ann, w, h)
            if line:
                lines.append(line)
        if lines:
            stats["test_img"] += 1
            stats["test_inst"] += len(lines)
            stem = os.path.splitext(fn)[0]
            (OUT_TEST / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            if link_images:
                src = src_images / fn
                if src.exists():
                    os.symlink(src, OUT_TEST / "images" / fn)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
