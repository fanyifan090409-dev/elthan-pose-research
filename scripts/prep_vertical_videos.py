#!/usr/bin/env python3
"""prep_vertical_videos.py — 竖屏居家视频 → 抽帧 + 当前最优模型自动预标注（task #98）

流程：ffmpeg 按 fps 抽帧 → YOLO pose 预测 → 转 YOLO 标注（person 类 + 17 关键点）
输出：datasets/vertical-v1/{images,labels}/{raw,auto}/ + summary.json
人工校正在 CVAT 做，本脚本只出预标注。
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

from ultralytics import YOLO

KPT = 17


def extract_frames(video, out_dir, fps):
    name = Path(video).stem
    dst = Path(out_dir) / name
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
         str(dst / f"{name}_%05d.jpg")],
        check=True, capture_output=True)
    return sorted(dst.glob("*.jpg"))


def auto_label(model, frames, out_lbl, conf):
    out_lbl = Path(out_lbl)
    out_lbl.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=[str(f) for f in frames], conf=conf, classes=[0],
        save=False, verbose=False, device=0)
    summary = []
    for res, f in zip(results, frames):
        n_person = 0
        lines = []
        if res.keypoints is not None and len(res.boxes) > 0:
            for box, kpts in zip(res.boxes, res.keypoints.data):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                W, H = res.orig_shape[1], res.orig_shape[0]
                cx = (x1 + w / 2) / W
                cy = (y1 + h / 2) / H
                kstr = []
                for kx, ky, kc in kpts.tolist():
                    v = 2 if kc >= conf else 0
                    kstr.append(f"{kx / W:.6f} {ky / H:.6f} {v}")
                lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f} " + " ".join(kstr))
                n_person += 1
        if lines:
            (out_lbl / (f.stem + ".txt")).write_text("\n".join(lines) + "\n")
        summary.append({"frame": f.name, "persons": n_person})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="视频文件或目录（目录则取 *.mp4/*.mov）")
    ap.add_argument("--model", default="/home/ubuntu/isef/runs/pose-round2-2/weights/best.pt")
    ap.add_argument("--out", default="/home/ubuntu/isef/datasets/vertical-v1")
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    src = Path(args.videos)
    if src.is_dir():
        videos = sorted(list(src.glob("*.mp4")) + list(src.glob("*.mov")))
    else:
        videos = [src]
    if not videos:
        raise SystemExit("未找到视频文件")

    model = YOLO(args.model)
    img_dir = Path(args.out) / "images"
    lbl_dir = Path(args.out) / "labels"
    total = {"videos": 0, "frames": 0, "frames_with_person": 0, "persons": 0}
    per_video = {}
    for v in videos:
        print(f"=== {v.name} ===")
        frames = extract_frames(v, img_dir, args.fps)
        s = auto_label(model, frames, lbl_dir, args.conf)
        n_person_frames = sum(1 for x in s if x["persons"] > 0)
        n_persons = sum(x["persons"] for x in s)
        per_video[v.name] = {"frames": len(frames), "frames_with_person": n_person_frames, "persons": n_persons}
        total["videos"] += 1
        total["frames"] += len(frames)
        total["frames_with_person"] += n_person_frames
        total["persons"] += n_persons
        print(f"  frames={len(frames)} with_person={n_person_frames} persons={n_persons}")

    out = {"summary": total, "per_video": per_video,
           "model": args.model, "fps": args.fps, "conf": args.conf}
    (Path(args.out) / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("=== SUMMARY ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
