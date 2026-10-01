#!/usr/bin/env python3
"""b14 修复重跑后的单行外部评测（CrowdPose val + OCHuman test，device 0）。

用法: python bench_round6_b14.py
结果写 /home/ubuntu/isef/logs/bench-round6-b14fix.json
"""
import json

from ultralytics import YOLO

CROWD_YAML = "/home/ubuntu/isef/scripts/crowdpose-val-data.yaml"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"
W = "/home/ubuntu/isef/runs/pose-round6-b14/weights/best.pt"

m = YOLO(W)
r1 = m.val(data=CROWD_YAML, imgsz=640, batch=64, device=0,
           project="/home/ubuntu/isef/runs/bench", name="round6_b14fix_crowdpose")
r2 = m.val(data=OCHUMAN_YAML, imgsz=640, batch=64, device=0,
           project="/home/ubuntu/isef/runs/bench", name="round6_b14fix_ochuman")
out = {
    "b14fix": {
        "crowdpose_mAP50-95(P)": r1.results_dict.get("metrics/mAP50-95(P)"),
        "crowdpose_mAP50(P)": r1.results_dict.get("metrics/mAP50(P)"),
        "ochuman_mAP50-95(P)": r2.results_dict.get("metrics/mAP50-95(P)"),
        "ochuman_mAP50(P)": r2.results_dict.get("metrics/mAP50(P)"),
    }
}
with open("/home/ubuntu/isef/logs/bench-round6-b14fix.json", "w") as f:
    json.dump(out, f, indent=2)
print(out)
