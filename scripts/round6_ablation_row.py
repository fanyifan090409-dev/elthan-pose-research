#!/usr/bin/env python3
"""round6 消融单行执行器：git 切到指定 tag → 按 hyp 开关训练 30 epoch。

用法: python round6_ablation_row.py <row>
rows: baseline(原版) b2(+B2') b12(+骨长) b13(+对称) b14(+角度)
      b15(+自动配权, 温度取环境变量 R6_T, 默认 1.0) b16s(手工批统计) b16l(可学习先验)

配方与 round5-2 完全一致（同配方锚）：batch=128 双卡 DDP（64/卡）、lr0=0.005、
seed 42、patience 30、close_mosaic 10、init=pose-round5-2/last.pt、data=pose-round5。
"""
import os
import subprocess
import sys

REPO = "/home/ubuntu/isef/ultralytics-dev"
INIT = "/home/ubuntu/isef/runs/pose-round5-2/weights/last.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round5/data.yaml"

ROWS = {
    # tag 决定代码版本：baseline 用 pristine round6-base，b2 用仅含 B2' 的 tag，其余用 master（开关默认全关）
    "baseline": {"tag": "round6-base", "hyp": {}},
    "b2": {"tag": "b2-vis-weight", "hyp": {}},
    "b12": {"tag": "master", "hyp": {"bone": 1.0}},
    "b13": {"tag": "master", "hyp": {"bone": 1.0, "symmetry": 1.0}},
    "b14": {"tag": "master", "hyp": {"bone": 1.0, "symmetry": 1.0, "angle_prior": 1.0}},
    # B1.5：三项全开 + 自动配权（手工权重退化为开门）；T 是量级参数，默认 1.0，可按行 5 的 loss 量级覆盖
    "b15": {
        "tag": "master",
        "hyp": {"bone": 1.0, "symmetry": 1.0, "angle_prior": 1.0,
                "anatomy_uncertainty": float(os.environ.get("R6_T", "1.0"))},
    },
    "b16s": {"tag": "master", "hyp": {"bone_stats": 1.0}},
    "b16l": {"tag": "master", "hyp": {"bone_prior": 1.0}},
}

if len(sys.argv) != 2 or sys.argv[1] not in ROWS:
    raise SystemExit(f"用法: python round6_ablation_row.py <{'|'.join(ROWS)}>")
row = sys.argv[1]
cfg = ROWS[row]

print(f"=== round6 row {row}: checkout {cfg['tag']} hyp={cfg['hyp']} ===", flush=True)
subprocess.run(["git", "-C", REPO, "checkout", cfg["tag"]], check=True)

from ultralytics import YOLO  # noqa: E402  在 git checkout 之后再 import，editable 安装解析当前工作区代码

model = YOLO(INIT)
model.train(
    data=DATA,
    epochs=30,
    batch=128,
    imgsz=640,
    device="0,1",
    lr0=0.005,
    patience=30,
    close_mosaic=10,
    seed=42,
    workers=8,
    plots=True,
    project="/home/ubuntu/isef/runs",
    name=f"pose-round6-{row}",
    **cfg["hyp"],
)
print(f"=== round6 row {row} done ===", flush=True)
