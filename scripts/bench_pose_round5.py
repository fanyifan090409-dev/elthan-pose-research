"""round5-2 best.pt 基准对拍：CrowdPose 遮挡代理 + OCHuman test 干净遮挡集。

- 训练时 COCO val 指标已在 results.csv（best epoch 45 = mAP50-95(P)）
- 本脚本跑两个独立 val：CrowdPose val（2000 图）与 OCHuman test（2230 图）
- 单卡 GPU0 跑（GPU1 有 vLLM），不干扰线上服务
"""
from ultralytics import YOLO

CKPT = "/home/ubuntu/isef/runs/pose-round5-2/weights/best.pt"
CROWD_YAML = "/home/ubuntu/isef/scripts/crowdpose-val-data.yaml"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"

m = YOLO(CKPT)

# 1) CrowdPose val 遮挡代理（round4 对照：Pose mAP50-95 0.543）
r1 = m.val(
    data=CROWD_YAML,
    imgsz=640,
    batch=64,
    device=0,
    project="/home/ubuntu/isef/runs/bench",
    name="round5_crowdpose",
)

# 2) OCHuman test 干净遮挡基准（首次建立）
r2 = m.val(
    data=OCHUMAN_YAML,
    imgsz=640,
    batch=64,
    device=0,
    project="/home/ubuntu/isef/runs/bench",
    name="round5_ochuman",
)

print("CrowdPose val:", r1.box.map50, r1.box.map, "| pose:", r1.pose.map50, r1.pose.map)
print("OCHuman test:", r2.box.map50, r2.box.map, "| pose:", r2.pose.map50, r2.pose.map)
