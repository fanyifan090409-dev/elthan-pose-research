from ultralytics import YOLO

CROWD_YAML = "/home/ubuntu/isef/scripts/crowdpose-val-data.yaml"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"

CKPTS = {
    "nocrowd": "/home/ubuntu/isef/runs/pose-round5-abl-nocrowd-2/weights/best.pt",
    "noochu": "/home/ubuntu/isef/runs/pose-round5-abl-noochu/weights/best.pt",
    "nootp": "/home/ubuntu/isef/runs/pose-round5-abl-nootp/weights/best.pt",
}

for tag, ckpt in CKPTS.items():
    m = YOLO(ckpt)
    r1 = m.val(data=CROWD_YAML, imgsz=640, batch=64, device=0,
               project="/home/ubuntu/isef/runs/bench", name=f"abl_{tag}_crowdpose")
    r2 = m.val(data=OCHUMAN_YAML, imgsz=640, batch=64, device=0,
               project="/home/ubuntu/isef/runs/bench", name=f"abl_{tag}_ochuman")
    print(f"ABL_{tag}: crowd_pose_mAP50-95={r1.results_dict.get('metrics/mAP50-95(P)')} "
          f"ochuman_pose_mAP50-95={r2.results_dict.get('metrics/mAP50-95(P)')}")
