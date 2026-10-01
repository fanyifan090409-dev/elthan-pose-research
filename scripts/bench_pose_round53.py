from ultralytics import YOLO

CKPT = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
CROWD_YAML = "/home/ubuntu/isef/scripts/crowdpose-val-data.yaml"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"

m = YOLO(CKPT)
r1 = m.val(data=CROWD_YAML, imgsz=640, batch=64, device=0,
           project="/home/ubuntu/isef/runs/bench", name="round53_crowdpose")
r2 = m.val(data=OCHUMAN_YAML, imgsz=640, batch=64, device=0,
           project="/home/ubuntu/isef/runs/bench", name="round53_ochuman")
