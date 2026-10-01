from ultralytics import YOLO

CKPT = "/home/ubuntu/isef/runs/pose-round3/weights/best.pt"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"

m = YOLO(CKPT)
r = m.val(data=OCHUMAN_YAML, imgsz=640, batch=64, device=0,
          project="/home/ubuntu/isef/runs/bench", name="round3_ochuman")
print("ROUND3_OCHUMAN_DONE", r.box.map50, r.box.map, r.results_dict.get("metrics/mAP50-95(P)"))
