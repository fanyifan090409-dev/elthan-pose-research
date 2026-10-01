import json
import sys
from ultralytics import YOLO

CROWD_YAML = "/home/ubuntu/isef/scripts/crowdpose-val-data.yaml"
OCHUMAN_YAML = "/home/ubuntu/isef/datasets/ochuman/yolo-test/data.yaml"

ALL = {
    "baseline": "/home/ubuntu/isef/runs/pose-round6-baseline/weights/best.pt",
    "b2": "/home/ubuntu/isef/runs/pose-round6-b2/weights/best.pt",
    "b12": "/home/ubuntu/isef/runs/pose-round6-b12/weights/best.pt",
    "b13": "/home/ubuntu/isef/runs/pose-round6-b13/weights/best.pt",
    "b14": "/home/ubuntu/isef/runs/pose-round6-b14/weights/best.pt",
    "b16s": "/home/ubuntu/isef/runs/pose-round6-b16s/weights/best.pt",
    "b16l": "/home/ubuntu/isef/runs/pose-round6-b16l/weights/best.pt",
}
SUBSETS = {
    "a": ["baseline", "b2", "b12", "b13"],
    "b": ["b14", "b16s", "b16l"],
}

device = int(sys.argv[1])
batch = int(sys.argv[2])
subset = sys.argv[3]
out = sys.argv[4]

results = {}
for tag in SUBSETS[subset]:
    m = YOLO(ALL[tag])
    r1 = m.val(data=CROWD_YAML, imgsz=640, batch=batch, device=device,
               project="/home/ubuntu/isef/runs/bench", name=f"round6_{tag}_crowdpose")
    r2 = m.val(data=OCHUMAN_YAML, imgsz=640, batch=batch, device=device,
               project="/home/ubuntu/isef/runs/bench", name=f"round6_{tag}_ochuman")
    results[tag] = {
        "crowdpose_mAP50-95(P)": r1.results_dict.get("metrics/mAP50-95(P)"),
        "crowdpose_mAP50(P)": r1.results_dict.get("metrics/mAP50(P)"),
        "ochuman_mAP50-95(P)": r2.results_dict.get("metrics/mAP50-95(P)"),
        "ochuman_mAP50(P)": r2.results_dict.get("metrics/mAP50(P)"),
    }
    print(f"R6_{tag}: {results[tag]}", flush=True)

with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"SUBSET_{subset}_DONE -> {out}", flush=True)
