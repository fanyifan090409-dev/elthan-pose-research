#!/usr/bin/env python3
"""Convert ultralytics results.csv -> TensorBoard events. Usage: csv2tb.py <run_dir> [<run_dir2> ...]"""
import csv, os, sys
from torch.utils.tensorboard import SummaryWriter

for run_dir in sys.argv[1:]:
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"skip {run_dir}: no results.csv"); continue
    w = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        epoch = int(float(row.get("epoch", i + 1)))
        for k, v in row.items():
            k = k.strip()
            if k in ("epoch", "time"): continue
            try: w.add_scalar(k, float(v), epoch)
            except ValueError: pass
    w.close()
    print(f"TB events written for {run_dir} ({len(rows)} epochs)")
