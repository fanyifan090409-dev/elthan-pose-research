#!/usr/bin/env python3
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/isef/data/raw/web-batch1")

rows = []
for j in sorted((BASE / "videos").rglob("*.info.json")):
    try:
        d = json.loads(j.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    stem = j.name[: -len(".info.json")]
    media = next((f for f in j.parent.glob(stem + ".*")
                  if f.name != j.name and f.suffix.lower() in (".mp4", ".mkv", ".flv", ".webm")), None)
    if media is None:
        continue
    frames = sum(1 for _ in (BASE / "images" / j.parent.name / stem).glob("*.jpg"))
    rows.append([
        j.parent.name,
        d.get("id", ""),
        d.get("title", ""),
        d.get("uploader", ""),
        d.get("webpage_url", ""),
        d.get("duration", ""),
        media.stat().st_size,
        media.name,
        frames,
        datetime.now().strftime("%F %T"),
    ])

out = BASE / "manifest.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["action", "id", "title", "uploader", "url", "duration_s",
                "size_bytes", "media_file", "frames", "downloaded_at"])
    w.writerows(rows)
print(f"manifest: {len(rows)} entries -> {out}")
