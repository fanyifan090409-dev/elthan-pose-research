import json
import os
from pathlib import Path

BASE = Path("/home/ubuntu/isef/data/raw/web-batch1")

removed = 0
for j in sorted(BASE.joinpath("videos").rglob("*.info.json")):
    try:
        info = json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        continue
    url = info.get("webpage_url") or info.get("original_url") or ""
    if not url.startswith("http"):
        j.unlink()
        print(f"removed {j.relative_to(BASE)}")
        removed += 1

print(f"removed: {removed}")
print("mp4:", sum(1 for _ in BASE.joinpath("videos").rglob("*.mp4")))
print("info.json:", sum(1 for _ in BASE.joinpath("videos").rglob("*.info.json")))
