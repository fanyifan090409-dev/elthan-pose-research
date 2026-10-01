import json
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/ubuntu/isef/data/raw/web-batch1")
VIDEOS = BASE / "videos"
ARCHIVE = BASE / ".archive"
REFERER = "https://www.bilibili.com/"
OUT = BASE / "retry3.log"

media_suffixes = {".mp4", ".mkv", ".flv", ".webm"}

missing = []
for j in sorted(VIDEOS.rglob("*.info.json")):
    stem = j.name[: -len(".info.json")]
    media = [f for f in j.parent.glob(stem + ".*")
             if f.suffix.lower() in media_suffixes]
    if not media:
        try:
            info = json.loads(j.read_text(encoding="utf-8"))
            url = info.get("webpage_url") or info.get("original_url")
            if url and url.startswith("http"):
                missing.append((str(j), url))
            else:
                print(f"skip no-url: {j}")
        except Exception as e:
            print(f"skip {j}: {e}")

print(f"missing media: {len(missing)}")
if not missing:
    sys.exit(0)

log = open(OUT, "a", encoding="utf-8")
for idx, (jpath, url) in enumerate(missing, 1):
    print(f"[{idx}/{len(missing)}] {jpath}")
    cmd = [
        "/home/ubuntu/Miniconda3/envs/py310/bin/yt-dlp",
        "--download-archive", str(ARCHIVE),
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--add-headers", f"Referer:{REFERER}",
        "--sleep-requests", "5",
        "--output", str(jpath)[: -len(".info.json")] + ".%(ext)s",
        url,
    ]
    log.write("\n" + "=" * 80 + "\n")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"  exit={r.returncode}")
log.close()
