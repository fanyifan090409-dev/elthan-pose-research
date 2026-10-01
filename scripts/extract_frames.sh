#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 3 ]; then
  echo "用法: extract_frames.sh <视频目录> <输出目录> <fps>"
  exit 1
fi
VDIR="$1"
OUT="$2"
FPS="$3"
mkdir -p "$OUT"
find "$VDIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | sort | while read -r v; do
  name=$(basename "$v")
  stem="${name%.*}"
  dest="$OUT/$stem"
  mkdir -p "$dest"
  echo "[$(date +%H:%M:%S)] 抽帧 $name -> $dest (fps=$FPS)"
  ffmpeg -hide_banner -loglevel error -i "$v" -vf "fps=$FPS" -q:v 2 "$dest/frame_%06d.jpg"
done
echo "完成。总帧数: $(find "$OUT" -name '*.jpg' | wc -l)"
