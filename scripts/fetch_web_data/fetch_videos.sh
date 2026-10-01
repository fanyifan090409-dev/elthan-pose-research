#!/bin/bash
# 第一实验组素材下载：B站搜索三大动作教学视频，video+audio 合成 mp4，720p 封顶
set -u
BASE="${1:-/home/ubuntu/isef/data/raw/web-batch1}"
YTDLP="/home/ubuntu/Miniconda3/envs/py310/bin/yt-dlp"
FFMPEG="$(/home/ubuntu/Miniconda3/bin/conda run -n py310 python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
MAX_PER_ACTION="${MAX_PER_ACTION:-10}"

ACTIONS="squat:深蹲 标准动作教学|徒手深蹲 教学|深蹲 错误示范
pushup:俯卧撑 标准动作|俯卧撑 教学|俯卧撑 错误姿势
deadlift:硬拉 教学|硬拉 标准动作|硬拉 错误示范"

mkdir -p "$BASE/videos"
LOG="$BASE/fetch.log"
: > "$LOG"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  action="${line%%:*}"
  kws="${line#*:}"
  dir="$BASE/videos/$action"
  mkdir -p "$dir"
  IFS='|' read -ra KW <<< "$kws"
  for kw in "${KW[@]}"; do
    got=$(ls "$dir"/*.mp4 2>/dev/null | wc -l)
    [ "$got" -ge "$MAX_PER_ACTION" ] && break
    "$YTDLP" --ffmpeg-location "$FFMPEG" --no-playlist --no-warnings \
      --match-filter "duration >= 30 & duration <= 900" \
      -f "bestvideo[height<=720]+bestaudio/bestvideo[height<=720]/best[height<=720]/best" \
      --merge-output-format mp4 \
      --download-archive "$BASE/.archive" \
      --write-info-json \
      --retries 5 --fragment-retries 5 -N 4 \
      -o "$dir/%(id)s_%(title)s.%(ext)s" \
      "bilisearch10:$kw" >>"$LOG" 2>&1 || true
  done
done <<< "$ACTIONS"

echo "videos: $(find "$BASE/videos" -name '*.mp4' | wc -l)"
echo "failures: $(grep -c ERROR "$LOG" 2>/dev/null || echo 0)"
