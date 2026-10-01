#!/usr/bin/env bash
set -u
exec 9>/tmp/dl_datasets.lock
flock -n 9 || { echo "another runner active, exit"; exit 0; }

PROJ=/home/ubuntu/isef
MANIFEST=$PROJ/datasets/manifest.tsv
STATUS=$PROJ/datasets/status.tsv
LOG=$PROJ/logs/datasets-auto.log
DL_DIR=$PROJ/datasets

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >> "$LOG"; }
mkdir -p "$DL_DIR"
touch "$STATUS" "$LOG"

done_names=$(awk -F'\t' '$1=="DONE"{print $2}' "$STATUS")

while IFS=$'\t' read -r name url dir fname expected conns; do
  case "$name" in ''|\#*) continue ;; esac
  if grep -qx "$name" <<<"$done_names"; then continue; fi

  dest="$DL_DIR/$dir"
  mkdir -p "$dest"
  path="$dest/$fname"

  mark() {
    sed -i "\#^[^\t]*\t$name\t#d" "$STATUS"
    printf '%s\t%s\t%s\n' "$1" "$name" "$2" >> "$STATUS"
  }

  mark DOWNLOADING "$(date '+%H:%M')"
  log "开始下载 $name -> $path (conns=$conns)"

  aria2c -x"$conns" -s16 -c --max-tries=8 --retry-wait=60 --timeout=60 \
    --console-log-level=warn --summary-interval=120 \
    --dir="$dest" --out="$fname" "$url" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    mark FAILED "aria2 rc=$rc $(date '+%H:%M')"
    log "$name 下载失败 rc=$rc"
    continue
  fi

  sz=$(stat -c%s "$path")
  if [ -n "$expected" ] && [ "$expected" != "0" ] && [ "$sz" -ne "$expected" ]; then
    mark FAILED "size $sz != expected $expected"
    log "$name 大小不符 $sz/$expected"
    continue
  fi

  log "$name 下载完成 ($sz bytes)，完整性校验中..."
  case "$fname" in
    *.zip) unzip -tq "$path" >> "$LOG" 2>&1 ;;
    *.tar.gz|*.tgz) tar -tzf "$path" > /dev/null 2>>"$LOG" ;;
    *.tar) tar -tf "$path" > /dev/null 2>>"$LOG" ;;
  esac
  if [ $? -ne 0 ]; then
    mark FAILED "校验失败 $(date '+%H:%M')"
    log "$name 完整性校验失败"
    continue
  fi

  mark DONE "$sz $(date '+%H:%M')"
  log "$name 校验通过 DONE"
done < "$MANIFEST"

log "本轮扫描结束"
