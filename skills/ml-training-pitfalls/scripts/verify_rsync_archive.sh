#!/bin/bash
# 归档零增量校验：复跑 rsync 并断言 transferred=0。
# 兼容 macOS openrsync（stats 字段 Number of files transferred）与 Linux rsync 3.x（Number of regular files transferred）。
# 用法: verify_rsync_archive.sh "描述|源|目标" ["描述|源|目标" ...]
# 示例: verify_rsync_archive.sh "runs|trainer:/home/ubuntu/isef/runs/|/Volumes/Expansion/姿态数据集/训练产物/"
set -u

RSYNC_OPTS="-a --stats --timeout=300 --partial"

extract_transferred() {
  local out="$1" n
  n=$(printf '%s\n' "$out" | sed -n 's/^Number of regular files transferred: //p')
  if [ -z "$n" ]; then
    n=$(printf '%s\n' "$out" | sed -n 's/^Number of files transferred: //p')
  fi
  printf '%s' "${n:-NA}"
}

if [ "$#" -lt 1 ]; then
  echo "用法: $0 \"描述|源|目标\" [\"描述|源|目标\" ...]" >&2
  echo "注意: 不要手写 -e 参数（双 -e 会静默失败），ssh 附加选项进 ~/.ssh/config" >&2
  exit 2
fi

fail=0
for spec in "$@"; do
  desc="${spec%%|*}"
  rest="${spec#*|}"
  src="${rest%%|*}"
  dst="${rest#*|}"
  [ -e "$dst" ] || { echo "!! [$desc] 目标不存在: $dst"; fail=1; continue; }
  out=$(rsync $RSYNC_OPTS "$src" "$dst" 2>&1)
  n=$(extract_transferred "$out")
  if [ "$n" = "0" ]; then
    echo "OK [$desc] 零增量校验通过"
  else
    echo "!! [$desc] 校验异常: transferred=${n}"
    fail=1
  fi
done

exit $fail
