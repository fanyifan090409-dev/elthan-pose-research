#!/usr/bin/env bash
PROJ=/home/ubuntu/isef
MANIFEST=$PROJ/datasets/manifest.tsv
STATUS=$PROJ/datasets/status.tsv

printf '%-20s %-12s %12s %14s %s\n' 数据集 状态 已下载 期望字节 说明
while IFS=$'\t' read -r name url dir fname expected conns; do
  case "$name" in ''|\#*) continue ;; esac
  path="$PROJ/datasets/$dir/$fname"
  sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
  st=$(awk -F'\t' -v n="$name" '$2==n{print $1}' "$STATUS" 2>/dev/null)
  note=$(awk -F'\t' -v n="$name" '$2==n{print $3}' "$STATUS" 2>/dev/null)
  printf '%-20s %-12s %12s %14s %s\n' "$name" "${st:-PENDING}" "$sz" "$expected" "$note"
done < "$MANIFEST"
