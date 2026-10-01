#!/bin/bash
# round6 消融顺序执行器（b15 单独跑：T 需按 b14 的 loss 量级定）
# 用法: bash round6_run.sh [起始行名]   # 默认从 baseline 开始
set -u
cd /home/ubuntu/isef
ROWS="baseline b2 b12 b13 b14 b16s b16l"
START="${1:-baseline}"
FOUND=0
for row in $ROWS; do
  if [ "$row" = "$START" ]; then FOUND=1; fi
  [ "$FOUND" = "1" ] || continue
  echo "=== round6 row $row start: $(date '+%F %T') ==="
  /home/ubuntu/Miniconda3/envs/py310/bin/python /home/ubuntu/isef/scripts/round6_ablation_row.py "$row"
  RC=$?
  echo "=== round6 row $row exit=$RC: $(date '+%F %T') ==="
  if [ "$RC" != "0" ]; then
    echo "=== row $row 失败，终止后续行，保留现场 ==="
    exit "$RC"
  fi
done
git -C /home/ubuntu/isef/ultralytics-dev checkout master
echo "=== round6 顺序矩阵全部结束，repo 已切回 master ==="
