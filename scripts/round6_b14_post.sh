#!/bin/bash
# b14 修复重跑链：训练 30 epoch（同矩阵配方）→ 成功则单行外部评测
set -u
cd /home/ubuntu/isef
/home/ubuntu/Miniconda3/envs/py310/bin/python /home/ubuntu/isef/scripts/round6_ablation_row.py b14
RC=$?
echo "=== row b14 rerun exit=$RC: $(date '+%F %T') ==="
if [ "$RC" != "0" ]; then
  echo "=== b14 重跑失败，跳过评测 ==="
  exit "$RC"
fi
/home/ubuntu/Miniconda3/envs/py310/bin/python /home/ubuntu/isef/scripts/bench_round6_b14.py
echo "=== b14 rerun + eval done: $(date '+%F %T') ==="
