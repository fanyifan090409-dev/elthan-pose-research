#!/bin/bash
# finish_round4_record.sh — pose-round4 训练会话结束后抓最终指标块（训练机侧保险，Mac 挂了也不丢）
SESS=train_pose4
LOG=/home/ubuntu/isef/logs/pose-round4-train.log
OUT=/home/ubuntu/isef/logs/round4-final-block.txt
while tmux has-session -t "$SESS" 2>/dev/null; do sleep 60; done
sleep 5
tr '\r' '\n' < "$LOG" | awk '
  /Results saved to/{last=NR}
  {lines[NR]=$0}
  END{
    if (last==0) { print "NO_RESULTS_SAVED_FOUND"; exit }
    for (i=last-30; i<=last; i++) if (lines[i]!="") print lines[i]
  }' > "$OUT"
echo "=== $(date '+%F %T') round4 final block extracted ===" >> "$LOG"
