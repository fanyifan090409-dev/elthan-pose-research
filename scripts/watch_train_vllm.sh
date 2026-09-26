#!/bin/bash
# watch_train_vllm.sh — 指定 tmux 训练会话结束后自动重启 vLLM
# 用法: watch_train_vllm.sh <训练tmux会话名>
SESS="${1:-train_pose3}"
LOG=/home/ubuntu/isef/logs/watch_${SESS}_vllm.log
echo "=== $(date '+%F %T') watch start ($SESS running) ===" >> "$LOG"
while tmux has-session -t "$SESS" 2>/dev/null; do
  sleep 60
done
echo "=== $(date '+%F %T') $SESS ended, restarting vLLM ===" >> "$LOG"
tmux new-session -d -s vlm8 'bash /home/ubuntu/isef/scripts/serve_qwen3vl8b.sh 2>&1 | tee /home/ubuntu/isef/logs/vllm_serve_8b.log'
sleep 30
if tmux has-session -t vlm8 2>/dev/null; then
  echo "=== $(date '+%F %T') vlm8 tmux restarted ===" >> "$LOG"
else
  echo "=== $(date '+%F %T') ERROR vlm8 tmux not running ===" >> "$LOG"
fi
