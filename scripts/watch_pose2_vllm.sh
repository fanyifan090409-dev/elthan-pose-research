#!/bin/bash
# watch_pose2_vllm.sh — pose-round2 训练结束后自动重启 vLLM（task #95）
LOG=/home/ubuntu/isef/logs/watch_pose2_vllm.log
echo "=== $(date '+%F %T') watch start (train_pose2 running) ===" >> "$LOG"
while tmux has-session -t train_pose2 2>/dev/null; do
  sleep 60
done
echo "=== $(date '+%F %T') train_pose2 ended, restarting vLLM ===" >> "$LOG"
tmux new-session -d -s vlm8 'bash /home/ubuntu/isef/scripts/serve_qwen3vl8b.sh 2>&1 | tee /home/ubuntu/isef/logs/vllm_serve_8b.log'
sleep 30
if tmux has-session -t vlm8 2>/dev/null; then
  echo "=== $(date '+%F %T') vlm8 tmux restarted ===" >> "$LOG"
else
  echo "=== $(date '+%F %T') ERROR vlm8 tmux not running ===" >> "$LOG"
fi
