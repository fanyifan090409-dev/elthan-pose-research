#!/bin/bash
# FYF 开机自恢复：已在跑的跳过，没跑的拉起。crontab @reboot 触发，也可手动跑。
S=/home/ubuntu/isef/scripts
L=/home/ubuntu/isef/logs
mkdir -p $L
tmux has-session -t vlm8   2>/dev/null || tmux new-session -d -s vlm8   "bash $S/serve_qwen3vl8b.sh 2>&1 | tee -a $L/vlm8.log"
tmux has-session -t rag    2>/dev/null || tmux new-session -d -s rag    "bash $S/serve_rag.sh 2>&1 | tee -a $L/rag.log"
tmux has-session -t agent  2>/dev/null || tmux new-session -d -s agent  "bash $S/serve_agent.sh 2>&1 | tee -a $L/agent_serve.log"
tmux has-session -t webui  2>/dev/null || tmux new-session -d -s webui  "bash $S/serve_webui.sh 2>&1 | tee -a $L/webui.log"
tmux has-session -t dualui 2>/dev/null || tmux new-session -d -s dualui "python3 -m http.server 3002 --directory /home/ubuntu/isef/webui-theme"
# MLflow：5000 端口已被占用则跳过（兼容裸跑实例），否则拉起 tmux 会话
if ss -tln 2>/dev/null | grep -q ':5000 '; then
  echo "[boot_all] mlflow 已在运行(:5000)，跳过"
else
  tmux has-session -t mlflow 2>/dev/null || tmux new-session -d -s mlflow "bash $S/serve_mlflow.sh 2>&1 | tee -a $L/mlflow.log"
  echo "[boot_all] mlflow tmux 会话已拉起"
fi
# Gitea：snap 托管服务(gitea.web)已随开机自启；若未起，用免密 sudo（/etc/sudoers.d/90-gitea-nopass）自动拉起
if ss -tln 2>/dev/null | grep -q ':3000 '; then
  echo "[boot_all] gitea 已在运行(:3000)"
else
  sudo -n /usr/bin/snap start gitea >> $L/boot_all.log 2>&1
  sleep 15
  if ss -tln 2>/dev/null | grep -q ':3000 '; then
    echo "[boot_all] gitea 已自动拉起(:3000)"
  else
    echo "[boot_all] 警告: gitea 自动拉起失败，需人工执行 sudo snap start gitea"
  fi
fi
# RAG 起来后刷新实验记录本向量集合（与 30 分钟 cron 共用一把锁，不会撞车）
nohup bash -c "sleep 40; flock -n /tmp/ingest_notebook.lock /usr/bin/python3 $S/ingest_notebook.py" >> $L/ingest_notebook.log 2>&1 &
echo "[boot_all] $(date '+%F %T') sessions: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
