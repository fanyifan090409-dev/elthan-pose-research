#!/bin/bash
# Elthan MLflow server launcher (allows Tailscale + LAN + loopback hosts)
export MLFLOW_SERVER_ALLOWED_HOSTS="localhost:*,127.0.0.1:*,192.168.*.*:*,100.64.*.*:*,100.86.*.*:*,ubuntu-20-1:*,ubuntu-20-1.local:*"
pkill -f "mlflow server" 2>/dev/null
sleep 2
nohup /home/ubuntu/Miniconda3/envs/py310/bin/mlflow server \
  --backend-store-uri sqlite:////home/ubuntu/isef/mlflow/mlflow.db \
  --default-artifact-root /home/ubuntu/isef/mlflow/artifacts \
  --host 0.0.0.0 --port 5000 \
  >> /home/ubuntu/isef/mlflow/server.log 2>&1 &
echo "mlflow pid=$!"
