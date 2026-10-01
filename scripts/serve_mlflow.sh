#!/bin/bash
# MLflow 追踪服务器（tmux 托管版）：sqlite 后端 + 工件目录
export MLFLOW_SERVER_ALLOWED_HOSTS="localhost:*,127.0.0.1:*,192.168.*.*:*,100.64.*.*:*,100.86.*.*:*,ubuntu-20-1:*,ubuntu-20-1.local:*"
cd /home/ubuntu/isef/mlflow
exec /home/ubuntu/Miniconda3/envs/py310/bin/mlflow server \
  --backend-store-uri sqlite:////home/ubuntu/isef/mlflow/mlflow.db \
  --default-artifact-root /home/ubuntu/isef/mlflow/artifacts \
  --host 0.0.0.0 --port 5000
