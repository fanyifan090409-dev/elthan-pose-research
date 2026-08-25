#!/bin/bash
# ISEF TensorBoard launcher
export PATH=/home/ubuntu/Miniconda3/envs/py310/bin:$PATH
pkill -f "tensorboard --logdir /home/ubuntu/isef/runs" 2>/dev/null
sleep 1
nohup /home/ubuntu/Miniconda3/envs/py310/bin/tensorboard --logdir /home/ubuntu/isef/runs --host 0.0.0.0 --port 6006 > /home/ubuntu/isef/mlflow/tensorboard.log 2>&1 &
echo "tensorboard pid=$!"
