#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate webui
export DATA_DIR=/home/ubuntu/isef/webui-data
mkdir -p $DATA_DIR
export OPENAI_API_BASE_URLS="http://127.0.0.1:8000/v1"
export OPENAI_API_KEYS="sk-elthan-vlm"
export WEBUI_NAME="Elthan 姿态矫正教练"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export ENABLE_OLLAMA_API=false
exec open-webui serve --host 0.0.0.0 --port 3001
