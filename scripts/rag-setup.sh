#!/bin/bash
set -x
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda create -n rag python=3.12 -y || exit 1
conda activate rag
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip || exit 1
pip install fastapi "uvicorn[standard]" chromadb sentence-transformers huggingface_hub || exit 1
python - <<'PY'
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
p = snapshot_download("BAAI/bge-m3", local_dir="/home/ubuntu/isef/models/bge-m3")
print("MODEL_OK", p)
PY
python - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("/home/ubuntu/isef/models/bge-m3")
e = m.encode(["深蹲时膝盖内扣怎么办"], normalize_embeddings=True)
print("EMB_OK", e.shape)
PY
echo SETUP_DONE
