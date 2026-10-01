#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate py310
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen3-VL-8B-Instruct", local_dir="/home/ubuntu/isef/models/qwen3-vl-8b-instruct", max_workers=8)
print("WEIGHTS_DOWNLOAD_DONE", p)
PYEOF
