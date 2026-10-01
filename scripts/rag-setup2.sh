#!/bin/bash
set -x
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate rag
python - <<'PY'
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
p = snapshot_download("BAAI/bge-m3",
  local_dir="/home/ubuntu/isef/models/bge-m3",
  allow_patterns=["*.json","*.txt","*.model","*.pt","*.safetensors","1_Pooling/*"],
  max_workers=4)
print("SNAP_OK", p)
PY
SZ=$(stat -c%s /home/ubuntu/isef/models/bge-m3/model.safetensors 2>/dev/null || echo 0)
echo "SAFETENSORS_SIZE=$SZ"
if [ "$SZ" -lt 1000000000 ]; then
  pip install modelscope -q
  python - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("BAAI/bge-m3", local_dir="/home/ubuntu/isef/models/bge-m3-ms")
print("MS_OK", p)
PY
fi
python - <<'PY'
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"]=""
from sentence_transformers import SentenceTransformer
cand = [d for d in ["/home/ubuntu/isef/models/bge-m3","/home/ubuntu/isef/models/bge-m3-ms"] if glob.glob(d+"/model.safetensors") or glob.glob(d+"/pytorch_model.bin")]
assert cand, "NO_WEIGHTS"
m = SentenceTransformer(cand[0])
e = m.encode(["深蹲时膝盖内扣怎么办","squat knee valgus correction"], normalize_embeddings=True)
print("EMB_OK", cand[0], e.shape)
PY
echo SETUP2_DONE
