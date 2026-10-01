#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate vlm
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCH_CUDA_ARCH_LIST="12.0"
exec vllm serve /home/ubuntu/isef/models/qwen3-vl-8b-instruct \
  --served-model-name qwen3-vl-8b \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --allowed-local-media-path /home/ubuntu/isef \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
