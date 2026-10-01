#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate vlm
pip install -U pip
pip install -U vllm
python -c "import vllm, torch; print('VLLM_INSTALL_DONE', vllm.__version__, torch.__version__)"
