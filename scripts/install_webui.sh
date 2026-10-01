#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate webui
pip install -U pip
pip install --no-cache-dir open-webui
python -c "import importlib.metadata as m; print('WEBUI_INSTALL_DONE', m.version('open-webui'))"
