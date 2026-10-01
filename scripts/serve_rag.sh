#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate rag
cd /home/ubuntu/isef/rag
exec python server.py
