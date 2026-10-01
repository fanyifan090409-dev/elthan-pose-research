#!/bin/bash
source /home/ubuntu/Miniconda3/etc/profile.d/conda.sh
conda activate py310
cd /home/ubuntu/isef/agent
exec python agent_server.py
