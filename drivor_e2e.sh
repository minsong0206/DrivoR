#!/bin/zsh

DRIVOR_PATH=/home/ms/DrivoR
cd ${DRIVOR_PATH} || exit 1
echo ${PWD}
# machine migrated conda -> uv; old /home/ms/miniconda3/envs/drivoR no longer
# exists. Replacement env built at /home/ms/uv-envs/drivoR/venv (uv pip
# install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# --index-strategy unsafe-best-match), torch==2.1.0+cu121, verified importable
# (navsim.agents.drivoR.drivor_model.DrivoRModel).
CUDA_VISIBLE_DEVICES=${1} PYTHONPATH=/home/ms/DrivoR:${PYTHONPATH} /home/ms/uv-envs/drivoR/venv/bin/python -u /home/ms/DrivoR/hugsim_drivor_client_raw.py --output="$2"
cd -
