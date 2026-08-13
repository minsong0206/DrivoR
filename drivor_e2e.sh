#!/bin/zsh
# Launched by sim/utils/launch_ad.py as: zsh drivor_e2e.sh <cuda_id> <output_dir>
# (see README_HUGSIM.md for the full obs_pipe/plan_pipe protocol this feeds into)

DRIVOR_PATH="${DRIVOR_PATH:-<path/to/your/DrivoR>}"
DRIVOR_PYTHON="${DRIVOR_PYTHON:-<path/to/your/DrivoR/venv>/bin/python}"
cd "${DRIVOR_PATH}" || exit 1
echo ${PWD}
# env migrated conda -> uv here; adjust DRIVOR_PYTHON to your own env's
# interpreter (uv pip install -r requirements.txt --extra-index-url
# https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match,
# torch==2.1.0+cu121, verified importable: navsim.agents.drivoR.drivor_model.DrivoRModel).
CUDA_VISIBLE_DEVICES=${1} PYTHONPATH="${DRIVOR_PATH}:${PYTHONPATH}" "${DRIVOR_PYTHON}" -u "${DRIVOR_PATH}/hugsim_drivor_client_raw.py" --output="$2"
cd -
