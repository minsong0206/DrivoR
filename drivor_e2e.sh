#!/bin/zsh
# Launched by sim/utils/launch_ad.py as: zsh drivor_e2e.sh <cuda_id> <output_dir>
# (see README_HUGSIM.md for the full obs_pipe/plan_pipe protocol this feeds into)

DRIVOR_PATH="${DRIVOR_PATH:-<path/to/your/DrivoR>}"
# Default: a venv created INSIDE the hugsim_v3 container itself (python3 -m
# venv .venv_docker --python /usr/bin/python3, then pip install -r
# requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121,
# torch==2.1.0+cu121, verified importable:
# navsim.agents.drivoR.drivor_model.DrivoRModel). This is a real python
# binary, not a symlink to some host-only path (e.g. a uv-managed interpreter
# cache under ~/.local/share/uv/ that may not be bind-mounted into the
# container this script actually runs in) -- since this whole repo runs as a
# subprocess of closed_loop_drivor.py inside hugsim_v3 (sim/utils/launch_ad.py
# just does subprocess.Popen, no separate docker exec), DRIVOR_PYTHON must
# resolve to something reachable from THERE, not just from your host shell.
# Override if you keep DrivoR's env elsewhere.
DRIVOR_PYTHON="${DRIVOR_PYTHON:-${DRIVOR_PATH}/.venv_docker/bin/python}"
cd "${DRIVOR_PATH}" || exit 1
echo ${PWD}
# LD_LIBRARY_PATH="": hugsim_v3's own env sets this to its OWN torch's CUDA
# libs (e.g. /usr/local/cuda-11.8/lib64, .../dist-packages/torch/lib for its
# torch==2.4.1+cu118) so subprocesses inherit it by default. DrivoR's venv
# needs a DIFFERENT torch build (2.1.0+cu121) with its own bundled CUDA libs
# under .venv_docker/lib/.../torch/lib -- inheriting hugsim_v3's path makes
# the dynamic linker resolve the wrong-version CUDA runtime first, which
# doesn't error loudly: torch imports, but partially-initializes and later
# fails with a confusing `AttributeError: module 'torch._C' has no attribute
# '_OutOfMemoryError'` the first time anything touches torch.cuda. Clearing
# it lets the venv's own torch/lib rpath resolve correctly instead.
CUDA_VISIBLE_DEVICES=${1} LD_LIBRARY_PATH="" PYTHONPATH="${DRIVOR_PATH}:${PYTHONPATH}" "${DRIVOR_PYTHON}" -u "${DRIVOR_PATH}/hugsim_drivor_client_raw.py" --output="$2"
cd -
