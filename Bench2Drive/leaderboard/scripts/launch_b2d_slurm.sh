#!/bin/bash
#SBATCH -A wjv@v100
#SBATCH --job-name=b2d_dynamo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
CKPT=$1
TASK=$2
SAVE=$3

module purge

# load image
module load python/3.8.2
module load gcc/10.1.0 
module load cuda/12.1.0

export PYTHONUSERBASE=$WORK/python_envs/ipad_original

export HYDRA_FULL_ERROR=1

export CARLA_ROOT="$WORK/code/carla_garage/carla"
export CARLA_CACHE_DIR="$SCRATCH/carla_cache"
export Bench2Drive_ROOT="${WORK}/code/dynamo_iPad_fork/Bench2Drive"
export WORK_DIR=$Bench2Drive_ROOT
export SCENARIO_RUNNER_ROOT="${WORK_DIR}/scenario_runner"
export LEADERBOARD_ROOT="${WORK_DIR}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export LD_LIBRARY_PATH=$WORK/local/lib:${LD_LIBRARY_PATH}
export PATH=/lustre/fswork/projects/rech/wjv/uwm61yq/local/bin:${PATH}
export TMPDIR=$SCRATCH/tmp

# move to the code directory 
# NOTE: need to adapt to your own path
cd $WORK/code/dynamo_iPad_fork/Bench2Drive

srun leaderboard/scripts/run_evaluation_pad.sh $CKPT $TASK $SAVE