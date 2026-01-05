#!/bin/bash
BASE_PORT=20000 #20000
BASE_TM_PORT=40000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
export TEAM_AGENT=leaderboard/pad_team_code/dynamo_b2d_agent_vis.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=leaderboard/pad_team_code/pad_config.py+$1
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=dynamo
SAVE_PATH=$3

i=$2

PORT=$((BASE_PORT + i * 150))
TM_PORT=$((BASE_TM_PORT + i * 150))
ROUTES="${BASE_ROUTES}_$2_${ALGO}_${PLANNER_TYPE}.xml"
CHECKPOINT_ENDPOINT="$3/${BASE_CHECKPOINT_ENDPOINT}_$2.json"
GPU_RANK=0 #$2

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=8
    python tools/split_xml.py --base_route $BASE_ROUTES --task_num $TASK_NUM --algo $ALGO --planner_type $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

if [ ! -d "$SAVE_PATH" ]; then
  mkdir "$SAVE_PATH"
  echo "Folder '$SAVE_PATH' created."
else
  echo "Folder '$SAVE_PATH' already exists."
fi

echo -e "\033[32m ALGO: $ALGO \033[0m"
echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
echo -e "\033[32m TASK_ID: $i \033[0m"
echo -e "\033[32m PORT: $PORT \033[0m"
echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
echo -e "***********************************************************************************"
bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK > $3/$2.log  2>&1 &

# gpu_count=$(nvidia-smi -L | wc -l)
# if [ "$gpu_count" -eq 6 ]; then
#     i=$((i + $(nvidia-smi -L | wc -l)))
#     PORT=$((BASE_PORT + i * 150))
#     TM_PORT=$((BASE_TM_PORT + i * 150))
#     ROUTES="${BASE_ROUTES}_${i}_${ALGO}_${PLANNER_TYPE}.xml"
#     CHECKPOINT_ENDPOINT="$3/${BASE_CHECKPOINT_ENDPOINT}_${i}.json"
#     bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK  > $3/${i}.log  2>&1 &
# else
#     echo "GPU count is $gpu_count, not 1. Exiting."
# fi
wait