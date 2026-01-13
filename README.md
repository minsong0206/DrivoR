# DrivoR: Driving on Registers
DrivoR: an end-to-end driving model by driving on registers.

<p align="center">
  <img src="/assets/architecture.jpg" alt="Pipeline">
</p>


[**Project**](https://valeoai.github.io/driving-on-registers/) |
[**Paper**](https://arxiv.org/abs/2601.05083)

# Data and weights

Please download the navsim organize the generated data in the same way as [HERE](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).
```bash
bash ./download/download_navtrain.sh
bash ./download/download_navhard_two_stage.sh
bash ./download/download_warmup_two_stage.sh
```
ViT-S dinoV2 pretrained model can be found in https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m/tree/main, please download all files and put them into *./weights/vit_small_patch14_reg4_dinov2.lvd142m*

The model weights are provided in *GitHub Releases*.
# Installations 
```bash
conda create -n drivoR python=3.8
conda activate drivoR
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./nuplan-devkit
pip install -e .
```

# Training

```bash
cd drivoR
conda activate drivoR
module load Ninja/1.11.1-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCC/12.2.0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/drivoR/dataset/maps"
export NAVSIM_EXP_ROOT="/PATH/TO/drivoR/exp"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/drivoR/"
export OPENSCENE_DATA_ROOT="/PATH/TO/drivoR/dataset"
```
Cache train metrics for pdm score calculation:
```bash
python navsim/planning/script/run_train_metric_caching.py
```
To train with 4xA100 and 10 epochs:

For NAVSIM-v1 model:
```bash
export HYDRA_FULL_ERROR=1 \
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py  \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2
```


For NAVSIM-v2 model:
```bash
export HYDRA_FULL_ERROR=1 \
EXPERIMENT=training_drivoR_Nav2_10epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py  \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=10 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 
    seed=2
```

# Evaluation
Cache train metrics for pdm score calculation:
```bash
bash scripts/evaluation/run_metric_caching.sh
```

```bash
cd drivoR
conda activate drivoR
module load Ninja/1.11.1-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load GCC/12.2.0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/drivoR/dataset/maps"
export NAVSIM_EXP_ROOT="/PATH/TO/drivoR/exp"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/drivoR/"
export OPENSCENE_DATA_ROOT="/PATH/TO/drivoR/dataset"
```

 (i) [NAVSIM-v1] **trained for 25 epochs with longer trajecotry loss**:
```bash
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py \
        train_test_split=navtest \
        agent=drivoR \
        agent.checkpoint_path=PATH/TO/Nav1_25epochs.pth \
        experiment_name=drivoR_nav1 \
    	agent.config.proposal_num=64 \
        agent.config.refiner_ls_values=0.0 \
        agent.config.image_backbone.focus_front_cam=false \
        agent.config.one_token_per_traj=true \
        agent.config.refiner_num_heads=1 \
        agent.config.tf_d_model=256 \
        agent.config.tf_d_ffn=1024 \
        agent.config.area_pred=false \
        agent.config.agent_pred=false \
        agent.config.ref_num=4 \
    	agent.config.noc=1 \
    	agent.config.dac=1\
    	agent.config.ddc=0.0 \
    	agent.config.ttc=5 \
    	agent.config.ep=5 \
    	agent.config.comfort=2

```

 (ii) [NAVSIM-v2] **train for 10 epochs** (You need to use [Navsim2 repo](https://github.com/autonomousvision/navsim) for evaluation: and copy the agent files, config to the evaluation repo.):
```bash
TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/navhard_two_stage_metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
CHECKPOINT=PATH/TO/Nav2_10epochs.pth
EXPERIMENT=drivoR_nav2
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py  \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXPERIMENT \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    agent=$AGENT \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.noc=10 \
    agent.config.dac=13 \
    agent.config.ddc=6 \
    agent.config.ttc=14 \
    agent.config.ep=15 \
    agent.config.comfort=2
```

# Acknowledgement 
The code takes inspiration from https://github.com/Kguo-cs/iPad. 
The NAVSIM-v2 evaluation code is adopted from https://github.com/autonomousvision/navsim.
