# Install
```bash
conda create -n ipad_no_mmcv python=3.11.10
conda activate ipad_no_mmcv
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ./nuplan-devkit
pip install -e .
```


# Jeanzay

```bash
export PYTHONUSERBASE=$CCFRWORK/python_envs/dynamo_ipad
mkdir -p "${PYTHONUSERBASE}"

module purge
module load pytorch-gpu/py3/2.6.0
```

```bash
pip install -e nuplan-devkit/ --user --no-cache-dir
```

```bash
pip install -e . --user --no-cache-dir
```

Download the Resnet34 weights from hugginface
```
mkdir weights
cd weights
module load git-lfs/3.3.0
git lfs install
git clone https://huggingface.co/timm/resnet34.a1_in1k
```


# Evaluation

> **Important:** Before you start, make sure to run:
> ```bash
> source jeanzay/environment_setup.sh
> ```
> to properly set up all required environment variables and modules.

1. (Optional) Adjust environment variables in `environment_setup.sh` if needed.  
   Example:
   ```bash
   export NAVSIM_EXP_ROOT="$zuw_ALL_CCFRWORK/dynamo_ipad_quan/pad_workspace/exp"
   export NAVSIM_DEVKIT_ROOT="$zuw_ALL_CCFRWORK/dynamo_ipad_quan/navsim"
   ```

2. Link the metric cache to skip redundant caching:
   ```bash
   ln -s $zuw_ALL_CCFRWORK/dynamo_ipad_alex/pad_workspace/exp/train_metric_cache \
         $zuw_ALL_CCFRWORK/dynamo_ipad_quan/pad_workspace/exp/train_metric_cache
   ```

3. Run the metric caching script (about 12 min on 40 CPUs, full V100 node):  
   **Note:** You can directly use the metric cache at `/lustre/fswork/projects/rech/zuw/commun/dynamo_ipad_quan/pad_workspace/exp/metric_cache` that has already been extracted. Read permissions are set for the `zuw` group.
   ```bash
   source jeanzay/environment_setup.sh
   bash metric_caching.sh
   ```

4. Run the evaluation script:
   ```bash
   source jeanzay/environment_setup.sh 
   bash eval.sh
   ```
