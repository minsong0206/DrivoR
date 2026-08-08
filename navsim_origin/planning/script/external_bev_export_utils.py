import logging
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import navsim

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivoR.utils.bev_export import export_prediction_dict
from navsim.common import dataclasses as navsim_dataclasses
from navsim.common.dataclasses import SceneFilter
from navsim.common.external_nuscenes_adapter import ProcessedNuScenesSceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset


logger = logging.getLogger(__name__)

DEFAULT_RELEASE_CHECKPOINT = (
    "/home/ms/DrivoR/weights/release_checkpoints/"
    "nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth"
)


def _status(message: str) -> None:
    print(f"[external_bev_export] {message}", flush=True)
    logger.info(message)


def _resolve_processed_scene_root(processed_root: Path, scene_name: str) -> Path:
    if (processed_root / scene_name).exists():
        return processed_root
    if processed_root.name == scene_name and (processed_root / "meta_data.json").exists():
        return processed_root.parent
    return processed_root


def _set_map_root(map_root: Path) -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
    navsim_dataclasses.NUPLAN_MAPS_ROOT = str(map_root)
    _status(f"NUPLAN_MAPS_ROOT set to: {map_root}")


def _build_trainer_params(cfg: DictConfig) -> Dict:
    trainer_params = dict(cfg.trainer.params)
    trainer_params["strategy"] = "auto"
    if not torch.cuda.is_available() and trainer_params.get("accelerator", "gpu") == "gpu":
        trainer_params["accelerator"] = "cpu"
        trainer_params["precision"] = "32-true"
        _status("CUDA unavailable. Falling back to CPU exporter runtime.")
    return trainer_params


def run_external_bev_export(cfg: DictConfig, ckpt_override: str = None) -> int:
    pl.seed_everything(cfg.seed, workers=True)
    _status(f"Resolved navsim package path: {Path(navsim.__file__).resolve()}")

    train_ckpt_path = ckpt_override or cfg.train_ckpt_path or DEFAULT_RELEASE_CHECKPOINT
    ckpt_path = Path(train_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    scene_name = cfg.export_bev_scene_name
    processed_root = _resolve_processed_scene_root(Path(cfg.external_processed_scene_dir), scene_name)
    nuscenes_root = Path(cfg.external_nuscenes_root)
    map_root = Path(getattr(cfg, "export_bev_map_root", "/home/ms/DrivoR/datasets/maps"))

    if not processed_root.exists():
        raise FileNotFoundError(f"Processed scenes root does not exist: {processed_root}")
    if not nuscenes_root.exists():
        raise FileNotFoundError(f"nuScenes root does not exist: {nuscenes_root}")
    if not map_root.exists():
        raise FileNotFoundError(f"nuPlan map root does not exist: {map_root}")

    _set_map_root(map_root)

    export_root = Path(cfg.output_dir) / cfg.export_bev_output_subdir / scene_name
    export_root.mkdir(parents=True, exist_ok=True)
    _status(f"Export root prepared at: {export_root}")

    cfg.agent.checkpoint_path = str(ckpt_path)
    cfg.agent.enable_training_runtime = False
    if hasattr(cfg.agent, "config"):
        cfg.agent.config.bev_map = True

    _status(
        f"Agent runtime config: checkpoint_path={cfg.agent.checkpoint_path}, "
        f"enable_training_runtime={getattr(cfg.agent, 'enable_training_runtime', None)}, "
        f"bev_map={getattr(cfg.agent.config, 'bev_map', None)}"
    )

    agent: AbstractAgent = instantiate(cfg.agent)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = ProcessedNuScenesSceneLoader(
        processed_scenes_root=processed_root,
        nuscenes_root=nuscenes_root,
        scene_name=scene_name,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        max_samples=cfg.export_bev_max_samples,
    )
    _status(f"External scene loader ready with {len(scene_loader)} tokens")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    _status(f"Dataset ready with {len(dataset)} samples")

    dataloader_params = dict(cfg.dataloader.params)
    dataloader_params["num_workers"] = 0
    if "prefetch_factor" in dataloader_params:
        dataloader_params.pop("prefetch_factor")
    dataloader = DataLoader(dataset, **dataloader_params, shuffle=False)

    trainer = pl.Trainer(**_build_trainer_params(cfg), callbacks=agent.get_training_callbacks())
    predict_module = AgentLightningModule(agent=agent, for_viz=True)

    _status("Loading checkpoint with strict=False for compatibility")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    checkpoint_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if not any("map_head" in key for key in checkpoint_state_dict.keys()):
        logger.warning(
            "Loaded checkpoint does not include map_head weights. "
            "Trajectory inference is available, but BEV prediction quality may be poor."
        )
    load_result = predict_module.load_state_dict(checkpoint_state_dict, strict=False)
    _status(
        f"Checkpoint load complete. missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}"
    )

    _status("Starting trainer.predict")
    predictions = trainer.predict(
        predict_module,
        dataloader,
        return_predictions=True,
        ckpt_path=None,
    )
    _status("trainer.predict finished")

    merged_predictions = {}
    for batch_prediction in predictions:
        merged_predictions.update(batch_prediction)
    _status(f"Merged predictions for {len(merged_predictions)} tokens")

    exported = export_prediction_dict(
        predictions=merged_predictions,
        output_root=export_root,
        include_compare=cfg.export_bev_include_compare,
        max_samples=cfg.export_bev_max_samples,
        scene_name=None,
        tokens=scene_loader.tokens,
    )
    _status(f"Exported {exported} BEV PNG samples to {export_root}")
    return exported
