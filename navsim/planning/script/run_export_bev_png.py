import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivoR.utils.bev_export import export_prediction_dict
from navsim.common import dataclasses as navsim_dataclasses
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
DEFAULT_RELEASE_CHECKPOINT = (
    "/home/ms/DrivoR/weights/release_checkpoints/"
    "nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth"
)


def _validate_required_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def _validate_required_dir(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _status(message: str) -> None:
    print(f"[run_export_bev_png] {message}", flush=True)
    logger.info(message)


def _split_list(input_list: List[Dict], num_frames: int, frame_interval: int) -> List[List[Dict]]:
    return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]


def _scene_matches(frame_value: str, query: str, match_mode: str) -> bool:
    if frame_value is None:
        return False
    if match_mode == "exact":
        return frame_value == query
    if match_mode == "suffix":
        return frame_value.endswith(query)
    if match_mode == "contains":
        return query in frame_value
    raise ValueError(f"Unsupported scene match mode: {match_mode}")


def _set_map_root(map_root: Path) -> None:
    os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
    navsim_dataclasses.NUPLAN_MAPS_ROOT = str(map_root)
    _status(f"NUPLAN_MAPS_ROOT set to: {map_root}")


def _required_camera_names_by_frame(sensor_config: Dict, num_frames: int) -> Dict[int, List[str]]:
    frame_to_cameras: Dict[int, List[str]] = {}
    for frame_idx in range(num_frames):
        sensors = sensor_config.get_sensors_at_iteration(frame_idx)
        camera_names = [sensor.upper() for sensor in sensors if sensor.startswith("cam_")]
        if camera_names:
            frame_to_cameras[frame_idx] = camera_names
    return frame_to_cameras


def _has_required_sensor_files(
    frame_list: List[Dict],
    sensor_blobs_path: Path,
    required_cameras_by_frame: Dict[int, List[str]],
) -> bool:
    for frame_idx, camera_names in required_cameras_by_frame.items():
        if frame_idx >= len(frame_list):
            return False
        camera_dict = frame_list[frame_idx].get("cams", {})
        for camera_name in camera_names:
            camera_info = camera_dict.get(camera_name)
            if camera_info is None:
                return False
            data_path = camera_info.get("data_path")
            if data_path is None:
                return False
            image_path = sensor_blobs_path / data_path
            if not image_path.exists():
                return False
    return True


def find_tokens_for_scene(
    data_path: Path,
    sensor_blobs_path: Path,
    scene_filter: SceneFilter,
    scene_name: str,
    limit: Optional[int],
    scene_match_key: str = "scene_name",
    scene_match_mode: str = "contains",
    required_cameras_by_frame: Optional[Dict[int, List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    matched_log_names: Set[str] = set()
    log_paths = sorted(path for path in data_path.iterdir() if path.suffix == ".pkl")
    _status(
        f"Scanning {len(log_paths)} log pickle files for scene '{scene_name}' "
        f"using {scene_match_key}/{scene_match_mode}"
    )
    for log_index, log_pickle_path in enumerate(log_paths, start=1):
        if log_index == 1 or log_index % 50 == 0:
            _status(f"Scanning log {log_index}/{len(log_paths)}: {log_pickle_path.name}")
        with open(log_pickle_path, "rb") as fp:
            import pickle

            scene_dict_list = pickle.load(fp)

        for frame_list in _split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            if len(frame_list) < scene_filter.num_frames:
                continue
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue
            current_frame = frame_list[scene_filter.num_history_frames - 1]
            current_scene_value = current_frame.get(scene_match_key)
            if not _scene_matches(current_scene_value, scene_name, scene_match_mode):
                continue
            if required_cameras_by_frame is not None:
                if not _has_required_sensor_files(
                    frame_list=frame_list,
                    sensor_blobs_path=sensor_blobs_path,
                    required_cameras_by_frame=required_cameras_by_frame,
                ):
                    continue

            token = current_frame["token"]
            tokens.append(token)
            log_name = current_frame.get("log_name")
            if log_name:
                matched_log_names.add(log_name)
            if len(tokens) <= 3:
                _status(f"Matched token {token} from {scene_match_key}={current_scene_value}")
            if limit is not None and len(tokens) >= limit:
                _status(f"Reached token limit {limit} while scanning scene '{scene_name}'")
                return tokens, sorted(matched_log_names)

    return tokens, sorted(matched_log_names)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _status("Starting BEV PNG export")

    if cfg.train_ckpt_path is None:
        cfg.train_ckpt_path = DEFAULT_RELEASE_CHECKPOINT
        _status(f"train_ckpt_path not set. Using default release checkpoint: {cfg.train_ckpt_path}")
    if str(cfg.train_ckpt_path) == "/path/to/your.ckpt":
        raise ValueError("Replace train_ckpt_path=/path/to/your.ckpt with a real checkpoint path")

    ckpt_path = Path(cfg.train_ckpt_path)
    navsim_log_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    scenario_path = Path(cfg.export_bev_scenario_path)
    scene_cfg_path = Path(cfg.export_bev_scene_cfg_path)
    map_root_str = getattr(cfg, "export_bev_map_root", None) or os.environ.get("NUPLAN_MAPS_ROOT")
    if not map_root_str:
        raise ValueError(
            "Map root is not configured. Set export_bev_map_root or NUPLAN_MAPS_ROOT."
        )
    map_root = Path(map_root_str)
    _validate_required_path(ckpt_path, "Checkpoint")
    _validate_required_dir(navsim_log_path, "NAVSIM/OpenScene log directory")
    _validate_required_dir(sensor_blobs_path, "NAVSIM/OpenScene sensor_blobs directory")
    _validate_required_path(scenario_path, "HUGSIM scenario yaml")
    _validate_required_path(scene_cfg_path, "HUGSIM scene cfg")
    _validate_required_dir(map_root, "nuPlan maps root")

    _status(f"Checkpoint: {ckpt_path}")
    _status(f"NAVSIM/OpenScene logs: {navsim_log_path}")
    _status(f"NAVSIM/OpenScene sensor_blobs: {sensor_blobs_path}")
    _status(f"Using scenario yaml: {scenario_path}")
    _status(f"Using scene asset cfg: {scene_cfg_path}")
    _set_map_root(map_root)

    export_root = Path(cfg.output_dir) / cfg.export_bev_output_subdir / cfg.export_bev_scene_name
    export_root.mkdir(parents=True, exist_ok=True)
    _status(f"Export root prepared at: {export_root}")

    _status("Instantiating agent and scene filter")
    if getattr(cfg.agent, "checkpoint_path", "") == "":
        cfg.agent.checkpoint_path = str(ckpt_path)
    cfg.agent.enable_training_runtime = False
    _status(
        f"Agent runtime config: checkpoint_path={cfg.agent.checkpoint_path}, "
        f"enable_training_runtime={getattr(cfg.agent, 'enable_training_runtime', None)}"
    )
    agent: AbstractAgent = instantiate(cfg.agent)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    required_cameras_by_frame = _required_camera_names_by_frame(
        agent.get_sensor_config(), scene_filter.num_frames
    )
    _status(f"Required camera checks enabled for frames: {required_cameras_by_frame}")

    scene_tokens = cfg.export_bev_tokens
    matched_log_names: List[str] = []
    if scene_tokens is None:
        scene_match_key = getattr(cfg, "export_bev_scene_match_key", "scene_name")
        scene_match_mode = getattr(cfg, "export_bev_scene_match_mode", "contains")
        _status(f"Finding tokens for scene '{cfg.export_bev_scene_name}'")
        scene_tokens, matched_log_names = find_tokens_for_scene(
            data_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
            scene_filter=scene_filter,
            scene_name=cfg.export_bev_scene_name,
            limit=cfg.export_bev_max_samples,
            scene_match_key=scene_match_key,
            scene_match_mode=scene_match_mode,
            required_cameras_by_frame=required_cameras_by_frame,
        )

    if not scene_tokens:
        raise ValueError(
            f"No NAVSIM/OpenScene tokens found for scene '{cfg.export_bev_scene_name}' under {cfg.navsim_log_path}"
        )

    _status(f"Selected {len(scene_tokens)} tokens for scene '{cfg.export_bev_scene_name}'")
    if matched_log_names:
        _status(f"Matched {len(matched_log_names)} logs; restricting SceneLoader log scan")
        scene_filter.log_names = matched_log_names
    scene_filter.tokens = scene_tokens
    _status("Building SceneLoader")
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    _status("Building Dataset")
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path if cfg.use_cache_without_dataset else None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )

    _status(f"Dataset ready with {len(dataset)} samples")
    dataloader_params = dict(cfg.dataloader.params)
    if dataloader_params.get("num_workers", 0) == 0 and "prefetch_factor" in dataloader_params:
        dataloader_params.pop("prefetch_factor")
        _status("Removed prefetch_factor because num_workers=0")
    dataloader = DataLoader(dataset, **dataloader_params, shuffle=False)
    _status("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
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

    if dist_ready():
        dist.barrier()

    world_size = dist.get_world_size() if dist_ready() else 1
    all_predictions = [None for _ in range(world_size)]
    if dist_ready():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions = [predictions]

    rank = dist.get_rank() if dist_ready() else 0
    if rank != 0:
        return

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for batch_prediction in proc_prediction:
            merged_predictions.update(batch_prediction)

    _status(f"Merged predictions for {len(merged_predictions)} tokens")
    exported = export_prediction_dict(
        predictions=merged_predictions,
        output_root=export_root,
        include_compare=cfg.export_bev_include_compare,
        max_samples=cfg.export_bev_max_samples,
        scene_name=None,
        tokens=scene_tokens,
    )
    _status(f"Exported {exported} BEV PNG samples to {export_root}")


if __name__ == "__main__":
    main()
