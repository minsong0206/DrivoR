from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import math
import json
import numpy as np
import pandas as pd

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.evaluate.pdm_score import transform_trajectory

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

# ----------------------------
# Trajectory extraction hook
# ----------------------------
def extract_predicted_trajectories(scene, agent: AbstractAgent) -> np.ndarray:
    """
    Return predicted trajectories for this scene as an array [N, T, 2] in ego frame (meters).
    - N: number of trajectory hypotheses / proposals
    - T: number of timesteps in horizon
    - 2: (x, y)

    You may need to adapt this to your agent. Keep the signature and return type.
    """
    # Example stub:
    # preds = agent.predict_scene(scene)  # -> list of arrays [T,2], or dict with 'trajectories'
    # return np.stack(preds, axis=0)

    # TEMP: raise a clear error so you only have to touch this one function
    raise NotImplementedError(
        "Implement `extract_predicted_trajectories(scene, agent)` to return [N, T, 2] in ego frame."
    )

# ----------------------------
# Feature engineering
# ----------------------------
def normalize_and_flatten_trajs(trajs_xy: np.ndarray, pad_to: int = None) -> np.ndarray:
    """
    Convert trajectories [N, T, 2] -> feature vectors [N, F].
    - Translate so first point is (0,0).
    - Optionally, rotate so initial heading aligns with +x (if step 1->2 exists).
    - Flatten to 1D.
    """
    assert trajs_xy.ndim == 3 and trajs_xy.shape[2] == 2, "Expected [N, T, 2]"

    N, T, _ = trajs_xy.shape
    # Translate
    origin = trajs_xy[:, :1, :]  # [N,1,2]
    rel = trajs_xy - origin

    # Rotate to initial heading (if non-degenerate)
    v0 = rel[:, 1, :] - rel[:, 0, :]  # [N,2]
    headings = np.arctan2(v0[:, 1], v0[:, 0])  # [N]
    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)
    R = np.stack([np.stack([cos_h, -sin_h], axis=-1),
                  np.stack([sin_h,  cos_h], axis=-1)], axis=1)  # [N,2,2]
    rel_rot = np.einsum("nij,ntj->nti", R, rel)  # [N,T,2]

    # Optional padding to a fixed T
    if pad_to is not None and T != pad_to:
        if T > pad_to:
            rel_rot = rel_rot[:, :pad_to, :]
            T = pad_to
        else:
            pad = np.repeat(rel_rot[:, -1:, :], repeats=pad_to - T, axis=1)
            rel_rot = np.concatenate([rel_rot, pad], axis=1)
            T = pad_to

    # Scale (robust): divide by 95th percentile of endpoint distances to reduce scale effects
    end_dists = np.linalg.norm(rel_rot[:, -1, :], axis=-1) + 1e-6
    scale = np.percentile(end_dists, 95)
    rel_rot = rel_rot / (scale if scale > 0 else 1.0)

    feats = rel_rot.reshape(N, T * 2)  # [N, 2T]
    return feats.astype(np.float32)

# ----------------------------
# Simple k-means (NumPy)
# ----------------------------
def kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: [N, D]
    Returns (centers [k, D], labels [N])
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    assert k <= N, "k must be <= number of samples"

    # k-means++ init
    centers = np.empty((k, D), dtype=X.dtype)
    # pick first at random
    idx0 = rng.randint(N)
    centers[0] = X[idx0]
    # remaining
    d2 = np.full(N, np.inf, dtype=X.dtype)
    for ci in range(1, k):
        d2 = np.minimum(d2, np.sum((X - centers[ci-1])**2, axis=1))
        probs = d2 / d2.sum()
        next_idx = rng.choice(N, p=probs)
        centers[ci] = X[next_idx]

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(max_iter):
        # assign
        dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # [N,k]
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centers[ci] = X[mask].mean(axis=0)
            # if empty cluster, re-seed randomly
            else:
                centers[ci] = X[rng.randint(N)]
    return centers, labels

# ----------------------------
# Worker: extract + cluster
# ----------------------------
def run_traj_cluster(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Worker that:
      - loads scenes for given tokens
      - extracts predicted trajectories
      - engineers features
      - runs k-means
      - writes parquet + centers + cluster summary
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, \
        "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    save_root = os.path.join(os.environ.get("VIZ_ROOT", "."), cfg.experiment_name, f"{timestamp}_clusters")
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, "ckpt_used.txt"), "w") as f:
        f.write(cfg.agent.checkpoint_path)

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    rows = []
    all_feats = []
    row_idx = 0

    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            trajectory = agent.compute_trajectory(agent_input, scene)
            initial_ego_state = metric_cache.ego_state

            trajs = transform_trajectory(trajectory, initial_ego_state)
            N, T, _ = trajs.shape
            feats = normalize_and_flatten_trajs(trajs)  # [N, 2T]
            all_feats.append(feats)

            # Keep per-trajectory metadata (token + index)
            for i in range(N):
                rows.append(
                    {
                        "token": token,
                        "traj_local_id": i,
                        "horizon_T": T,
                        # Store compact representation; raw traj for that ID will be saved separately
                        "endpoint_x": float(trajs[i, -1, 0]),
                        "endpoint_y": float(trajs[i, -1, 1]),
                    }
                )
                row_idx += 1

            # Save raw trajectories per token for provenance
            np.save(os.path.join(save_root, f"{token}_trajs.npy"), trajs)

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    if len(all_feats) == 0:
        logger.error("No features collected; aborting clustering.")
        return []

    X = np.concatenate(all_feats, axis=0)  # [M, D]
    df = pd.DataFrame(rows)

    # Determine number of clusters
    # Option A: from config; Option B: heuristic
    k = getattr(cfg, "traj_cluster_k", None)
    if k is None or not isinstance(k, int) or k < 2:
        # simple heuristic: sqrt(M/2), clipped
        M = X.shape[0]
        k = max(2, min(30, int(math.sqrt(max(4, M // 2)))))

    logger.info(f"Clustering {X.shape[0]} trajectories into k={k} clusters (feature dim {X.shape[1]})")

    centers, labels = kmeans_numpy(X, k=k, max_iter=100, seed=getattr(cfg, "seed", 0))
    df["cluster_id"] = labels

    # Write artifacts
    out_parquet = os.path.join(save_root, "trajectories.parquet")
    out_centers = os.path.join(save_root, "cluster_centers.npy")
    df.to_parquet(out_parquet, index=False)
    np.save(out_centers, centers)

    # Summary
    summary = (
        df.groupby("cluster_id")
          .agg(count=("token", "count"),
               mean_end_x=("endpoint_x", "mean"),
               mean_end_y=("endpoint_y", "mean"))
          .reset_index()
          .sort_values("count", ascending=False)
    )
    summary_path = os.path.join(save_root, "clusters_summary.csv")
    summary.to_csv(summary_path, index=False)

    logger.info(f"Saved: {out_parquet}")
    logger.info(f"Saved: {out_centers}")
    logger.info(f"Saved: {summary_path}")
    return [{"save_root": save_root, "k": k, "num_samples": X.shape[0]}]

# ----------------------------
# Hydra entry
# ----------------------------
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    # Discover tokens (same as your original, but we don't need sensor blobs here)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting trajectory clustering for %s scenarios...", str(len(tokens_to_evaluate)))

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    worker_map(worker, run_traj_cluster, data_points)

if __name__ == "__main__":
    main()
