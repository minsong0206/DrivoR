#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bench2Drive PAD cache builder (refactored)

- Iterates over train/val splits from MMCV dataset (B2D_VAD_Dataset wrapper).
- For each sample:
  * Saves PAD features -> <cache>/<split>/<idx>/pad_feature.gz
  * Saves PAD targets  -> <cache>/<split>/<idx>/pad_target.gz
  * Returns future agent box corners for T=6, accumulated into <cache>/<split>_fut_boxes.gz

Optional (off by default):
- Build BEV semantic map & agent states (train only).
- Quick debug plotting helpers.
"""

from __future__ import annotations

import os
import gzip
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional dependencies used only in optional helpers:
import cv2  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

from bench2driveMMCV.datasets.B2D_vad_dataset import B2D_VAD_Dataset
from pad_config import (
    train_pipeline,
    test_pipeline,
    modality,
    class_names,
    NameMapping,
    eval_cfg,
    point_cloud_range,
)

# ----------------------------------------------------------------------
# Config & paths (env-driven)
# ----------------------------------------------------------------------

DATA_ROOT = os.environ["B2D_DATA_ROOT"]
INFOS_ROOT =  os.path.join(os.environ["B2D_DATA_ROOT"], "infos", "b2d_")
CACHE_ROOT = os.environ["NAVSIM_EXP_ROOT"] + "/B2d_cache/"
MAP_FILE = INFOS_ROOT + "map_infos.pkl"

# ----------------------------------------------------------------------
# IO utilities
# ----------------------------------------------------------------------

def dump_gz_pickle(path: Path | str, data: Any, compresslevel: int = 1) -> None:
    """Write python object to gzip-pickled file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(data, f)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def compute_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Compute oriented 2D rectangle corners.
    Args:
        boxes: (N,5) array [x, y, width, length, yaw]
    Returns:
        (N,4,2) float array of corners in order: FL(0), RL(1), RR(2), FR(3)
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    half_w = boxes[:, 2] / 2.0
    half_l = boxes[:, 3] / 2.0
    yaw = boxes[:, 4]

    cos_yaw = np.cos(yaw)[..., None]
    sin_yaw = np.sin(yaw)[..., None]

    # local rectangle
    corners_x = np.stack([half_l, -half_l, -half_l, half_l], axis=-1)
    corners_y = np.stack([half_w, half_w, -half_w, -half_w], axis=-1)

    # rotate
    rot_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_y = sin_yaw * corners_x + cos_yaw * corners_y

    # translate
    corners = np.stack((rot_x + x[..., None], rot_y + y[..., None]), axis=-1)
    return corners


# ----------------------------------------------------------------------
# Optional: BEV helpers (moved from commented block)
# ----------------------------------------------------------------------

def coords_to_bev_pixels(
    coords: np.ndarray,
    bev_pixel_width: int,
    bev_pixel_height: int,
    bev_pixel_size: float,
) -> np.ndarray:
    """
    Convert (x,y) local coords to BEV pixel indices with front-centered layout.
    Args:
        coords: (..., 2) array
    """
    # center (row 0, col width/2), note we swap yx -> (row,col) later as needed by cv2
    pixel_center = np.array([[0, bev_pixel_width / 2.0]])
    # coords[..., ::-1] converts (x,y) -> (y,x) for image-style indexing
    idcs = (coords[..., ::-1] / bev_pixel_size) + pixel_center
    return idcs.astype(np.int32)


def build_train_extras_bev_and_agents(
    *,
    gt_bboxes_3d_bev: np.ndarray,  # (N,5) [x,y,w,l,heading]
    gt_attr_labels: torch.Tensor,  # used for agent categories
    data: Dict[str, Any],
    bev_h: int,
    bev_w: int,
    bev_px: float,
) -> Dict[str, Any]:
    """
    Optional: build BEV semantic map and top-30 closest agent states/labels.
    Mirrors the previously commented section.
    """
    # Sort closest 30 agents by distance in BEV
    distances = np.linalg.norm(gt_bboxes_3d_bev[:, :2], axis=-1)
    order = np.argsort(distances)
    kept = order[:30]
    gt_bboxes_3d_sort = gt_bboxes_3d_bev[kept]
    gt_bboxes_3d_all = np.zeros((30, 5), dtype=gt_bboxes_3d_bev.dtype)
    gt_bboxes_3d_all[: len(gt_bboxes_3d_sort)] = gt_bboxes_3d_sort

    agent_labels = np.zeros((30,), dtype=np.float32)
    agent_labels[: len(gt_bboxes_3d_sort)] = 1.0

    # Map elements
    map_gt_bboxes_3d = data["map_gt_bboxes_3d"].data.instance_list  # shapely linestrings
    map_gt_labels_3d = data["map_gt_labels_3d"].data  # ints (0..5)
    bev_semantic_map = np.zeros((bev_h, bev_w), dtype=np.int64)

    # Draw lane-like map layers
    for map_label in range(6):
        mask = np.zeros((bev_w, bev_h), dtype=np.uint8)  # note (W,H) for cv2 drawing
        for label, linestring in zip(map_gt_labels_3d, map_gt_bboxes_3d):
            if label != map_label:
                continue
            points = np.array(linestring.coords).reshape((-1, 1, 2))  # (N,1,2)
            points_px = coords_to_bev_pixels(points, bev_w, bev_h, bev_px)
            cv2.polylines(mask, [points_px], isClosed=False, color=255, thickness=2)
        mask = np.rot90(mask)[::-1]  # rotate into (H,W)
        bev_semantic_map[mask > 0] = map_label + 1

    # Draw agent boxes per class (0..7) over the map
    corners = compute_corners(gt_bboxes_3d_bev)  # (N,4,2)
    category_index = gt_attr_labels[:, 27].to(int)  # original field usage

    for agent_label in range(8):
        mask = np.zeros((bev_w, bev_h), dtype=np.uint8)
        for label, coords in zip(category_index, corners):
            if label != agent_label:
                continue
            exterior = coords.reshape((-1, 1, 2))
            exterior_px = coords_to_bev_pixels(exterior, bev_w, bev_h, bev_px)
            cv2.fillPoly(mask, [exterior_px], color=255)
        mask = np.rot90(mask)[::-1]
        bev_semantic_map[mask > 0] = agent_label + 7

    return {
        "agent_states": gt_bboxes_3d_all,
        "agent_labels": agent_labels,
        "bev_semantic_map": bev_semantic_map,
    }


# ----------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------

class CustomB2DDataset(B2D_VAD_Dataset):
    """
    Thin wrapper that:
      - Computes future agent box corners (T=6 default)
      - Writes per-sample PAD feature/target caches
      - Returns {token: fut_boxes} for caller accumulation
    """

    def __init__(
        self,
        split: str,
        ann_file: str,
        pipeline,
        modality,
        *,
        cache_root: str,
        map_file: str,
        data_root: str,
        future_frames: int = 8,
        past_frames: int = 2,
        bev_w: int = 256,
        bev_h: int = 128,
        bev_px: float = 0.25,
        enable_train_extras: bool = False,
        debug_plot: bool = False,
    ):
        super().__init__(
            point_cloud_range=point_cloud_range,
            queue_length=1,
            data_root=data_root,
            ann_file=ann_file,
            eval_cfg=eval_cfg,
            map_file=map_file,
            pipeline=pipeline,
            name_mapping=NameMapping,
            modality=modality,
            classes=class_names,
            future_frames=future_frames,
            past_frames=past_frames
        )
        self.split = split
        self._cache_path = os.path.join(cache_root, split)
        self.future_frames = future_frames
        self.bev_pixel_width = bev_w
        self.bev_pixel_height = bev_h
        self.bev_pixel_size = bev_px
        self.enable_train_extras = enable_train_extras
        self.debug_plot = debug_plot

    # ---- core: future boxes ----
    def get_fut_box(
        self,
        gt_agent_feats: torch.Tensor,        # (A, 34) when T=6
        gt_agent_boxes: torch.Tensor,        # (A, 9) [x,y,z,w,l,h,yaw,vx,vy]
        T: int,
    ) -> np.ndarray:
        A = gt_agent_feats.shape[0]

        fut_xy = gt_agent_feats[..., : T * 2].reshape(-1, T, 2)
        fut_mask = gt_agent_feats[..., T * 2 : T * 3].reshape(-1, T)
        fut_yaw = gt_agent_feats[..., T * 3 + 10 : T * 4 + 10].reshape(-1, T, 1)

        fut_xy = np.cumsum(fut_xy, axis=1)
        fut_yaw = np.cumsum(fut_yaw, axis=1)

        boxes = gt_agent_boxes.clone()
        boxes[:, 6:7] = -1 * (boxes[:, 6:7] + np.pi / 2)  # world->LiDAR yaw

        fut_xy = fut_xy + boxes[:, None, 0:2]
        fut_yaw = fut_yaw + boxes[:, None, 6:7]

        x = fut_xy[:, :, 0]
        y = fut_xy[:, :, 1]
        yaw = fut_yaw[:, :, 0]

        w = np.repeat(boxes[:, 3:4], T, axis=1)  # width
        l = np.repeat(boxes[:, 4:5], T, axis=1)  # length

        fut_boxes = torch.stack(
            [x, y,
             w, l,
             yaw],
            dim=-1,
        )  # (A, T, 5)

        fut_boxes = fut_boxes * fut_mask[:, :, None]
        corners = compute_corners(fut_boxes.numpy().reshape(-1, 5)).reshape(A, T, 4, 2)
        return corners.astype(np.float32)

    # ---- sample retrieval ----
    def __getitem__(self, idx: int) -> Dict[str, Optional[np.ndarray]]:
        # Pull parent data bundle
        data = self.prepare_train_data(idx) if self.split == "train" else self.prepare_test_data(idx)
        token = str(idx)
        fut_boxes: Optional[np.ndarray] = None

        if data is None:
            return {token: fut_boxes}

        # Unwrap containers (train/test pack differently)
        fut_valid_flag = data["fut_valid_flag"]
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_attr_labels = data["gt_attr_labels"]
        ego_fut_cmd = data["ego_fut_cmd"]
        img = data["img"]
        img_metas = data["img_metas"]
        ego_fut_trajs = data["ego_fut_trajs"]

        if self.split == "train":
            gt_bboxes_3d = gt_bboxes_3d.data
            gt_attr_labels = gt_attr_labels.data
            ego_fut_cmd = ego_fut_cmd.data
            img = img.data[0]
            img_metas = img_metas.data[0]
            ego_fut_trajs = ego_fut_trajs.data[0]
        else:
            fut_valid_flag = fut_valid_flag[0]
            gt_bboxes_3d = gt_bboxes_3d[0].data
            gt_attr_labels = gt_attr_labels[0].data
            ego_fut_cmd = ego_fut_cmd[0].data
            img = img[0].data
            img_metas = img_metas[0].data
            ego_fut_trajs = ego_fut_trajs[0].data[0]

        if not fut_valid_flag:
            return {token: fut_boxes}

        # ---- compute future agent boxes ----
        agent_boxes = gt_bboxes_3d.tensor  # (A,9)
        fut_boxes = self.get_fut_box(gt_attr_labels, agent_boxes, T=self.future_frames)

        # ---- build features ----
        ann_info = self.data_infos[idx]
        ego_vel = ann_info["ego_vel"][:1]          # (1,)
        ego_accel = ann_info["ego_accel"][:2]      # (2,)
        ego_translation = ann_info["ego_translation"]

        # world -> local (LiDAR) command_near in xy
        command_near_xy = np.array([
            ann_info["command_near_xy"][0] - ego_translation[0],
            ann_info["command_near_xy"][1] - ego_translation[1],
        ])
        yaw = ann_info["ego_yaw"]
        theta_to_lidar = -(yaw - np.pi / 2)
        R = np.array([[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                      [np.sin(theta_to_lidar),  np.cos(theta_to_lidar)]])
        local_command_xy = R @ command_near_xy

        gt_ego_fut_cmd = ego_fut_cmd.reshape(6)  # keep T=6

        features: Dict[str, Any] = {}
        features["ego_status"] = torch.cat([
            torch.tensor(ego_vel),
            torch.tensor(ego_accel),
            torch.tensor(local_command_xy),
            gt_ego_fut_cmd,
        ])[None].to(torch.float32)

        # assume (V, C, H, W) and we want first 4 cameras
        features["camera_feature"] = img[:4]

        image_shape = np.zeros((1, 2), dtype=np.int32)
        image_shape[:, 0] = img.shape[-2]  # H
        image_shape[:, 1] = img.shape[-1]  # W
        features["img_shape"] = image_shape

        features["lidar2img"] = np.array(img_metas["lidar2img"])[:4]

        # write features
        token_path = Path(self._cache_path) / token
        dump_gz_pickle(token_path / "pad_feature.gz", features)

        # ---- build targets ----
        targets: Dict[str, Any] = {}
        target_traj = ego_fut_trajs.cumsum(dim=-2)  # make absolute along time
        target_traj[:, 2] += np.pi / 2              # heading to LiDAR conv.
        targets["trajectory"] = target_traj
        targets["token"] = token
        targets["town_name"] = ann_info["town_name"]

        world2lidar = np.array(ann_info["sensors"]["LIDAR_TOP"]["world2lidar"])
        targets["lidar2world"] = np.linalg.inv(world2lidar)

        # ---- optional: train extras (bev map & agent states) ----
        if self.enable_train_extras and self.split == "train":
            gt_bboxes_3d_bev = gt_bboxes_3d.bev  # (N,5) x,y,w,l,heading
            extras = build_train_extras_bev_and_agents(
                gt_bboxes_3d_bev=gt_bboxes_3d_bev,
                gt_attr_labels=gt_attr_labels,
                data=data,
                bev_h=self.bev_pixel_height,
                bev_w=self.bev_pixel_width,
                bev_px=self.bev_pixel_size,
            )
            targets.update(extras)

        # write targets
        dump_gz_pickle(token_path / "pad_target.gz", targets)

        # ---- optional: debug plot (kept as hook) ----
        if self.debug_plot:
            pass  # add your quick viz here if desired

        return {token: fut_boxes}


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------

def passthrough_collate(batch):
    """Keep dataset outputs as a list; we unwrap per item in the main loop."""
    return batch


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser("PAD cache builder (refactored)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=28)
    parser.add_argument("--prefetch-factor", type=int, default=32)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--enable-train-extras", action="store_true",
                        help="Build BEV semantic map & agent states for train samples.", default=False)
    parser.add_argument("--debug-plot", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Limit samples per split for smoke tests.")
    parser.add_argument("--future_frames", type=int)
    parser.add_argument("--past_frames", type=int)
    args = parser.parse_args()

    # Ensure cache root exists
    Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Cache root: {CACHE_ROOT}")

    for split in ["train", "val"]:
        fut_box: Dict[str, Optional[np.ndarray]] = {}

        ann_file = INFOS_ROOT + f"infos_{split}.pkl"
        pipeline = train_pipeline if split == "train" else test_pipeline

        dataset = CustomB2DDataset(
            split=split,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            cache_root=CACHE_ROOT,
            map_file=MAP_FILE,
            data_root=DATA_ROOT,
            enable_train_extras=args.enable_train_extras,
            debug_plot=args.debug_plot,
            future_frames=args.future_frames,
            past_frames=args.past_frames,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            collate_fn=passthrough_collate,
        )

        n = 0
        for batch in tqdm(loader, desc=f"Building {split} cache"):
            # passthrough_collate -> list of items
            sample = batch[0]
            for key, value in sample.items():
                fut_box[key] = value
            n += 1
            if args.max_samples > 0 and n >= args.max_samples:
                break

        split_path = Path(CACHE_ROOT) / f"{split}_fut_boxes.gz"
        dump_gz_pickle(split_path, fut_box)
        print(f"Saved {split} fut boxes to: {split_path}")

    print("Done.")

if __name__ == "__main__":
    main()
