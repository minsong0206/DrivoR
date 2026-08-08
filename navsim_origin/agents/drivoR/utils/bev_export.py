from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image


DEFAULT_BEV_PALETTE = np.array(
    [
        [0, 0, 0],
        [70, 70, 70],
        [128, 64, 128],
        [220, 20, 60],
        [244, 35, 232],
        [250, 170, 30],
        [107, 142, 35],
        [0, 0, 142],
        [220, 20, 60],
        [119, 11, 32],
        [255, 140, 0],
        [190, 153, 153],
        [220, 220, 0],
        [70, 130, 180],
        [152, 251, 152],
        [255, 0, 0],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def label_map_to_rgb(label_map: np.ndarray, palette: np.ndarray = DEFAULT_BEV_PALETTE) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int64)
    clipped = np.clip(label_map, 0, len(palette) - 1)
    return palette[clipped]


def save_label_png(path: Path, label_map: np.ndarray, palette: np.ndarray = DEFAULT_BEV_PALETTE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_map_to_rgb(label_map, palette)).save(path)


def save_compare_png(
    path: Path,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    palette: np.ndarray = DEFAULT_BEV_PALETTE,
) -> None:
    gt_rgb = label_map_to_rgb(gt_map, palette)
    pred_rgb = label_map_to_rgb(pred_map, palette)
    compare = np.concatenate([gt_rgb, pred_rgb], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(compare).save(path)


def export_bev_prediction(
    output_root: Path,
    token: str,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    include_compare: bool = True,
    palette: np.ndarray = DEFAULT_BEV_PALETTE,
) -> None:
    token_dir = output_root / token
    save_label_png(token_dir / "gt.png", gt_map, palette)
    save_label_png(token_dir / "pred.png", pred_map, palette)
    if include_compare:
        save_compare_png(token_dir / "compare.png", gt_map, pred_map, palette)


def export_prediction_dict(
    predictions: Dict[str, Dict],
    output_root: Path,
    include_compare: bool = True,
    max_samples: Optional[int] = None,
    scene_name: Optional[str] = None,
    tokens: Optional[Iterable[str]] = None,
) -> int:
    selected_tokens = set(tokens) if tokens is not None else None
    exported = 0

    for token, payload in predictions.items():
        if selected_tokens is not None and token not in selected_tokens:
            continue
        if scene_name is not None and payload.get("scene_name") != scene_name:
            continue
        if payload.get("gt_bev_semantic_map") is None or payload.get("pred_bev_semantic_map") is None:
            continue

        export_bev_prediction(
            output_root=output_root,
            token=token,
            gt_map=np.asarray(payload["gt_bev_semantic_map"]),
            pred_map=np.asarray(payload["pred_bev_semantic_map"]),
            include_compare=include_compare,
        )
        exported += 1
        if max_samples is not None and exported >= max_samples:
            break

    return exported
