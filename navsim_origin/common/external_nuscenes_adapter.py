import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig


logger = logging.getLogger(__name__)

NUSC_TO_NUPLAN_MAP = {
    "boston-seaport": "us-ma-boston",
    "singapore-onenorth": "sg-one-north",
    "singapore-hollandvillage": "sg-one-north",
    "singapore-queenstown": "sg-one-north",
}

CAMERA_CHANNEL_TO_NAVSIM = {
    "CAM_FRONT": "CAM_F0",
    "CAM_FRONT_LEFT": "CAM_L0",
    "CAM_FRONT_RIGHT": "CAM_R0",
    "CAM_BACK": "CAM_B0",
}

ALL_NAVSIM_CAMERAS = ["CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"]


def _load_json(path: Path):
    with open(path, "r") as fp:
        return json.load(fp)


def _pose_matrix(rotation_wxyz: List[float], translation_xyz: List[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Quaternion(*rotation_wxyz).rotation_matrix
    pose[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return pose


def _map_category_to_navsim_name(category_name: str) -> Optional[str]:
    if category_name.startswith("vehicle.car") or category_name.startswith("vehicle.bus") or category_name.startswith("vehicle.truck") or category_name.startswith("vehicle.trailer") or category_name.startswith("vehicle.construction"):
        return "vehicle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    if category_name.startswith("movable_object.debris") or category_name.startswith("movable_object.pushable_pullable"):
        return "generic_object"
    return None


class NuScenesTableIndex:
    """Minimal nuScenes table loader that avoids external devkit dependency."""

    def __init__(self, nuscenes_root: Path, version: str = "v1.0-trainval"):
        self.nuscenes_root = Path(nuscenes_root)
        self.table_root = self.nuscenes_root / version
        if not self.table_root.exists():
            raise FileNotFoundError(f"nuScenes table directory does not exist: {self.table_root}")

        self.scene = _load_json(self.table_root / "scene.json")
        self.sample = _load_json(self.table_root / "sample.json")
        self.sample_data = _load_json(self.table_root / "sample_data.json")
        self.sensor = _load_json(self.table_root / "sensor.json")
        self.calibrated_sensor = _load_json(self.table_root / "calibrated_sensor.json")
        self.ego_pose = _load_json(self.table_root / "ego_pose.json")
        self.sample_annotation = _load_json(self.table_root / "sample_annotation.json")
        self.instance = _load_json(self.table_root / "instance.json")
        self.category = _load_json(self.table_root / "category.json")
        self.log = _load_json(self.table_root / "log.json")

        self.scene_by_token = {entry["token"]: entry for entry in self.scene}
        self.scene_by_name = {entry["name"]: entry for entry in self.scene}
        self.sample_by_token = {entry["token"]: entry for entry in self.sample}
        self.sample_data_by_token = {entry["token"]: entry for entry in self.sample_data}
        self.sensor_by_token = {entry["token"]: entry for entry in self.sensor}
        self.calibrated_sensor_by_token = {entry["token"]: entry for entry in self.calibrated_sensor}
        self.ego_pose_by_token = {entry["token"]: entry for entry in self.ego_pose}
        self.sample_annotation_by_token = {entry["token"]: entry for entry in self.sample_annotation}
        self.instance_by_token = {entry["token"]: entry for entry in self.instance}
        self.category_by_token = {entry["token"]: entry for entry in self.category}
        self.log_by_token = {entry["token"]: entry for entry in self.log}
        self.channel_by_calibrated_sensor_token = {
            calibrated["token"]: self.sensor_by_token[calibrated["sensor_token"]]["channel"]
            for calibrated in self.calibrated_sensor
        }

        self.sample_data_by_sample_token: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        for sample_data_entry in self.sample_data:
            channel = self.channel_by_calibrated_sensor_token[sample_data_entry["calibrated_sensor_token"]]
            self.sample_data_by_sample_token[sample_data_entry["sample_token"]][channel] = sample_data_entry

        self.annotation_tokens_by_sample_token: Dict[str, List[str]] = defaultdict(list)
        for annotation in self.sample_annotation:
            self.annotation_tokens_by_sample_token[annotation["sample_token"]].append(annotation["token"])

    def get_scene_by_name(self, scene_name: str) -> Dict:
        if scene_name not in self.scene_by_name:
            raise KeyError(f"Scene '{scene_name}' not found in nuScenes tables.")
        return self.scene_by_name[scene_name]

    def iter_scene_samples(self, scene_token: str) -> List[Dict]:
        scene = self.scene_by_token[scene_token]
        sample_token = scene["first_sample_token"]
        out = []
        while sample_token:
            sample = self.sample_by_token[sample_token]
            out.append(sample)
            sample_token = sample["next"]
        return out

    def infer_nuplan_map_name(self, log_token: str) -> str:
        location = self.log_by_token[log_token]["location"]
        if location in NUSC_TO_NUPLAN_MAP:
            return NUSC_TO_NUPLAN_MAP[location]
        if location.startswith("singapore-"):
            return "sg-one-north"
        if location.startswith("boston-"):
            return "us-ma-boston"
        raise ValueError(f"Unsupported nuScenes location '{location}' for nuPlan map conversion.")

    def get_sample_data(self, sample_token: str, channel: str) -> Dict:
        if sample_token not in self.sample_data_by_sample_token:
            raise KeyError(f"Sample token '{sample_token}' has no sample_data rows.")
        sample_data_per_channel = self.sample_data_by_sample_token[sample_token]
        if channel not in sample_data_per_channel:
            raise KeyError(f"Channel '{channel}' not available for sample '{sample_token}'.")
        return sample_data_per_channel[channel]

    def get_annotation_tokens(self, sample_token: str) -> List[str]:
        return self.annotation_tokens_by_sample_token.get(sample_token, [])


class ProcessedNuScenesSceneLoader:
    """Builds NAVSIM-compatible scenes from processed_scenes + raw nuScenes tables."""

    def __init__(
        self,
        processed_scenes_root: Path,
        nuscenes_root: Path,
        scene_name: str,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig,
        max_samples: Optional[int] = None,
        timestamp_tolerance_s: float = 0.12,
    ):
        self._processed_scenes_root = Path(processed_scenes_root)
        self._nuscenes_root = Path(nuscenes_root)
        self._scene_name = scene_name
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self._timestamp_tolerance_s = timestamp_tolerance_s
        self._max_samples = max_samples

        self._processed_scene_dir = self._processed_scenes_root / self._scene_name
        if not self._processed_scene_dir.exists():
            raise FileNotFoundError(f"Processed scene directory does not exist: {self._processed_scene_dir}")
        self._meta_data_path = self._processed_scene_dir / "meta_data.json"
        if not self._meta_data_path.exists():
            raise FileNotFoundError(f"meta_data.json does not exist: {self._meta_data_path}")

        self._table_index = NuScenesTableIndex(self._nuscenes_root)
        self._scene_record = self._table_index.get_scene_by_name(self._scene_name)
        self._sample_chain = self._table_index.iter_scene_samples(self._scene_record["token"])
        self._map_name = self._table_index.infer_nuplan_map_name(self._scene_record["log_token"])

        self._processed_frame_groups = self._build_processed_frame_groups()
        self._matched_timestamps = self._match_timestamps_to_samples()
        self._ego_dynamic_state_by_sample_token = self._build_ego_dynamic_state_lookup()
        self.scene_frames_dicts = self._build_scene_windows()

    @property
    def tokens(self) -> List[str]:
        return list(self.scene_frames_dicts.keys())

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        return Scene.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._processed_scene_dir,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._processed_scene_dir,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
        )

    def _build_processed_frame_groups(self) -> Dict[float, Dict[str, Dict]]:
        meta_data = _load_json(self._meta_data_path)
        frames = meta_data["frames"]
        grouped: Dict[float, Dict[str, Dict]] = {}
        for frame in frames:
            timestamp = float(frame["timestamp"])
            channel = Path(frame["rgb_path"]).parts[-2]
            grouped.setdefault(timestamp, {})
            grouped[timestamp][channel] = frame
        return grouped

    def _match_timestamps_to_samples(self) -> List[Tuple[float, int]]:
        processed_timestamps = sorted(self._processed_frame_groups.keys())
        sample_times_s = np.array(
            [(sample["timestamp"] - self._sample_chain[0]["timestamp"]) / 1e6 for sample in self._sample_chain],
            dtype=np.float64,
        )

        matched: List[Tuple[float, int]] = []
        used_sample_indices = set()
        for timestamp in processed_timestamps:
            nearest_idx = int(np.argmin(np.abs(sample_times_s - timestamp)))
            if abs(sample_times_s[nearest_idx] - timestamp) > self._timestamp_tolerance_s:
                continue
            if nearest_idx in used_sample_indices:
                continue
            used_sample_indices.add(nearest_idx)
            matched.append((timestamp, nearest_idx))
        if not matched:
            raise RuntimeError(
                f"Failed to match processed timestamps for scene '{self._scene_name}'. "
                f"Check processed_scenes and nuScenes root alignment."
            )
        return matched

    def _build_ego_dynamic_state_lookup(self) -> Dict[str, List[float]]:
        times = []
        positions = []
        sample_tokens = []
        for sample in self._sample_chain:
            lidar_sd = self._table_index.get_sample_data(sample["token"], "LIDAR_TOP")
            ego = self._table_index.ego_pose_by_token[lidar_sd["ego_pose_token"]]
            sample_tokens.append(sample["token"])
            times.append(sample["timestamp"] / 1e6)
            positions.append(np.asarray(ego["translation"][:2], dtype=np.float64))
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        velocities = np.zeros_like(positions)
        for i in range(len(times)):
            if i == 0 and len(times) > 1:
                dt = max(times[i + 1] - times[i], 1e-3)
                velocities[i] = (positions[i + 1] - positions[i]) / dt
            elif i == len(times) - 1 and len(times) > 1:
                dt = max(times[i] - times[i - 1], 1e-3)
                velocities[i] = (positions[i] - positions[i - 1]) / dt
            elif len(times) > 2:
                dt = max(times[i + 1] - times[i - 1], 1e-3)
                velocities[i] = (positions[i + 1] - positions[i - 1]) / dt

        accelerations = np.zeros_like(positions)
        for i in range(len(times)):
            if i == 0 and len(times) > 1:
                dt = max(times[i + 1] - times[i], 1e-3)
                accelerations[i] = (velocities[i + 1] - velocities[i]) / dt
            elif i == len(times) - 1 and len(times) > 1:
                dt = max(times[i] - times[i - 1], 1e-3)
                accelerations[i] = (velocities[i] - velocities[i - 1]) / dt
            elif len(times) > 2:
                dt = max(times[i + 1] - times[i - 1], 1e-3)
                accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / dt

        output = {}
        for token, vel, acc in zip(sample_tokens, velocities, accelerations):
            output[token] = [float(vel[0]), float(vel[1]), float(acc[0]), float(acc[1])]
        return output

    def _build_scene_windows(self) -> Dict[str, List[Dict]]:
        num_history = self._scene_filter.num_history_frames
        num_future = self._scene_filter.num_future_frames
        num_frames = num_history + num_future

        if len(self._matched_timestamps) < num_frames:
            raise RuntimeError(
                f"Not enough matched frames ({len(self._matched_timestamps)}) for required window size ({num_frames})."
            )

        windows: Dict[str, List[Dict]] = {}
        center_start = num_history - 1
        center_end = len(self._matched_timestamps) - num_future
        for center in range(center_start, center_end):
            window_pairs = self._matched_timestamps[center - (num_history - 1) : center + num_future + 1]
            center_sample_idx = window_pairs[num_history - 1][1]
            center_sample = self._sample_chain[center_sample_idx]
            token = center_sample["token"]
            windows[token] = [self._build_frame_dict(ts, sample_idx) for ts, sample_idx in window_pairs]
            if self._max_samples is not None and len(windows) >= self._max_samples:
                break
        return windows

    def _build_frame_dict(self, processed_timestamp: float, sample_idx: int) -> Dict:
        sample = self._sample_chain[sample_idx]
        sample_token = sample["token"]
        processed_channels = self._processed_frame_groups[processed_timestamp]

        lidar_sd = self._table_index.get_sample_data(sample_token, "LIDAR_TOP")
        lidar_calib = self._table_index.calibrated_sensor_by_token[lidar_sd["calibrated_sensor_token"]]
        ego_pose = self._table_index.ego_pose_by_token[lidar_sd["ego_pose_token"]]

        cams = self._build_camera_dict(sample, lidar_calib, processed_channels)
        anns = self._build_annotation_dict(sample, ego_pose)

        return {
            "token": sample_token,
            "frame_idx": sample_idx,
            "timestamp": int(sample["timestamp"]),
            "log_name": f"external-{self._scene_name}",
            "log_token": self._scene_record["log_token"],
            "scene_name": self._scene_name,
            "scene_token": self._scene_record["token"],
            "map_location": self._map_name,
            "roadblock_ids": [],
            "vehicle_name": "external",
            "ego2global_translation": ego_pose["translation"],
            "ego2global_rotation": ego_pose["rotation"],
            "ego_dynamic_state": self._ego_dynamic_state_by_sample_token.get(sample_token, [0.0, 0.0, 0.0, 0.0]),
            "traffic_lights": [],
            "driving_command": [0, 1, 0, 0],
            "cams": cams,
            "lidar_path": "",
            "anns": anns,
            "sample_prev": sample["prev"],
            "sample_next": sample["next"],
        }

    def _build_camera_dict(self, sample: Dict, lidar_calib: Dict, processed_channels: Dict[str, Dict]) -> Dict[str, Dict]:
        camera_dict: Dict[str, Dict] = {
            cam_name: {
                "data_path": "",
                "sensor2lidar_rotation": np.eye(3).tolist(),
                "sensor2lidar_translation": [0.0, 0.0, 0.0],
                "cam_intrinsic": np.eye(3).tolist(),
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
            for cam_name in ALL_NAVSIM_CAMERAS
        }

        t_lidar_ego = _pose_matrix(lidar_calib["rotation"], lidar_calib["translation"])
        t_ego_lidar = np.linalg.inv(t_lidar_ego)

        for nusc_channel, navsim_name in CAMERA_CHANNEL_TO_NAVSIM.items():
            if nusc_channel not in processed_channels:
                continue
            try:
                cam_sd = self._table_index.get_sample_data(sample["token"], nusc_channel)
            except KeyError:
                continue
            cam_calib = self._table_index.calibrated_sensor_by_token[cam_sd["calibrated_sensor_token"]]
            t_cam_ego = _pose_matrix(cam_calib["rotation"], cam_calib["translation"])
            t_cam_lidar = t_ego_lidar @ t_cam_ego

            processed_frame = processed_channels[nusc_channel]
            intrinsics = np.asarray(processed_frame["intrinsics"], dtype=np.float64)
            cam_k = intrinsics[:3, :3].tolist()

            camera_dict[navsim_name] = {
                "data_path": str(processed_frame["rgb_path"]).lstrip("./"),
                "sensor2lidar_rotation": t_cam_lidar[:3, :3].tolist(),
                "sensor2lidar_translation": t_cam_lidar[:3, 3].tolist(),
                "cam_intrinsic": cam_k,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        return camera_dict

    def _build_annotation_dict(self, sample: Dict, ego_pose: Dict) -> Dict[str, List]:
        gt_boxes = []
        gt_names = []
        gt_velocity_3d = []
        instance_tokens = []
        track_tokens = []

        t_ego_global = _pose_matrix(ego_pose["rotation"], ego_pose["translation"])
        t_global_ego = np.linalg.inv(t_ego_global)

        for ann_token in self._table_index.get_annotation_tokens(sample["token"]):
            ann = self._table_index.sample_annotation_by_token[ann_token]
            instance = self._table_index.instance_by_token[ann["instance_token"]]
            category_name = self._table_index.category_by_token[instance["category_token"]]["name"]
            navsim_name = _map_category_to_navsim_name(category_name)
            if navsim_name is None:
                continue

            t_box_global = _pose_matrix(ann["rotation"], ann["translation"])
            t_box_ego = t_global_ego @ t_box_global
            yaw = float(np.arctan2(t_box_ego[1, 0], t_box_ego[0, 0]))

            size_wlh = ann["size"]
            box_length = float(size_wlh[1])
            box_width = float(size_wlh[0])
            box_height = float(size_wlh[2])

            gt_boxes.append(
                [
                    float(t_box_ego[0, 3]),
                    float(t_box_ego[1, 3]),
                    float(t_box_ego[2, 3]),
                    box_length,
                    box_width,
                    box_height,
                    yaw,
                ]
            )
            gt_names.append(navsim_name)
            gt_velocity_3d.append([0.0, 0.0, 0.0])
            instance_tokens.append(ann["instance_token"])
            track_tokens.append(ann["instance_token"])

        return {
            "gt_boxes": gt_boxes,
            "gt_names": gt_names,
            "gt_velocity_3d": gt_velocity_3d,
            "instance_tokens": instance_tokens,
            "track_tokens": track_tokens,
        }
