import argparse
import os
import pickle
from collections import deque
import yaml
from typing import Any, List, Dict, Union
import math
import glob

import torch
import numpy as np
import scipy.spatial.transform as spt
from scipy.spatial.transform import Rotation as SCR
import cv2
from omegaconf import OmegaConf

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.common.dataclasses import AgentInput, EgoStatus, Cameras, Camera, Lidar


OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
OPENCV2LIDAR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
nusc_cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK']
nuplan_cameras = ["cam_f0", "cam_r0", "cam_l0", "cam_b0"]

_checkpoint_path = '/home/ms/DrivoR/weights/release_checkpoints/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth'
_config_path = "/home/ms/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml"

def rt2pose(r, t, degrees=False):
    pose = np.eye(4)
    pose[:3, :3] = SCR.from_euler('XYZ', r, degrees=degrees).as_matrix()
    pose[:3, 3] = t
    return pose

def pose2rt(pose, degrees=False):
    r = SCR.from_matrix(pose[:3, :3]).as_euler('XYZ', degrees=degrees)
    t = pose[:3, 3]
    return r, t

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def create_cam(intrinsic, l2c, image):
    fovx, fovy = intrinsic['fovx'], intrinsic['fovy']
    h, w = intrinsic['H'], intrinsic['W']
    K = np.eye(3)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrinsic['cx'], intrinsic['cy']
    c2l = np.linalg.inv(l2c)
    cam = Camera(image=image, sensor2lidar_rotation=c2l[:3, :3], sensor2lidar_translation=c2l[:3, 3], intrinsics=K)
    return cam

def save_frame(raw_image, traj_poses, cam_params, save_fn):
    plan_traj = np.stack([
        traj_poses[:, 0],
        1.4*np.ones_like(traj_poses[:, 2]),
        traj_poses[:, 1],
    ], axis=1)
    
    intrinsic_dict = cam_params['CAM_FRONT']['intrinsic']
    intrinsic = np.eye(4)
    fx = fov2focal(intrinsic_dict['fovx'], intrinsic_dict['W'])
    fy = fov2focal(intrinsic_dict['fovy'], intrinsic_dict['H'])
    intrinsic[0, 0], intrinsic[1, 1] = fx, fy
    intrinsic[0, 2], intrinsic[1, 2] = intrinsic_dict['cx'], intrinsic_dict['cy']
    uvz_traj = (intrinsic[:3, :3] @ plan_traj.T).T
    uv_traj = uvz_traj[:, :2] / uvz_traj[:, 2][:, None]
    uv_traj_int = uv_traj.astype(int)
    
    images = []
    for cam in nusc_cameras:
        im = raw_image[cam]
        if cam == 'CAM_FRONT':
            for point in uv_traj_int:
                im = cv2.circle(im, point, radius=5, color=(192, 152, 7), thickness=-1)
        images.append(im)
    images = cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_fn+'.jpg', images)

def to_video(folder_path, fps=5, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height//downsample))
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
        
    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join(folder_path, 'video.mp4')
    if os.path.exists(mp4_path): 
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path, 
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def parse_args():

    parser = argparse.ArgumentParser(
        description='Evaluate DrivoR in closed-loop simulation')
    
    # Logging and output
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    return args

def main(args):

    config = OmegaConf.load(_config_path)

    drivor_model = DrivoRModel(config.config).cuda()

    if _checkpoint_path != "":
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(_checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(_checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"]
        drivor_model.load_state_dict({k.replace("agent._drivor_model.", ""): v for k, v in state_dict.items()})

    feature_builder = DrivoRFeatureBuilder(config.config)

    # Connect to pipes for HUGSIM
    obs_pipe = os.path.join(args.output, 'obs_pipe')
    plan_pipe = os.path.join(args.output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for recieving')

    cnt = 0
    output_folder = os.path.join(args.output, 'drivor')
    os.makedirs(output_folder, exist_ok=True)

    while True:

        with open(obs_pipe, "rb") as pipe:
            raw_data = pipe.read()
            raw_data = pickle.loads(raw_data)
        print('received')
        
        if raw_data == 'Done':
            print('Waiting for visualize tasks...')
            to_video(output_folder)
            exit(0)

        obs, info = raw_data

        imgs = {}
        raw_imgs = {}
        
        for cam in nusc_cameras:
            im = obs['rgb'][cam]
            print(f"[DEBUG] {cam}: {type(im)}, {im.shape if im is not None else 'NONE'}")
            # resize_im = cv2.resize(im, (1920, 1080))
            resize_im = im
            raw_imgs[cam] = im
            imgs[cam] = resize_im

        velo, acc = np.zeros(2), np.zeros(2)
        command = np.zeros(4)
        # print('cmd', info['command'])
        command[info['command']] = 1
        # command[1] = 1
        
        ego_pose = OPENCV2IMU @ info['ego_pos']
        # yaw = -info['ego_rot'][1]
        yaw = -info['ego_steer']
        forward_velo = info['ego_velo']
        forward_acc = info['accelerate']
        velo[0] = forward_velo * np.cos(yaw)
        velo[1] = forward_velo * np.sin(yaw)
        acc[0] = forward_acc * np.cos(yaw)
        acc[1] = forward_acc * np.sin(yaw)
        ego_status = EgoStatus(ego_pose, velo, acc, command)
    
        cameras_dict = {}
        for nusc_cam, nuplan_cam in zip(nusc_cameras, nuplan_cameras):
            curr_cam_params = info['cam_params'][nusc_cam]
            cameras_dict[nuplan_cam] = create_cam(curr_cam_params['intrinsic'], curr_cam_params['l2c'], imgs[nusc_cam])
            
        cameras = Cameras(
            cam_f0=Camera(),
            cam_l0=Camera(),
            cam_l1=Camera(),
            cam_l2=Camera(),
            cam_r0=Camera(),
            cam_r1=Camera(),
            cam_r2=Camera(),
            cam_b0=Camera(),
        )
        #cameras = Cameras(*([None] * 8))
        for cam_name, cam in cameras_dict.items():
            setattr(cameras, cam_name, cam)
        
        lidar = Lidar(np.zeros((6, 100), dtype=np.float32))
        
        agent_input = AgentInput([ego_status], [cameras], [lidar])
    
        features = feature_builder.compute_features(agent_input)

        features['image'] = features['image'].unsqueeze(0).cuda()
        features['cam_K'] = features['cam_K'].unsqueeze(0).cuda()
        features['world_2_cam'] = features['world_2_cam'].unsqueeze(0).cuda()
        features['ego_status'] = features['ego_status'].unsqueeze(0).cuda()

        predictions = drivor_model(features)
        pred_traj = predictions['trajectory'].detach().cpu().numpy()[0] # (8, 3)

        way_points = pred_traj[:, [1, 0, 2]]
        way_points[:, 0] *= -1
            
        save_fn = os.path.join(output_folder, f'{str(cnt).zfill(4)}')
        save_frame(raw_imgs, way_points, info['cam_params'], save_fn)

        with open(plan_pipe, "wb") as pipe:
            pipe.write(pickle.dumps(way_points[:, :2]))
        print('sent')

        cnt += 1 

if __name__ == '__main__':
    args = parse_args()
    main(args)
