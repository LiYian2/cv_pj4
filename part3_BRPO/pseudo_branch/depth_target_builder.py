# -*- coding: utf-8 -*-
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

def load_depth(path, depth_scale=5000.0):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    elif path.suffix in (".png", ".jpg"):
        img = np.array(Image.open(path))
        if img.dtype == np.uint16:
            return img.astype(np.float32) / depth_scale
        return img.astype(np.float32) / 255.0
    raise ValueError("Unsupported depth format: " + str(path))

def load_camera(camera_path):
    with open(camera_path) as f:
        return json.load(f)

def get_intrinsic_matrix(intr, height=512, width=512):
    fx = intr.get("fx", 500)
    fy = intr.get("fy", fx)
    cx = intr.get("cx", width / 2)
    cy = intr.get("cy", height / 2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K

def reproject_depth(depth_src, K_src, K_dst, pose_src_c2w, pose_dst_c2w, height=512, width=512):
    w2c_dst = np.linalg.inv(pose_dst_c2w)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    depth_flat = depth_src.reshape(-1)
    valid_src = depth_flat > 0.001
    rays = (np.linalg.inv(K_src) @ uv.T).T
    pts_cam = rays * depth_flat[:, None]
    pts_world = (pose_src_c2w[:3, :3] @ pts_cam.T + pose_src_c2w[:3, 3:4]).T
    pts_dst = (w2c_dst[:3, :3] @ pts_world.T + w2c_dst[:3, 3:4]).T
    pts_2d = (K_dst @ pts_dst.T).T
    z = pts_2d[:, 2]
    valid_z = z > 0.01
    u_dst = pts_2d[:, 0] / (z + 1e-8)
    v_dst = pts_2d[:, 1] / (z + 1e-8)
    valid_u = (u_dst >= 0) & (u_dst < width)
    valid_v = (v_dst >= 0) & (v_dst < height)
    valid = valid_src & valid_z & valid_u & valid_v
    depth_reproj = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)
    u_idx = np.clip(np.round(u_dst[valid]).astype(int), 0, width-1)
    v_idx = np.clip(np.round(v_dst[valid]).astype(int), 0, height-1)
    z_vals = z[valid]
    for i in range(len(u_idx)):
        ui, vi = u_idx[i], v_idx[i]
        if valid_mask[vi, ui]:
            if z_vals[i] < depth_reproj[vi, ui]:
                depth_reproj[vi, ui] = z_vals[i]
        else:
            depth_reproj[vi, ui] = z_vals[i]
            valid_mask[vi, ui] = True
    return depth_reproj, valid_mask
