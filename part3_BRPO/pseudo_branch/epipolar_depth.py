# -*- coding: utf-8 -*-
"""Epipolar Depth Priors (EDP) - compute target_depth from RGB images."""
import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from pathlib import Path


def load_camera(camera_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract K (3x3 intrinsic) and pose (4x4 c2w) from camera dict."""
    intr = camera_dict.get("intrinsics_px", {})
    H = camera_dict.get("image_size", {}).get("height", 512)
    W = camera_dict.get("image_size", {}).get("width", 512)
    
    fx = intr.get("fx", 500)
    fy = intr.get("fy", fx)
    cx = intr.get("cx", W / 2)
    cy = intr.get("cy", H / 2)
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    pose = np.array(camera_dict["pose_c2w"], dtype=np.float64)
    
    return K, pose


def pose_c2w_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    """Convert c2w to w2c."""
    return np.linalg.inv(pose_c2w)


def compute_fundamental_matrix(K1: np.ndarray, pose1: np.ndarray, 
                                K2: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """Compute fundamental matrix from camera parameters."""
    w2c1 = pose_c2w_to_w2c(pose1)
    w2c2 = pose_c2w_to_w2c(pose2)
    
    R_rel = w2c2[:3, :3] @ np.linalg.inv(w2c1[:3, :3])
    t_rel = w2c2[:3, 3] - R_rel @ w2c1[:3, 3]
    
    t_skew = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    E = t_skew @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    
    return F


def compute_epipolar_distance(pts: np.ndarray, F: np.ndarray, pts_other: np.ndarray) -> np.ndarray:
    """
    Compute point-to-epipolar-line distance.
    
    For each point in pts_other, compute distance to epipolar line from pts.
    """
    # Epipolar line: l = F @ [x, y, 1]^T
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    lines = (F @ pts_h.T).T  # (N, 3)
    
    # Distance from pts_other to lines
    pts_other_h = np.hstack([pts_other, np.ones((pts_other.shape[0], 1))])
    
    # Distance = |ax + by + c| / sqrt(a^2 + b^2)
    distances = np.abs(np.sum(lines * pts_other_h, axis=1)) / (
        np.sqrt(lines[:, 0]**2 + lines[:, 1]**2) + 1e-8
    )
    
    return distances


def triangulate_depth(pts1: np.ndarray, pts2: np.ndarray,
                      K1: np.ndarray, pose1: np.ndarray,
                      K2: np.ndarray, pose2: np.ndarray,
                      H: int = 512, W: int = 512) -> np.ndarray:
    """Triangulate depth from point correspondences."""
    w2c1 = pose_c2w_to_w2c(pose1)
    w2c2 = pose_c2w_to_w2c(pose2)
    
    P1 = K1 @ w2c1[:3, :]
    P2 = K2 @ w2c2[:3, :]
    
    pts1_h = pts1.astype(np.float64).T
    pts2_h = pts2.astype(np.float64).T
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = points_4d[:3] / (points_4d[3:4] + 1e-8)
    
    points_cam1 = w2c1[:3, :3] @ points_3d + w2c1[:3, 3:4]
    depths = points_cam1[2, :]
    
    depth_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    for i, (pt, d) in enumerate(zip(pts1, depths)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < W and 0 <= y < H and d > 0.01:
            depth_map[y, x] += d
            count_map[y, x] += 1
    
    valid = count_map > 0
    depth_map[valid] /= count_map[valid]
    
    return depth_map


def compute_edp_depth(
    pseudo_rgb_path: str,
    pseudo_camera: Dict,
    ref_rgb_path: str,
    ref_camera: Dict,
    flow_matcher=None,
    confidence_threshold: float = 0.5,
    size: int = 512
) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[Dict]]:
    """
    Compute epipolar depth for pseudo view using one reference frame.
    
    Returns:
        depth_epi: (H, W) depth map
        confidence: (H, W) confidence map
        stats: statistics dict
        diag_extra: extra info for visualization (pts, epipolar_distances, etc.)
    """
    from flow_matcher import FlowMatcher
    
    if flow_matcher is None:
        flow_matcher = FlowMatcher()
    
    # Get matches
    pts_pseudo, pts_ref, match_conf = flow_matcher.match_pair(pseudo_rgb_path, ref_rgb_path, size=size)
    
    if len(pts_pseudo) < 100:
        return (
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            {"error": "insufficient matches", "num_matches": len(pts_pseudo)},
            None
        )
    
    # Filter by confidence
    valid = match_conf > confidence_threshold
    pts_pseudo = pts_pseudo[valid]
    pts_ref = pts_ref[valid]
    match_conf = match_conf[valid]
    
    if len(pts_pseudo) < 50:
        return (
            np.zeros((size, size), dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
            {"error": "insufficient matches after filtering", "num_matches": len(pts_pseudo)},
            None
        )
    
    # Load cameras
    K_pseudo, pose_pseudo = load_camera(pseudo_camera)
    K_ref, pose_ref = load_camera(ref_camera)
    
    # Compute fundamental matrix
    F = compute_fundamental_matrix(K_pseudo, pose_pseudo, K_ref, pose_ref)
    
    # Compute epipolar distance
    epipolar_distances = compute_epipolar_distance(pts_pseudo, F, pts_ref)
    
    # Epipolar confidence
    epipolar_conf = np.exp(-epipolar_distances / 5.0)
    
    # Combined confidence
    combined_conf = match_conf * epipolar_conf
    
    # Normalize to standard weights in [0, 1] on valid matches only.
    # MASt3R desc_conf is not a probability; using it raw makes downstream
    # confidence_mask scale unstable. Percentile-clip then normalize.
    if combined_conf.size > 0:
        hi = np.percentile(combined_conf, 95.0)
        if hi > 1e-8:
            combined_conf = np.clip(combined_conf, 0.0, hi) / hi
        else:
            combined_conf = np.zeros_like(combined_conf, dtype=np.float32)
    combined_conf = combined_conf.astype(np.float32)
    
    # Triangulate depth
    depth_epi = triangulate_depth(
        pts_pseudo, pts_ref,
        K_pseudo, pose_pseudo, K_ref, pose_ref,
        H=size, W=size
    )
    
    # Create confidence map
    confidence_map = np.zeros((size, size), dtype=np.float32)
    for pt, c in zip(pts_pseudo, combined_conf):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < size and 0 <= y < size:
            confidence_map[y, x] = max(confidence_map[y, x], c)
    
    stats = {
        "num_matches": len(pts_pseudo),
        "mean_epipolar_dist": float(np.mean(epipolar_distances)),
        "mean_match_conf": float(np.mean(match_conf)),
        "depth_valid_pixels": int((depth_epi > 0).sum()),
    }
    
    # Extra info for visualization
    diag_extra = {
        "pts_pseudo": pts_pseudo,
        "pts_ref": pts_ref,
        "epipolar_distances": epipolar_distances,
        "match_confidence": match_conf,
        "img1_path": pseudo_rgb_path,
        "img2_path": ref_rgb_path,
    }
    
    return depth_epi, confidence_map, stats, diag_extra


def compute_edp_depth_bidirectional(
    pseudo_rgb_path: str,
    pseudo_camera: Dict,
    left_rgb_path: str,
    left_camera: Dict,
    right_rgb_path: str,
    right_camera: Dict,
    flow_matcher=None,
    size: int = 512
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute EDP depth using both left and right reference frames.
    """
    # Compute from left reference
    depth_left, conf_left, stats_left, diag_left = compute_edp_depth(
        pseudo_rgb_path, pseudo_camera, left_rgb_path, left_camera, flow_matcher, size=size
    )
    
    # Compute from right reference
    depth_right, conf_right, stats_right, diag_right = compute_edp_depth(
        pseudo_rgb_path, pseudo_camera, right_rgb_path, right_camera, flow_matcher, size=size
    )
    
    # Fuse: take maximum confidence
    target_depth = np.where(conf_left >= conf_right, depth_left, depth_right)
    confidence = np.maximum(conf_left, conf_right)
    
    # Combined stats
    stats = {
        "left": stats_left,
        "right": stats_right,
        "fused_valid_pixels": int((target_depth > 0).sum()),
        "mean_confidence": float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0,
    }
    
    # Return diag_extra from the better side
    if diag_left and diag_right:
        # Use left as primary (arbitrary choice)
        stats["diag_primary"] = "left"
    elif diag_left:
        stats["diag_primary"] = "left"
    elif diag_right:
        stats["diag_primary"] = "right"
    
    return target_depth, confidence, stats
