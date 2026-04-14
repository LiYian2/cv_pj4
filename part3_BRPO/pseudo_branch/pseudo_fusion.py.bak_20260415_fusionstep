# -*- coding: utf-8 -*-
"""Pseudo RGB fusion: combine left/right conditioned restorations."""
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, Tuple, Optional


def compute_branch_score(
    confidence: np.ndarray,
    depth_gate: np.ndarray,
    view_weight: float
) -> np.ndarray:
    """Compute per-pixel branch score S = C * G * V."""
    return confidence * depth_gate * view_weight


def compute_depth_gate(
    render_depth: np.ndarray,
    branch_depth: np.ndarray,
    tau_d: float = 0.1
) -> np.ndarray:
    """Compute depth consistency gate G = exp(-|D_r - D_branch| / (tau_d * D_r))."""
    valid = branch_depth > 0.001
    gate = np.zeros_like(render_depth)
    if valid.any():
        rel_diff = np.abs(render_depth[valid] - branch_depth[valid]) / (render_depth[valid] * tau_d + 1e-8)
        gate[valid] = np.exp(-rel_diff)
    return gate


def compute_view_weight(
    mean_confidence: float,
    valid_ratio: float,
    alpha1: float = 0.7,
    alpha2: float = 0.3
) -> float:
    """Compute view-level weight V = alpha1 * mean_conf + alpha2 * valid_ratio."""
    return alpha1 * mean_confidence + alpha2 * valid_ratio


def compute_agreement_map_rgb(
    I_L: np.ndarray,
    I_R: np.ndarray,
    tau_rgb: float = 10.0
) -> np.ndarray:
    """Compute RGB agreement A_rgb = exp(-||I_L - I_R||_1 / tau_rgb)."""
    diff = np.abs(I_L - I_R).astype(np.float32)
    l1_per_pixel = np.sum(diff, axis=2)
    agreement = np.exp(-l1_per_pixel / tau_rgb)
    return agreement


def compute_agreement_map_depth(
    D_L: np.ndarray,
    D_R: np.ndarray,
    render_depth: np.ndarray,
    tau_ld: float = 0.05
) -> np.ndarray:
    """Compute depth agreement A_depth = exp(-|D_L - D_R| / (tau_ld * D_r))."""
    valid = (D_L > 0.001) & (D_R > 0.001)
    agreement = np.zeros_like(render_depth)
    if valid.any():
        rel_diff = np.abs(D_L[valid] - D_R[valid]) / (render_depth[valid] * tau_ld + 1e-8)
        agreement[valid] = np.exp(-rel_diff)
    return agreement


def compute_agreement_map(
    I_L: np.ndarray,
    I_R: np.ndarray,
    D_L: Optional[np.ndarray],
    D_R: Optional[np.ndarray],
    render_depth: Optional[np.ndarray],
    use_depth_agreement: bool = False,
    tau_rgb: float = 10.0,
    tau_ld: float = 0.05
) -> np.ndarray:
    """Compute full agreement map A = A_rgb * A_depth (or just A_rgb)."""
    A_rgb = compute_agreement_map_rgb(I_L, I_R, tau_rgb)
    if use_depth_agreement and D_L is not None and D_R is not None and render_depth is not None:
        A_depth = compute_agreement_map_depth(D_L, D_R, render_depth, tau_ld)
        return A_rgb * A_depth
    return A_rgb


def fuse_residual_targets(
    I_render: np.ndarray,
    I_L: np.ndarray,
    I_R: np.ndarray,
    W_L: np.ndarray,
    W_R: np.ndarray
) -> np.ndarray:
    """Fuse residuals: I_F = I_r + W_L * (I_L - I_r) + W_R * (I_R - I_r)."""
    R_L = I_L.astype(np.float32) - I_render.astype(np.float32)
    R_R = I_R.astype(np.float32) - I_render.astype(np.float32)
    W_L_3 = W_L[:, :, None]
    W_R_3 = W_R[:, :, None]
    I_fused = I_render.astype(np.float32) + W_L_3 * R_L + W_R_3 * R_R
    return np.clip(I_fused, 0, 255).astype(np.uint8)


def build_fused_confidence(
    W_L: np.ndarray,
    W_R: np.ndarray,
    A: np.ndarray,
    tau_c: float = 1.0
) -> np.ndarray:
    """Compute fused confidence C_F = min(1, (W_L + W_R) / tau_c) * A."""
    raw_weight = W_L + W_R
    normalized = np.minimum(raw_weight / tau_c, 1.0)
    return normalized * A


def run_fusion_for_sample(
    render_rgb_path: str,
    target_rgb_left_path: str,
    target_rgb_right_path: str,
    depth_left: Optional[np.ndarray],
    conf_left: Optional[np.ndarray],
    depth_right: Optional[np.ndarray],
    conf_right: Optional[np.ndarray],
    render_depth: Optional[np.ndarray],
    output_dir: str,
    use_depth_agreement: bool = False,
    tau_rgb: float = 10.0,
    tau_d: float = 0.1,
    tau_ld: float = 0.05,
    tau_c: float = 1.0,
    alpha1: float = 0.7,
    alpha2: float = 0.3
) -> Dict:
    """Run full fusion pipeline for a single pseudo sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    I_render = np.array(Image.open(render_rgb_path).convert('RGB'))
    I_L = np.array(Image.open(target_rgb_left_path).convert('RGB'))
    I_R = np.array(Image.open(target_rgb_right_path).convert('RGB'))
    H, W = I_render.shape[:2]
    
    # View weights
    if conf_left is not None:
        mean_conf_L = float(conf_left[conf_left > 0].mean()) if (conf_left > 0).any() else 0.0
        valid_ratio_L = float((conf_left > 0).sum()) / (H * W)
    else:
        mean_conf_L, valid_ratio_L = 0.0, 0.0
    
    if conf_right is not None:
        mean_conf_R = float(conf_right[conf_right > 0].mean()) if (conf_right > 0).any() else 0.0
        valid_ratio_R = float((conf_right > 0).sum()) / (H * W)
    else:
        mean_conf_R, valid_ratio_R = 0.0, 0.0
    
    V_L = compute_view_weight(mean_conf_L, valid_ratio_L, alpha1, alpha2)
    V_R = compute_view_weight(mean_conf_R, valid_ratio_R, alpha1, alpha2)
    
    # Depth gates
    if depth_left is not None and render_depth is not None:
        G_L = compute_depth_gate(render_depth, depth_left, tau_d)
    else:
        G_L = np.ones((H, W), dtype=np.float32)
    if depth_right is not None and render_depth is not None:
        G_R = compute_depth_gate(render_depth, depth_right, tau_d)
    else:
        G_R = np.ones((H, W), dtype=np.float32)
    
    # Branch scores
    S_L = compute_branch_score(conf_left if conf_left is not None else np.zeros((H,W)), G_L, V_L)
    S_R = compute_branch_score(conf_right if conf_right is not None else np.zeros((H,W)), G_R, V_R)
    
    # Agreement
    A = compute_agreement_map(I_L, I_R, depth_left, depth_right, render_depth,
                              use_depth_agreement=use_depth_agreement,
                              tau_rgb=tau_rgb, tau_ld=tau_ld)
    
    # Weights
    W_L = A * S_L
    W_R = A * S_R
    W_sum = W_L + W_R + 1e-8
    W_L_norm = W_L / W_sum
    W_R_norm = W_R / W_sum
    
    # Fuse
    I_fused = fuse_residual_targets(I_render, I_L, I_R, W_L_norm, W_R_norm)
    C_fused = build_fused_confidence(W_L, W_R, A, tau_c)
    
    # Save
    Image.fromarray(I_fused).save(output_dir / 'target_rgb_fused.png')
    np.save(output_dir / 'confidence_mask_fused.npy', C_fused)
    conf_img = (np.clip(C_fused, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(conf_img).save(output_dir / 'confidence_mask_fused.png')
    
    stats = {
        'use_depth_agreement': use_depth_agreement,
        'tau_rgb': tau_rgb, 'tau_d': tau_d, 'tau_ld': tau_ld, 'tau_c': tau_c,
        'alpha1': alpha1, 'alpha2': alpha2,
        'V_L': V_L, 'V_R': V_R,
        'mean_agreement': float(A[A > 0].mean()) if (A > 0).any() else 0.0,
        'mean_conf_fused': float(C_fused[C_fused > 0].mean()) if (C_fused > 0).any() else 0.0,
        'support_ratio_fused': float((C_fused > 0.01).sum()) / (H * W),
        'support_pixels_fused': int((C_fused > 0.01).sum()),
    }
    
    with open(output_dir / 'fusion_meta.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats, I_render, I_L, I_R, W_L, W_R, A, C_fused


def get_fusion_diag_images(I_L, I_R, W_L, W_R, A, C_fused, conf_left, conf_right):
    """Generate diagnostic images for fusion."""
    diag = {}
    diff = np.abs(I_L - I_R).astype(np.float32)
    diag['rgb_disagreement'] = np.clip(np.sum(diff, axis=2), 0, 255).astype(np.uint8)
    diag['weight_left'] = (np.clip(W_L, 0, 1) * 255).astype(np.uint8)
    diag['weight_right'] = (np.clip(W_R, 0, 1) * 255).astype(np.uint8)
    diag['agreement'] = (np.clip(A, 0, 1) * 255).astype(np.uint8)
    diag['confidence_fused'] = (np.clip(C_fused, 0, 1) * 255).astype(np.uint8)
    if conf_left is not None:
        diag['support_left'] = ((conf_left > 0.01).astype(np.float32) * 255).astype(np.uint8)
    if conf_right is not None:
        diag['support_right'] = ((conf_right > 0.01).astype(np.float32) * 255).astype(np.uint8)
    diag['support_fused'] = ((C_fused > 0.01).astype(np.float32) * 255).astype(np.uint8)
    return diag
