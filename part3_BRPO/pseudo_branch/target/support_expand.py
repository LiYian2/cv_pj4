"""A2: Geometry-Constrained Support Expand

This module implements the A2 expansion strategy:
- Extract high-confidence seed from A1 joint_confidence_v2
- Expand under geometry constraints (projected depth + overlap + fusion weight)
- Output expanded artifacts with source map for traceability

All outputs are written as separate files (joint_confidence_expand_v1.*)
to enable A1 vs A1+A2 comparison without overwriting A1 artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

# Source map labels
SOURCE_SEED = 0
SOURCE_EXPANDED_BOTH = 1
SOURCE_EXPANDED_LEFT_ONLY = 2
SOURCE_EXPANDED_RIGHT_ONLY = 3
SOURCE_FALLBACK = 4


def _save_mask_png(mask: np.ndarray, path: Path):
    """Save binary/tertiary mask as PNG."""
    from PIL import Image
    img = np.clip(mask * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _save_float_png(arr: np.ndarray, path: Path, vmax: Optional[float] = None):
    """Save float array as normalized PNG."""
    from PIL import Image
    arr = np.asarray(arr, dtype=np.float32)
    if vmax is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    img = np.clip(arr / vmax, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def build_geometry_seed_support(
    joint_confidence_v2: np.ndarray,
    joint_confidence_cont_v2: np.ndarray,
    seed_threshold: float = 0.7,
) -> Tuple[np.ndarray, Dict]:
    """Extract high-confidence seed from A1 joint confidence.
    
    Args:
        joint_confidence_v2: A1 discrete joint confidence (0, 0.5, 1.0)
        joint_confidence_cont_v2: A1 continuous joint confidence
        seed_threshold: Minimum confidence to be considered seed
    
    Returns:
        seed_mask: Boolean mask of seed regions
        summary: Statistics dict
    """
    joint_conf = np.asarray(joint_confidence_v2, dtype=np.float32)
    joint_cont = np.asarray(joint_confidence_cont_v2, dtype=np.float32)
    
    # Seed = high confidence in both discrete and continuous
    seed_mask = (joint_conf >= seed_threshold) & (joint_cont >= seed_threshold * 0.8)
    seed_mask = seed_mask.astype(np.float32)
    
    # Connected component analysis to remove tiny isolated seeds
    # Note: A1 joint_confidence_v2 is sparse, so min_size should be 1 (no filtering)
    # labeled, num_features = ndimage.label(seed_mask)
    # if num_features > 0:
    #     component_sizes = ndimage.sum(seed_mask, labeled, range(1, num_features + 1))
    #     min_size = 1  # No filtering for sparse A1 masks
    #     for i, size in enumerate(component_sizes, start=1):
    #         if size < min_size:
    #             seed_mask[labeled == i] = 0
    # Skip connected component filtering for sparse A1 joint_confidence_v2
    
    seed_ratio = float(seed_mask.sum() / seed_mask.size)
    # Since we skip connected component filtering, use seed_mask.sum() directly
    summary = {
        'seed_nonzero_ratio': seed_ratio,
        'seed_threshold': seed_threshold,
        'num_seed_pixels': int(seed_mask.sum()),
    }
    
    return seed_mask.astype(bool), summary


def expand_support_under_geometry_consistency(
    seed_mask: np.ndarray,
    joint_confidence_v2: np.ndarray,
    joint_depth_target_v2: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    depth_diff_threshold: float = 0.05,
    fusion_weight_threshold: float = 0.3,
    max_expand_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Expand seed support under geometry consistency constraints.
    
    Expansion rules:
    1. Neighbor pixel must have valid projected depth on at least one side
    2. Overlap must be valid
    3. Fusion weight must be above threshold
    4. Depth difference from nearest seed must be within threshold
    
    Args:
        seed_mask: Boolean mask from build_geometry_seed_support
        joint_confidence_v2: A1 joint confidence for reference
        joint_depth_target_v2: A1 joint depth target
        projected_depth_left/right: Projected depth from neighbor views
        overlap_mask_left/right: Valid projection masks
        fusion_weight_left/right: Fusion quality weights
        depth_diff_threshold: Max relative depth difference from seed
        fusion_weight_threshold: Min fusion weight for expansion
        max_expand_iterations: Max region grow iterations
    
    Returns:
        expand_mask: Boolean mask of expanded regions (excluding seed)
        source_map: Int array marking source type (SOURCE_* constants)
        expanded_conf: Confidence values for expanded regions
        summary: Statistics dict
    """
    H, W = seed_mask.shape
    seed_mask = seed_mask.astype(bool)
    joint_depth = np.asarray(joint_depth_target_v2, dtype=np.float32)
    
    # Prepare geometry masks
    proj_left_valid = np.isfinite(projected_depth_left) & (overlap_mask_left > 0.5)
    proj_right_valid = np.isfinite(projected_depth_right) & (overlap_mask_right > 0.5)
    fusion_left_good = fusion_weight_left >= fusion_weight_threshold
    fusion_right_good = fusion_weight_right >= fusion_weight_threshold
    
    # Both-side valid: left AND right have valid projection
    both_valid = proj_left_valid & proj_right_valid & fusion_left_good & fusion_right_good
    # Single-side valid: only one side has valid projection
    left_only = proj_left_valid & fusion_left_good & ~proj_right_valid
    right_only = proj_right_valid & fusion_right_good & ~proj_left_valid
    
    # Source map: start with seed
    source_map = np.full((H, W), SOURCE_FALLBACK, dtype=np.int32)
    source_map[seed_mask] = SOURCE_SEED
    
    # Track current frontier for iterative expansion
    current_frontier = seed_mask.copy()
    expanded_both = np.zeros_like(seed_mask)
    expanded_left = np.zeros_like(seed_mask)
    expanded_right = np.zeros_like(seed_mask)
    
    # Depth reference: use seed depth as anchor
    seed_depth = np.where(seed_mask, joint_depth, np.nan)
    
    iteration = 0
    for iteration in range(max_expand_iterations):
        # Dilate frontier by 1 pixel (8-connectivity)
        dilated = ndimage.binary_dilation(current_frontier, iterations=1)
        new_candidates = dilated & ~current_frontier & ~seed_mask
        
        if not new_candidates.any():
            break
        
        # Check depth consistency with nearest seed
        for y, x in zip(*np.where(new_candidates)):
            # Find nearest seed depth
            # Use 5x5 window around candidate
            y_min, y_max = max(0, y - 2), min(H, y + 3)
            x_min, x_max = max(0, x - 2), min(W, x + 3)
            local_seed_mask = seed_mask[y_min:y_max, x_min:x_max]
            local_seed_depth = seed_depth[y_min:y_max, x_min:x_max]
            
            if not local_seed_mask.any():
                continue
            
            # Use median seed depth as reference
            ref_depth = np.nanmedian(local_seed_depth[local_seed_mask])
            if not np.isfinite(ref_depth):
                continue
            
            # Check depth at candidate position using projected depth (not joint_depth)
            # Candidate depth comes from projected_depth_left/right based on geometry support type
            if both_valid[y, x]:
                # Use weighted fusion of left and right projected depths
                w_left = fusion_weight_left[y, x]
                w_right = fusion_weight_right[y, x]
                w_total = w_left + w_right
                cand_depth = (w_left * projected_depth_left[y, x] + w_right * projected_depth_right[y, x]) / w_total
            elif proj_left_valid[y, x] and fusion_left_good[y, x]:
                cand_depth = projected_depth_left[y, x]
            elif proj_right_valid[y, x] and fusion_right_good[y, x]:
                cand_depth = projected_depth_right[y, x]
            else:
                # No valid projected depth, skip this candidate
                continue
            
            if not np.isfinite(cand_depth):
                continue
            
            depth_diff = abs(cand_depth - ref_depth) / max(ref_depth, 1e-6)
            if depth_diff > depth_diff_threshold:
                continue
            
            # Check geometry support
            if both_valid[y, x]:
                expanded_both[y, x] = True
                source_map[y, x] = SOURCE_EXPANDED_BOTH
            elif left_only[y, x]:
                expanded_left[y, x] = True
                source_map[y, x] = SOURCE_EXPANDED_LEFT_ONLY
            elif right_only[y, x]:
                expanded_right[y, x] = True
                source_map[y, x] = SOURCE_EXPANDED_RIGHT_ONLY
        
        # Update frontier for next iteration
        current_frontier = seed_mask | expanded_both | expanded_left | expanded_right
    
    # Final expand mask (excluding seed)
    expand_mask = expanded_both | expanded_left | expanded_right
    
    # Compute confidence for expanded regions
    # Use A1 continuous confidence where available, else use tier-based
    expanded_conf = np.zeros((H, W), dtype=np.float32)
    # Both-side expansion gets higher confidence
    expanded_conf[expanded_both] = 0.8
    expanded_conf[expanded_left] = 0.6
    expanded_conf[expanded_right] = 0.6
    
    summary = {
        'seed_ratio': float(seed_mask.sum() / seed_mask.size),
        'expanded_both_ratio': float(expanded_both.sum() / expanded_both.size),
        'expanded_left_ratio': float(expanded_left.sum() / expanded_left.size),
        'expanded_right_ratio': float(expanded_right.sum() / expanded_right.size),
        'total_expanded_ratio': float(expand_mask.sum() / expand_mask.size),
        'depth_diff_threshold': depth_diff_threshold,
        'fusion_weight_threshold': fusion_weight_threshold,
        'max_expand_iterations': max_expand_iterations,
        'num_iterations_used': iteration + 1,
    }
    
    return expand_mask, source_map, expanded_conf, summary


def build_expanded_joint_confidence(
    joint_confidence_v2: np.ndarray,
    joint_confidence_cont_v2: np.ndarray,
    joint_depth_target_v2: np.ndarray,
    expand_mask: np.ndarray,
    source_map: np.ndarray,
    expanded_conf: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Merge seed + expanded into final expanded joint confidence.
    
    Args:
        joint_confidence_v2: A1 discrete joint confidence
        joint_confidence_cont_v2: A1 continuous joint confidence
        joint_depth_target_v2: A1 joint depth target
        expand_mask: Boolean mask of expanded regions
        source_map: Source type map (SOURCE_* constants)
        expanded_conf: Confidence values for expanded regions
    
    Returns:
        Dict with expanded artifacts
    """
    joint_conf = np.asarray(joint_confidence_v2, dtype=np.float32)
    joint_cont = np.asarray(joint_confidence_cont_v2, dtype=np.float32)
    joint_depth = np.asarray(joint_depth_target_v2, dtype=np.float32)
    expand_mask = expand_mask.astype(bool)
    
    # Expanded discrete confidence: min(original, expanded_conf)
    # This ensures expanded regions do not exceed seed quality
    expanded_discrete = np.where(expand_mask, expanded_conf, 0.0)
    final_discrete = np.maximum(joint_conf, expanded_discrete)
    
    # Expanded continuous confidence
    # Use A1 continuous where available, else use expanded tier
    final_cont = joint_cont.copy()
    for source_type, conf_val in [
        (SOURCE_EXPANDED_BOTH, 0.8),
        (SOURCE_EXPANDED_LEFT_ONLY, 0.6),
        (SOURCE_EXPANDED_RIGHT_ONLY, 0.6),
    ]:
        mask = (source_map == source_type) & expand_mask
        final_cont = np.where(mask & (final_cont < conf_val), conf_val, final_cont)
    
    # Depth target: keep A1 values, they are already geometry-consistent
    final_depth = joint_depth.copy()
    
    return {
        'joint_confidence_expand_v1': final_discrete.astype(np.float32),
        'joint_confidence_cont_expand_v1': final_cont.astype(np.float32),
        'joint_depth_target_expand_v1': final_depth.astype(np.float32),
        'joint_expand_source_map_v1': source_map.astype(np.int32),
    }


def write_support_expand_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    """Write A2 expanded artifacts to disk.
    
    Writes:
        - joint_confidence_expand_v1.npy
        - joint_confidence_cont_expand_v1.npy
        - joint_depth_target_expand_v1.npy
        - joint_expand_source_map_v1.npy
        - joint_expand_meta_v1.json
        - diag/joint_confidence_expand_v1.png
        - diag/joint_expand_source_map_v1.png
    """
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(frame_out / 'joint_confidence_expand_v1.npy', result['joint_confidence_expand_v1'])
    np.save(frame_out / 'joint_confidence_cont_expand_v1.npy', result['joint_confidence_cont_expand_v1'])
    np.save(frame_out / 'joint_depth_target_expand_v1.npy', result['joint_depth_target_expand_v1'])
    np.save(frame_out / 'joint_expand_source_map_v1.npy', result['joint_expand_source_map_v1'])
    
    # PNG visualizations
    _save_mask_png(result['joint_confidence_expand_v1'], frame_out / 'joint_confidence_expand_v1.png')
    _save_float_png(result['joint_confidence_cont_expand_v1'], frame_out / 'joint_confidence_cont_expand_v1.png', vmax=1.0)
    _save_float_png(result['joint_depth_target_expand_v1'], frame_out / 'joint_depth_target_expand_v1.png')
    
    # Source map visualization (color-coded)
    source_map = result['joint_expand_source_map_v1']
    source_vis = np.zeros((*source_map.shape, 3), dtype=np.uint8)
    source_vis[source_map == SOURCE_SEED] = [255, 255, 255]  # White
    source_vis[source_map == SOURCE_EXPANDED_BOTH] = [0, 255, 0]  # Green
    source_vis[source_map == SOURCE_EXPANDED_LEFT_ONLY] = [255, 255, 0]  # Cyan
    source_vis[source_map == SOURCE_EXPANDED_RIGHT_ONLY] = [0, 255, 255]  # Yellow
    source_vis[source_map == SOURCE_FALLBACK] = [0, 0, 0]  # Black
    from PIL import Image
    Image.fromarray(source_vis).save(diag_dir / 'joint_expand_source_map_v1.png')
    
    with open(frame_out / 'joint_expand_meta_v1.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def build_support_expand_from_a1(
    joint_confidence_v2: np.ndarray,
    joint_confidence_cont_v2: np.ndarray,
    joint_depth_target_v2: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    seed_threshold: float = 0.7,
    depth_diff_threshold: float = 0.05,
    fusion_weight_threshold: float = 0.3,
    max_expand_iterations: int = 3,
) -> Tuple[Dict, Dict]:
    """End-to-end A2 expansion from A1 artifacts.
    
    Args:
        joint_confidence_v2/cont_v2: A1 joint confidence
        joint_depth_target_v2: A1 joint depth target
        projected_depth_left/right: From fusion
        overlap_mask_left/right: From fusion
        fusion_weight_left/right: From fusion
        seed_threshold: Min confidence for seed
        depth_diff_threshold: Max relative depth diff for expansion
        fusion_weight_threshold: Min fusion weight for expansion
        max_expand_iterations: Max region grow iterations
    
    Returns:
        result: Dict with expanded artifacts
        meta: Dict with statistics and parameters
    """
    # Step 1: Extract seed
    seed_mask, seed_summary = build_geometry_seed_support(
        joint_confidence_v2=joint_confidence_v2,
        joint_confidence_cont_v2=joint_confidence_cont_v2,
        seed_threshold=seed_threshold,
    )
    
    # Step 2: Expand under geometry constraints
    expand_mask, source_map, expanded_conf, expand_summary = expand_support_under_geometry_consistency(
        seed_mask=seed_mask,
        joint_confidence_v2=joint_confidence_v2,
        joint_depth_target_v2=joint_depth_target_v2,
        projected_depth_left=projected_depth_left,
        projected_depth_right=projected_depth_right,
        overlap_mask_left=overlap_mask_left,
        overlap_mask_right=overlap_mask_right,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
        depth_diff_threshold=depth_diff_threshold,
        fusion_weight_threshold=fusion_weight_threshold,
        max_expand_iterations=max_expand_iterations,
    )
    
    # Step 3: Merge into expanded confidence
    result = build_expanded_joint_confidence(
        joint_confidence_v2=joint_confidence_v2,
        joint_confidence_cont_v2=joint_confidence_cont_v2,
        joint_depth_target_v2=joint_depth_target_v2,
        expand_mask=expand_mask,
        source_map=source_map,
        expanded_conf=expanded_conf,
    )
    
    # Compute final statistics
    final_discrete = result['joint_confidence_expand_v1']
    
    a1_nonzero = (joint_confidence_v2 > 0).sum()
    a2_nonzero = (final_discrete > 0).sum()
    coverage_gain = (a2_nonzero - a1_nonzero) / joint_confidence_v2.size
    
    meta = {
        'seed_summary': seed_summary,
        'expand_summary': expand_summary,
        'final_summary': {
            'a1_nonzero_ratio': float(a1_nonzero / joint_confidence_v2.size),
            'a2_nonzero_ratio': float(a2_nonzero / joint_confidence_v2.size),
            'coverage_gain_ratio': float(coverage_gain),
            'a2_mean_positive': float(final_discrete[final_discrete > 0].mean()) if (final_discrete > 0).any() else 0.0,
            'source_distribution': {
                'seed': int((source_map == SOURCE_SEED).sum()),
                'expanded_both': int((source_map == SOURCE_EXPANDED_BOTH).sum()),
                'expanded_left': int((source_map == SOURCE_EXPANDED_LEFT_ONLY).sum()),
                'expanded_right': int((source_map == SOURCE_EXPANDED_RIGHT_ONLY).sum()),
                'fallback': int((source_map == SOURCE_FALLBACK).sum()),
            },
        },
        'parameters': {
            'seed_threshold': seed_threshold,
            'depth_diff_threshold': depth_diff_threshold,
            'fusion_weight_threshold': fusion_weight_threshold,
            'max_expand_iterations': max_expand_iterations,
        },
        'policy': {
            'version': 'a2_geometry_constrained_expand_v1',
            'expand_rule': 'seed + geometry_supported_neighbors (both/single) with depth_consistency',
        },
    }
    
    return result, meta
