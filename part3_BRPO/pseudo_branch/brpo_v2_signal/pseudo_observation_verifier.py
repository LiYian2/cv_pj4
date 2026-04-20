from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def _normalize(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def _relative_consistency(a: np.ndarray, b: np.ndarray, valid: np.ndarray, tol: float = 0.10) -> np.ndarray:
    rel = np.zeros_like(a, dtype=np.float32)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)
    rel[valid] = np.abs(a[valid] - b[valid]) / denom[valid]
    return np.clip(1.0 - rel / max(float(tol), 1e-6), 0.0, 1.0)


def _save_mask_png(mask: np.ndarray, path: Path):
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_float_png(arr: np.ndarray, path: Path, vmax: float | None = None):
    arr = np.asarray(arr, dtype=np.float32)
    if vmax is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    img = np.clip(arr / vmax, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def build_pseudo_observation_verifier(
    depth_target: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    render_depth: np.ndarray | None = None,
    reference_confidence_joint: np.ndarray | None = None,
    relative_depth_tol: float = 0.10,
    xor_confidence: float = 0.5,
    min_verify_confidence: float = 0.05,
) -> Dict[str, np.ndarray | Dict]:
    depth_target = np.asarray(depth_target, dtype=np.float32)
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    overlap_mask_left = _normalize(overlap_mask_left)
    overlap_mask_right = _normalize(overlap_mask_right)
    render_depth = np.asarray(render_depth, dtype=np.float32) if render_depth is not None else None
    reference_confidence_joint = np.asarray(reference_confidence_joint, dtype=np.float32) if reference_confidence_joint is not None else None

    target_valid = depth_target > 1e-6
    valid_left = (projected_depth_left > 1e-6) & (overlap_mask_left > 0.5) & target_valid
    valid_right = (projected_depth_right > 1e-6) & (overlap_mask_right > 0.5) & target_valid

    left_consistency = _relative_consistency(projected_depth_left, depth_target, valid_left, tol=relative_depth_tol)
    right_consistency = _relative_consistency(projected_depth_right, depth_target, valid_right, tol=relative_depth_tol)
    verify_left = valid_left & (left_consistency > 0.0)
    verify_right = valid_right & (right_consistency > 0.0)

    verify_both = verify_left & verify_right
    verify_xor = verify_left ^ verify_right
    verify_union = verify_left | verify_right

    verify_confidence = np.zeros_like(depth_target, dtype=np.float32)
    verify_confidence[verify_both] = 1.0
    verify_confidence[verify_xor] = float(xor_confidence)
    valid_mask = (verify_confidence > float(min_verify_confidence)) & target_valid
    verify_confidence = np.where(valid_mask, verify_confidence, 0.0).astype(np.float32)

    render_consistency = None
    if render_depth is not None:
        valid_render = (render_depth > 1e-6) & target_valid
        render_consistency = _relative_consistency(render_depth, depth_target, valid_render, tol=max(float(relative_depth_tol), 1e-6) * 1.5).astype(np.float32)

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'verify_left_ratio': float(verify_left.mean()),
        'verify_right_ratio': float(verify_right.mean()),
        'verify_both_ratio': float(verify_both.mean()),
        'verify_xor_ratio': float(verify_xor.mean()),
        'verify_union_ratio': float(verify_union.mean()),
        'verify_confidence_nonzero_ratio': float((verify_confidence > 0).mean()),
        'verify_confidence_mean_positive': float(verify_confidence[verify_confidence > 0].mean()) if (verify_confidence > 0).any() else 0.0,
        'left_consistency_mean_positive': float(left_consistency[verify_left].mean()) if verify_left.any() else 0.0,
        'right_consistency_mean_positive': float(right_consistency[verify_right].mean()) if verify_right.any() else 0.0,
        'target_nonzero_ratio': float(target_valid.mean()),
        'xor_confidence': float(xor_confidence),
        'relative_depth_tol': float(relative_depth_tol),
        'policy': {
            'version': 'brpo_verify_v1_proxy',
            'confidence_rule': 'both->1.0 xor->0.5 none->0.0',
            'confidence_source': 'independent left/right geometric verification proxy against current depth_target',
            'target_reused_from': 'pseudo_depth_target_joint_v1',
        },
    }
    if render_consistency is not None:
        render_valid = (render_depth > 1e-6) & target_valid
        summary['render_consistency_mean_positive'] = float(render_consistency[render_valid].mean()) if render_valid.any() else 0.0
    if reference_confidence_joint is not None:
        reference_positive = reference_confidence_joint > 0
        verify_positive = verify_confidence > 0
        disagreement = np.logical_xor(reference_positive, verify_positive)
        summary['reference_joint_nonzero_ratio'] = float(reference_positive.mean())
        summary['verify_vs_joint_disagreement_ratio'] = float(disagreement.mean())
        summary['verify_positive_joint_positive_overlap_ratio'] = float((verify_positive & reference_positive).mean())

    return {
        'pseudo_confidence_verify_v1': verify_confidence,
        'pseudo_valid_mask_verify_v1': valid_mask.astype(np.float32),
        'pseudo_verify_left_v1': verify_left.astype(np.float32),
        'pseudo_verify_right_v1': verify_right.astype(np.float32),
        'pseudo_verify_both_v1': verify_both.astype(np.float32),
        'pseudo_verify_xor_v1': verify_xor.astype(np.float32),
        'pseudo_verify_union_v1': verify_union.astype(np.float32),
        'pseudo_verify_left_consistency_v1': left_consistency.astype(np.float32),
        'pseudo_verify_right_consistency_v1': right_consistency.astype(np.float32),
        'pseudo_verify_render_consistency_v1': render_consistency if render_consistency is not None else np.zeros_like(depth_target, dtype=np.float32),
        'summary': summary,
    }


def write_pseudo_observation_verifier_outputs(frame_out: Path, result: Dict, meta: Dict):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'pseudo_confidence_verify_v1.npy', result['pseudo_confidence_verify_v1'])
    np.save(frame_out / 'pseudo_valid_mask_verify_v1.npy', result['pseudo_valid_mask_verify_v1'])
    np.save(diag_dir / 'pseudo_verify_left_v1.npy', result['pseudo_verify_left_v1'])
    np.save(diag_dir / 'pseudo_verify_right_v1.npy', result['pseudo_verify_right_v1'])
    np.save(diag_dir / 'pseudo_verify_both_v1.npy', result['pseudo_verify_both_v1'])
    np.save(diag_dir / 'pseudo_verify_xor_v1.npy', result['pseudo_verify_xor_v1'])
    np.save(diag_dir / 'pseudo_verify_union_v1.npy', result['pseudo_verify_union_v1'])
    np.save(diag_dir / 'pseudo_verify_left_consistency_v1.npy', result['pseudo_verify_left_consistency_v1'])
    np.save(diag_dir / 'pseudo_verify_right_consistency_v1.npy', result['pseudo_verify_right_consistency_v1'])
    np.save(diag_dir / 'pseudo_verify_render_consistency_v1.npy', result['pseudo_verify_render_consistency_v1'])

    _save_float_png(result['pseudo_confidence_verify_v1'], frame_out / 'pseudo_confidence_verify_v1.png', vmax=1.0)
    _save_mask_png(result['pseudo_valid_mask_verify_v1'], frame_out / 'pseudo_valid_mask_verify_v1.png')
    _save_mask_png(result['pseudo_verify_left_v1'], diag_dir / 'pseudo_verify_left_v1.png')
    _save_mask_png(result['pseudo_verify_right_v1'], diag_dir / 'pseudo_verify_right_v1.png')
    _save_mask_png(result['pseudo_verify_both_v1'], diag_dir / 'pseudo_verify_both_v1.png')
    _save_mask_png(result['pseudo_verify_xor_v1'], diag_dir / 'pseudo_verify_xor_v1.png')
    _save_mask_png(result['pseudo_verify_union_v1'], diag_dir / 'pseudo_verify_union_v1.png')
    _save_float_png(result['pseudo_verify_left_consistency_v1'], diag_dir / 'pseudo_verify_left_consistency_v1.png', vmax=1.0)
    _save_float_png(result['pseudo_verify_right_consistency_v1'], diag_dir / 'pseudo_verify_right_consistency_v1.png', vmax=1.0)
    _save_float_png(result['pseudo_verify_render_consistency_v1'], diag_dir / 'pseudo_verify_render_consistency_v1.png', vmax=1.0)

    with open(frame_out / 'pseudo_verify_meta_v1.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
