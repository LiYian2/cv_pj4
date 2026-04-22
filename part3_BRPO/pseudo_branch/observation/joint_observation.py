from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from pseudo_branch.target.depth_supervision_v2 import (
    SOURCE_BOTH_WEIGHTED,
    SOURCE_LEFT,
    SOURCE_NONE,
    SOURCE_RENDER_FALLBACK,
    SOURCE_RIGHT,
)


CANDIDATE_NAMES = ('left', 'right', 'both_weighted', 'render_prior')
SOURCE_IDS = {
    'left': SOURCE_LEFT,
    'right': SOURCE_RIGHT,
    'both_weighted': SOURCE_BOTH_WEIGHTED,
    'render_prior': SOURCE_RENDER_FALLBACK,
}
SOURCE_PRIOR = {
    'left': 0.82,
    'right': 0.82,
    'both_weighted': 1.00,
    'render_prior': 0.22,
}


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


def _save_source_map_png(source_map: np.ndarray, path: Path):
    arr = np.asarray(source_map, dtype=np.uint8)
    vmax = max(int(arr.max()), 1)
    img = (arr.astype(np.float32) / float(vmax) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def _normalize(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def _relative_consistency(a: np.ndarray, b: np.ndarray, valid: np.ndarray, tol: float = 0.10) -> np.ndarray:
    rel = np.zeros_like(a, dtype=np.float32)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)
    rel[valid] = np.abs(a[valid] - b[valid]) / denom[valid]
    return np.clip(1.0 - rel / max(float(tol), 1e-6), 0.0, 1.0)


def collect_joint_observation_candidates(
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
) -> Dict[str, np.ndarray]:
    render_depth = np.asarray(render_depth, dtype=np.float32)
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    fusion_weight_left = np.asarray(fusion_weight_left, dtype=np.float32)
    fusion_weight_right = np.asarray(fusion_weight_right, dtype=np.float32)
    overlap_mask_left = _normalize(overlap_mask_left)
    overlap_mask_right = _normalize(overlap_mask_right)

    valid_left = (projected_depth_left > 1e-6) & (overlap_mask_left > 0.5)
    valid_right = (projected_depth_right > 1e-6) & (overlap_mask_right > 0.5)
    valid_render = render_depth > 1e-6
    valid_both = valid_left & valid_right

    fusion_sum = fusion_weight_left + fusion_weight_right
    fusion_safe = np.maximum(fusion_sum, 1e-8)
    both_depth = (fusion_weight_left * projected_depth_left + fusion_weight_right * projected_depth_right) / fusion_safe
    both_depth = np.where(valid_both, both_depth, 0.0).astype(np.float32)

    return {
        'left_depth': projected_depth_left.astype(np.float32),
        'right_depth': projected_depth_right.astype(np.float32),
        'both_depth': both_depth,
        'render_depth': render_depth.astype(np.float32),
        'valid_left': valid_left,
        'valid_right': valid_right,
        'valid_both': valid_both,
        'valid_render': valid_render,
        'overlap_left': overlap_mask_left.astype(np.float32),
        'overlap_right': overlap_mask_right.astype(np.float32),
        'fusion_weight_left': fusion_weight_left.astype(np.float32),
        'fusion_weight_right': fusion_weight_right.astype(np.float32),
    }


def score_joint_observation_candidates(
    raw_rgb_confidence: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray | None,
    candidate_pack: Dict[str, np.ndarray],
    w_appearance: float = 0.35,
    w_geometry: float = 0.35,
    w_support: float = 0.20,
    w_prior: float = 0.10,
) -> Dict[str, np.ndarray]:
    raw_rgb_confidence = _normalize(raw_rgb_confidence)
    appearance = _normalize(raw_rgb_confidence_cont) if raw_rgb_confidence_cont is not None else raw_rgb_confidence
    support_seed = (raw_rgb_confidence > 0).astype(np.float32)

    valid_left = candidate_pack['valid_left']
    valid_right = candidate_pack['valid_right']
    valid_both = candidate_pack['valid_both']
    valid_render = candidate_pack['valid_render']
    overlap_left = candidate_pack['overlap_left']
    overlap_right = candidate_pack['overlap_right']
    left_depth = candidate_pack['left_depth']
    right_depth = candidate_pack['right_depth']
    render_depth = candidate_pack['render_depth']

    both_consistency = _relative_consistency(left_depth, right_depth, valid_both, tol=0.08)
    left_consistency = _relative_consistency(left_depth, render_depth, valid_left & valid_render, tol=0.18)
    right_consistency = _relative_consistency(right_depth, render_depth, valid_right & valid_render, tol=0.18)

    appearance_scores = {
        'left': appearance,
        'right': appearance,
        'both_weighted': np.clip(0.6 * appearance + 0.4 * raw_rgb_confidence, 0.0, 1.0),
        'render_prior': np.clip(0.5 * appearance + 0.1, 0.0, 1.0),
    }
    geometry_scores = {
        'left': np.where(valid_left, 0.55 + 0.20 * overlap_left + 0.25 * left_consistency, 0.0),
        'right': np.where(valid_right, 0.55 + 0.20 * overlap_right + 0.25 * right_consistency, 0.0),
        'both_weighted': np.where(valid_both, 0.65 + 0.35 * both_consistency, 0.0),
        'render_prior': np.where(valid_render, np.clip(0.08 + 0.18 * appearance, 0.0, 0.35), 0.0),
    }
    support_scores = {
        'left': np.where(valid_left, 0.55 + 0.25 * overlap_left, 0.0),
        'right': np.where(valid_right, 0.55 + 0.25 * overlap_right, 0.0),
        'both_weighted': np.where(valid_both, 1.0, 0.0),
        'render_prior': np.where(valid_render, 0.18, 0.0),
    }
    valid_masks = {
        'left': valid_left,
        'right': valid_right,
        'both_weighted': valid_both,
        'render_prior': valid_render,
    }

    score_map = {}
    for name in CANDIDATE_NAMES:
        valid = valid_masks[name].astype(np.float32)
        score = (
            float(w_appearance) * appearance_scores[name]
            + float(w_geometry) * geometry_scores[name]
            + float(w_support) * support_scores[name]
            + float(w_prior) * float(SOURCE_PRIOR[name])
        )
        score_map[name] = np.clip(score * valid * support_seed, 0.0, 1.0).astype(np.float32)

    return {
        'appearance_evidence': appearance.astype(np.float32),
        'score_map': score_map,
        'both_consistency': both_consistency.astype(np.float32),
    }


def build_joint_observation_from_candidates(
    raw_rgb_confidence: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray | None,
    render_depth: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    min_joint_confidence: float = 0.05,
) -> Dict[str, np.ndarray | Dict]:
    candidate_pack = collect_joint_observation_candidates(
        render_depth=render_depth,
        projected_depth_left=projected_depth_left,
        projected_depth_right=projected_depth_right,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
        overlap_mask_left=overlap_mask_left,
        overlap_mask_right=overlap_mask_right,
    )
    score_pack = score_joint_observation_candidates(
        raw_rgb_confidence=raw_rgb_confidence,
        raw_rgb_confidence_cont=raw_rgb_confidence_cont,
        candidate_pack=candidate_pack,
    )

    depth_stack = np.stack([
        candidate_pack['left_depth'],
        candidate_pack['right_depth'],
        candidate_pack['both_depth'],
        candidate_pack['render_depth'],
    ], axis=0)
    score_stack = np.stack([score_pack['score_map'][name] for name in CANDIDATE_NAMES], axis=0)
    best_idx = np.argmax(score_stack, axis=0)
    best_score = np.take_along_axis(score_stack, best_idx[None, ...], axis=0)[0]
    best_depth = np.take_along_axis(depth_stack, best_idx[None, ...], axis=0)[0]

    depth_valid_stack = (depth_stack > 1e-6).astype(np.float32)
    score_shift = score_stack - np.max(score_stack, axis=0, keepdims=True)
    score_exp = np.exp(score_shift / 0.12) * depth_valid_stack
    score_sum = np.maximum(score_exp.sum(axis=0, keepdims=True), 1e-8)
    score_prob = score_exp / score_sum
    fused_depth = (score_prob * depth_stack).sum(axis=0).astype(np.float32)

    confidence_rgb = np.clip(np.take_along_axis(np.stack([
        score_pack['appearance_evidence'],
        score_pack['appearance_evidence'],
        score_pack['appearance_evidence'],
        score_pack['appearance_evidence'],
    ], axis=0), best_idx[None, ...], axis=0)[0], 0.0, 1.0).astype(np.float32)
    confidence_depth = np.where(best_depth > 1e-6, best_score, 0.0).astype(np.float32)
    confidence_joint = np.sqrt(np.clip(confidence_rgb * confidence_depth, 0.0, 1.0)).astype(np.float32)
    valid_mask = (confidence_joint > float(min_joint_confidence)) & (fused_depth > 1e-6)

    source_ids = np.array([SOURCE_IDS[name] for name in CANDIDATE_NAMES], dtype=np.int16)
    source_map = source_ids[best_idx]
    source_map = np.where(valid_mask, source_map, SOURCE_NONE).astype(np.int16)

    confidence_rgb = np.where(valid_mask, confidence_rgb, 0.0).astype(np.float32)
    confidence_depth = np.where(valid_mask, confidence_depth, 0.0).astype(np.float32)
    confidence_joint = np.where(valid_mask, confidence_joint, 0.0).astype(np.float32)
    uncertainty = np.where(valid_mask, 1.0 - confidence_joint, 1.0).astype(np.float32)
    depth_target = np.where(valid_mask, fused_depth, 0.0).astype(np.float32)

    selected_both = source_map == SOURCE_BOTH_WEIGHTED
    selected_left = source_map == SOURCE_LEFT
    selected_right = source_map == SOURCE_RIGHT
    selected_render = source_map == SOURCE_RENDER_FALLBACK

    old_min_rule = np.minimum(_normalize(raw_rgb_confidence), np.where(selected_both, 1.0, np.where(selected_left | selected_right, 0.5, 0.0)))
    old_like_depth = np.where(selected_both, candidate_pack['both_depth'], np.where(selected_left, candidate_pack['left_depth'], np.where(selected_right, candidate_pack['right_depth'], candidate_pack['render_depth'])))
    joint_diff_ratio = float(np.mean(np.abs(confidence_joint - old_min_rule) > 1e-6))
    depth_diff_ratio = float(np.mean(valid_mask & (np.abs(depth_target - old_like_depth) > 1e-6)))

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'joint_confidence_nonzero_ratio': float((confidence_joint > 0).mean()),
        'joint_confidence_mean_positive': float(confidence_joint[confidence_joint > 0].mean()) if (confidence_joint > 0).any() else 0.0,
        'rgb_confidence_mean_positive': float(confidence_rgb[confidence_rgb > 0].mean()) if (confidence_rgb > 0).any() else 0.0,
        'depth_confidence_mean_positive': float(confidence_depth[confidence_depth > 0].mean()) if (confidence_depth > 0).any() else 0.0,
        'selected_left_ratio': float(selected_left.mean()),
        'selected_right_ratio': float(selected_right.mean()),
        'selected_both_ratio': float(selected_both.mean()),
        'selected_render_prior_ratio': float(selected_render.mean()),
        'old_min_rule_diff_ratio': joint_diff_ratio,
        'new_depth_vs_old_like_diff_ratio': depth_diff_ratio,
        'render_prior_usage_inside_valid_ratio': float(selected_render.sum() / max(int(valid_mask.sum()), 1)),
        'candidate_score_mean': {name: float(score_pack['score_map'][name].mean()) for name in CANDIDATE_NAMES},
        'policy': {
            'version': 'brpo_joint_v1_observation_rewrite',
            'depth_target_reuses_old_target': False,
            'joint_confidence_is_old_min_rule': False,
            'selection_rule': 'candidate competition over {left,right,both_weighted,render_prior}',
            'joint_confidence_rule': 'sqrt(conf_rgb * conf_depth)',
            'uncertainty_rule': '1 - conf_joint',
            'consumer_locked': True,
            'weights': {
                'appearance': 0.35,
                'geometry': 0.35,
                'support': 0.20,
                'prior': 0.10,
            },
        },
    }

    return {
        'pseudo_depth_target_joint_v1': depth_target,
        'pseudo_confidence_joint_v1': confidence_joint,
        'pseudo_confidence_rgb_joint_v1': confidence_rgb,
        'pseudo_confidence_depth_joint_v1': confidence_depth,
        'pseudo_uncertainty_joint_v1': uncertainty,
        'pseudo_source_map_joint_v1': source_map,
        'pseudo_valid_mask_joint_v1': valid_mask.astype(np.float32),
        'joint_candidate_index_v1': best_idx.astype(np.int16),
        'joint_candidate_score_left_v1': score_pack['score_map']['left'],
        'joint_candidate_score_right_v1': score_pack['score_map']['right'],
        'joint_candidate_score_both_v1': score_pack['score_map']['both_weighted'],
        'joint_candidate_score_render_v1': score_pack['score_map']['render_prior'],
        'joint_both_consistency_v1': score_pack['both_consistency'],
        'summary': summary,
    }


def write_joint_observation_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'pseudo_depth_target_joint_v1.npy', result['pseudo_depth_target_joint_v1'])
    np.save(frame_out / 'pseudo_confidence_joint_v1.npy', result['pseudo_confidence_joint_v1'])
    np.save(frame_out / 'pseudo_confidence_rgb_joint_v1.npy', result['pseudo_confidence_rgb_joint_v1'])
    np.save(frame_out / 'pseudo_confidence_depth_joint_v1.npy', result['pseudo_confidence_depth_joint_v1'])
    np.save(frame_out / 'pseudo_uncertainty_joint_v1.npy', result['pseudo_uncertainty_joint_v1'])
    np.save(frame_out / 'pseudo_source_map_joint_v1.npy', result['pseudo_source_map_joint_v1'])
    np.save(frame_out / 'pseudo_valid_mask_joint_v1.npy', result['pseudo_valid_mask_joint_v1'])

    np.save(diag_dir / 'joint_candidate_index_v1.npy', result['joint_candidate_index_v1'])
    np.save(diag_dir / 'joint_candidate_score_left_v1.npy', result['joint_candidate_score_left_v1'])
    np.save(diag_dir / 'joint_candidate_score_right_v1.npy', result['joint_candidate_score_right_v1'])
    np.save(diag_dir / 'joint_candidate_score_both_v1.npy', result['joint_candidate_score_both_v1'])
    np.save(diag_dir / 'joint_candidate_score_render_v1.npy', result['joint_candidate_score_render_v1'])
    np.save(diag_dir / 'joint_both_consistency_v1.npy', result['joint_both_consistency_v1'])

    _save_float_png(result['pseudo_depth_target_joint_v1'], frame_out / 'pseudo_depth_target_joint_v1.png')
    _save_float_png(result['pseudo_confidence_joint_v1'], frame_out / 'pseudo_confidence_joint_v1.png', vmax=1.0)
    _save_float_png(result['pseudo_confidence_rgb_joint_v1'], diag_dir / 'pseudo_confidence_rgb_joint_v1.png', vmax=1.0)
    _save_float_png(result['pseudo_confidence_depth_joint_v1'], diag_dir / 'pseudo_confidence_depth_joint_v1.png', vmax=1.0)
    _save_float_png(result['pseudo_uncertainty_joint_v1'], diag_dir / 'pseudo_uncertainty_joint_v1.png', vmax=1.0)
    _save_mask_png(result['pseudo_valid_mask_joint_v1'], frame_out / 'pseudo_valid_mask_joint_v1.png')
    _save_source_map_png(result['pseudo_source_map_joint_v1'], frame_out / 'pseudo_source_map_joint_v1.png')
    _save_float_png(result['joint_candidate_score_left_v1'], diag_dir / 'joint_candidate_score_left_v1.png', vmax=1.0)
    _save_float_png(result['joint_candidate_score_right_v1'], diag_dir / 'joint_candidate_score_right_v1.png', vmax=1.0)
    _save_float_png(result['joint_candidate_score_both_v1'], diag_dir / 'joint_candidate_score_both_v1.png', vmax=1.0)
    _save_float_png(result['joint_candidate_score_render_v1'], diag_dir / 'joint_candidate_score_render_v1.png', vmax=1.0)
    _save_float_png(result['joint_both_consistency_v1'], diag_dir / 'joint_both_consistency_v1.png', vmax=1.0)

    with open(frame_out / 'joint_observation_meta_v1.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
