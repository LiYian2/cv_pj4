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
    SOURCE_RIGHT,
)


def _normalize(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


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


def _relative_consistency(a: np.ndarray, b: np.ndarray, valid: np.ndarray, tol: float) -> np.ndarray:
    rel = np.zeros_like(a, dtype=np.float32)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-6)
    rel[valid] = np.abs(a[valid] - b[valid]) / denom[valid]
    return np.clip(1.0 - rel / max(float(tol), 1e-6), 0.0, 1.0)


def _blend_with_stable_target(
    primary_depth: np.ndarray,
    stable_depth: np.ndarray,
    alpha: np.ndarray,
    valid_primary: np.ndarray,
    valid_stable: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(primary_depth, dtype=np.float32)
    blend_mask = valid_primary & valid_stable
    primary_only = valid_primary & (~valid_stable)
    out[blend_mask] = (
        np.clip(alpha[blend_mask], 0.0, 1.0) * primary_depth[blend_mask]
        + (1.0 - np.clip(alpha[blend_mask], 0.0, 1.0)) * stable_depth[blend_mask]
    )
    out[primary_only] = primary_depth[primary_only]
    return out



def _build_exact_brpo_cm_support_sets(
    support_left: np.ndarray,
    support_right: np.ndarray,
) -> Dict[str, np.ndarray]:
    support_left = _normalize(support_left) > 0.5
    support_right = _normalize(support_right) > 0.5
    verify_both = support_left & support_right
    verify_left_only = support_left & (~support_right)
    verify_right_only = support_right & (~support_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = support_left | support_right

    confidence = np.zeros_like(support_left, dtype=np.float32)
    confidence[verify_both] = 1.0
    confidence[verify_xor] = 0.5
    return {
        'support_left': support_left,
        'support_right': support_right,
        'verify_both': verify_both,
        'verify_left_only': verify_left_only,
        'verify_right_only': verify_right_only,
        'verify_xor': verify_xor,
        'verify_union': verify_union,
        'confidence': confidence.astype(np.float32),
    }


def _build_exact_brpo_cm_observation_with_target(
    version: str,
    support_left: np.ndarray,
    support_right: np.ndarray,
    depth_target: np.ndarray,
    source_map: np.ndarray,
    depth_target_rule: str,
    strict_scope: str = 'cm_only',
) -> Dict[str, np.ndarray | Dict]:
    verify = _build_exact_brpo_cm_support_sets(support_left=support_left, support_right=support_right)
    depth_target = np.asarray(depth_target, dtype=np.float32)
    source_map = np.asarray(source_map, dtype=np.int16)
    confidence = verify['confidence'].astype(np.float32)
    valid_mask = verify['verify_union'].astype(np.float32)

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'cm_nonzero_ratio': float((confidence > 0).mean()),
        'cm_mean_positive': float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0,
        'cm_both_ratio': float(verify['verify_both'].mean()),
        'cm_single_ratio': float(verify['verify_xor'].mean()),
        'verify_left_ratio': float(verify['support_left'].mean()),
        'verify_right_ratio': float(verify['support_right'].mean()),
        'depth_target_nonzero_ratio': float((depth_target > 1e-6).mean()),
        'source_left_ratio': float((source_map == SOURCE_LEFT).mean()),
        'source_right_ratio': float((source_map == SOURCE_RIGHT).mean()),
        'source_both_ratio': float((source_map == SOURCE_BOTH_WEIGHTED).mean()),
        'policy': {
            'version': version,
            'confidence_rule': 'strict BRPO-style C_m from fused pseudo-frame correspondence support sets only: both->1.0 xor->0.5 none->0.0',
            'depth_target_rule': depth_target_rule,
            'strict_brpo_scope': strict_scope,
            'target_confidence_same_source': False,
        },
    }

    return {
        f'pseudo_depth_target_{version}': depth_target.astype(np.float32),
        f'pseudo_confidence_{version}': confidence.astype(np.float32),
        f'pseudo_source_map_{version}': source_map.astype(np.int16),
        f'pseudo_valid_mask_{version}': valid_mask.astype(np.float32),
        f'pseudo_verify_left_{version}': verify['support_left'].astype(np.float32),
        f'pseudo_verify_right_{version}': verify['support_right'].astype(np.float32),
        f'pseudo_verify_both_{version}': verify['verify_both'].astype(np.float32),
        f'pseudo_verify_xor_{version}': verify['verify_xor'].astype(np.float32),
        f'pseudo_verify_union_{version}': verify['verify_union'].astype(np.float32),
        'summary': summary,
    }


def build_exact_brpo_cm_old_target_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    target_depth_for_refine_v2_brpo: np.ndarray,
    target_depth_source_map_v2_brpo: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    return _build_exact_brpo_cm_observation_with_target(
        version='exact_brpo_cm_old_target_v1',
        support_left=support_left,
        support_right=support_right,
        depth_target=target_depth_for_refine_v2_brpo,
        source_map=target_depth_source_map_v2_brpo,
        depth_target_rule='reuse target_depth_for_refine_v2_brpo and target_depth_source_map_v2_brpo; strict BRPO alignment applies to C_m only',
    )


def build_exact_brpo_cm_hybrid_target_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    hybrid_depth_target: np.ndarray,
    hybrid_source_map: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    return _build_exact_brpo_cm_observation_with_target(
        version='exact_brpo_cm_hybrid_target_v1',
        support_left=support_left,
        support_right=support_right,
        depth_target=hybrid_depth_target,
        source_map=hybrid_source_map,
        depth_target_rule='reuse current hybrid brpo_direct_v1 target/source_map so the ablation isolates exact C_m vs hybrid geometry-gated C_m',
    )


def build_exact_brpo_cm_stable_target_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    stable_depth_target: np.ndarray,
    stable_source_map: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    return _build_exact_brpo_cm_observation_with_target(
        version='exact_brpo_cm_stable_target_v1',
        support_left=support_left,
        support_right=support_right,
        depth_target=stable_depth_target,
        source_map=stable_source_map,
        depth_target_rule='reuse brpo_style_v2 stable-target blend depth/source contract so the ablation isolates exact C_m from fallback/stabilization effects',
    )


def build_exact_brpo_full_target_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    verify = _build_exact_brpo_cm_support_sets(support_left=support_left, support_right=support_right)
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    fusion_weight_left = np.asarray(fusion_weight_left, dtype=np.float32)
    fusion_weight_right = np.asarray(fusion_weight_right, dtype=np.float32)

    confidence = verify['confidence'].astype(np.float32)
    valid_mask = verify['verify_union'].astype(np.float32)

    left_available = projected_depth_left > 1e-6
    right_available = projected_depth_right > 1e-6

    both_weight_sum = np.maximum(fusion_weight_left + fusion_weight_right, 1e-8)
    both_available = verify['verify_both'] & left_available & right_available
    left_from_both_only = verify['verify_both'] & left_available & (~right_available)
    right_from_both_only = verify['verify_both'] & right_available & (~left_available)
    left_single = verify['verify_left_only'] & left_available
    right_single = verify['verify_right_only'] & right_available

    depth_target = np.zeros_like(projected_depth_left, dtype=np.float32)
    depth_target[both_available] = (
        fusion_weight_left[both_available] * projected_depth_left[both_available]
        + fusion_weight_right[both_available] * projected_depth_right[both_available]
    ) / both_weight_sum[both_available]
    depth_target[left_from_both_only | left_single] = projected_depth_left[left_from_both_only | left_single]
    depth_target[right_from_both_only | right_single] = projected_depth_right[right_from_both_only | right_single]

    source_map = np.full_like(projected_depth_left, fill_value=SOURCE_NONE, dtype=np.int16)
    source_map[left_from_both_only | left_single] = SOURCE_LEFT
    source_map[right_from_both_only | right_single] = SOURCE_RIGHT
    source_map[both_available] = SOURCE_BOTH_WEIGHTED

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'cm_nonzero_ratio': float((confidence > 0).mean()),
        'cm_mean_positive': float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0,
        'cm_both_ratio': float(verify['verify_both'].mean()),
        'cm_single_ratio': float(verify['verify_xor'].mean()),
        'verify_left_ratio': float(verify['support_left'].mean()),
        'verify_right_ratio': float(verify['support_right'].mean()),
        'depth_target_nonzero_ratio': float((depth_target > 1e-6).mean()),
        'depth_target_nonzero_within_cm_ratio': float((depth_target[verify['verify_union']] > 1e-6).mean()) if verify['verify_union'].any() else 0.0,
        'source_left_ratio': float((source_map == SOURCE_LEFT).mean()),
        'source_right_ratio': float((source_map == SOURCE_RIGHT).mean()),
        'source_both_ratio': float((source_map == SOURCE_BOTH_WEIGHTED).mean()),
        'source_none_ratio': float((source_map == SOURCE_NONE).mean()),
        'policy': {
            'version': 'exact_brpo_full_target_v1',
            'confidence_rule': 'strict BRPO-style C_m from fused pseudo-frame correspondence support sets only: both->1.0 xor->0.5 none->0.0',
            'depth_target_rule': 'strict BRPO target-side proxy: direct projected-depth composition under exact C_m support sets with fusion weights in both-supported regions, no stable-target blending and no old render-fallback contract',
            'strict_brpo_scope': 'cm_and_target',
            'target_confidence_same_source': True,
            'recommended_stageA_depth_loss_mode': 'legacy',
        },
    }

    return {
        'pseudo_depth_target_exact_brpo_full_target_v1': depth_target.astype(np.float32),
        'pseudo_confidence_exact_brpo_full_target_v1': confidence.astype(np.float32),
        'pseudo_source_map_exact_brpo_full_target_v1': source_map.astype(np.int16),
        'pseudo_valid_mask_exact_brpo_full_target_v1': valid_mask.astype(np.float32),
        'pseudo_verify_left_exact_brpo_full_target_v1': verify['support_left'].astype(np.float32),
        'pseudo_verify_right_exact_brpo_full_target_v1': verify['support_right'].astype(np.float32),
        'pseudo_verify_both_exact_brpo_full_target_v1': verify['verify_both'].astype(np.float32),
        'pseudo_verify_xor_exact_brpo_full_target_v1': verify['verify_xor'].astype(np.float32),
        'pseudo_verify_union_exact_brpo_full_target_v1': verify['verify_union'].astype(np.float32),
        'summary': summary,
    }

def build_brpo_style_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
) -> Dict[str, np.ndarray | Dict]:
    support_left = _normalize(support_left) > 0.5
    support_right = _normalize(support_right) > 0.5
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    fusion_weight_left = np.asarray(fusion_weight_left, dtype=np.float32)
    fusion_weight_right = np.asarray(fusion_weight_right, dtype=np.float32)
    overlap_mask_left = _normalize(overlap_mask_left) > 0.5
    overlap_mask_right = _normalize(overlap_mask_right) > 0.5

    valid_left = support_left & overlap_mask_left & (projected_depth_left > 1e-6)
    valid_right = support_right & overlap_mask_right & (projected_depth_right > 1e-6)

    verify_both = valid_left & valid_right
    verify_left_only = valid_left & (~valid_right)
    verify_right_only = valid_right & (~valid_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = valid_left | valid_right

    fused_depth = np.zeros_like(projected_depth_left, dtype=np.float32)
    both_weight_sum = np.maximum(fusion_weight_left + fusion_weight_right, 1e-8)
    fused_depth[verify_both] = (
        fusion_weight_left[verify_both] * projected_depth_left[verify_both]
        + fusion_weight_right[verify_both] * projected_depth_right[verify_both]
    ) / both_weight_sum[verify_both]
    fused_depth[verify_left_only] = projected_depth_left[verify_left_only]
    fused_depth[verify_right_only] = projected_depth_right[verify_right_only]

    confidence = np.zeros_like(fused_depth, dtype=np.float32)
    confidence[verify_both] = 1.0
    confidence[verify_xor] = 0.5
    valid_mask = verify_union & (fused_depth > 1e-6)
    confidence = np.where(valid_mask, confidence, 0.0).astype(np.float32)
    depth_target = np.where(valid_mask, fused_depth, 0.0).astype(np.float32)

    source_map = np.full_like(projected_depth_left, fill_value=SOURCE_NONE, dtype=np.int16)
    source_map[verify_left_only] = SOURCE_LEFT
    source_map[verify_right_only] = SOURCE_RIGHT
    source_map[verify_both] = SOURCE_BOTH_WEIGHTED

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'cm_nonzero_ratio': float((confidence > 0).mean()),
        'cm_both_ratio': float(verify_both.mean()),
        'cm_single_ratio': float(verify_xor.mean()),
        'verify_left_ratio': float(valid_left.mean()),
        'verify_right_ratio': float(valid_right.mean()),
        'depth_target_nonzero_ratio': float((depth_target > 1e-6).mean()),
        'source_left_ratio': float((source_map == SOURCE_LEFT).mean()),
        'source_right_ratio': float((source_map == SOURCE_RIGHT).mean()),
        'source_both_ratio': float((source_map == SOURCE_BOTH_WEIGHTED).mean()),
        'policy': {
            'version': 'brpo_style_v1',
            'confidence_rule': 'C_m from direct fused-reference support sets: both->1.0 xor->0.5 none->0.0',
            'depth_target_rule': 'verified projected depth composition under the same support sets',
            'target_confidence_same_source': False,
        },
    }

    return {
        'pseudo_depth_target_brpo_style_v1': depth_target,
        'pseudo_confidence_brpo_style_v1': confidence,
        'pseudo_source_map_brpo_style_v1': source_map,
        'pseudo_valid_mask_brpo_style_v1': valid_mask.astype(np.float32),
        'pseudo_verify_left_brpo_style_v1': valid_left.astype(np.float32),
        'pseudo_verify_right_brpo_style_v1': valid_right.astype(np.float32),
        'pseudo_verify_both_brpo_style_v1': verify_both.astype(np.float32),
        'pseudo_verify_xor_brpo_style_v1': verify_xor.astype(np.float32),
        'pseudo_verify_union_brpo_style_v1': verify_union.astype(np.float32),
        'summary': summary,
    }


def build_brpo_style_observation_v2(
    support_left: np.ndarray,
    support_right: np.ndarray,
    support_conf_left: np.ndarray,
    support_conf_right: np.ndarray,
    raw_rgb_confidence_cont: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    stable_depth_target: np.ndarray,
    render_depth: np.ndarray,
    min_confidence: float = 0.05,
) -> Dict[str, np.ndarray | Dict]:
    support_left = _normalize(support_left) > 0.5
    support_right = _normalize(support_right) > 0.5
    support_conf_left = _normalize(support_conf_left)
    support_conf_right = _normalize(support_conf_right)
    raw_rgb_confidence_cont = _normalize(raw_rgb_confidence_cont)

    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    fusion_weight_left = np.asarray(fusion_weight_left, dtype=np.float32)
    fusion_weight_right = np.asarray(fusion_weight_right, dtype=np.float32)
    overlap_mask_left = _normalize(overlap_mask_left)
    overlap_mask_right = _normalize(overlap_mask_right)
    stable_depth_target = np.asarray(stable_depth_target, dtype=np.float32)
    render_depth = np.asarray(render_depth, dtype=np.float32)

    valid_left = support_left & (overlap_mask_left > 0.5) & (projected_depth_left > 1e-6)
    valid_right = support_right & (overlap_mask_right > 0.5) & (projected_depth_right > 1e-6)
    verify_both = valid_left & valid_right
    verify_left_only = valid_left & (~valid_right)
    verify_right_only = valid_right & (~valid_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = valid_left | valid_right

    stable_depth = np.where(stable_depth_target > 1e-6, stable_depth_target, np.where(render_depth > 1e-6, render_depth, 0.0)).astype(np.float32)
    stable_valid = stable_depth > 1e-6

    both_weight_sum = np.maximum(fusion_weight_left + fusion_weight_right, 1e-8)
    projected_depth_both = np.zeros_like(projected_depth_left, dtype=np.float32)
    projected_depth_both[verify_both] = (
        fusion_weight_left[verify_both] * projected_depth_left[verify_both]
        + fusion_weight_right[verify_both] * projected_depth_right[verify_both]
    ) / both_weight_sum[verify_both]

    left_stable_consistency = _relative_consistency(projected_depth_left, stable_depth, valid_left & stable_valid, tol=0.18)
    right_stable_consistency = _relative_consistency(projected_depth_right, stable_depth, valid_right & stable_valid, tol=0.18)
    both_lr_consistency = _relative_consistency(projected_depth_left, projected_depth_right, verify_both, tol=0.10)

    left_support_strength = np.where(
        valid_left,
        np.clip(0.55 * support_conf_left + 0.25 * overlap_mask_left + 0.20 * raw_rgb_confidence_cont, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    right_support_strength = np.where(
        valid_right,
        np.clip(0.55 * support_conf_right + 0.25 * overlap_mask_right + 0.20 * raw_rgb_confidence_cont, 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    left_quality = np.where(valid_left, np.clip(0.5 * left_support_strength + 0.5 * left_stable_consistency, 0.0, 1.0), 0.0).astype(np.float32)
    right_quality = np.where(valid_right, np.clip(0.5 * right_support_strength + 0.5 * right_stable_consistency, 0.0, 1.0), 0.0).astype(np.float32)
    both_quality = np.where(
        verify_both,
        np.clip(np.sqrt(np.clip(left_quality * right_quality, 0.0, 1.0)) * (0.55 + 0.45 * both_lr_consistency), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    confidence = np.zeros_like(projected_depth_left, dtype=np.float32)
    confidence[verify_both] = 0.5 + 0.5 * both_quality[verify_both]
    confidence[verify_left_only] = 0.25 + 0.25 * left_quality[verify_left_only]
    confidence[verify_right_only] = 0.25 + 0.25 * right_quality[verify_right_only]

    both_alpha = np.clip(0.55 + 0.35 * both_quality, 0.0, 0.95).astype(np.float32)
    left_alpha = np.clip(0.35 + 0.40 * left_quality, 0.0, 0.90).astype(np.float32)
    right_alpha = np.clip(0.35 + 0.40 * right_quality, 0.0, 0.90).astype(np.float32)

    depth_target = np.zeros_like(projected_depth_left, dtype=np.float32)
    depth_target += _blend_with_stable_target(projected_depth_both, stable_depth, both_alpha, verify_both, stable_valid)
    depth_target += _blend_with_stable_target(projected_depth_left, stable_depth, left_alpha, verify_left_only, stable_valid)
    depth_target += _blend_with_stable_target(projected_depth_right, stable_depth, right_alpha, verify_right_only, stable_valid)

    valid_mask = verify_union & (depth_target > 1e-6) & (confidence > float(min_confidence))
    confidence = np.where(valid_mask, confidence, 0.0).astype(np.float32)
    depth_target = np.where(valid_mask, depth_target, 0.0).astype(np.float32)

    source_map = np.full_like(projected_depth_left, fill_value=SOURCE_NONE, dtype=np.int16)
    source_map[verify_left_only] = SOURCE_LEFT
    source_map[verify_right_only] = SOURCE_RIGHT
    source_map[verify_both] = SOURCE_BOTH_WEIGHTED
    source_map = np.where(valid_mask, source_map, SOURCE_NONE).astype(np.int16)

    stable_blend_ratio = np.zeros_like(projected_depth_left, dtype=np.float32)
    stable_blend_ratio[verify_both] = 1.0 - both_alpha[verify_both]
    stable_blend_ratio[verify_left_only] = 1.0 - left_alpha[verify_left_only]
    stable_blend_ratio[verify_right_only] = 1.0 - right_alpha[verify_right_only]
    stable_blend_ratio = np.where(valid_mask, stable_blend_ratio, 0.0).astype(np.float32)

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'cm_nonzero_ratio': float((confidence > 0).mean()),
        'cm_mean_positive': float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0,
        'cm_both_ratio': float(verify_both.mean()),
        'cm_single_ratio': float(verify_xor.mean()),
        'verify_left_ratio': float(valid_left.mean()),
        'verify_right_ratio': float(valid_right.mean()),
        'depth_target_nonzero_ratio': float((depth_target > 1e-6).mean()),
        'source_left_ratio': float((source_map == SOURCE_LEFT).mean()),
        'source_right_ratio': float((source_map == SOURCE_RIGHT).mean()),
        'source_both_ratio': float((source_map == SOURCE_BOTH_WEIGHTED).mean()),
        'left_quality_mean_positive': float(left_quality[left_quality > 0].mean()) if (left_quality > 0).any() else 0.0,
        'right_quality_mean_positive': float(right_quality[right_quality > 0].mean()) if (right_quality > 0).any() else 0.0,
        'both_quality_mean_positive': float(both_quality[both_quality > 0].mean()) if (both_quality > 0).any() else 0.0,
        'stable_blend_ratio_mean_positive': float(stable_blend_ratio[stable_blend_ratio > 0].mean()) if (stable_blend_ratio > 0).any() else 0.0,
        'policy': {
            'version': 'brpo_style_v2',
            'confidence_rule': 'shared C_m with BRPO-style floors (both>=0.5, single>=0.25) modulated by continuous verifier quality',
            'depth_target_rule': 'verified projected depth blended with stable old-depth target / render fallback according to verifier quality',
            'target_confidence_same_source': False,
            'stable_target_source': 'target_depth_for_refine_v2_brpo_then_render_depth',
        },
    }

    return {
        'pseudo_depth_target_brpo_style_v2': depth_target,
        'pseudo_confidence_brpo_style_v2': confidence,
        'pseudo_source_map_brpo_style_v2': source_map,
        'pseudo_valid_mask_brpo_style_v2': valid_mask.astype(np.float32),
        'pseudo_verify_left_brpo_style_v2': valid_left.astype(np.float32),
        'pseudo_verify_right_brpo_style_v2': valid_right.astype(np.float32),
        'pseudo_verify_both_brpo_style_v2': verify_both.astype(np.float32),
        'pseudo_verify_xor_brpo_style_v2': verify_xor.astype(np.float32),
        'pseudo_verify_union_brpo_style_v2': verify_union.astype(np.float32),
        'pseudo_verify_quality_left_brpo_style_v2': left_quality.astype(np.float32),
        'pseudo_verify_quality_right_brpo_style_v2': right_quality.astype(np.float32),
        'pseudo_verify_quality_both_brpo_style_v2': both_quality.astype(np.float32),
        'pseudo_stable_blend_ratio_brpo_style_v2': stable_blend_ratio.astype(np.float32),
        'summary': summary,
    }


def build_brpo_direct_observation(
    support_left: np.ndarray,
    support_right: np.ndarray,
    projected_depth_left: np.ndarray,
    projected_depth_right: np.ndarray,
    overlap_mask_left: np.ndarray,
    overlap_mask_right: np.ndarray,
    overlap_conf_left: np.ndarray,
    overlap_conf_right: np.ndarray,
    min_overlap_conf: float = 1e-6,
) -> Dict[str, np.ndarray | Dict]:
    support_left = _normalize(support_left) > 0.5
    support_right = _normalize(support_right) > 0.5
    projected_depth_left = np.asarray(projected_depth_left, dtype=np.float32)
    projected_depth_right = np.asarray(projected_depth_right, dtype=np.float32)
    overlap_mask_left = _normalize(overlap_mask_left) > 0.5
    overlap_mask_right = _normalize(overlap_mask_right) > 0.5
    overlap_conf_left = _normalize(overlap_conf_left)
    overlap_conf_right = _normalize(overlap_conf_right)
    min_overlap_conf = max(float(min_overlap_conf), 0.0)

    valid_left = support_left & overlap_mask_left & (projected_depth_left > 1e-6) & (overlap_conf_left > min_overlap_conf)
    valid_right = support_right & overlap_mask_right & (projected_depth_right > 1e-6) & (overlap_conf_right > min_overlap_conf)

    verify_both = valid_left & valid_right
    verify_left_only = valid_left & (~valid_right)
    verify_right_only = valid_right & (~valid_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = valid_left | valid_right

    depth_target = np.zeros_like(projected_depth_left, dtype=np.float32)
    weight_left = np.where(valid_left, overlap_conf_left, 0.0).astype(np.float32)
    weight_right = np.where(valid_right, overlap_conf_right, 0.0).astype(np.float32)
    both_weight_sum = np.maximum(weight_left + weight_right, 1e-8)
    depth_target[verify_both] = (
        weight_left[verify_both] * projected_depth_left[verify_both]
        + weight_right[verify_both] * projected_depth_right[verify_both]
    ) / both_weight_sum[verify_both]
    depth_target[verify_left_only] = projected_depth_left[verify_left_only]
    depth_target[verify_right_only] = projected_depth_right[verify_right_only]

    confidence = np.zeros_like(depth_target, dtype=np.float32)
    confidence[verify_both] = 1.0
    confidence[verify_xor] = 0.5
    valid_mask = verify_union & (depth_target > 1e-6)
    confidence = np.where(valid_mask, confidence, 0.0).astype(np.float32)
    depth_target = np.where(valid_mask, depth_target, 0.0).astype(np.float32)

    source_map = np.full_like(projected_depth_left, fill_value=SOURCE_NONE, dtype=np.int16)
    source_map[verify_left_only] = SOURCE_LEFT
    source_map[verify_right_only] = SOURCE_RIGHT
    source_map[verify_both] = SOURCE_BOTH_WEIGHTED
    source_map = np.where(valid_mask, source_map, SOURCE_NONE).astype(np.int16)

    summary = {
        'valid_ratio': float(valid_mask.mean()),
        'cm_nonzero_ratio': float((confidence > 0).mean()),
        'cm_mean_positive': float(confidence[confidence > 0].mean()) if (confidence > 0).any() else 0.0,
        'cm_both_ratio': float(verify_both.mean()),
        'cm_single_ratio': float(verify_xor.mean()),
        'verify_left_ratio': float(valid_left.mean()),
        'verify_right_ratio': float(valid_right.mean()),
        'depth_target_nonzero_ratio': float((depth_target > 1e-6).mean()),
        'source_left_ratio': float((source_map == SOURCE_LEFT).mean()),
        'source_right_ratio': float((source_map == SOURCE_RIGHT).mean()),
        'source_both_ratio': float((source_map == SOURCE_BOTH_WEIGHTED).mean()),
        'overlap_conf_left_mean_verified': float(overlap_conf_left[valid_left].mean()) if valid_left.any() else 0.0,
        'overlap_conf_right_mean_verified': float(overlap_conf_right[valid_right].mean()) if valid_right.any() else 0.0,
        'policy': {
            'version': 'brpo_direct_v1',
            'semantic_label': 'hybrid_brpo_cm_geo_v1',
            'strict_brpo': False,
            'confidence_rule': 'hybrid C_m: fused-frame support sets additionally gated by overlap validity and overlap-confidence / projected-depth availability',
            'depth_target_rule': 'projected depth composed directly from per-side overlap confidence weights under the same geometry-gated support sets',
            'target_confidence_same_source': True,
            'fusion_weight_source': 'overlap_conf_left/right',
        },
    }

    return {
        'pseudo_depth_target_brpo_direct_v1': depth_target,
        'pseudo_confidence_brpo_direct_v1': confidence,
        'pseudo_source_map_brpo_direct_v1': source_map,
        'pseudo_valid_mask_brpo_direct_v1': valid_mask.astype(np.float32),
        'pseudo_verify_left_brpo_direct_v1': valid_left.astype(np.float32),
        'pseudo_verify_right_brpo_direct_v1': valid_right.astype(np.float32),
        'pseudo_verify_both_brpo_direct_v1': verify_both.astype(np.float32),
        'pseudo_verify_xor_brpo_direct_v1': verify_xor.astype(np.float32),
        'pseudo_verify_union_brpo_direct_v1': verify_union.astype(np.float32),
        'summary': summary,
    }


def write_brpo_style_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'pseudo_depth_target_brpo_style_v1.npy', result['pseudo_depth_target_brpo_style_v1'])
    np.save(frame_out / 'pseudo_confidence_brpo_style_v1.npy', result['pseudo_confidence_brpo_style_v1'])
    np.save(frame_out / 'pseudo_source_map_brpo_style_v1.npy', result['pseudo_source_map_brpo_style_v1'])
    np.save(frame_out / 'pseudo_valid_mask_brpo_style_v1.npy', result['pseudo_valid_mask_brpo_style_v1'])
    np.save(diag_dir / 'pseudo_verify_left_brpo_style_v1.npy', result['pseudo_verify_left_brpo_style_v1'])
    np.save(diag_dir / 'pseudo_verify_right_brpo_style_v1.npy', result['pseudo_verify_right_brpo_style_v1'])
    np.save(diag_dir / 'pseudo_verify_both_brpo_style_v1.npy', result['pseudo_verify_both_brpo_style_v1'])
    np.save(diag_dir / 'pseudo_verify_xor_brpo_style_v1.npy', result['pseudo_verify_xor_brpo_style_v1'])
    np.save(diag_dir / 'pseudo_verify_union_brpo_style_v1.npy', result['pseudo_verify_union_brpo_style_v1'])

    _save_float_png(result['pseudo_depth_target_brpo_style_v1'], frame_out / 'pseudo_depth_target_brpo_style_v1.png')
    _save_float_png(result['pseudo_confidence_brpo_style_v1'], frame_out / 'pseudo_confidence_brpo_style_v1.png', vmax=1.0)
    _save_source_map_png(result['pseudo_source_map_brpo_style_v1'], frame_out / 'pseudo_source_map_brpo_style_v1.png')
    _save_mask_png(result['pseudo_valid_mask_brpo_style_v1'], frame_out / 'pseudo_valid_mask_brpo_style_v1.png')
    _save_mask_png(result['pseudo_verify_left_brpo_style_v1'], diag_dir / 'pseudo_verify_left_brpo_style_v1.png')
    _save_mask_png(result['pseudo_verify_right_brpo_style_v1'], diag_dir / 'pseudo_verify_right_brpo_style_v1.png')
    _save_mask_png(result['pseudo_verify_both_brpo_style_v1'], diag_dir / 'pseudo_verify_both_brpo_style_v1.png')
    _save_mask_png(result['pseudo_verify_xor_brpo_style_v1'], diag_dir / 'pseudo_verify_xor_brpo_style_v1.png')
    _save_mask_png(result['pseudo_verify_union_brpo_style_v1'], diag_dir / 'pseudo_verify_union_brpo_style_v1.png')

    with open(frame_out / 'brpo_style_observation_meta_v1.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def write_brpo_style_observation_outputs_v2(frame_out: Path, result: Dict, meta: Dict):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / 'pseudo_depth_target_brpo_style_v2.npy', result['pseudo_depth_target_brpo_style_v2'])
    np.save(frame_out / 'pseudo_confidence_brpo_style_v2.npy', result['pseudo_confidence_brpo_style_v2'])
    np.save(frame_out / 'pseudo_source_map_brpo_style_v2.npy', result['pseudo_source_map_brpo_style_v2'])
    np.save(frame_out / 'pseudo_valid_mask_brpo_style_v2.npy', result['pseudo_valid_mask_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_left_brpo_style_v2.npy', result['pseudo_verify_left_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_right_brpo_style_v2.npy', result['pseudo_verify_right_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_both_brpo_style_v2.npy', result['pseudo_verify_both_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_xor_brpo_style_v2.npy', result['pseudo_verify_xor_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_union_brpo_style_v2.npy', result['pseudo_verify_union_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_quality_left_brpo_style_v2.npy', result['pseudo_verify_quality_left_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_quality_right_brpo_style_v2.npy', result['pseudo_verify_quality_right_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_verify_quality_both_brpo_style_v2.npy', result['pseudo_verify_quality_both_brpo_style_v2'])
    np.save(diag_dir / 'pseudo_stable_blend_ratio_brpo_style_v2.npy', result['pseudo_stable_blend_ratio_brpo_style_v2'])

    _save_float_png(result['pseudo_depth_target_brpo_style_v2'], frame_out / 'pseudo_depth_target_brpo_style_v2.png')
    _save_float_png(result['pseudo_confidence_brpo_style_v2'], frame_out / 'pseudo_confidence_brpo_style_v2.png', vmax=1.0)
    _save_source_map_png(result['pseudo_source_map_brpo_style_v2'], frame_out / 'pseudo_source_map_brpo_style_v2.png')
    _save_mask_png(result['pseudo_valid_mask_brpo_style_v2'], frame_out / 'pseudo_valid_mask_brpo_style_v2.png')
    _save_mask_png(result['pseudo_verify_left_brpo_style_v2'], diag_dir / 'pseudo_verify_left_brpo_style_v2.png')
    _save_mask_png(result['pseudo_verify_right_brpo_style_v2'], diag_dir / 'pseudo_verify_right_brpo_style_v2.png')
    _save_mask_png(result['pseudo_verify_both_brpo_style_v2'], diag_dir / 'pseudo_verify_both_brpo_style_v2.png')
    _save_mask_png(result['pseudo_verify_xor_brpo_style_v2'], diag_dir / 'pseudo_verify_xor_brpo_style_v2.png')
    _save_mask_png(result['pseudo_verify_union_brpo_style_v2'], diag_dir / 'pseudo_verify_union_brpo_style_v2.png')
    _save_float_png(result['pseudo_verify_quality_left_brpo_style_v2'], diag_dir / 'pseudo_verify_quality_left_brpo_style_v2.png', vmax=1.0)
    _save_float_png(result['pseudo_verify_quality_right_brpo_style_v2'], diag_dir / 'pseudo_verify_quality_right_brpo_style_v2.png', vmax=1.0)
    _save_float_png(result['pseudo_verify_quality_both_brpo_style_v2'], diag_dir / 'pseudo_verify_quality_both_brpo_style_v2.png', vmax=1.0)
    _save_float_png(result['pseudo_stable_blend_ratio_brpo_style_v2'], diag_dir / 'pseudo_stable_blend_ratio_brpo_style_v2.png', vmax=1.0)

    with open(frame_out / 'brpo_style_observation_meta_v2.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)




def _write_basic_observation_outputs(frame_out: Path, result: Dict, meta: Dict, prefix: str, meta_filename: str):
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / 'diag'
    diag_dir.mkdir(parents=True, exist_ok=True)

    np.save(frame_out / f'pseudo_depth_target_{prefix}.npy', result[f'pseudo_depth_target_{prefix}'])
    np.save(frame_out / f'pseudo_confidence_{prefix}.npy', result[f'pseudo_confidence_{prefix}'])
    np.save(frame_out / f'pseudo_source_map_{prefix}.npy', result[f'pseudo_source_map_{prefix}'])
    np.save(frame_out / f'pseudo_valid_mask_{prefix}.npy', result[f'pseudo_valid_mask_{prefix}'])
    np.save(diag_dir / f'pseudo_verify_left_{prefix}.npy', result[f'pseudo_verify_left_{prefix}'])
    np.save(diag_dir / f'pseudo_verify_right_{prefix}.npy', result[f'pseudo_verify_right_{prefix}'])
    np.save(diag_dir / f'pseudo_verify_both_{prefix}.npy', result[f'pseudo_verify_both_{prefix}'])
    np.save(diag_dir / f'pseudo_verify_xor_{prefix}.npy', result[f'pseudo_verify_xor_{prefix}'])
    np.save(diag_dir / f'pseudo_verify_union_{prefix}.npy', result[f'pseudo_verify_union_{prefix}'])

    _save_float_png(result[f'pseudo_depth_target_{prefix}'], frame_out / f'pseudo_depth_target_{prefix}.png')
    _save_float_png(result[f'pseudo_confidence_{prefix}'], frame_out / f'pseudo_confidence_{prefix}.png', vmax=1.0)
    _save_source_map_png(result[f'pseudo_source_map_{prefix}'], frame_out / f'pseudo_source_map_{prefix}.png')
    _save_mask_png(result[f'pseudo_valid_mask_{prefix}'], frame_out / f'pseudo_valid_mask_{prefix}.png')
    _save_mask_png(result[f'pseudo_verify_left_{prefix}'], diag_dir / f'pseudo_verify_left_{prefix}.png')
    _save_mask_png(result[f'pseudo_verify_right_{prefix}'], diag_dir / f'pseudo_verify_right_{prefix}.png')
    _save_mask_png(result[f'pseudo_verify_both_{prefix}'], diag_dir / f'pseudo_verify_both_{prefix}.png')
    _save_mask_png(result[f'pseudo_verify_xor_{prefix}'], diag_dir / f'pseudo_verify_xor_{prefix}.png')
    _save_mask_png(result[f'pseudo_verify_union_{prefix}'], diag_dir / f'pseudo_verify_union_{prefix}.png')

    with open(frame_out / meta_filename, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def write_exact_brpo_cm_old_target_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    _write_basic_observation_outputs(frame_out, result, meta, 'exact_brpo_cm_old_target_v1', 'exact_brpo_cm_old_target_meta_v1.json')


def write_exact_brpo_cm_hybrid_target_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    _write_basic_observation_outputs(frame_out, result, meta, 'exact_brpo_cm_hybrid_target_v1', 'exact_brpo_cm_hybrid_target_meta_v1.json')


def write_exact_brpo_cm_stable_target_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    _write_basic_observation_outputs(frame_out, result, meta, 'exact_brpo_cm_stable_target_v1', 'exact_brpo_cm_stable_target_meta_v1.json')


def write_exact_brpo_full_target_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    _write_basic_observation_outputs(frame_out, result, meta, 'exact_brpo_full_target_v1', 'exact_brpo_full_target_meta_v1.json')


def write_brpo_direct_observation_outputs(frame_out: Path, result: Dict, meta: Dict):
    _write_basic_observation_outputs(frame_out, result, meta, 'brpo_direct_v1', 'brpo_direct_observation_meta_v1.json')

def build_exact_brpo_upstream_target_observation(
    support_left_exact: np.ndarray,
    support_right_exact: np.ndarray,
    projected_depth_left_exact: np.ndarray,
    projected_depth_right_exact: np.ndarray,
    confidence_left_exact: np.ndarray,
    confidence_right_exact: np.ndarray,
    fusion_weight_left: np.ndarray,
    fusion_weight_right: np.ndarray,
    provenance_left: np.ndarray | None = None,
    provenance_right: np.ndarray | None = None,
    use_confidence_weighted_composition: bool = True,
) -> Dict[str, np.ndarray | Dict]:
    """Exact BRPO upstream target observation builder.
    
    This is the final Phase T2 target: exact M~ + exact upstream T~.
    
    Key differences from build_exact_brpo_full_target_observation:
    1. Uses exact backend bundle (not proxy projected_depth)
    2. No render_depth fallback
    3. Continuous confidence from exact backend
    4. Provenance tracked throughout
    """    
    from pseudo_branch.target.depth_supervision_v2 import build_exact_upstream_depth_target
    
    # Build exact upstream depth target
    depth_result = build_exact_upstream_depth_target(
        support_left_exact=support_left_exact,
        support_right_exact=support_right_exact,
        projected_depth_left_exact=projected_depth_left_exact,
        projected_depth_right_exact=projected_depth_right_exact,
        confidence_left_exact=confidence_left_exact,
        confidence_right_exact=confidence_right_exact,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
        provenance_left=provenance_left,
        provenance_right=provenance_right,
        use_confidence_weighted_composition=use_confidence_weighted_composition,
    )
    
    # Build exact C_m from support sets (same as exact_brpo_full_target)
    support_left = np.asarray(support_left_exact, dtype=np.float32) > 0.5
    support_right = np.asarray(support_right_exact, dtype=np.float32) > 0.5
    
    verify_both = support_left & support_right
    verify_left_only = support_left & (~support_right)
    verify_right_only = support_right & (~support_left)
    verify_xor = verify_left_only | verify_right_only
    verify_union = support_left | support_right
    
    # Strict BRPO C_m: both->1.0, xor->0.5, none->0.0
    confidence_cm = np.zeros_like(support_left, dtype=np.float32)
    confidence_cm[verify_both] = 1.0
    confidence_cm[verify_xor] = 0.5
    
    # Combine with depth target result
    depth_target = depth_result["pseudo_depth_target_exact_upstream_v1"]
    source_map = depth_result["pseudo_source_map_exact_upstream_v1"]
    valid_mask = depth_result["pseudo_valid_mask_exact_upstream_v1"]
    target_confidence = depth_result["pseudo_confidence_exact_upstream_v1"]
    
    summary = {
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": "exact_upstream_v1",
        "target_loss_contract": "exact_shared_cm_v1",
        "valid_ratio": float(valid_mask.mean()),
        "cm_nonzero_ratio": float((confidence_cm > 0).mean()),
        "cm_mean_positive": float(confidence_cm[confidence_cm > 0].mean()) if (confidence_cm > 0).any() else 0.0,
        "cm_both_ratio": float(verify_both.mean()),
        "cm_single_ratio": float(verify_xor.mean()),
        "depth_target_filled_ratio": float((depth_target > 1e-6).mean()),
        "depth_target_filled_within_cm_ratio": float((depth_target[verify_union] > 1e-6).mean()) if verify_union.any() else 0.0,
        "avg_target_confidence": float(target_confidence[valid_mask > 0].mean()) if (valid_mask > 0).any() else 0.0,
        "source_counts": depth_result["summary"]["source_counts"],
        "no_render_fallback": True,
        "policy": {
            "version": "exact_brpo_upstream_target_v1",
            "confidence_rule": "strict BRPO-style C_m from exact backend support sets",
            "depth_target_rule": "exact upstream projected-depth composition with continuous confidence, no render fallback",
            "strict_brpo_scope": "cm_and_target_and_upstream_backend",
            "upstream_backend": "exact_branch_native_v1",
            "target_confidence_same_source": True,
            "recommended_stageA_depth_loss_mode": "exact_shared_cm_v1",  # NOT legacy
        },
    }
    
    return {
        "pseudo_depth_target_exact_brpo_upstream_target_v1": depth_target.astype(np.float32),
        "pseudo_confidence_exact_brpo_upstream_target_v1": confidence_cm.astype(np.float32),
        "pseudo_source_map_exact_brpo_upstream_target_v1": source_map.astype(np.int16),
        "pseudo_valid_mask_exact_brpo_upstream_target_v1": valid_mask.astype(np.float32),
        "pseudo_target_confidence_exact_brpo_upstream_target_v1": target_confidence.astype(np.float32),
        "pseudo_verify_left_exact_brpo_upstream_target_v1": support_left.astype(np.float32),
        "pseudo_verify_right_exact_brpo_upstream_target_v1": support_right.astype(np.float32),
        "pseudo_verify_both_exact_brpo_upstream_target_v1": verify_both.astype(np.float32),
        "pseudo_verify_xor_exact_brpo_upstream_target_v1": verify_xor.astype(np.float32),
        "pseudo_verify_union_exact_brpo_upstream_target_v1": verify_union.astype(np.float32),
        "summary": summary,
    }


def write_exact_brpo_upstream_target_observation_outputs(
    frame_out: Path,
    result: Dict,
    meta: Dict,
):
    """Write exact BRPO upstream target observation outputs."""    
    frame_out.mkdir(parents=True, exist_ok=True)
    diag_dir = frame_out / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(frame_out / "pseudo_depth_target_exact_brpo_upstream_target_v1.npy", result["pseudo_depth_target_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_confidence_exact_brpo_upstream_target_v1.npy", result["pseudo_confidence_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_source_map_exact_brpo_upstream_target_v1.npy", result["pseudo_source_map_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_valid_mask_exact_brpo_upstream_target_v1.npy", result["pseudo_valid_mask_exact_brpo_upstream_target_v1"])
    np.save(frame_out / "pseudo_target_confidence_exact_brpo_upstream_target_v1.npy", result["pseudo_target_confidence_exact_brpo_upstream_target_v1"])
    
    _save_float_png(result["pseudo_depth_target_exact_brpo_upstream_target_v1"], frame_out / "pseudo_depth_target_exact_brpo_upstream_target_v1.png")
    _save_mask_png(result["pseudo_confidence_exact_brpo_upstream_target_v1"], frame_out / "pseudo_confidence_exact_brpo_upstream_target_v1.png")
    _save_source_map_png(result["pseudo_source_map_exact_brpo_upstream_target_v1"], frame_out / "pseudo_source_map_exact_brpo_upstream_target_v1.png")
    _save_mask_png(result["pseudo_valid_mask_exact_brpo_upstream_target_v1"], frame_out / "pseudo_valid_mask_exact_brpo_upstream_target_v1.png")
    _save_float_png(result["pseudo_target_confidence_exact_brpo_upstream_target_v1"], diag_dir / "pseudo_target_confidence_exact_brpo_upstream_target_v1.png", vmax=1.0)
    
    meta["observation_builder"] = "build_exact_brpo_upstream_target_observation"
    meta["summary"] = result["summary"]
    
    with open(frame_out / "exact_brpo_upstream_target_observation_meta_v1.json", "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
