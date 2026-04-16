from __future__ import annotations

from typing import Iterable, List

from .gating_schema import PseudoLocalGatingConfig


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_min_correction(view: dict) -> float:
    depth_meta = view.get('depth_meta') or {}
    source_meta = view.get('source_meta') or {}
    candidates = [
        depth_meta.get('mean_abs_rel_correction_verified'),
        depth_meta.get('mean_abs_rel_correction'),
        source_meta.get('mean_abs_rel_correction_verified'),
        source_meta.get('mean_abs_rel_correction'),
    ]
    for value in candidates:
        if value is not None:
            return _safe_float(value, 0.0)
    return 0.0


def _soft_component_positive(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    return max(0.0, min(1.0, value / max(threshold, 1e-8)))


def _soft_component_negative(value: float, threshold: float) -> float:
    if threshold >= 1.0:
        return 1.0
    if value <= threshold:
        return 1.0
    span = max(1e-8, 1.0 - threshold)
    return max(0.0, min(1.0, 1.0 - ((value - threshold) / span)))


def evaluate_single_view_signal_gate(view: dict, cfg: PseudoLocalGatingConfig) -> dict:
    verified_ratio = _safe_float(view.get('target_depth_verified_ratio'), 0.0)
    rgb_mask_ratio = _safe_float(view.get('rgb_confidence_nonzero_ratio'), 0.0)
    fallback_ratio = _safe_float(view.get('target_depth_render_fallback_ratio'), 1.0)
    min_correction = _extract_min_correction(view)

    reasons: List[str] = []
    if verified_ratio < cfg.min_verified_ratio:
        reasons.append('verified_ratio')
    if rgb_mask_ratio < cfg.min_rgb_mask_ratio:
        reasons.append('rgb_mask_ratio')
    if fallback_ratio > cfg.max_fallback_ratio:
        reasons.append('fallback_ratio')
    if cfg.min_correction > 0 and min_correction < cfg.min_correction:
        reasons.append('min_correction')

    if cfg.is_soft():
        components = [
            _soft_component_positive(verified_ratio, cfg.min_verified_ratio),
            _soft_component_positive(rgb_mask_ratio, cfg.min_rgb_mask_ratio),
            _soft_component_negative(fallback_ratio, cfg.max_fallback_ratio),
        ]
        if cfg.min_correction > 0:
            components.append(_soft_component_positive(min_correction, cfg.min_correction))
        base_score = sum(components) / max(len(components), 1)
        weight = float(max(0.0, min(1.0, base_score ** max(cfg.soft_power, 1e-8))))
        accepted = weight > 0.0
    else:
        accepted = len(reasons) == 0
        weight = 1.0 if accepted else 0.0

    return {
        'sample_id': int(view['sample_id']),
        'frame_id': int(view.get('frame_id', view['sample_id'])),
        'accepted': bool(accepted),
        'weight': float(weight),
        'reasons': reasons,
        'metrics': {
            'target_depth_verified_ratio': verified_ratio,
            'rgb_confidence_nonzero_ratio': rgb_mask_ratio,
            'target_depth_render_fallback_ratio': fallback_ratio,
            'mean_abs_rel_correction_verified': min_correction,
        },
    }


def evaluate_sampled_views_for_local_gating(sampled_views: Iterable[dict], cfg: PseudoLocalGatingConfig) -> List[dict]:
    if not cfg.enabled():
        results = []
        for view in sampled_views:
            results.append(
                {
                    'sample_id': int(view['sample_id']),
                    'frame_id': int(view.get('frame_id', view['sample_id'])),
                    'accepted': True,
                    'weight': 1.0,
                    'reasons': [],
                    'metrics': {
                        'target_depth_verified_ratio': _safe_float(view.get('target_depth_verified_ratio'), 0.0),
                        'rgb_confidence_nonzero_ratio': _safe_float(view.get('rgb_confidence_nonzero_ratio'), 0.0),
                        'target_depth_render_fallback_ratio': _safe_float(view.get('target_depth_render_fallback_ratio'), 1.0),
                        'mean_abs_rel_correction_verified': _extract_min_correction(view),
                    },
                }
            )
        return results
    return [evaluate_single_view_signal_gate(view, cfg) for view in sampled_views]
