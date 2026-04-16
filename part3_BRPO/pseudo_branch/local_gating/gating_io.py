from __future__ import annotations

from typing import List


def build_iteration_gating_summary(mode: str, params: str, sampled_ids: List[int], gate_results: List[dict], visibility_stats: dict, grad_stats: dict) -> dict:
    accepted_ids = [int(item['sample_id']) for item in gate_results if item.get('weight', 0.0) > 0]
    rejected = [item for item in gate_results if item.get('weight', 0.0) <= 0]
    rejected_ids = [int(item['sample_id']) for item in rejected]
    rejected_reasons = {str(int(item['sample_id'])): list(item.get('reasons', [])) for item in rejected if item.get('reasons')}
    accepted_signal_weights = {str(int(item['sample_id'])): float(item.get('weight', 0.0)) for item in gate_results if item.get('weight', 0.0) > 0}
    metrics = {
        str(int(item['sample_id'])): dict(item.get('metrics', {}))
        for item in gate_results
    }
    return {
        'pseudo_local_gating_mode': mode,
        'pseudo_local_gating_params': params,
        'sampled_pseudo_sample_ids': [int(x) for x in sampled_ids],
        'accepted_pseudo_sample_ids': accepted_ids,
        'rejected_pseudo_sample_ids': rejected_ids,
        'rejected_reasons': rejected_reasons,
        'accepted_signal_weights': accepted_signal_weights,
        'sample_signal_metrics': metrics,
        'visible_union_ratio': visibility_stats.get('visible_union_ratio'),
        'visible_union_weight_mean': visibility_stats.get('visible_union_weight_mean'),
        'accepted_visibility_count': visibility_stats.get('accepted_count'),
        'grad_keep_ratio_xyz': grad_stats.get('grad_keep_ratio_xyz'),
        'grad_keep_ratio_opacity': grad_stats.get('grad_keep_ratio_opacity'),
        'grad_weight_mean_xyz': grad_stats.get('grad_weight_mean_xyz'),
        'grad_weight_mean_opacity': grad_stats.get('grad_weight_mean_opacity'),
        'grad_norm_xyz_pre_mask': grad_stats.get('grad_norm_xyz_pre_mask'),
        'grad_norm_xyz_post_mask': grad_stats.get('grad_norm_xyz_post_mask'),
        'grad_norm_opacity_pre_mask': grad_stats.get('grad_norm_opacity_pre_mask'),
        'grad_norm_opacity_post_mask': grad_stats.get('grad_norm_opacity_post_mask'),
    }
