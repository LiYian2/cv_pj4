"""G~ packages and Gaussian-side control helpers."""

from .gaussian_param_groups import build_micro_gaussian_param_groups
from .local_gating import (
    PseudoLocalGatingConfig,
    evaluate_sampled_views_for_local_gating,
    build_visibility_weight_map,
    apply_gaussian_grad_mask,
    build_iteration_gating_summary,
)
from .spgm import (
    collect_spgm_stats,
    build_spgm_importance_score,
    build_spgm_grad_weights,
    build_spgm_update_policy,
    apply_spgm_state_management,
)

__all__ = [
    'build_micro_gaussian_param_groups',
    'PseudoLocalGatingConfig',
    'evaluate_sampled_views_for_local_gating',
    'build_visibility_weight_map',
    'apply_gaussian_grad_mask',
    'build_iteration_gating_summary',
    'collect_spgm_stats',
    'build_spgm_importance_score',
    'build_spgm_grad_weights',
    'build_spgm_update_policy',
    'apply_spgm_state_management',
]
