from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PseudoLocalGatingConfig:
    # Core mode and params
    mode: str = 'off'
    params: str = 'xyz'

    # Legacy gating thresholds
    min_verified_ratio: float = 0.01
    min_rgb_mask_ratio: float = 0.01
    max_fallback_ratio: float = 0.995
    min_correction: float = 0.0
    soft_power: float = 1.0
    log_interval: int = 20

    # SPGM-specific hyperparameters
    spgm_num_clusters: int = 3
    spgm_alpha_depth: float = 0.5
    spgm_beta_entropy: float = 0.5
    spgm_gamma_entropy: float = 0.5
    spgm_support_eta: float = 0.5
    spgm_weight_floor: float = 0.05
    spgm_entropy_bins: int = 32
    spgm_density_mode: str = 'opacity_support'
    spgm_cluster_keep_near: float = 1.0
    spgm_cluster_keep_mid: float = 0.8
    spgm_cluster_keep_far: float = 0.6
    spgm_policy_mode: str = 'dense_keep'
    spgm_ranking_mode: str = 'v1'
    spgm_lambda_support_rank: float = 0.0
    spgm_selector_keep_ratio_near: float = 1.0
    spgm_selector_keep_ratio_mid: float = 1.0
    spgm_selector_keep_ratio_far: float = 1.0
    spgm_selector_min_keep: int = 1

    # B3 deterministic manager controls
    spgm_manager_mode: str = 'summary_only'
    spgm_state_candidate_quantile: float = 0.5
    spgm_state_base_scale_near: float = 1.0
    spgm_state_base_scale_mid: float = 0.95
    spgm_state_base_scale_far: float = 0.90
    spgm_state_participation_keep_near: float = 1.0
    spgm_state_participation_keep_mid: float = 0.9
    spgm_state_participation_keep_far: float = 0.75
    spgm_state_opacity_floor_near: float = 1.0
    spgm_state_opacity_floor_mid: float = 0.98
    spgm_state_opacity_floor_far: float = 0.92

    def enabled(self) -> bool:
        return (self.mode or 'off') != 'off'

    def uses_visibility_union(self) -> bool:
        return self.mode in {'hard_visible_union_signal', 'soft_visible_union_signal'}

    def uses_spgm(self) -> bool:
        return self.mode in {'spgm_keep', 'spgm_soft'}

    def is_soft(self) -> bool:
        return self.mode in {'soft_visible_union_signal', 'spgm_soft'}

    def as_dict(self) -> dict:
        return asdict(self)
