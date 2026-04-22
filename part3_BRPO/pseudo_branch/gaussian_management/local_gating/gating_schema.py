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

    # B3 deterministic manager controls (legacy)
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

    # === Direct BRPO semantic axes (G-BRPO-0) ===
    # Score semantics: legacy_v1 uses state_score/participation_score; brpo_unified_v1 uses paper unified S_i
    spgm_score_semantics: str = 'legacy_v1'
    # Control universe: 'active' uses accepted-pseudo active_mask; 'population_active' uses scene-level population_active_mask
    spgm_control_universe: str = 'active'
    # Action semantics: inherit_manager_mode preserves legacy CLI behavior; direct BRPO sets explicit action semantics.
    spgm_action_semantics: str = 'inherit_manager_mode'
    # Timing semantics: 'delayed' (iter t decide, iter t+1 apply) vs 'current_step_probe_loss' (probe->decide->loss in same iter)
    spgm_timing_mode: str = 'delayed'

    # Direct BRPO hyperparameters
    # Global drop rate r (BRPO paper: p_i^{drop} = r * w_cluster * S_i)
    spgm_drop_rate_global: float = 0.05
    # Cluster weights for drop probability scaling
    spgm_cluster_weight_near: float = 1.0
    spgm_cluster_weight_mid: float = 1.0
    spgm_cluster_weight_far: float = 1.0
    # Seed mode for Bernoulli sampling: 'per_iter' (new seed each iter) or 'fixed' (deterministic)
    spgm_sample_seed_mode: str = 'per_iter'

    # BRPO unified score alpha: S_i = alpha * hat_s_i^{(z)} + (1-alpha) * hat_s_i^{(rho)}
    spgm_brpo_alpha: float = 0.5

    def enabled(self) -> bool:
        return (self.mode or 'off') != 'off'

    def uses_visibility_union(self) -> bool:
        return self.mode in {'hard_visible_union_signal', 'soft_visible_union_signal'}

    def uses_spgm(self) -> bool:
        # Direct BRPO path also uses SPGM infrastructure
        return self.mode in {'spgm_keep', 'spgm_soft'}

    def is_soft(self) -> bool:
        return self.mode in {'soft_visible_union_signal', 'spgm_soft'}

    def uses_direct_brpo(self) -> bool:
        """Check if using direct BRPO semantic path (not legacy deterministic)."""
        return (
            self.spgm_score_semantics == 'brpo_unified_v1'
            or self.spgm_action_semantics == 'stochastic_bernoulli_opacity'
            or self.spgm_timing_mode == 'current_step_probe_loss'
        )

    def as_dict(self) -> dict:
        return asdict(self)