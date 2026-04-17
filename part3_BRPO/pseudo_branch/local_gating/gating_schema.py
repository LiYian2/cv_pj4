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
    
    # SPGM-specific hyperparameters (Phase 0 plumbing - v1 defaults)
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

    def enabled(self) -> bool:
        return (self.mode or 'off') != 'off'
    
    def uses_visibility_union(self) -> bool:
        """Check if mode uses legacy visibility-union path."""
        return self.mode in {'hard_visible_union_signal', 'soft_visible_union_signal'}

    def uses_spgm(self) -> bool:
        """Check if mode uses SPGM path."""
        return self.mode in {'spgm_keep', 'spgm_soft'}

    def is_soft(self) -> bool:
        """Check if mode uses soft gating (for signal gate behavior)."""
        # Covers both legacy soft_visible_union_signal and spgm_soft
        return self.mode in {'soft_visible_union_signal', 'spgm_soft'}

    def as_dict(self) -> dict:
        return asdict(self)
