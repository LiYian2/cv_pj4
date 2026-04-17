"""SPGM (Scene Perception Gaussian Management) module for Part3 BRPO.

Phase 0 plumbing: skeleton exports for wiring verification.
Actual implementation will be filled in subsequent phases.
"""

from .stats import collect_spgm_stats
from .score import build_spgm_importance_score
from .policy import build_spgm_grad_weights

__all__ = [
    'collect_spgm_stats',
    'build_spgm_importance_score',
    'build_spgm_grad_weights',
]
