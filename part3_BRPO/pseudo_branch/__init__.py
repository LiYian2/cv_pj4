# -*- coding: utf-8 -*-
"""Pseudo branch module for EDP.

Only the EDP/common surface remains package-level here. Legacy depth-target
helpers are intentionally excluded from the live GitHub subset so package
import stays independent of archived compatibility modules.
"""

from .common.flow_matcher import FlowMatcher
from .common.epipolar_depth import compute_edp_depth, compute_edp_depth_bidirectional
# Legacy depth-target helpers intentionally not exported in the live GitHub subset.
# from legacy_or_archive.retired_pseudo_branch_mask_target.depth_target_builder import (
#     get_intrinsic_matrix,
#     load_depth,
#     reproject_depth,
# )

__all__ = [
    "FlowMatcher",
    "compute_edp_depth",
    "compute_edp_depth_bidirectional",
]
