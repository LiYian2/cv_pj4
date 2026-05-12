# -*- coding: utf-8 -*-
"""Pseudo branch module for EDP.

Only the EDP/common surface remains package-level here. Retired depth-target
helpers are imported directly from legacy_or_archive so package import does not
pull standalone/compatibility target modules into the live online path.
"""

from .common.flow_matcher import FlowMatcher
from .common.epipolar_depth import compute_edp_depth, compute_edp_depth_bidirectional
from legacy_or_archive.retired_pseudo_branch_mask_target.depth_target_builder import (
    get_intrinsic_matrix,
    load_depth,
    reproject_depth,
)

__all__ = [
    "FlowMatcher",
    "compute_edp_depth",
    "compute_edp_depth_bidirectional",
    "load_depth",
    "get_intrinsic_matrix",
    "reproject_depth",
]
