# -*- coding: utf-8 -*-
"""2IMG + PAIR-proxy adaptive scale depth generation.

This module provides dense depth supervision within C_m (confidence mask) region,
where PAIR projected depth has sparse coverage (~3-20%).

Flow:
1. MASt3R(pseudo, pseudo) -> depth_2img (100% coverage, unknown scale)
2. PAIR projected_depth -> scale anchor (metric depth in matching regions)
3. Adaptive scale calibration per depth range
4. C_m cap -> only enable depth within confidence mask

Key constraint: depth effective mask does NOT exceed C_m boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mast3r_pair_forward import MASt3RPairForward, MASt3RPairBundle


DEFAULT_DEPTH_RANGES = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
MIN_SAMPLES_PER_RANGE = 50  # Minimum pixels to compute scale for a range
MIN_DEPTH_VALID = 0.1  # Minimum depth value considered valid


@dataclass
class TwoImgPairProxyDepthResult:
    """Result of 2IMG + PAIR-proxy depth generation."""
    depth_2img_raw: np.ndarray  # Raw 2IMG depth (unknown scale)
    depth_pair_anchor: Optional[np.ndarray]  # PAIR projected depth (scale anchor)
    confidence_pair: Optional[np.ndarray]  # PAIR matching confidence
    depth_calibrated: np.ndarray  # Calibrated depth (after PAIR-proxy adaptive scale)
    depth_effective: np.ndarray  # Final depth (C_m capped, ready for supervision)
    cm_mask: np.ndarray  # Confidence mask (C_m from RGB matching)
    scale_by_range: Dict[Tuple[float, float], float]  # Per-range scale factors
    fallback_scale: float  # Global fallback scale for uncovered pixels
    metadata: Dict[str, float]  # Coverage ratios
    
    def save(self, output_dir: Path) -> None:
        """Save all outputs to directory."""
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "depth_2img_raw.npy", self.depth_2img_raw.astype(np.float32))
        if self.depth_pair_anchor is not None:
            np.save(output_dir / "depth_pair_anchor.npy", self.depth_pair_anchor.astype(np.float32))
        if self.confidence_pair is not None:
            np.save(output_dir / "confidence_pair.npy", self.confidence_pair.astype(np.float32))
        np.save(output_dir / "depth_calibrated.npy", self.depth_calibrated.astype(np.float32))
        np.save(output_dir / "depth_effective.npy", self.depth_effective.astype(np.float32))
        np.save(output_dir / "cm_mask.npy", self.cm_mask.astype(np.float32))
        
        # Convert scales to native Python floats for JSON
        scale_dict = {f"{r[0]}-{r[1]}": float(s) for r, s in self.scale_by_range.items()}
        with open(output_dir / "scale_by_range.json", "w") as f:
            json.dump({"scales": scale_dict, "fallback": float(self.fallback_scale)}, f, indent=2)
        
        # Convert metadata to native Python floats
        meta_native = {k: float(v) for k, v in self.metadata.items()}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta_native, f, indent=2)


def run_2img_forward(
    forwarder: MASt3RPairForward,
    pseudo_rgb_path: str,
    size: int = 512,
) -> MASt3RPairBundle:
    """Run MASt3R(pseudo, pseudo) to get 2IMG depth.
    
    Returns pts3d with z-component as depth in pseudo's own coordinate frame.
    This depth has 100% coverage but unknown scale (MASt3R's internal metric).
    """
    return forwarder.run_pair(img1_path=pseudo_rgb_path, img2_path=pseudo_rgb_path, size=size)


def compute_pair_proxy_adaptive_scale(
    depth_2img: np.ndarray,
    depth_pair_anchor: np.ndarray,
    depth_ranges: List[Tuple[float, float]] = DEFAULT_DEPTH_RANGES,
    min_samples: int = MIN_SAMPLES_PER_RANGE,
    min_depth_valid: float = MIN_DEPTH_VALID,
) -> Tuple[Dict[Tuple[float, float], float], float, np.ndarray]:
    """Compute adaptive scale per depth range using PAIR projected depth as anchor.
    
    Args:
        depth_2img: Raw 2IMG depth (H, W), unknown scale
        depth_pair_anchor: PAIR projected depth (H, W), metric scale (only valid in matching regions)
        depth_ranges: List of (min, max) depth ranges for scale calibration
        min_samples: Minimum pixels per range to compute scale
        min_depth_valid: Minimum depth value considered valid
        
    Returns:
        scales_by_range: Dict mapping depth range -> scale factor
        fallback_scale: Global scale for pixels not covered by any range-specific scale
        calibrated_depth: Depth after applying range-specific scales + fallback
    """
    h, w = depth_2img.shape
    scales_by_range: Dict[Tuple[float, float], float] = {}
    
    # Compute global fallback scale from all valid anchor pixels
    valid_anchor = depth_pair_anchor > min_depth_valid
    valid_2img = depth_2img > min_depth_valid
    valid_global = valid_anchor & valid_2img
    
    if valid_global.sum() > min_samples:
        fallback_scale = float(np.median(depth_pair_anchor[valid_global]) / np.median(depth_2img[valid_global]))
    else:
        # No valid anchor, estimate scale from overall depth ratio
        fallback_scale = 1.0
    
    # Compute scale per range (only for valid anchor pixels in that range)
    for range_tuple in depth_ranges:
        r_min, r_max = range_tuple
        # Mask: pixels where PAIR projected depth is valid AND falls in this range
        mask = (depth_pair_anchor > min_depth_valid) &                (depth_pair_anchor >= r_min) &                (depth_pair_anchor < r_max) &                (depth_2img > min_depth_valid)
        
        if mask.sum() >= min_samples:
            # Scale = median(PAIR depth) / median(2IMG depth) in this range
            scale = float(np.median(depth_pair_anchor[mask]) / np.median(depth_2img[mask]))
            scales_by_range[range_tuple] = scale
    
    # Apply adaptive scale to create calibrated depth
    calibrated_depth = np.zeros_like(depth_2img)
    
    # First, apply range-specific scales where valid PAIR anchor exists
    anchor_valid = depth_pair_anchor > min_depth_valid
    for range_tuple in depth_ranges:
        if range_tuple in scales_by_range:
            r_min, r_max = range_tuple
            scale = scales_by_range[range_tuple]
            mask = anchor_valid & (depth_pair_anchor >= r_min) & (depth_pair_anchor < r_max)
            calibrated_depth[mask] = depth_2img[mask] * scale
    
    # For pixels without valid PAIR anchor, use fallback scale (full image)
    uncovered = (calibrated_depth == 0) & (depth_2img > min_depth_valid)
    calibrated_depth[uncovered] = depth_2img[uncovered] * fallback_scale
    
    return scales_by_range, fallback_scale, calibrated_depth


def apply_cm_cap(
    depth_calibrated: np.ndarray,
    cm_mask: np.ndarray,
    fill_with_zero: bool = True,
) -> np.ndarray:
    """Apply C_m support cap without changing depth values.

    C_m is a supervision weight map: both=1.0, single=0.5, none=0.0.
    It must not scale the depth target values themselves.  The cap only decides
    where a depth target exists; the loss consumer applies C_m as weight later.
    """
    if fill_with_zero:
        support = (np.asarray(cm_mask, dtype=np.float32) > 0.0).astype(np.float32)
        return depth_calibrated * support
    else:
        return depth_calibrated


def compute_coverage_metadata(
    cm_mask: np.ndarray,
    depth_pair_anchor: np.ndarray,
    depth_effective: np.ndarray,
    min_depth_valid: float = MIN_DEPTH_VALID,
) -> Dict[str, float]:
    """Compute the three coverage ratios required by design.
    
    Returns:
        cm_nonzero_ratio: Fraction of image pixels with C_m > 0
        projected_depth_union_ratio: Fraction of C_m pixels with PAIR projected depth
        twoimg_depth_effective_ratio_after_cm_cap: Fraction of C_m pixels with effective depth
    """
    total_pixels = cm_mask.size
    
    cm_nonzero = cm_mask > 0.5
    cm_nonzero_ratio = float(cm_nonzero.sum() / total_pixels)
    
    # PAIR projected depth coverage within C_m
    pair_valid_in_cm = (depth_pair_anchor > min_depth_valid) & cm_nonzero
    projected_depth_union_ratio = float(pair_valid_in_cm.sum() / max(cm_nonzero.sum(), 1))
    
    # 2IMG effective depth coverage within C_m
    effective_in_cm = (depth_effective > min_depth_valid) & cm_nonzero
    twoimg_depth_effective_ratio_after_cm_cap = float(effective_in_cm.sum() / max(cm_nonzero.sum(), 1))
    
    return {
        "cm_nonzero_ratio": cm_nonzero_ratio,
        "projected_depth_union_ratio": projected_depth_union_ratio,
        "twoimg_depth_effective_ratio_after_cm_cap": twoimg_depth_effective_ratio_after_cm_cap,
    }


def build_twoimg_pair_proxy_depth(
    forwarder: MASt3RPairForward,
    pseudo_rgb_path: str,
    depth_pair_anchor: np.ndarray,  # From exact_backend projected_depth
    cm_mask: np.ndarray,  # From signal_v2 confidence
    depth_ranges: List[Tuple[float, float]] = DEFAULT_DEPTH_RANGES,
    size: int = 512,
    output_dir: Optional[Path] = None,
) -> TwoImgPairProxyDepthResult:
    """Main entry point: Generate 2IMG + PAIR-proxy calibrated depth.
    
    Args:
        forwarder: Shared MASt3R forwarder
        pseudo_rgb_path: Path to pseudo RGB image
        depth_pair_anchor: PAIR projected depth (from exact_backend)
        cm_mask: Confidence mask C_m (from signal_v2)
        depth_ranges: Depth ranges for adaptive scale
        size: Image size for MASt3R
        output_dir: Optional directory to save outputs
        
    Returns:
        TwoImgPairProxyDepthResult with all intermediate and final depths
    """
    # Step 1: Get 2IMG depth
    bundle_2img = run_2img_forward(forwarder, pseudo_rgb_path, size)
    depth_2img = bundle_2img.pts3d_1[..., 2].astype(np.float32)
    
    # Step 2: Apply PAIR-proxy adaptive scale
    scales_by_range, fallback_scale, depth_calibrated = compute_pair_proxy_adaptive_scale(
        depth_2img=depth_2img,
        depth_pair_anchor=depth_pair_anchor,
        depth_ranges=depth_ranges,
    )
    
    # Step 3: Apply C_m cap
    depth_effective = apply_cm_cap(depth_calibrated, cm_mask)
    
    # Step 4: Compute metadata
    metadata = compute_coverage_metadata(cm_mask, depth_pair_anchor, depth_effective)
    
    result = TwoImgPairProxyDepthResult(
        depth_2img_raw=depth_2img,
        depth_pair_anchor=depth_pair_anchor,
        confidence_pair=None,  # Not computed here, comes from exact_backend
        depth_calibrated=depth_calibrated,
        depth_effective=depth_effective,
        cm_mask=cm_mask,
        scale_by_range=scales_by_range,
        fallback_scale=fallback_scale,
        metadata=metadata,
    )
    
    if output_dir is not None:
        result.save(output_dir)
    
    return result
