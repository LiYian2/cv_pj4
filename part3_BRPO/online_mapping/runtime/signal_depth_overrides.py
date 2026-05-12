from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .runtime_exact_backend import RuntimeExactBackendBundle
from .slot_selector import RuntimePseudoSlot


def _build_twoimg_pair_proxy_depth_override(
    *,
    slot: RuntimePseudoSlot,
    exact_bundle: RuntimeExactBackendBundle,
    frame_root: Path,
    pair_forwarder=None,
) -> dict[str, Any]:
    """Build 2IMG+PAIR-proxy calibrated depth for twoimg_pair_proxy_cm_capped_v1 mode.
    
    This runs MASt3R(pseudo, pseudo) to get dense depth, then calibrates using
    PAIR projected_depth as scale anchor, and finally caps to C_m boundary.
    
    Returns kwargs for build_exact_brpo_upstream_target_observation.
    """
    from pseudo_branch.common.mast3r_pair_forward import get_shared_mast3r_pair_forward
    from pseudo_branch.common.twoimg_pair_proxy_depth import build_twoimg_pair_proxy_depth
    
    # Get pair forwarder
    matcher_model_name = str(exact_bundle.exact_meta.get("matcher", {}).get("matcher_model_name", ""))
    matcher_device = str(exact_bundle.exact_meta.get("matcher", {}).get("matcher_device", "cuda"))
    if pair_forwarder is None:
        pair_forwarder = get_shared_mast3r_pair_forward(
            model_name=matcher_model_name,
            device=matcher_device,
        )
    
    # Build initial C_m from support (before depth override)
    support_left = np.asarray(exact_bundle.left_result["support_mask"], dtype=np.float32) > 0.5
    support_right = np.asarray(exact_bundle.right_result["support_mask"], dtype=np.float32) > 0.5
    verify_both = support_left & support_right
    verify_xor = (support_left & (~support_right)) | (support_right & (~support_left))
    cm_mask = np.zeros_like(support_left, dtype=np.float32)
    cm_mask[verify_both] = 1.0
    cm_mask[verify_xor] = 0.5
    
    # Get PAIR projected depth as scale anchor (use left branch)
    depth_pair_anchor = np.asarray(exact_bundle.left_result["projected_depth_map"], dtype=np.float32)
    
    # Find pseudo RGB path
    pseudo_rgb_path = str(exact_bundle.pseudo_rgb_path)
    
    # Build 2IMG+PAIR-proxy depth
    twoimg_dir = frame_root / "twoimg_pair_proxy_v1"
    twoimg_result = build_twoimg_pair_proxy_depth(
        forwarder=pair_forwarder,
        pseudo_rgb_path=pseudo_rgb_path,
        depth_pair_anchor=depth_pair_anchor,
        cm_mask=cm_mask,
        output_dir=twoimg_dir,
    )
    
    # Save twoimg metadata
    twoimg_meta = {
        "frame_id": int(slot.frame_id),
        "mode": "twoimg_pair_proxy_cm_capped_v1",
        "pseudo_rgb_path": pseudo_rgb_path,
        "anchor_source": "projected_depth_left_exact",
        "cm_source": "exact_backend_support",
        "metadata": twoimg_result.metadata,
        "scale_by_range": {f"{r[0]}-{r[1]}": s for r, s in twoimg_result.scale_by_range.items()},
        "fallback_scale": float(twoimg_result.fallback_scale),
    }
    with open(twoimg_dir / "twoimg_pair_proxy_meta.json", "w") as f:
        json.dump(twoimg_meta, f, indent=2)
    
    # Return kwargs for build_exact_brpo_upstream_target_observation
    # Note: we use the same depth for left and right override since 2IMG generates single-view depth
    depth_effective = twoimg_result.depth_effective
    
    return {
        "target_depth_left_override": depth_effective,
        "target_depth_right_override": depth_effective,
        "target_field_semantics": "exact_upstream_twoimg_pair_proxy_v1",
        "depth_input_semantics": "twoimg_pair_proxy_depth_capped_by_cm",
        "depth_target_rule": "replace exact upstream target_depth with 2IMG+PAIR-proxy calibrated depth capped to C_m; keep exact C_m, valid_mask, and target_confidence semantics unchanged",
        "twoimg_metadata": twoimg_result.metadata,
    }


def _depth_generation_kwargs(
    *,
    depth_generation_mode: str,
    direct_depth_left: np.ndarray | None,
    direct_depth_right: np.ndarray | None,
) -> dict[str, Any]:
    mode = str(depth_generation_mode or "projected")
    if mode == "mast3r_direct_exact_anchor_v1" and direct_depth_left is not None and direct_depth_right is not None:
        return {
            "target_depth_left_override": np.asarray(direct_depth_left, dtype=np.float32),
            "target_depth_right_override": np.asarray(direct_depth_right, dtype=np.float32),
            "target_field_semantics": "exact_upstream_directdepth_v1",
            "depth_input_semantics": "mast3r_direct_depth_anchored_by_exact_projected",
            "depth_target_rule": "replace exact upstream target_depth values with MASt3R direct pseudo depth anchored by exact projected depth; keep exact C_m, valid_mask, and target_confidence semantics unchanged",
        }
    # twoimg_pair_proxy_cm_capped_v1 requires additional context, handled in build_runtime_exact_signal_bundle
    return {
        "target_field_semantics": "exact_upstream_v1",
        "depth_input_semantics": "projected_depth_exact",
        "depth_target_rule": "exact upstream projected-depth composition with continuous confidence, no render fallback",
    }


