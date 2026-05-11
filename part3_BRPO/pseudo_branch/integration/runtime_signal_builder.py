from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pseudo_branch.observation.pseudo_observation_brpo_style import (
    build_exact_brpo_upstream_target_observation,
    write_exact_brpo_upstream_target_observation_outputs,
)

from .runtime_exact_backend import RuntimeExactBackendBundle, load_exact_backend_frame_bundle
from .runtime_slot_selector import RuntimePseudoSlot


@dataclass
class RuntimeSignalBundle:
    slot: RuntimePseudoSlot
    signal_frame_out: Path
    result: dict[str, Any]
    meta: dict[str, Any]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_runtime_exact_signal_bundle(
    *,
    slot: RuntimePseudoSlot,
    frame_root: str | Path,
    exact_bundle: RuntimeExactBackendBundle,
    fusion_weight_left: np.ndarray | None = None,
    fusion_weight_right: np.ndarray | None = None,
    weight_source: str = "exact_confidence_fallback_v1",
    pair_forwarder=None,
) -> RuntimeSignalBundle:
    frame_root = Path(frame_root)
    signal_frame_out = frame_root / "signal_v2"
    signal_frame_out.mkdir(parents=True, exist_ok=True)

    if fusion_weight_left is None:
        fusion_weight_left = np.asarray(exact_bundle.fusion_weight_left, dtype=np.float32)
    if fusion_weight_right is None:
        fusion_weight_right = np.asarray(exact_bundle.fusion_weight_right, dtype=np.float32)

    depth_generation_mode = str(exact_bundle.exact_meta.get("depth_generation", {}).get("mode", "projected"))
    
    # Handle twoimg_pair_proxy_cm_capped_v1 mode specially
    if depth_generation_mode == "twoimg_pair_proxy_cm_capped_v1":
        depth_kwargs = _build_twoimg_pair_proxy_depth_override(
            slot=slot,
            exact_bundle=exact_bundle,
            frame_root=frame_root,
            pair_forwarder=pair_forwarder,
        )
    else:
        depth_kwargs = _depth_generation_kwargs(
            depth_generation_mode=depth_generation_mode,
            direct_depth_left=exact_bundle.direct_depth_left,
            direct_depth_right=exact_bundle.direct_depth_right,
        )

    twoimg_metadata = depth_kwargs.pop("twoimg_metadata", None)

    # Check for confidence_cm_override from cm_expansion
    confidence_cm_override = None
    if "confidence_cm_override" in exact_bundle.left_result:
        confidence_cm_override = np.asarray(exact_bundle.left_result["confidence_cm_override"], dtype=np.float32)
    
    result = build_exact_brpo_upstream_target_observation(
        support_left_exact=np.asarray(exact_bundle.left_result["support_mask"], dtype=np.float32),
        support_right_exact=np.asarray(exact_bundle.right_result["support_mask"], dtype=np.float32),
        projected_depth_left_exact=np.asarray(exact_bundle.left_result["projected_depth_map"], dtype=np.float32),
        projected_depth_right_exact=np.asarray(exact_bundle.right_result["projected_depth_map"], dtype=np.float32),
        confidence_left_exact=np.asarray(exact_bundle.left_result["confidence_map"], dtype=np.float32),
        confidence_right_exact=np.asarray(exact_bundle.right_result["confidence_map"], dtype=np.float32),
        fusion_weight_left=np.asarray(fusion_weight_left, dtype=np.float32),
        fusion_weight_right=np.asarray(fusion_weight_right, dtype=np.float32),
        provenance_left=np.asarray(exact_bundle.left_result["provenance_map"], dtype=np.float32),
        provenance_right=np.asarray(exact_bundle.right_result["provenance_map"], dtype=np.float32),
        confidence_cm_override=confidence_cm_override,
        **depth_kwargs,
    )
    
    # Add twoimg metadata to result summary if present
    if twoimg_metadata is not None:
        result["summary"]["twoimg_pair_proxy"] = twoimg_metadata
    
    meta = {
        "frame_id": int(slot.frame_id),
        "image_name": exact_bundle.pseudo_state.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": str(depth_kwargs.get("target_field_semantics", "exact_upstream_v1")),
        "depth_generation_mode": depth_generation_mode,
        "exact_backend_bundle_path": str(exact_bundle.exact_frame_out),
        "fusion_weight_source": str(weight_source),
        "consumer_contract": {
            "pseudo_observation_mode": "exact_brpo_upstream_target_v1",
            "shared_confidence": "pseudo_confidence_exact_brpo_upstream_target_v1",
            "depth_target": "pseudo_depth_target_exact_brpo_upstream_target_v1",
            "source_map": "pseudo_source_map_exact_brpo_upstream_target_v1",
            "upstream_backend": "exact_branch_native_v1",
        },
        "cm_expansion_override_applied": confidence_cm_override is not None,
    }
    write_exact_brpo_upstream_target_observation_outputs(signal_frame_out, result, meta)
    return RuntimeSignalBundle(slot=slot, signal_frame_out=signal_frame_out, result=result, meta=meta)


def rebuild_runtime_exact_signal_from_existing_roots(
    *,
    slot: RuntimePseudoSlot,
    frame_root: str | Path,
    exact_backend_frame_root: str | Path,
    existing_signal_meta_path: str | Path,
    pair_forwarder=None,
) -> RuntimeSignalBundle:
    signal_meta = _load_json(existing_signal_meta_path)
    fusion_weight_left_path = signal_meta["fusion_weight_left_path"]
    fusion_weight_right_path = signal_meta["fusion_weight_right_path"]
    exact_inputs = load_exact_backend_frame_bundle(exact_backend_frame_root)
    exact_backend_meta_path = Path(exact_backend_frame_root) / "exact_backend_meta.json"
    exact_backend_meta = _load_json(exact_backend_meta_path) if exact_backend_meta_path.exists() else {}
    depth_generation_mode = str(exact_backend_meta.get("depth_generation", {}).get("mode", signal_meta.get("depth_generation_mode", "projected")))
    
    # Handle twoimg_pair_proxy_cm_capped_v1 mode specially
    if depth_generation_mode == "twoimg_pair_proxy_cm_capped_v1":
        # Build pseudo_bundle-like object for twoimg generation
        from dataclasses import dataclass
        import numpy as np
        
        # Reconstruct minimal exact_bundle structure needed for twoimg
        @dataclass
        class MinimalExactBundle:
            left_result: dict
            right_result: dict
            exact_meta: dict
            pseudo_rgb_path: str
        
        # Find pseudo RGB path from runtime_inputs
        runtime_inputs_dir = Path(frame_root) / "runtime_inputs"
        pseudo_rgb_path = None
        for candidate in ["pseudo_fused_rgb.png", "pseudo_render_rgb_runtime.png", "pseudo_gt_rgb_runtime.png"]:
            if (runtime_inputs_dir / candidate).exists():
                pseudo_rgb_path = str(runtime_inputs_dir / candidate)
                break
        
        minimal_bundle = MinimalExactBundle(
            left_result={
                "support_mask": exact_inputs["support_left_exact"],
                "projected_depth_map": exact_inputs["projected_depth_left_exact"],
            },
            right_result={
                "support_mask": exact_inputs["support_right_exact"],
            },
            exact_meta=exact_backend_meta,
            pseudo_rgb_path=pseudo_rgb_path or "",
        )
        
        depth_kwargs = _build_twoimg_pair_proxy_depth_override(
            slot=slot,
            exact_bundle=minimal_bundle,
            frame_root=Path(frame_root),
            pair_forwarder=pair_forwarder,
        )
    else:
        depth_kwargs = _depth_generation_kwargs(
            depth_generation_mode=depth_generation_mode,
            direct_depth_left=exact_inputs.get("direct_depth_left"),
            direct_depth_right=exact_inputs.get("direct_depth_right"),
        )
    
    twoimg_metadata = depth_kwargs.pop("twoimg_metadata", None)

    result = build_exact_brpo_upstream_target_observation(
        support_left_exact=exact_inputs["support_left_exact"],
        support_right_exact=exact_inputs["support_right_exact"],
        projected_depth_left_exact=exact_inputs["projected_depth_left_exact"],
        projected_depth_right_exact=exact_inputs["projected_depth_right_exact"],
        confidence_left_exact=exact_inputs["confidence_left_exact"],
        confidence_right_exact=exact_inputs["confidence_right_exact"],
        fusion_weight_left=np.load(fusion_weight_left_path).astype(np.float32),
        fusion_weight_right=np.load(fusion_weight_right_path).astype(np.float32),
        provenance_left=exact_inputs["provenance_left"],
        provenance_right=exact_inputs["provenance_right"],
        confidence_cm_override=exact_inputs.get("confidence_cm_override"),
        **depth_kwargs,
    )
    
    # Add twoimg metadata if present
    if twoimg_metadata is not None:
        result["summary"]["twoimg_pair_proxy"] = twoimg_metadata
    
    frame_root = Path(frame_root)
    signal_frame_out = frame_root / "signal_v2"
    signal_frame_out.mkdir(parents=True, exist_ok=True)
    meta = {
        "frame_id": int(slot.frame_id),
        "image_name": signal_meta.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": str(depth_kwargs.get("target_field_semantics", "exact_upstream_v1")),
        "depth_generation_mode": depth_generation_mode,
        "exact_backend_bundle_path": str(exact_backend_frame_root),
        "fusion_weight_left_path": str(fusion_weight_left_path),
        "fusion_weight_right_path": str(fusion_weight_right_path),
        "consumer_contract": signal_meta.get("consumer_contract", {}),
        "cm_expansion_override_applied": exact_inputs.get("confidence_cm_override") is not None,
    }
    write_exact_brpo_upstream_target_observation_outputs(signal_frame_out, result, meta)
    return RuntimeSignalBundle(slot=slot, signal_frame_out=signal_frame_out, result=result, meta=meta)
