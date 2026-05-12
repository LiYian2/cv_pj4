from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core_shared.targets.exact_upstream_observation import (
    build_exact_brpo_upstream_target_observation,
    write_exact_brpo_upstream_target_observation_outputs,
)

from .runtime_exact_backend import RuntimeExactBackendBundle, load_exact_backend_frame_bundle
from .slot_selector import RuntimePseudoSlot
from .signal_depth_overrides import _build_twoimg_pair_proxy_depth_override, _depth_generation_kwargs


@dataclass
class RuntimeSignalBundle:
    slot: RuntimePseudoSlot
    signal_frame_out: Path
    result: dict[str, Any]
    meta: dict[str, Any]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
