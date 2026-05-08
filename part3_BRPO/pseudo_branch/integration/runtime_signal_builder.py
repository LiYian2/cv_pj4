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
) -> RuntimeSignalBundle:
    frame_root = Path(frame_root)
    signal_frame_out = frame_root / "signal_v2"
    signal_frame_out.mkdir(parents=True, exist_ok=True)

    if fusion_weight_left is None:
        fusion_weight_left = np.asarray(exact_bundle.fusion_weight_left, dtype=np.float32)
    if fusion_weight_right is None:
        fusion_weight_right = np.asarray(exact_bundle.fusion_weight_right, dtype=np.float32)

    depth_generation_mode = str(exact_bundle.exact_meta.get("depth_generation", {}).get("mode", "projected"))
    depth_kwargs = _depth_generation_kwargs(
        depth_generation_mode=depth_generation_mode,
        direct_depth_left=exact_bundle.direct_depth_left,
        direct_depth_right=exact_bundle.direct_depth_right,
    )

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
        **depth_kwargs,
    )
    meta = {
        "frame_id": int(slot.frame_id),
        "image_name": exact_bundle.pseudo_state.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": str(depth_kwargs["target_field_semantics"]),
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
    }
    write_exact_brpo_upstream_target_observation_outputs(signal_frame_out, result, meta)
    return RuntimeSignalBundle(slot=slot, signal_frame_out=signal_frame_out, result=result, meta=meta)


def rebuild_runtime_exact_signal_from_existing_roots(
    *,
    slot: RuntimePseudoSlot,
    frame_root: str | Path,
    exact_backend_frame_root: str | Path,
    existing_signal_meta_path: str | Path,
) -> RuntimeSignalBundle:
    signal_meta = _load_json(existing_signal_meta_path)
    fusion_weight_left_path = signal_meta["fusion_weight_left_path"]
    fusion_weight_right_path = signal_meta["fusion_weight_right_path"]
    exact_inputs = load_exact_backend_frame_bundle(exact_backend_frame_root)
    exact_backend_meta_path = Path(exact_backend_frame_root) / "exact_backend_meta.json"
    exact_backend_meta = _load_json(exact_backend_meta_path) if exact_backend_meta_path.exists() else {}
    depth_generation_mode = str(exact_backend_meta.get("depth_generation", {}).get("mode", signal_meta.get("depth_generation_mode", "projected")))
    depth_kwargs = _depth_generation_kwargs(
        depth_generation_mode=depth_generation_mode,
        direct_depth_left=exact_inputs.get("direct_depth_left"),
        direct_depth_right=exact_inputs.get("direct_depth_right"),
    )
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
        **depth_kwargs,
    )
    frame_root = Path(frame_root)
    signal_frame_out = frame_root / "signal_v2"
    signal_frame_out.mkdir(parents=True, exist_ok=True)
    meta = {
        "frame_id": int(slot.frame_id),
        "image_name": signal_meta.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": str(depth_kwargs["target_field_semantics"]),
        "depth_generation_mode": depth_generation_mode,
        "exact_backend_bundle_path": str(exact_backend_frame_root),
        "fusion_weight_left_path": str(fusion_weight_left_path),
        "fusion_weight_right_path": str(fusion_weight_right_path),
        "consumer_contract": signal_meta.get("consumer_contract", {}),
    }
    write_exact_brpo_upstream_target_observation_outputs(signal_frame_out, result, meta)
    return RuntimeSignalBundle(slot=slot, signal_frame_out=signal_frame_out, result=result, meta=meta)
