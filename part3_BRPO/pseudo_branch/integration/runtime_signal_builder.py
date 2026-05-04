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
    )
    meta = {
        "frame_id": int(slot.frame_id),
        "image_name": exact_bundle.pseudo_state.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "verifier_backend_semantics": "exact_branch_native_v1",
        "target_field_semantics": "exact_upstream_v1",
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
        "target_field_semantics": "exact_upstream_v1",
        "exact_backend_bundle_path": str(exact_backend_frame_root),
        "fusion_weight_left_path": str(fusion_weight_left_path),
        "fusion_weight_right_path": str(fusion_weight_right_path),
        "consumer_contract": signal_meta.get("consumer_contract", {}),
    }
    write_exact_brpo_upstream_target_observation_outputs(signal_frame_out, result, meta)
    return RuntimeSignalBundle(slot=slot, signal_frame_out=signal_frame_out, result=result, meta=meta)
