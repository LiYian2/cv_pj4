from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from core_shared.verification.brpo_reprojection_verify import create_viewpoint_from_state
from core_shared.records.backend_pseudo_view_loader import BackendPseudoViewRecord
from core_shared.pose.pseudo_camera_state import make_viewpoint_trainable

from .runtime_debug_export import write_runtime_pseudo_record_frame
from online_mapping.runtime.slot_selector import RuntimePseudoSlot

if TYPE_CHECKING:
    from pseudo_branch.integration.runtime_signal_builder import RuntimeSignalBundle


@dataclass
class RuntimePseudoRecordBundle:
    slot: RuntimePseudoSlot
    record_frame_out: Path
    record: BackendPseudoViewRecord
    record_meta: dict[str, Any]


def _camera_json_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "pose_c2w": state["pose_c2w"],
        "intrinsics_px": {
            "fx": float(state["fx"]),
            "fy": float(state["fy"]),
            "cx": float(state["cx"]),
            "cy": float(state["cy"]),
        },
        "image_size": {
            "width": int(state["image_width"]),
            "height": int(state["image_height"]),
        },
    }


def build_runtime_pseudo_record_bundle(
    *,
    slot: RuntimePseudoSlot,
    frame_root: str | Path,
    pseudo_state: dict[str, Any],
    pseudo_render_rgb: np.ndarray,
    signal_bundle: RuntimeSignalBundle,
    stageA_scene_scale: float | None = None,
) -> RuntimePseudoRecordBundle:
    frame_root = Path(frame_root)
    record_frame_out = frame_root / "runtime_pseudo_record"
    result = signal_bundle.result
    viewpoint = create_viewpoint_from_state(pseudo_state)
    make_viewpoint_trainable(viewpoint)

    record = BackendPseudoViewRecord(
        sample_id=int(slot.frame_id),
        frame_id=int(slot.frame_id),
        viewpoint=viewpoint,
        target_rgb=np.asarray(pseudo_render_rgb, dtype=np.float32),
        target_depth=np.asarray(result["pseudo_depth_target_exact_brpo_upstream_target_v1"], dtype=np.float32),
        confidence_mask=np.asarray(result["pseudo_confidence_exact_brpo_upstream_target_v1"], dtype=np.float32),
        source_map=np.asarray(result["pseudo_source_map_exact_brpo_upstream_target_v1"], dtype=np.int16),
        valid_mask=np.asarray(result["pseudo_valid_mask_exact_brpo_upstream_target_v1"], dtype=np.float32),
        target_confidence=np.asarray(result["pseudo_target_confidence_exact_brpo_upstream_target_v1"], dtype=np.float32),
        support_both_mask=np.asarray(result["pseudo_verify_both_exact_brpo_upstream_target_v1"], dtype=np.float32),
        stageA_scene_scale=stageA_scene_scale,
        target_rgb_path=str(record_frame_out / "target_rgb_runtime.png"),
        target_depth_path=str(record_frame_out / "target_depth_runtime.npy"),
        confidence_path=str(record_frame_out / "confidence_mask_runtime.npy"),
        observation_meta_path=str(signal_bundle.signal_frame_out / "exact_brpo_upstream_target_observation_meta_v1.json"),
        # Phase 2 fix: pass reference frame IDs
        left_ref_frame_id=int(slot.left_ref_frame_id),
        right_ref_frame_id=int(slot.right_ref_frame_id),
    )
    record_meta = {
        "frame_id": int(slot.frame_id),
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "placement": slot.placement,
        "source": "runtime_signal_builder",
        "stageA_scene_scale": None if stageA_scene_scale is None else float(stageA_scene_scale),
    }
    write_runtime_pseudo_record_frame(
        record_frame_out=record_frame_out,
        target_rgb=record.target_rgb,
        target_depth=record.target_depth,
        confidence_mask=record.confidence_mask,
        source_map=record.source_map,
        valid_mask=record.valid_mask,
        target_confidence=record.target_confidence,
        support_both_mask=record.support_both_mask,
        camera_json=_camera_json_from_state(pseudo_state),
        record_meta=record_meta,
    )
    return RuntimePseudoRecordBundle(slot=slot, record_frame_out=record_frame_out, record=record, record_meta=record_meta)
