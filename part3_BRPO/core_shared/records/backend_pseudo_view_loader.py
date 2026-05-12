from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.camera_utils import Camera

from core_shared.records.backend_pseudo_bundle import LoadedPseudoBundleSample, PseudoBundleBatch, PseudoBundleSample
from core_shared.pose.pseudo_camera_state import make_viewpoint_trainable


@dataclass
class BackendPseudoViewRecord:
    sample_id: int
    frame_id: int
    viewpoint: Any
    target_rgb: np.ndarray
    target_depth: np.ndarray
    confidence_mask: np.ndarray
    source_map: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    target_confidence: Optional[np.ndarray]
    support_both_mask: Optional[np.ndarray]
    stageA_scene_scale: Optional[float] = None
    target_rgb_path: Optional[str] = None
    target_depth_path: Optional[str] = None
    confidence_path: Optional[str] = None
    observation_meta_path: Optional[str] = None
    # Phase 2 fix: reference frame IDs for pose propagation
    left_ref_frame_id: Optional[int] = None
    right_ref_frame_id: Optional[int] = None

    def to_loss_inputs(self) -> dict[str, Any]:
        return {
            "target_rgb": self.target_rgb,
            "target_depth": self.target_depth,
            "confidence_mask": self.confidence_mask,
            "source_map": self.source_map,
            "valid_mask": self.valid_mask,
            "target_confidence": self.target_confidence,
            "support_both_mask": self.support_both_mask,
            "viewpoint": self.viewpoint,
            "scene_scale": self.stageA_scene_scale,
        }



def build_record_from_loaded_sample(
    sample: PseudoBundleSample,
    loaded: LoadedPseudoBundleSample,
    viewpoint: Any,
) -> BackendPseudoViewRecord:
    make_viewpoint_trainable(viewpoint)
    return BackendPseudoViewRecord(
        sample_id=int(sample.sample_id),
        frame_id=int(sample.frame_id),
        viewpoint=viewpoint,
        target_rgb=loaded.target_rgb,
        target_depth=loaded.target_depth,
        confidence_mask=loaded.confidence_mask,
        source_map=loaded.source_map,
        valid_mask=loaded.valid_mask,
        target_confidence=loaded.target_confidence,
        support_both_mask=loaded.support_both_mask,
        stageA_scene_scale=sample.stageA_scene_scale,
        target_rgb_path=str(sample.target_rgb_path),
        target_depth_path=str(sample.target_depth_path),
        confidence_path=str(sample.confidence_path),
        observation_meta_path=str(sample.observation_meta_path) if sample.observation_meta_path is not None else None,
    )



def _build_camera_from_sample(sample: PseudoBundleSample, loaded: LoadedPseudoBundleSample) -> Any:
    sample_dir = sample.target_rgb_path.parent
    camera_json = sample_dir / "camera.json"
    if not camera_json.exists():
        raise FileNotFoundError(f"Missing camera.json for sample_id={sample.sample_id}: {camera_json}")
    camera_data = json.loads(camera_json.read_text())
    intr = camera_data.get("intrinsics_px", {})
    image_size = camera_data.get("image_size", {})
    width = int(image_size.get("width", loaded.target_rgb.shape[1]))
    height = int(image_size.get("height", loaded.target_rgb.shape[0]))
    fx = float(intr.get("fx"))
    fy = float(intr.get("fy", fx))
    cx = float(intr.get("cx", width / 2.0))
    cy = float(intr.get("cy", height / 2.0))
    fovx = 2.0 * np.arctan(width / (2.0 * fx))
    fovy = 2.0 * np.arctan(height / (2.0 * fy))
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=width,
        H=height,
    ).transpose(0, 1)
    pose_w2c = np.linalg.inv(np.asarray(camera_data["pose_c2w"], dtype=np.float32))
    color_t = torch.from_numpy(loaded.target_rgb).to(dtype=torch.float32, device="cuda").permute(2, 0, 1)
    viewpoint = Camera(
        uid=int(sample.frame_id),
        color=color_t,
        depth=None,
        mono_depth=loaded.target_depth,
        gt_T=torch.from_numpy(pose_w2c).to(dtype=torch.float32, device="cuda"),
        projection_matrix=projection_matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fovx=float(fovx),
        fovy=float(fovy),
        image_height=height,
        image_width=width,
        device="cuda",
    )
    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
    viewpoint.R0 = viewpoint.R.detach().clone()
    viewpoint.T0 = viewpoint.T.detach().clone()
    make_viewpoint_trainable(viewpoint)
    if sample.view_state is not None:
        loaded_w2c = torch.tensor(sample.view_state["pose_w2c"], device="cuda", dtype=torch.float32)
        viewpoint.update_RT(loaded_w2c[:3, :3], loaded_w2c[:3, 3])
        viewpoint.R0 = viewpoint.R.detach().clone()
        viewpoint.T0 = viewpoint.T.detach().clone()
        with torch.no_grad():
            viewpoint.cam_rot_delta.zero_()
            viewpoint.cam_trans_delta.zero_()
            viewpoint.exposure_a.copy_(torch.tensor([float(sample.view_state.get("exposure_a", 0.0))], device="cuda", dtype=torch.float32))
            viewpoint.exposure_b.copy_(torch.tensor([float(sample.view_state.get("exposure_b", 0.0))], device="cuda", dtype=torch.float32))
    else:
        viewpoint.R0 = viewpoint.R.detach().clone()
        viewpoint.T0 = viewpoint.T.detach().clone()
    return viewpoint



def build_records_from_pseudo_bundle(batch: PseudoBundleBatch) -> list[BackendPseudoViewRecord]:
    records: list[BackendPseudoViewRecord] = []
    for sample in batch.samples:
        loaded = sample.load()
        viewpoint = _build_camera_from_sample(sample, loaded)
        records.append(build_record_from_loaded_sample(sample, loaded, viewpoint))
    return records



def normalize_stageA_pseudo_views(
    pseudo_views: list[dict[str, Any]],
    *,
    require_exact_upstream: bool = False,
) -> list[BackendPseudoViewRecord]:
    records: list[BackendPseudoViewRecord] = []
    for view in pseudo_views:
        effective_mode = view.get("pseudo_observation_mode_effective")
        if require_exact_upstream and effective_mode != "exact_brpo_upstream_target_v1":
            raise ValueError(
                f"sample_id={view.get('sample_id')} expected exact_brpo_upstream_target_v1, got {effective_mode}"
            )
        vp = view["vp"]
        make_viewpoint_trainable(vp)
        exact_bundle = view.get("exact_upstream_bundle") or {}
        records.append(
            BackendPseudoViewRecord(
                sample_id=int(view["sample_id"]),
                frame_id=int(view.get("frame_id", view["sample_id"])),
                viewpoint=vp,
                target_rgb=view["rgb"],
                target_depth=view["depth_for_refine"],
                confidence_mask=view["conf"],
                source_map=view.get("target_depth_source_map"),
                valid_mask=exact_bundle.get("valid_mask"),
                target_confidence=exact_bundle.get("target_confidence"),
                support_both_mask=exact_bundle.get("support_both_mask"),
                stageA_scene_scale=view.get("stageA_scene_scale"),
                target_rgb_path=view.get("target_rgb_path"),
                target_depth_path=view.get("target_depth_for_refine_path"),
                confidence_path=view.get("confidence_path"),
                observation_meta_path=view.get("pseudo_observation_meta_path"),
            )
        )
    return records
