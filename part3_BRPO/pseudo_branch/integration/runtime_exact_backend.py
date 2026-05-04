from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.gaussian_renderer import render
from pseudo_branch.common import DEFAULT_MODEL_NAME, build_pair_matcher
from pseudo_branch.observation.brpo_reprojection_verify import create_viewpoint_from_state, verify_single_branch_exact

from .runtime_debug_export import ensure_dir, save_rgb_png, tensor_chw_to_hwc_numpy, write_runtime_exact_backend_frame
from .runtime_slot_selector import RuntimePseudoSlot


@dataclass(frozen=True)
class RuntimeExactBackendConfig:
    matcher_mode: str = "sparse_desc_2d"
    matcher_model_name: str = DEFAULT_MODEL_NAME
    matcher_device: str = "cuda"
    dense3d_conf_quantile: float = 0.90
    tau_reproj_px: float = 4.0
    tau_rel_depth: float = 0.15
    verification_mode: str = "branch_first"
    verifier_backend_semantics: str = "exact_branch_native_v1"


@dataclass
class RuntimeExactBackendBundle:
    slot: RuntimePseudoSlot
    exact_frame_out: Path
    pseudo_state: dict[str, Any]
    left_state: dict[str, Any]
    right_state: dict[str, Any]
    pseudo_rgb_path: Path
    pseudo_depth_path: Path
    left_ref_rgb_path: Path
    right_ref_rgb_path: Path
    pseudo_render_rgb: np.ndarray
    pseudo_render_depth: np.ndarray
    left_result: dict[str, Any]
    right_result: dict[str, Any]
    left_ref_depth: np.ndarray
    right_ref_depth: np.ndarray
    exact_meta: dict[str, Any]
    fusion_weight_left: np.ndarray
    fusion_weight_right: np.ndarray


def _build_matcher(cfg: RuntimeExactBackendConfig):
    return build_pair_matcher(
        matcher_mode=cfg.matcher_mode,
        model_name=cfg.matcher_model_name,
        device=cfg.matcher_device,
        dense3d_conf_quantile=float(cfg.dense3d_conf_quantile),
    )


def _pose_c2w_from_state_like(state: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(state["pose_c2w"], dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"pose_c2w must be 4x4, got {pose.shape}")
    return pose


def render_rgb_depth_from_state(
    *,
    state: dict[str, Any],
    gaussians,
    pipe,
    background,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    viewpoint = create_viewpoint_from_state(state, device=device)
    render_pkg = render(viewpoint, gaussians, pipe, background)
    rgb = tensor_chw_to_hwc_numpy(render_pkg["render"])
    depth = render_pkg["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    return rgb, depth


def _confidence_fusion_weight(result: dict[str, Any]) -> np.ndarray:
    support = np.asarray(result["support_mask"], dtype=np.float32)
    conf = np.asarray(result["confidence_map"], dtype=np.float32)
    out = np.where(support > 0.5, np.maximum(conf, 1e-6), 0.0)
    return out.astype(np.float32)


def load_exact_backend_frame_bundle(exact_frame_root: str | Path) -> dict[str, np.ndarray]:
    root = Path(exact_frame_root)
    return {
        "support_left_exact": np.load(root / "support_left_exact.npy").astype(np.float32),
        "support_right_exact": np.load(root / "support_right_exact.npy").astype(np.float32),
        "projected_depth_left_exact": np.load(root / "projected_depth_left_exact.npy").astype(np.float32),
        "projected_depth_right_exact": np.load(root / "projected_depth_right_exact.npy").astype(np.float32),
        "confidence_left_exact": np.load(root / "confidence_left_exact.npy").astype(np.float32),
        "confidence_right_exact": np.load(root / "confidence_right_exact.npy").astype(np.float32),
        "provenance_left": np.load(root / "provenance_left.npy").astype(np.float32) if (root / "provenance_left.npy").exists() else None,
        "provenance_right": np.load(root / "provenance_right.npy").astype(np.float32) if (root / "provenance_right.npy").exists() else None,
    }


def build_runtime_exact_backend_bundle(
    *,
    slot: RuntimePseudoSlot,
    states_by_id: dict[int, dict[str, Any]],
    gaussians,
    pipe,
    background,
    frame_root: str | Path,
    cfg: RuntimeExactBackendConfig,
    matcher=None,
) -> RuntimeExactBackendBundle:
    pseudo_state = dict(states_by_id[int(slot.frame_id)])
    left_state = dict(states_by_id[int(slot.left_ref_frame_id)])
    right_state = dict(states_by_id[int(slot.right_ref_frame_id)])

    frame_root = Path(frame_root)
    inputs_out = ensure_dir(frame_root / "runtime_inputs")
    exact_frame_out = ensure_dir(frame_root / "exact_backend_v1")

    if matcher is None:
        matcher = _build_matcher(cfg)

    pseudo_render_rgb, pseudo_render_depth = render_rgb_depth_from_state(
        state=pseudo_state,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        device=str(cfg.matcher_device),
    )
    left_ref_depth = render_rgb_depth_from_state(
        state=left_state,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        device=str(cfg.matcher_device),
    )[1]
    right_ref_depth = render_rgb_depth_from_state(
        state=right_state,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        device=str(cfg.matcher_device),
    )[1]

    pseudo_rgb_path = inputs_out / "pseudo_render_rgb_runtime.png"
    pseudo_depth_path = inputs_out / "pseudo_render_depth_runtime.npy"
    left_ref_rgb_path = inputs_out / "left_ref_rgb_runtime.png"
    right_ref_rgb_path = inputs_out / "right_ref_rgb_runtime.png"
    save_rgb_png(pseudo_rgb_path, pseudo_render_rgb)
    np.save(pseudo_depth_path, pseudo_render_depth.astype(np.float32))

    left_ref_img = np.asarray(Image.open(Path(left_state["image_path"])).convert("RGB"), dtype=np.float32) / 255.0
    right_ref_img = np.asarray(Image.open(Path(right_state["image_path"])).convert("RGB"), dtype=np.float32) / 255.0
    save_rgb_png(left_ref_rgb_path, left_ref_img)
    save_rgb_png(right_ref_rgb_path, right_ref_img)

    pseudo_state["image_path"] = str(pseudo_rgb_path)
    left_state["image_path"] = str(left_ref_rgb_path)
    right_state["image_path"] = str(right_ref_rgb_path)

    pts_pseudo, pts_ref_left, _ = matcher.match_pair(str(pseudo_rgb_path), str(left_ref_rgb_path), size=int(pseudo_state["image_width"]))
    left_match_meta = matcher.get_last_match_meta() if hasattr(matcher, "get_last_match_meta") else {}
    left_result = verify_single_branch_exact(
        pseudo_state=pseudo_state,
        ref_state=left_state,
        pseudo_depth=pseudo_render_depth,
        ref_depth=left_ref_depth,
        pts_pseudo=pts_pseudo,
        pts_ref=pts_ref_left,
        tau_reproj_px=float(cfg.tau_reproj_px),
        tau_rel_depth=float(cfg.tau_rel_depth),
        ref_side="left",
        ref_frame_id=int(slot.left_ref_frame_id),
    )

    pts_pseudo, pts_ref_right, _ = matcher.match_pair(str(pseudo_rgb_path), str(right_ref_rgb_path), size=int(pseudo_state["image_width"]))
    right_match_meta = matcher.get_last_match_meta() if hasattr(matcher, "get_last_match_meta") else {}
    right_result = verify_single_branch_exact(
        pseudo_state=pseudo_state,
        ref_state=right_state,
        pseudo_depth=pseudo_render_depth,
        ref_depth=right_ref_depth,
        pts_pseudo=pts_pseudo,
        pts_ref=pts_ref_right,
        tau_reproj_px=float(cfg.tau_reproj_px),
        tau_rel_depth=float(cfg.tau_rel_depth),
        ref_side="right",
        ref_frame_id=int(slot.right_ref_frame_id),
    )

    fusion_weight_left = _confidence_fusion_weight(left_result)
    fusion_weight_right = _confidence_fusion_weight(right_result)

    exact_meta = {
        "frame_id": int(slot.frame_id),
        "image_name": pseudo_state.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "verifier_backend_semantics": str(cfg.verifier_backend_semantics),
        "target_proxy_semantics": "runtime_render_placeholder_exact",
        "verification_mode": str(cfg.verification_mode),
        "matcher": {
            "matcher_mode": str(cfg.matcher_mode),
            "matcher_model_name": str(cfg.matcher_model_name),
            "matcher_device": str(cfg.matcher_device),
            "dense3d_conf_quantile": float(cfg.dense3d_conf_quantile),
            "type": type(matcher).__name__,
            "left_match_meta": left_match_meta,
            "right_match_meta": right_match_meta,
        },
        "left_ref_frame_id": int(slot.left_ref_frame_id),
        "right_ref_frame_id": int(slot.right_ref_frame_id),
        "tau_reproj_px": float(cfg.tau_reproj_px),
        "tau_rel_depth": float(cfg.tau_rel_depth),
        "backend_version": "runtime-exact-v1",
        "left_stats": left_result["stats"],
        "right_stats": right_result["stats"],
        "runtime_inputs": {
            "pseudo_rgb_path": str(pseudo_rgb_path),
            "pseudo_depth_path": str(pseudo_depth_path),
            "left_ref_rgb_path": str(left_ref_rgb_path),
            "right_ref_rgb_path": str(right_ref_rgb_path),
        },
        "policy": {
            "multi_hit_resolve": "best_confidence",
            "occlusion_handling": "explicit_invalid",
            "provenance_tracking": True,
            "confidence_continuous": True,
            "fusion_weight_source": "exact_confidence_fallback_v1",
        },
    }
    write_runtime_exact_backend_frame(
        exact_frame_out=exact_frame_out,
        left_result=left_result,
        right_result=right_result,
        exact_meta=exact_meta,
        left_ref_depth=left_ref_depth,
        right_ref_depth=right_ref_depth,
    )

    return RuntimeExactBackendBundle(
        slot=slot,
        exact_frame_out=exact_frame_out,
        pseudo_state=pseudo_state,
        left_state=left_state,
        right_state=right_state,
        pseudo_rgb_path=pseudo_rgb_path,
        pseudo_depth_path=pseudo_depth_path,
        left_ref_rgb_path=left_ref_rgb_path,
        right_ref_rgb_path=right_ref_rgb_path,
        pseudo_render_rgb=pseudo_render_rgb,
        pseudo_render_depth=pseudo_render_depth,
        left_result=left_result,
        right_result=right_result,
        left_ref_depth=left_ref_depth,
        right_ref_depth=right_ref_depth,
        exact_meta=exact_meta,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
    )
