from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.gaussian_renderer import render
from pseudo_branch.common import DEFAULT_MODEL_NAME, build_pair_matcher, get_shared_mast3r_pair_forward
from pseudo_branch.observation.brpo_reprojection_verify import create_viewpoint_from_state, verify_single_branch_exact
from online_mapping.mask.rgb_mask_inference import _accumulate_match_maps
from online_mapping.mask.dense_match_densify import build_dense_match_maps
from online_mapping.mask.cm_local_expansion import apply_cm_local_expansion, write_cm_expansion_outputs

from online_mapping.records.runtime_debug_export import ensure_dir, save_rgb_png, tensor_chw_to_hwc_numpy, write_json, write_runtime_exact_backend_frame
from .slot_selector import RuntimePseudoSlot
from .difix_rgb import run_single_difix_pil, run_difix_restoration
from .depth_variants import _optional_load_npy, _compute_scale_anchor, _build_mast3r_direct_depth_branch



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
    depth_generation_mode: str = "projected"
    # RGB source for pseudo target/matching: "render" (default) or "gt" (upper-bound oracle).
    pseudo_rgb_source: str = "render"
    # Difix restoration parameters
    difix_prompt: str = ""
    difix_height: int = 512
    difix_width: int = 512
    # Fusion parameters
    difix_fusion_mode: str = "brpo_overlap_confidence"
    depth_consistency_tau: float = 0.15
    translation_scale_tau: float = 1.0
    # Paper route: RGB-only verification (no depth check in C_m generation)
    rgb_only_verification: bool = False
    rgb_only_support_mode: str = "reciprocal_seed"  # reciprocal_seed | dense_match_v1
    cm_dense_point_radius: int = 2
    cm_dense_blur_sigma: float = 2.0
    cm_dense_blur_kernel: int = 0  # 0 => auto
    cm_dense_corr_threshold: float = 0.15
    cm_dense_seed_mode: str = "binary"  # binary | confidence_weighted
    cm_dense_normalize_mode: str = "max"  # max | p99 | none

    # C_m local expansion parameters (default disabled)
    cm_expansion_mode: str = "none"  # none | local_soft_v1
    cm_expansion_radius: int = 1
    cm_expansion_weight: float = 0.5
    cm_expansion_tau_rgb_l1: float = 0.08
    cm_expansion_tau_depth_rel: float = 0.05
    cm_expansion_min_seed_conf: float = 0.0
    cm_expansion_min_expanded_conf: float = 0.05
    cm_expanded_both_weight: float = 0.6
    cm_raw_exp_agree_weight: float = 0.5
    cm_expanded_single_weight: float = 0.25
    cm_expansion_apply_to_depth_scope: bool = False

    # ABLATION: Disable confidence mask (A2)
    # When True, confidence_mask becomes all ones (no masking effect)
    disable_confidence_mask: bool = False

    # ABLATION: Single-side difix fusion (A3)
    # Valid modes: "brpo_overlap_confidence", "left_only", "right_only"
    # "left_only" / "right_only" bypass fusion, use single reference's difix result directly

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
    direct_depth_left: np.ndarray | None = None
    direct_depth_right: np.ndarray | None = None
    direct_depth_meta: dict[str, Any] | None = None


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


def render_depth_from_state_only(
    *,
    state: dict[str, Any],
    gaussians,
    pipe,
    background,
    device: str = "cuda",
) -> np.ndarray:
    viewpoint = create_viewpoint_from_state(state, device=device)
    render_pkg = render(viewpoint, gaussians, pipe, background)
    return render_pkg["depth"].squeeze().detach().cpu().numpy().astype(np.float32)


def load_rgb_float_from_path(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as pil:
        return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def _confidence_fusion_weight(result: dict[str, Any]) -> np.ndarray:
    support = np.asarray(result["support_mask"], dtype=np.float32)
    conf = np.asarray(result["confidence_map"], dtype=np.float32)
    out = np.where(support > 0.5, np.maximum(conf, 1e-6), 0.0)
    return out.astype(np.float32)


def load_exact_backend_frame_bundle(exact_frame_root: str | Path) -> dict[str, np.ndarray | None]:
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
        "direct_depth_left": _optional_load_npy(root / "direct_depth_left_mast3r_exact_anchor.npy"),
        "direct_depth_right": _optional_load_npy(root / "direct_depth_right_mast3r_exact_anchor.npy"),
        "confidence_cm_override": (
            _optional_load_npy(root / "confidence_cm_local_soft_v1.npy")
            if (root / "confidence_cm_local_soft_v1.npy").exists()
            else _optional_load_npy(root / "cm_expansion_v1" / "cm_expanded_soft.npy")
        ),
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
    difix_model=None,  # Difix model bundle for RGB restoration
) -> RuntimeExactBackendBundle:
    pseudo_state = dict(states_by_id[int(slot.frame_id)])
    left_state = dict(states_by_id[int(slot.left_ref_frame_id)])
    right_state = dict(states_by_id[int(slot.right_ref_frame_id)])
    pseudo_gt_rgb_path = str(pseudo_state.get("image_path", ""))

    frame_root = Path(frame_root)
    inputs_out = ensure_dir(frame_root / "runtime_inputs")
    exact_frame_out = ensure_dir(frame_root / "exact_backend_v1")

    if matcher is None:
        matcher = _build_matcher(cfg)

    pseudo_rgb_source = str(getattr(cfg, "pseudo_rgb_source", "render") or "render").lower()
    if pseudo_rgb_source in {"gt", "gt_rgb", "ground_truth", "image"}:
        if not pseudo_gt_rgb_path:
            raise RuntimeError("pseudo_rgb_source=gt requires pseudo_state.image_path")
        # Upper-bound/oracle pseudo RGB: use the dataset image directly for
        # matching, C_m, and pseudo RGB supervision.  Keep E5's projected depth
        # route by still rendering pseudo/ref depth for geometric verification.
        pseudo_render_rgb = load_rgb_float_from_path(pseudo_gt_rgb_path)
        pseudo_render_depth = render_depth_from_state_only(
            state=pseudo_state,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            device=str(cfg.matcher_device),
        )
    else:
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

    pseudo_rgb_path = inputs_out / ("pseudo_gt_rgb_runtime.png" if pseudo_rgb_source in {"gt", "gt_rgb", "ground_truth", "image"} else "pseudo_render_rgb_runtime.png")
    pseudo_depth_path = inputs_out / "pseudo_render_depth_runtime.npy"
    left_ref_rgb_path = inputs_out / "left_ref_rgb_runtime.png"
    right_ref_rgb_path = inputs_out / "right_ref_rgb_runtime.png"
    save_rgb_png(pseudo_rgb_path, pseudo_render_rgb)
    np.save(pseudo_depth_path, pseudo_render_depth.astype(np.float32))

    with Image.open(Path(left_state["image_path"])) as left_ref_pil:
        left_ref_img = np.asarray(left_ref_pil.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(Path(right_state["image_path"])) as right_ref_pil:
        right_ref_img = np.asarray(right_ref_pil.convert("RGB"), dtype=np.float32) / 255.0
    save_rgb_png(left_ref_rgb_path, left_ref_img)
    save_rgb_png(right_ref_rgb_path, right_ref_img)

    # ==================== Difix Restoration + RGB Fusion ====================
    difix_out = ensure_dir(frame_root / "difix")
    fusion_out = ensure_dir(frame_root / "fusion")

    # Prepare uint8 images for Difix / residual fusion while keeping float RGB
    # for debug export and downstream pseudo supervision.
    left_ref_rgb_uint8 = np.clip(left_ref_img * 255.0, 0.0, 255.0).astype(np.uint8)
    right_ref_rgb_uint8 = np.clip(right_ref_img * 255.0, 0.0, 255.0).astype(np.uint8)
    pseudo_rgb_uint8 = np.clip(pseudo_render_rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    if difix_model is not None:
        # Execute bidirectional Difix restoration.
        left_fixed_rgb, right_fixed_rgb = run_difix_restoration(
            model_bundle=difix_model,
            pseudo_rgb=pseudo_rgb_uint8,
            left_ref_rgb=left_ref_rgb_uint8,
            right_ref_rgb=right_ref_rgb_uint8,
            cfg=cfg,
        )
        save_rgb_png(difix_out / "left_fixed.png", left_fixed_rgb.astype(np.float32) / 255.0)
        save_rgb_png(difix_out / "right_fixed.png", right_fixed_rgb.astype(np.float32) / 255.0)

        # Compute fusion weights using depth-guided overlap confidence.
        from pseudo_branch.observation.pseudo_fusion import (
            compute_overlap_confidence_map,
            normalize_branch_weights,
            fuse_residual_targets,
        )

        left_geom = compute_overlap_confidence_map(
            pseudo_state=pseudo_state,
            ref_state=left_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=left_ref_depth,
            depth_consistency_tau=float(cfg.depth_consistency_tau),
            translation_scale_tau=float(cfg.translation_scale_tau),
        )
        right_geom = compute_overlap_confidence_map(
            pseudo_state=pseudo_state,
            ref_state=right_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=right_ref_depth,
            depth_consistency_tau=float(cfg.depth_consistency_tau),
            translation_scale_tau=float(cfg.translation_scale_tau),
        )

        w_left, w_right, fused_conf = normalize_branch_weights(
            left_geom["overlap_confidence"],
            right_geom["overlap_confidence"],
        )

        # ABLATION: Single-side fusion mode (A3)
        fusion_mode = str(cfg.difix_fusion_mode)
        if fusion_mode == "left_only":
            # Use left reference's difix result only
            fused_rgb_uint8 = left_fixed_rgb
            w_left = np.ones(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
            w_right = np.zeros(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
            fused_conf = np.ones(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
        elif fusion_mode == "right_only":
            # Use right reference's difix result only
            fused_rgb_uint8 = right_fixed_rgb
            w_left = np.zeros(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
            w_right = np.ones(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
            fused_conf = np.ones(pseudo_rgb_uint8.shape[:2], dtype=np.float32)
        else:
            # Default: BRPO overlap confidence fusion
            fused_rgb_uint8 = fuse_residual_targets(
                I_render=pseudo_rgb_uint8,
                I_L=left_fixed_rgb,
                I_R=right_fixed_rgb,
                W_L=w_left,
                W_R=w_right,
            )
        fused_rgb = fused_rgb_uint8.astype(np.float32) / 255.0

        # Save fusion results.
        save_rgb_png(fusion_out / "fused_rgb.png", fused_rgb)
        np.save(fusion_out / "fusion_weight_left.npy", w_left.astype(np.float32))
        np.save(fusion_out / "fusion_weight_right.npy", w_right.astype(np.float32))
        np.save(fusion_out / "confidence_mask_fused.npy", fused_conf.astype(np.float32))

        # Use fused RGB for matching.
        fused_rgb_path = inputs_out / "pseudo_fused_rgb.png"
        save_rgb_png(fused_rgb_path, fused_rgb)
        pseudo_state["image_path"] = str(fused_rgb_path)
    else:
        fused_rgb = pseudo_render_rgb.astype(np.float32)
        w_left = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)
        w_right = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)
        fused_conf = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)
        pseudo_state["image_path"] = str(pseudo_rgb_path)
    
    left_state["image_path"] = str(left_ref_rgb_path)
    right_state["image_path"] = str(right_ref_rgb_path)

    # ==================== MASt3R Matching (using fused RGB if difix enabled) ====================
    pseudo_input_for_match = pseudo_state["image_path"]
    h, w = int(pseudo_state["image_height"]), int(pseudo_state["image_width"])
    write_json(frame_root / "sink_cfg_debug.json", {
        "module_file": __file__,
        "frame_id": int(slot.frame_id),
        "rgb_only_verification": bool(cfg.rgb_only_verification),
        "received_rgb_only_support_mode": str(cfg.rgb_only_support_mode),
        "received_cm_dense_point_radius": int(cfg.cm_dense_point_radius),
        "received_cm_dense_blur_sigma": float(cfg.cm_dense_blur_sigma),
        "received_cm_dense_corr_threshold": float(cfg.cm_dense_corr_threshold),
        "expect_dense_env": os.environ.get("E9_EXPECT_DENSE_MATCH_V1", ""),
    })

    # Get match points and confidence from MASt3R
    pts_pseudo_left, pts_ref_left, match_conf_left = matcher.match_pair(
        str(pseudo_input_for_match), str(left_ref_rgb_path), size=w
    )
    left_match_meta = matcher.get_last_match_meta() if hasattr(matcher, "get_last_match_meta") else {}

    pts_pseudo_right, pts_ref_right, match_conf_right = matcher.match_pair(
        str(pseudo_input_for_match), str(right_ref_rgb_path), size=w
    )
    right_match_meta = matcher.get_last_match_meta() if hasattr(matcher, "get_last_match_meta") else {}

    # ==================== Verification: RGB-only vs Exact ====================
    if cfg.rgb_only_verification:
        # Paper route: RGB-only C_m (confidence_cm) generation
        # - support_mask comes from _accumulate_match_maps (RGB only, no depth check)
        # - projected_depth_map still comes from verify_single_branch_exact (for depth target)
        # This decouples C_m from depth verification while keeping depth target intact

        # Step 1: Get RGB-only mask for C_m
        rgb_only_support_mode = str(cfg.rgb_only_support_mode)
        if os.environ.get("E9_EXPECT_DENSE_MATCH_V1", "") == "1" and rgb_only_support_mode != "dense_match_v1":
            raise RuntimeError(f"E9 sink mismatch: expected dense_match_v1 at runtime_exact_backend, got {rgb_only_support_mode}")
        if rgb_only_support_mode == "dense_match_v1" and str(cfg.cm_expansion_mode) != "none":
            raise ValueError("dense_match_v1 is a standalone RGB-only support mode and must not be combined with cm_expansion_mode")
        dense_match_meta = None
        if rgb_only_support_mode == "reciprocal_seed":
            left_rgb_maps = _accumulate_match_maps(
                image_shape=(h, w),
                pts_fused=pts_pseudo_left,
                conf=match_conf_left,
            )
            right_rgb_maps = _accumulate_match_maps(
                image_shape=(h, w),
                pts_fused=pts_pseudo_right,
                conf=match_conf_right,
            )
        elif rgb_only_support_mode == "dense_match_v1":
            left_rgb_maps = build_dense_match_maps(
                image_shape=(h, w),
                pts_fused=pts_pseudo_left,
                conf=match_conf_left,
                point_radius=int(cfg.cm_dense_point_radius),
                blur_sigma=float(cfg.cm_dense_blur_sigma),
                blur_kernel=(None if int(cfg.cm_dense_blur_kernel) <= 0 else int(cfg.cm_dense_blur_kernel)),
                normalize_mode=str(cfg.cm_dense_normalize_mode),
                corr_threshold=float(cfg.cm_dense_corr_threshold),
                seed_mode=str(cfg.cm_dense_seed_mode),
            )
            right_rgb_maps = build_dense_match_maps(
                image_shape=(h, w),
                pts_fused=pts_pseudo_right,
                conf=match_conf_right,
                point_radius=int(cfg.cm_dense_point_radius),
                blur_sigma=float(cfg.cm_dense_blur_sigma),
                blur_kernel=(None if int(cfg.cm_dense_blur_kernel) <= 0 else int(cfg.cm_dense_blur_kernel)),
                normalize_mode=str(cfg.cm_dense_normalize_mode),
                corr_threshold=float(cfg.cm_dense_corr_threshold),
                seed_mode=str(cfg.cm_dense_seed_mode),
            )
            dense_match_meta = {
                "mode": "dense_match_v1",
                "point_radius": int(cfg.cm_dense_point_radius),
                "blur_sigma": float(cfg.cm_dense_blur_sigma),
                "blur_kernel": (None if int(cfg.cm_dense_blur_kernel) <= 0 else int(cfg.cm_dense_blur_kernel)),
                "corr_threshold": float(cfg.cm_dense_corr_threshold),
                "seed_mode": str(cfg.cm_dense_seed_mode),
                "normalize_mode": str(cfg.cm_dense_normalize_mode),
                "left": left_rgb_maps.get("summary", {}),
                "right": right_rgb_maps.get("summary", {}),
            }
        else:
            raise ValueError(f"Unsupported rgb_only_support_mode: {rgb_only_support_mode}")

        # Step 2: Get projected_depth from exact verification (for depth target)
        left_exact_result = verify_single_branch_exact(
            pseudo_state=pseudo_state,
            ref_state=left_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=left_ref_depth,
            pts_pseudo=pts_pseudo_left,
            pts_ref=pts_ref_left,
            tau_reproj_px=float(cfg.tau_reproj_px),
            tau_rel_depth=float(cfg.tau_rel_depth),
            ref_side="left",
            ref_frame_id=int(slot.left_ref_frame_id),
        )
        right_exact_result = verify_single_branch_exact(
            pseudo_state=pseudo_state,
            ref_state=right_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=right_ref_depth,
            pts_pseudo=pts_pseudo_right,
            pts_ref=pts_ref_right,
            tau_reproj_px=float(cfg.tau_reproj_px),
            tau_rel_depth=float(cfg.tau_rel_depth),
            ref_side="right",
            ref_frame_id=int(slot.right_ref_frame_id),
        )

        # Step 3: Merge: RGB-only support_mask + exact projected_depth
        left_result = {
            "support_mask": left_rgb_maps["support_mask"],  # RGB-only for C_m
            "confidence_map": left_rgb_maps["conf_map"],    # RGB-only confidence
            "projected_depth_map": left_exact_result["projected_depth_map"],  # exact for depth target
            "projected_depth_valid_mask": left_exact_result["projected_depth_valid_mask"],
            "reproj_error_map": left_exact_result["reproj_error_map"],
            "rel_depth_error_map": left_exact_result["rel_depth_error_map"],
            "match_density": left_rgb_maps["match_density"],
            "provenance_map": left_exact_result["provenance_map"],
            "hit_count": left_exact_result["hit_count"],
            "occlusion_reason_map": left_exact_result["occlusion_reason_map"],
            "depth_variance_map": left_exact_result["depth_variance_map"],
            "stats": {
                "num_matches": int(pts_pseudo_left.shape[0]),
                "num_valid_ref_depth": int(left_exact_result["stats"]["num_valid_ref_depth"]),
                "num_valid_pseudo_depth": int(left_exact_result["stats"]["num_valid_pseudo_depth"]),
                "num_support": int(left_rgb_maps["support_mask"].sum()),  # RGB-only count
                "num_projected_depth": int(left_exact_result["stats"]["num_projected_depth"]),
                "support_ratio_vs_matches": float(left_rgb_maps["support_mask"].sum() / max(pts_pseudo_left.shape[0], 1)),
                "support_ratio_vs_image": float(left_rgb_maps["support_mask"].mean()),
                "projected_depth_ratio_vs_image": float(left_exact_result["stats"]["projected_depth_ratio_vs_image"]),
                "mean_reproj_error": left_exact_result["stats"]["mean_reproj_error"],
                "mean_rel_depth_error": left_exact_result["stats"]["mean_rel_depth_error"],
                "tau_reproj_px": float(cfg.tau_reproj_px),
                "tau_rel_depth": float(cfg.tau_rel_depth),
                "ref_side": "left",
                "ref_frame_id": int(slot.left_ref_frame_id),
                "avg_confidence": float(left_rgb_maps["conf_map"][left_rgb_maps["support_mask"] > 0].mean()) if (left_rgb_maps["support_mask"] > 0).any() else 0.0,
                "avg_depth_variance": float(left_exact_result["stats"]["avg_depth_variance"]),
                "multi_hit_pixels": int(left_exact_result["stats"]["multi_hit_pixels"]),
                "occlusion_breakdown": left_exact_result["stats"]["occlusion_breakdown"],
                "rgb_only_verification": True,
                "rgb_only_support_mode": str(cfg.rgb_only_support_mode),
                "cm_source": "rgb_only",
                "depth_target_source": "exact_backend",
            },
        }

        right_result = {
            "support_mask": right_rgb_maps["support_mask"],  # RGB-only for C_m
            "confidence_map": right_rgb_maps["conf_map"],    # RGB-only confidence
            "projected_depth_map": right_exact_result["projected_depth_map"],  # exact for depth target
            "projected_depth_valid_mask": right_exact_result["projected_depth_valid_mask"],
            "reproj_error_map": right_exact_result["reproj_error_map"],
            "rel_depth_error_map": right_exact_result["rel_depth_error_map"],
            "match_density": right_rgb_maps["match_density"],
            "provenance_map": right_exact_result["provenance_map"],
            "hit_count": right_exact_result["hit_count"],
            "occlusion_reason_map": right_exact_result["occlusion_reason_map"],
            "depth_variance_map": right_exact_result["depth_variance_map"],
            "stats": {
                "num_matches": int(pts_pseudo_right.shape[0]),
                "num_valid_ref_depth": int(right_exact_result["stats"]["num_valid_ref_depth"]),
                "num_valid_pseudo_depth": int(right_exact_result["stats"]["num_valid_pseudo_depth"]),
                "num_support": int(right_rgb_maps["support_mask"].sum()),  # RGB-only count
                "num_projected_depth": int(right_exact_result["stats"]["num_projected_depth"]),
                "support_ratio_vs_matches": float(right_rgb_maps["support_mask"].sum() / max(pts_pseudo_right.shape[0], 1)),
                "support_ratio_vs_image": float(right_rgb_maps["support_mask"].mean()),
                "projected_depth_ratio_vs_image": float(right_exact_result["stats"]["projected_depth_ratio_vs_image"]),
                "mean_reproj_error": right_exact_result["stats"]["mean_reproj_error"],
                "mean_rel_depth_error": right_exact_result["stats"]["mean_rel_depth_error"],
                "tau_reproj_px": float(cfg.tau_reproj_px),
                "tau_rel_depth": float(cfg.tau_rel_depth),
                "ref_side": "right",
                "ref_frame_id": int(slot.right_ref_frame_id),
                "avg_confidence": float(right_rgb_maps["conf_map"][right_rgb_maps["support_mask"] > 0].mean()) if (right_rgb_maps["support_mask"] > 0).any() else 0.0,
                "avg_depth_variance": float(right_exact_result["stats"]["avg_depth_variance"]),
                "multi_hit_pixels": int(right_exact_result["stats"]["multi_hit_pixels"]),
                "occlusion_breakdown": right_exact_result["stats"]["occlusion_breakdown"],
                "rgb_only_verification": True,
                "rgb_only_support_mode": str(cfg.rgb_only_support_mode),
                "cm_source": "rgb_only",
                "depth_target_source": "exact_backend",
            },
        }

        if str(cfg.rgb_only_support_mode) == "dense_match_v1":
            dense_match_out = ensure_dir(exact_frame_out / "dense_match_v1")
            dense_match_meta = dense_match_meta or {}
            np.save(exact_frame_out / "support_left_raw_reciprocal.npy", np.asarray(left_rgb_maps["raw_support_mask"], dtype=np.float32))
            np.save(exact_frame_out / "support_right_raw_reciprocal.npy", np.asarray(right_rgb_maps["raw_support_mask"], dtype=np.float32))
            np.save(dense_match_out / "support_left_dense_match_v1.npy", np.asarray(left_rgb_maps["dense_support_mask"], dtype=np.float32))
            np.save(dense_match_out / "support_right_dense_match_v1.npy", np.asarray(right_rgb_maps["dense_support_mask"], dtype=np.float32))
            np.save(dense_match_out / "confidence_left_dense_match_v1.npy", np.asarray(left_rgb_maps["conf_map"], dtype=np.float32))
            np.save(dense_match_out / "confidence_right_dense_match_v1.npy", np.asarray(right_rgb_maps["conf_map"], dtype=np.float32))
            np.save(dense_match_out / "dense_seed_left_v1.npy", np.asarray(left_rgb_maps["dense_seed_mask"], dtype=np.float32))
            np.save(dense_match_out / "dense_seed_right_v1.npy", np.asarray(right_rgb_maps["dense_seed_mask"], dtype=np.float32))
            write_json(dense_match_out / "dense_match_meta.json", dense_match_meta)
            left_result["support_mask_raw_reciprocal"] = np.asarray(left_rgb_maps["raw_support_mask"], dtype=np.float32)
            right_result["support_mask_raw_reciprocal"] = np.asarray(right_rgb_maps["raw_support_mask"], dtype=np.float32)
            left_result["confidence_map_raw_reciprocal"] = np.asarray(left_rgb_maps["raw_conf_map"], dtype=np.float32)
            right_result["confidence_map_raw_reciprocal"] = np.asarray(right_rgb_maps["raw_conf_map"], dtype=np.float32)
            left_result["dense_match_soft_map"] = np.asarray(left_rgb_maps["dense_soft_map"], dtype=np.float32)
            right_result["dense_match_soft_map"] = np.asarray(right_rgb_maps["dense_soft_map"], dtype=np.float32)
            left_result["dense_match_seed_mask"] = np.asarray(left_rgb_maps["dense_seed_mask"], dtype=np.float32)
            right_result["dense_match_seed_mask"] = np.asarray(right_rgb_maps["dense_seed_mask"], dtype=np.float32)
            left_result["stats"]["raw_support_ratio_vs_image"] = float(np.asarray(left_rgb_maps["raw_support_mask"], dtype=np.float32).mean())
            right_result["stats"]["raw_support_ratio_vs_image"] = float(np.asarray(right_rgb_maps["raw_support_mask"], dtype=np.float32).mean())
            left_result["stats"]["dense_match_summary"] = left_rgb_maps.get("summary", {})
            right_result["stats"]["dense_match_summary"] = right_rgb_maps.get("summary", {})
    else:
        # Exact route: RGB + depth verification (original behavior)
        left_result = verify_single_branch_exact(
            pseudo_state=pseudo_state,
            ref_state=left_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=left_ref_depth,
            pts_pseudo=pts_pseudo_left,
            pts_ref=pts_ref_left,
            tau_reproj_px=float(cfg.tau_reproj_px),
            tau_rel_depth=float(cfg.tau_rel_depth),
            ref_side="left",
            ref_frame_id=int(slot.left_ref_frame_id),
        )

        right_result = verify_single_branch_exact(
            pseudo_state=pseudo_state,
            ref_state=right_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=right_ref_depth,
            pts_pseudo=pts_pseudo_right,
            pts_ref=pts_ref_right,
            tau_reproj_px=float(cfg.tau_reproj_px),
            tau_rel_depth=float(cfg.tau_rel_depth),
            ref_side="right",
            ref_frame_id=int(slot.right_ref_frame_id),
        )

    fusion_weight_left = _confidence_fusion_weight(left_result)
    fusion_weight_right = _confidence_fusion_weight(right_result)

    if not cfg.rgb_only_verification:
        dense_match_meta = None

    # ==================== C_m Local Expansion ====================
    cm_expansion_result = None
    cm_expansion_meta = None
    if str(cfg.cm_expansion_mode) == "local_soft_v1":
        # Load pseudo RGB for expansion
        pseudo_rgb_for_exp = fused_rgb if difix_model is not None else pseudo_render_rgb.astype(np.float32)
        
        # Apply expansion
        cm_expansion_result = apply_cm_local_expansion(
            raw_support_left=np.asarray(left_result["support_mask"], dtype=np.float32),
            raw_support_right=np.asarray(right_result["support_mask"], dtype=np.float32),
            confidence_left=np.asarray(left_result["confidence_map"], dtype=np.float32),
            confidence_right=np.asarray(right_result["confidence_map"], dtype=np.float32),
            pseudo_rgb=pseudo_rgb_for_exp,
            pseudo_depth=pseudo_render_depth,
            radius=int(cfg.cm_expansion_radius),
            expansion_weight=float(cfg.cm_expansion_weight),
            tau_rgb_l1=float(cfg.cm_expansion_tau_rgb_l1),
            tau_depth_rel=float(cfg.cm_expansion_tau_depth_rel),
            min_seed_conf=float(cfg.cm_expansion_min_seed_conf),
            min_expanded_conf=float(cfg.cm_expansion_min_expanded_conf),
            raw_both_weight=1.0,
            raw_single_weight=0.5,
            expanded_both_weight=float(cfg.cm_expanded_both_weight),
            raw_exp_agree_weight=float(cfg.cm_raw_exp_agree_weight),
            expanded_single_weight=float(cfg.cm_expanded_single_weight),
        )
        
        # Write expansion side products.  Never overwrite raw reciprocal support by default.
        cm_exp_out = ensure_dir(exact_frame_out / "cm_expansion_v1")
        confidence_cm_local_soft = cm_expansion_result["cm_composition"]["confidence_cm"].astype(np.float32)

        # ABLATION: Disable confidence mask (A2)
        if bool(cfg.disable_confidence_mask):
            h, w = confidence_cm_local_soft.shape
            confidence_cm_local_soft = np.ones((h, w), dtype=np.float32)

        cm_expansion_meta = {
            "frame_id": int(slot.frame_id),
            "expansion_mode": "local_soft_v1",
            "apply_to_depth_scope": bool(cfg.cm_expansion_apply_to_depth_scope),
            "confidence_cm_override_path": str(exact_frame_out / "confidence_cm_local_soft_v1.npy"),
            "sidecar_dir": str(cm_exp_out),
            "summary": cm_expansion_result["summary"],
        }
        write_cm_expansion_outputs(cm_exp_out, cm_expansion_result, cm_expansion_meta)
        np.save(exact_frame_out / "confidence_cm_local_soft_v1.npy", confidence_cm_local_soft)
        write_json(exact_frame_out / "cm_expansion_meta.json", cm_expansion_meta)
        
        # Save raw support for provenance and expose expanded support separately.
        left_raw_support = np.asarray(left_result["support_mask"], dtype=np.float32)
        right_raw_support = np.asarray(right_result["support_mask"], dtype=np.float32)
        np.save(exact_frame_out / "support_left_raw_reciprocal.npy", left_raw_support)
        np.save(exact_frame_out / "support_right_raw_reciprocal.npy", right_raw_support)
        left_result["support_mask_raw_reciprocal"] = left_raw_support
        right_result["support_mask_raw_reciprocal"] = right_raw_support
        left_result["support_mask_expanded"] = cm_expansion_result["cm_composition"]["support_left_expanded"].astype(np.float32)
        right_result["support_mask_expanded"] = cm_expansion_result["cm_composition"]["support_right_expanded"].astype(np.float32)
        
        # Only widen the support/depth target scope for an explicitly named arm.
        # E8/default local_soft_v1 should change soft C_m/RGB weights, not depth scope.
        if bool(cfg.cm_expansion_apply_to_depth_scope):
            left_result["support_mask"] = left_result["support_mask_expanded"]
            right_result["support_mask"] = right_result["support_mask_expanded"]
        
        # Add confidence_cm_override for signal builder; this is the intended runtime effect.
        left_result["confidence_cm_override"] = confidence_cm_local_soft
        right_result["confidence_cm_override"] = confidence_cm_local_soft

    # ABLATION: Disable confidence mask when cm_expansion is off (A2 fallback)
    # When disable_confidence_mask=True but cm_expansion_mode="none",
    # we need to inject all-ones override for RGB/depth loss weighting.
    # Also save the file so load_exact_backend_frame_bundle can pick it up.
    if bool(cfg.disable_confidence_mask) and str(cfg.cm_expansion_mode) == "none":
        h = int(pseudo_state["image_height"])
        w = int(pseudo_state["image_width"])
        all_ones = np.ones((h, w), dtype=np.float32)
        left_result["confidence_cm_override"] = all_ones
        right_result["confidence_cm_override"] = all_ones
        # Save to expected location for downstream signal builder
        np.save(exact_frame_out / "confidence_cm_local_soft_v1.npy", all_ones)
        write_json(exact_frame_out / "ablation_no_mask_meta.json", {
            "frame_id": int(slot.frame_id),
            "disable_confidence_mask": True,
            "cm_expansion_mode": "none",
            "override_source": "ablation_all_ones",
        })


    direct_depth_left = None
    direct_depth_right = None
    direct_depth_meta: dict[str, Any] | None = None
    if str(cfg.depth_generation_mode) == "mast3r_direct_exact_anchor_v1":
        if not pseudo_gt_rgb_path:
            raise RuntimeError("depth_generation_mode=mast3r_direct_exact_anchor_v1 requires pseudo_state.image_path (GT RGB) to be present")
        pair_forwarder = get_shared_mast3r_pair_forward(
            model_name=str(cfg.matcher_model_name),
            device=str(cfg.matcher_device),
        )
        left_direct = _build_mast3r_direct_depth_branch(
            pseudo_gt_rgb_path=str(pseudo_gt_rgb_path),
            ref_rgb_path=str(left_ref_rgb_path),
            anchor_depth=np.asarray(left_result["projected_depth_map"], dtype=np.float32),
            anchor_mask=np.asarray(left_result["support_mask"], dtype=np.float32),
            cfg=cfg,
            pair_forwarder=pair_forwarder,
        )
        right_direct = _build_mast3r_direct_depth_branch(
            pseudo_gt_rgb_path=str(pseudo_gt_rgb_path),
            ref_rgb_path=str(right_ref_rgb_path),
            anchor_depth=np.asarray(right_result["projected_depth_map"], dtype=np.float32),
            anchor_mask=np.asarray(right_result["support_mask"], dtype=np.float32),
            cfg=cfg,
            pair_forwarder=pair_forwarder,
        )
        direct_depth_left = np.asarray(left_direct["scaled_depth"], dtype=np.float32)
        direct_depth_right = np.asarray(right_direct["scaled_depth"], dtype=np.float32)
        direct_depth_meta = {
            "mode": "mast3r_direct_exact_anchor_v1",
            "pseudo_gt_rgb_path": str(pseudo_gt_rgb_path),
            "left": {
                "scale": left_direct["scale_meta"],
                "pair_meta": left_direct["pair_meta"],
            },
            "right": {
                "scale": right_direct["scale_meta"],
                "pair_meta": right_direct["pair_meta"],
            },
        }
        np.save(exact_frame_out / "direct_depth_left_mast3r_raw.npy", left_direct["raw_depth"].astype(np.float32))
        np.save(exact_frame_out / "direct_depth_right_mast3r_raw.npy", right_direct["raw_depth"].astype(np.float32))
        np.save(exact_frame_out / "direct_depth_left_mast3r_exact_anchor.npy", direct_depth_left)
        np.save(exact_frame_out / "direct_depth_right_mast3r_exact_anchor.npy", direct_depth_right)
        np.save(exact_frame_out / "direct_depth_left_mast3r_confidence.npy", left_direct["pseudo_confidence"].astype(np.float32))
        np.save(exact_frame_out / "direct_depth_right_mast3r_confidence.npy", right_direct["pseudo_confidence"].astype(np.float32))
        write_json(exact_frame_out / "direct_depth_mast3r_exact_anchor_meta.json", direct_depth_meta)

    exact_meta = {
        "frame_id": int(slot.frame_id),
        "image_name": pseudo_state.get("image_name", f"{int(slot.frame_id):05d}.png"),
        "verifier_backend_semantics": str(cfg.verifier_backend_semantics),
        "target_proxy_semantics": "runtime_gt_rgb_projected_depth_exact" if pseudo_rgb_source in {"gt", "gt_rgb", "ground_truth", "image"} else "runtime_render_placeholder_exact",
        "pseudo_rgb_source": str(pseudo_rgb_source),
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
            "pseudo_gt_rgb_path": str(pseudo_gt_rgb_path),
            "pseudo_rgb_source": str(pseudo_rgb_source),
        },
        "depth_generation": {
            "mode": str(cfg.depth_generation_mode),
            "uses_direct_target_depth": direct_depth_left is not None and direct_depth_right is not None,
            "target_depth_scope": "replace_target_depth_only_keep_exact_cm_valid_confidence" if direct_depth_left is not None else "projected_depth_exact_upstream",
            "direct_depth_meta_path": str(exact_frame_out / "direct_depth_mast3r_exact_anchor_meta.json") if direct_depth_meta is not None else None,
        },
        "policy": {
            "multi_hit_resolve": "best_confidence",
            "occlusion_handling": "explicit_invalid",
            "provenance_tracking": True,
            "confidence_continuous": True,
            "fusion_weight_source": "exact_confidence_fallback_v1",
        },
        # Difix restoration info
        "difix_enabled": difix_model is not None,
        "difix_fusion_mode": str(cfg.difix_fusion_mode) if difix_model is not None else "none",
        "pseudo_input_for_match": str(pseudo_input_for_match),
        # Paper route indicator
        "rgb_only_verification": bool(cfg.rgb_only_verification),
        "cm_generation_mode": "rgb_only" if cfg.rgb_only_verification else "exact_backend",
        "rgb_only_support_mode": str(cfg.rgb_only_support_mode) if cfg.rgb_only_verification else None,
        "dense_match_meta": dense_match_meta,
        # C_m expansion info
        "cm_expansion_mode": str(cfg.cm_expansion_mode),
        "cm_expansion_enabled": str(cfg.cm_expansion_mode) != "none",
        "cm_expansion_meta": cm_expansion_meta,
    }
    write_runtime_exact_backend_frame(
        exact_frame_out=exact_frame_out,
        left_result=left_result,
        right_result=right_result,
        exact_meta=exact_meta,
        left_ref_depth=left_ref_depth,
        right_ref_depth=right_ref_depth,
    )

    # Use float RGB in [0, 1] for downstream pseudo supervision.
    final_pseudo_rgb = fused_rgb if difix_model is not None else pseudo_render_rgb.astype(np.float32)

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
        pseudo_render_rgb=final_pseudo_rgb,  # Fused RGB if difix enabled
        pseudo_render_depth=pseudo_render_depth,
        left_result=left_result,
        right_result=right_result,
        left_ref_depth=left_ref_depth,
        right_ref_depth=right_ref_depth,
        exact_meta=exact_meta,
        fusion_weight_left=fusion_weight_left,
        fusion_weight_right=fusion_weight_right,
        direct_depth_left=direct_depth_left,
        direct_depth_right=direct_depth_right,
        direct_depth_meta=direct_depth_meta,
    )
