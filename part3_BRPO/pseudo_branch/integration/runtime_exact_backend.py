from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from gaussian_splatting.gaussian_renderer import render
from pseudo_branch.common import DEFAULT_MODEL_NAME, build_pair_matcher, get_shared_mast3r_pair_forward
from pseudo_branch.observation.brpo_reprojection_verify import create_viewpoint_from_state, verify_single_branch_exact
from pseudo_branch.mask.rgb_mask_inference import _accumulate_match_maps

from .runtime_debug_export import ensure_dir, save_rgb_png, tensor_chw_to_hwc_numpy, write_json, write_runtime_exact_backend_frame
from .runtime_slot_selector import RuntimePseudoSlot



def run_single_difix_pil(model_bundle, image, ref_image, prompt, height, width):
    """Difix restoration for single branch (PIL input)."""
    if model_bundle is None:
        return image
    if model_bundle["kind"] == "hf_pipeline":
        pipe = model_bundle["obj"]
        # Ensure pipe is on the correct CUDA device (remapped by CUDA_VISIBLE_DEVICES)
        import torch
        import os
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible is not None:
            pipe = pipe.to(torch.device("cuda:0"))  # cuda:0 maps to visible GPU
        out = pipe(
            prompt,
            image=image,
            ref_image=ref_image,
            height=height,
            width=width,
            num_inference_steps=1,
            timesteps=[model_bundle["timestep"]],
            guidance_scale=0.0,
        ).images[0]
    else:
        model = model_bundle["obj"]
        out = model.sample(image=image, ref_image=ref_image, prompt=prompt, height=height, width=width)
    if out.size != image.size:
        out = out.resize(image.size, Image.LANCZOS)
    return out


def run_difix_restoration(
    model_bundle,
    pseudo_rgb: np.ndarray,
    left_ref_rgb: np.ndarray,
    right_ref_rgb: np.ndarray,
    cfg: RuntimeExactBackendConfig,
) -> tuple:
    """Execute bidirectional Difix restoration.
    
    Args:
        model_bundle: Difix model bundle (None if disabled)
        pseudo_rgb: Coarse render RGB (H, W, 3) uint8
        left_ref_rgb: Left reference RGB (H, W, 3) uint8
        right_ref_rgb: Right reference RGB (H, W, 3) uint8
        cfg: Config with prompt, height, width
    
    Returns:
        left_fixed_rgb: Left-branch restored RGB (H, W, 3) uint8
        right_fixed_rgb: Right-branch restored RGB (H, W, 3) uint8
    """
    if model_bundle is None:
        return pseudo_rgb, pseudo_rgb
    
    pseudo_img = Image.fromarray(pseudo_rgb.astype(np.uint8))
    left_ref_img = Image.fromarray(left_ref_rgb.astype(np.uint8))
    right_ref_img = Image.fromarray(right_ref_rgb.astype(np.uint8))
    
    left_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=left_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )
    right_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=right_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )
    
    return np.array(left_fixed), np.array(right_fixed)

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


def _confidence_fusion_weight(result: dict[str, Any]) -> np.ndarray:
    support = np.asarray(result["support_mask"], dtype=np.float32)
    conf = np.asarray(result["confidence_map"], dtype=np.float32)
    out = np.where(support > 0.5, np.maximum(conf, 1e-6), 0.0)
    return out.astype(np.float32)


def _optional_load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def _compute_scale_anchor(
    raw_depth: np.ndarray,
    anchor_depth: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    anchor_depth = np.asarray(anchor_depth, dtype=np.float32)
    anchor_mask = np.asarray(anchor_mask, dtype=np.float32) > 0.5
    valid = anchor_mask & np.isfinite(raw_depth) & np.isfinite(anchor_depth) & (raw_depth > 1e-6) & (anchor_depth > 1e-6)

    scale_factor = 1.0
    if valid.any():
        ratios = anchor_depth[valid] / np.maximum(raw_depth[valid], 1e-6)
        positive = ratios[np.isfinite(ratios) & (ratios > 1e-6)]
        if positive.size:
            scale_factor = float(np.median(positive))

    scaled_depth = np.where(np.isfinite(raw_depth) & (raw_depth > 1e-6), raw_depth * scale_factor, 0.0).astype(np.float32)
    rel_err = None
    if valid.any():
        rel_err_arr = np.abs(scaled_depth[valid] - anchor_depth[valid]) / np.maximum(anchor_depth[valid], 1e-6)
        rel_err = float(np.median(rel_err_arr)) if rel_err_arr.size else None

    return scaled_depth, {
        "scale_factor": float(scale_factor),
        "anchor_count": int(valid.sum()),
        "anchor_ratio": float(valid.mean()),
        "post_scale_anchor_relerr_median": rel_err,
    }


def _build_mast3r_direct_depth_branch(
    *,
    pseudo_gt_rgb_path: str,
    ref_rgb_path: str,
    anchor_depth: np.ndarray,
    anchor_mask: np.ndarray,
    cfg: RuntimeExactBackendConfig,
    pair_forwarder=None,
) -> dict[str, Any]:
    forwarder = pair_forwarder or get_shared_mast3r_pair_forward(
        model_name=str(cfg.matcher_model_name),
        device=str(cfg.matcher_device),
    )
    bundle = forwarder.run_pair(
        img1_path=str(pseudo_gt_rgb_path),
        img2_path=str(ref_rgb_path),
        size=int(anchor_depth.shape[1]),
    )
    pts3d_1 = np.asarray(bundle.pts3d_1, dtype=np.float32)
    conf1 = np.asarray(bundle.conf1, dtype=np.float32)
    raw_depth = np.where(np.isfinite(pts3d_1[..., 2]) & (pts3d_1[..., 2] > 1e-6), pts3d_1[..., 2], 0.0).astype(np.float32)
    scaled_depth, scale_meta = _compute_scale_anchor(raw_depth, anchor_depth, anchor_mask)
    return {
        "raw_depth": raw_depth,
        "scaled_depth": scaled_depth,
        "pseudo_confidence": conf1,
        "pair_meta": dict(bundle.meta),
        "scale_meta": scale_meta,
    }


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

        # Fuse RGB using residual fusion.
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
                "cm_source": "rgb_only",
                "depth_target_source": "exact_backend",
            },
        }
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
            "pseudo_gt_rgb_path": str(pseudo_gt_rgb_path),
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
