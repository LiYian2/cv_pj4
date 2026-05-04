from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .backend_pseudo_view_loader import BackendPseudoViewRecord
from .pseudo_loss_v2 import build_stageA_loss_exact_shared_cm, build_stageA_loss_paper_brpo_split


@dataclass
class BackendPseudoLossConfig:
    beta_rgb: float = 1.0
    lambda_pose: float = 0.0
    lambda_exp: float = 0.0
    trans_weight: float = 1.0
    lambda_depth: float = 1.0
    use_depth: bool = True
    lambda_abs_pose: float = 0.0
    lambda_abs_t: float = 0.0
    lambda_abs_r: float = 0.0
    abs_pose_robust: str = "charbonnier"
    depth_loss_mode: str = "exact_shared_cm_v1"



def compute_backend_pseudo_exact_loss(
    *,
    render_rgb: torch.Tensor,
    render_depth: torch.Tensor,
    record: BackendPseudoViewRecord,
    cfg: BackendPseudoLossConfig,
    viewpoint: Any | None = None,
    return_terms: bool = False,
):
    vp = record.viewpoint if viewpoint is None else viewpoint
    scene_scale = 1.0 if record.stageA_scene_scale is None else float(record.stageA_scene_scale)
    if cfg.depth_loss_mode == "paper_brpo_split_v1":
        return build_stageA_loss_paper_brpo_split(
            render_rgb=render_rgb,
            render_depth=render_depth,
            target_rgb=record.target_rgb,
            target_depth=record.target_depth,
            confidence_mask=record.confidence_mask,
            viewpoint=vp,
            beta_rgb=cfg.beta_rgb,
            lambda_pose=cfg.lambda_pose,
            lambda_exp=cfg.lambda_exp,
            trans_weight=cfg.trans_weight,
            lambda_depth=cfg.lambda_depth,
            use_depth=cfg.use_depth,
            lambda_abs_pose=cfg.lambda_abs_pose,
            lambda_abs_t=cfg.lambda_abs_t,
            lambda_abs_r=cfg.lambda_abs_r,
            abs_pose_robust=cfg.abs_pose_robust,
            scene_scale=scene_scale,
            return_terms=return_terms,
            depth_confidence=None,
        )
    if cfg.depth_loss_mode == "paper_brpo_split_depthconf_v1":
        return build_stageA_loss_paper_brpo_split(
            render_rgb=render_rgb,
            render_depth=render_depth,
            target_rgb=record.target_rgb,
            target_depth=record.target_depth,
            confidence_mask=record.confidence_mask,
            viewpoint=vp,
            beta_rgb=cfg.beta_rgb,
            lambda_pose=cfg.lambda_pose,
            lambda_exp=cfg.lambda_exp,
            trans_weight=cfg.trans_weight,
            lambda_depth=cfg.lambda_depth,
            use_depth=cfg.use_depth,
            lambda_abs_pose=cfg.lambda_abs_pose,
            lambda_abs_t=cfg.lambda_abs_t,
            lambda_abs_r=cfg.lambda_abs_r,
            abs_pose_robust=cfg.abs_pose_robust,
            scene_scale=scene_scale,
            return_terms=return_terms,
            depth_confidence=record.target_confidence,
        )
    return build_stageA_loss_exact_shared_cm(
        render_rgb=render_rgb,
        render_depth=render_depth,
        target_rgb=record.target_rgb,
        target_depth=record.target_depth,
        confidence_mask=record.confidence_mask,
        viewpoint=vp,
        beta_rgb=cfg.beta_rgb,
        lambda_pose=cfg.lambda_pose,
        lambda_exp=cfg.lambda_exp,
        trans_weight=cfg.trans_weight,
        lambda_depth=cfg.lambda_depth,
        use_depth=cfg.use_depth,
        lambda_abs_pose=cfg.lambda_abs_pose,
        lambda_abs_t=cfg.lambda_abs_t,
        lambda_abs_r=cfg.lambda_abs_r,
        abs_pose_robust=cfg.abs_pose_robust,
        scene_scale=scene_scale,
        return_terms=return_terms,
        valid_mask=record.valid_mask,
        target_confidence=record.target_confidence,
    )
