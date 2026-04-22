from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from pseudo_branch.refine.pseudo_camera_state import viewpoint_optimizer_groups
from pseudo_branch.gaussian_management.gaussian_param_groups import build_micro_gaussian_param_groups


@dataclass
class StageAConfig:
    num_iterations: int = 300
    beta_rgb: float = 0.7
    lambda_pose: float = 0.01
    lambda_exp: float = 0.001
    # legacy single-scalar abs prior (kept for backward compatibility)
    lambda_abs_pose: float = 0.0
    # new split abs prior
    lambda_abs_t: float = 0.0
    lambda_abs_r: float = 0.0
    abs_pose_robust: str = "charbonnier"
    trans_reg_weight: float = 1.0
    lr_rot: float = 0.003
    lr_trans: float = 0.001
    lr_exp: float = 0.01
    num_pseudo_views: int = 4


@dataclass
class StageA5Config(StageAConfig):
    trainable_params: str = "xyz"
    lr_xyz: float = 1e-4
    lr_opacity: float = 5e-4
    disable_densify: bool = True
    disable_prune: bool = True


def build_stageA_optimizer(pseudo_views: List[dict], cfg: StageAConfig) -> torch.optim.Optimizer:
    groups = []
    for i, view in enumerate(pseudo_views):
        groups.extend(
            viewpoint_optimizer_groups(
                view["vp"],
                lr_rot=cfg.lr_rot,
                lr_trans=cfg.lr_trans,
                lr_exp=cfg.lr_exp,
                uid_prefix=f"pseudo_{i}_{int(view['sample_id'])}",
            )
        )
    return torch.optim.Adam(groups)


def build_stageA5_optimizers(pseudo_views: List[dict], gaussians, cfg: StageA5Config) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    pseudo_opt = build_stageA_optimizer(pseudo_views, cfg)
    gaussian_groups = build_micro_gaussian_param_groups(
        gaussians=gaussians,
        mode=cfg.trainable_params,
        lr_xyz=cfg.lr_xyz,
        lr_opacity=cfg.lr_opacity,
    )
    gaussian_opt = torch.optim.Adam(gaussian_groups)
    return pseudo_opt, gaussian_opt
