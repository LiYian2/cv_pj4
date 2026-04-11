from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from utils.pose_utils import SE3_exp


@dataclass
class ExportedPseudoCameraState:
    sample_id: int
    frame_id: int
    pose_c2w: list
    pose_w2c: list
    cam_rot_delta: list
    cam_trans_delta: list
    exposure_a: float
    exposure_b: float
    target_rgb_path: str
    target_depth_path: str
    confidence_path: str | None
    confidence_source_kind: str | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': int(self.sample_id),
            'frame_id': int(self.frame_id),
            'pose_c2w': self.pose_c2w,
            'pose_w2c': self.pose_w2c,
            'cam_rot_delta': self.cam_rot_delta,
            'cam_trans_delta': self.cam_trans_delta,
            'exposure_a': float(self.exposure_a),
            'exposure_b': float(self.exposure_b),
            'target_rgb_path': self.target_rgb_path,
            'target_depth_path': self.target_depth_path,
            'confidence_path': self.confidence_path,
            'confidence_source_kind': self.confidence_source_kind,
        }


def _ensure_parameter(x: torch.Tensor, shape: tuple[int, ...]):
    if isinstance(x, nn.Parameter):
        return x
    if x is None:
        x = torch.zeros(shape, device='cuda', dtype=torch.float32)
    x = x.detach().clone().float().to('cuda')
    if tuple(x.shape) != shape:
        x = x.reshape(shape)
    return nn.Parameter(x, requires_grad=True)


def make_viewpoint_trainable(vp):
    vp.cam_rot_delta = _ensure_parameter(getattr(vp, 'cam_rot_delta', None), (3,))
    vp.cam_trans_delta = _ensure_parameter(getattr(vp, 'cam_trans_delta', None), (3,))
    vp.exposure_a = _ensure_parameter(getattr(vp, 'exposure_a', None), (1,))
    vp.exposure_b = _ensure_parameter(getattr(vp, 'exposure_b', None), (1,))
    return vp


def viewpoint_optimizer_groups(vp, lr_rot: float, lr_trans: float, lr_exp: float, uid_prefix: str):
    return [
        {'params': [vp.cam_rot_delta], 'lr': float(lr_rot), 'name': f'{uid_prefix}_cam_rot_delta'},
        {'params': [vp.cam_trans_delta], 'lr': float(lr_trans), 'name': f'{uid_prefix}_cam_trans_delta'},
        {'params': [vp.exposure_a], 'lr': float(lr_exp), 'name': f'{uid_prefix}_exposure_a'},
        {'params': [vp.exposure_b], 'lr': float(lr_exp), 'name': f'{uid_prefix}_exposure_b'},
    ]


def current_w2c(vp) -> torch.Tensor:
    base = torch.eye(4, device=vp.R.device, dtype=vp.R.dtype)
    base[:3, :3] = vp.R
    base[:3, 3] = vp.T
    tau = torch.cat([vp.cam_trans_delta, vp.cam_rot_delta], dim=0)
    return SE3_exp(tau) @ base


def current_c2w(vp) -> torch.Tensor:
    return torch.linalg.inv(current_w2c(vp))


def export_view_state(view: Dict[str, Any]) -> Dict[str, Any]:
    vp = view['vp']
    w2c = current_w2c(vp).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    state = ExportedPseudoCameraState(
        sample_id=int(view['sample_id']),
        frame_id=int(view['frame_id']),
        pose_c2w=c2w.tolist(),
        pose_w2c=w2c.tolist(),
        cam_rot_delta=vp.cam_rot_delta.detach().cpu().tolist(),
        cam_trans_delta=vp.cam_trans_delta.detach().cpu().tolist(),
        exposure_a=float(vp.exposure_a.detach().cpu().item()),
        exposure_b=float(vp.exposure_b.detach().cpu().item()),
        target_rgb_path=str(view.get('target_rgb_path', '')),
        target_depth_path=str(view.get('target_depth_for_refine_path', view.get('target_depth_path', ''))),
        confidence_path=view.get('confidence_path'),
        confidence_source_kind=view.get('confidence_source_kind'),
    )
    return state.to_dict()
