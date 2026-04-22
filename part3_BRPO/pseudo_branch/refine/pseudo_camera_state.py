from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getWorld2View2
from utils.pose_utils import SE3_exp, SE3_log


@dataclass
class ExportedPseudoCameraState:
    sample_id: int
    frame_id: int
    pose_c2w: list
    pose_w2c: list
    pose_w2c_initial: list | None
    cam_rot_delta: list
    cam_trans_delta: list
    abs_pose_rho: list | None
    abs_pose_theta: list | None
    abs_pose_norm: float | None
    abs_pose_rho_norm: float | None
    abs_pose_theta_norm: float | None
    scene_scale_used: float | None
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
            'pose_w2c_initial': self.pose_w2c_initial,
            'cam_rot_delta': self.cam_rot_delta,
            'cam_trans_delta': self.cam_trans_delta,
            'abs_pose_rho': self.abs_pose_rho,
            'abs_pose_theta': self.abs_pose_theta,
            'abs_pose_norm': float(self.abs_pose_norm) if self.abs_pose_norm is not None else None,
            'abs_pose_rho_norm': float(self.abs_pose_rho_norm) if self.abs_pose_rho_norm is not None else None,
            'abs_pose_theta_norm': float(self.abs_pose_theta_norm) if self.abs_pose_theta_norm is not None else None,
            'scene_scale_used': float(self.scene_scale_used) if self.scene_scale_used is not None else None,
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
    if not hasattr(vp, 'R0') or vp.R0 is None:
        vp.R0 = vp.R.detach().clone()
    if not hasattr(vp, 'T0') or vp.T0 is None:
        vp.T0 = vp.T.detach().clone()
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


def refresh_viewpoint_transforms_(vp):
    world_view_transform = getWorld2View2(vp.R, vp.T).transpose(0, 1)
    vp.world_view_transform = world_view_transform
    vp.full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        vp.projection_matrix.unsqueeze(0)
    ).squeeze(0)
    vp.camera_center = world_view_transform.inverse()[3, :3]
    return vp


def apply_pose_residual_(vp, converged_threshold: float = 1e-4) -> bool:
    tau = torch.cat([vp.cam_trans_delta, vp.cam_rot_delta], dim=0)
    new_w2c = current_w2c(vp)
    vp.R = new_w2c[:3, :3].detach().clone()
    vp.T = new_w2c[:3, 3].detach().clone()
    refresh_viewpoint_transforms_(vp)
    converged = bool(torch.norm(tau).detach().cpu().item() < converged_threshold)
    with torch.no_grad():
        vp.cam_rot_delta.zero_()
        vp.cam_trans_delta.zero_()
    return converged


def load_exported_view_states(path: str | Path) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f'Expected a list of exported pseudo camera states in {path}')
    out = {}
    for item in data:
        sid = int(item['sample_id'])
        if sid in out:
            raise ValueError(f'Duplicate sample_id={sid} in {path}')
        out[sid] = item
    return out


def apply_loaded_view_state_(vp, state: Dict[str, Any], reference_mode: str = 'keep'):
    w2c = torch.tensor(state['pose_w2c'], device=vp.R.device, dtype=vp.R.dtype)
    vp.R = w2c[:3, :3].detach().clone()
    vp.T = w2c[:3, 3].detach().clone()
    refresh_viewpoint_transforms_(vp)
    with torch.no_grad():
        vp.cam_rot_delta.zero_()
        vp.cam_trans_delta.zero_()
        vp.exposure_a.copy_(torch.tensor([float(state.get('exposure_a', 0.0))], device=vp.exposure_a.device, dtype=vp.exposure_a.dtype))
        vp.exposure_b.copy_(torch.tensor([float(state.get('exposure_b', 0.0))], device=vp.exposure_b.device, dtype=vp.exposure_b.dtype))
    if reference_mode == 'reset_to_loaded':
        vp.R0 = vp.R.detach().clone()
        vp.T0 = vp.T.detach().clone()
    elif reference_mode != 'keep':
        raise ValueError(f'Unsupported reference_mode={reference_mode}')
    return vp


def summarize_true_pose_deltas(initial_states: list[Dict[str, Any]], final_states: list[Dict[str, Any]]) -> Dict[str, Any]:
    init_by = {int(x['sample_id']): x for x in initial_states}
    final_by = {int(x['sample_id']): x for x in final_states}
    rows = []
    for sid in sorted(init_by):
        if sid not in final_by:
            continue
        a = np.asarray(init_by[sid]['pose_w2c'], dtype=np.float64)
        b = np.asarray(final_by[sid]['pose_w2c'], dtype=np.float64)
        delta = b @ np.linalg.inv(a)
        trans_norm = float(np.linalg.norm(delta[:3, 3]))
        rot_fro = float(np.linalg.norm(delta[:3, :3] - np.eye(3)))
        rows.append({
            'sample_id': int(sid),
            'frame_id': int(final_by[sid].get('frame_id', init_by[sid].get('frame_id', sid))),
            'trans_norm': trans_norm,
            'rot_fro_norm': rot_fro,
            'abs_pose_norm_final': float(final_by[sid].get('abs_pose_norm') or 0.0),
        })
    trans = [r['trans_norm'] for r in rows]
    rot = [r['rot_fro_norm'] for r in rows]
    absn = [r['abs_pose_norm_final'] for r in rows]
    aggregate = {
        'num_views': int(len(rows)),
        'mean_trans_norm': float(np.mean(trans)) if trans else 0.0,
        'max_trans_norm': float(np.max(trans)) if trans else 0.0,
        'mean_rot_fro_norm': float(np.mean(rot)) if rot else 0.0,
        'max_rot_fro_norm': float(np.max(rot)) if rot else 0.0,
        'mean_abs_pose_norm_final': float(np.mean(absn)) if absn else 0.0,
        'max_abs_pose_norm_final': float(np.max(absn)) if absn else 0.0,
    }
    return {'per_view': rows, 'aggregate': aggregate}


def export_view_state(view: Dict[str, Any]) -> Dict[str, Any]:
    vp = view['vp']
    w2c = current_w2c(vp).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)

    w2c0_list = None
    abs_pose_rho = None
    abs_pose_theta = None
    abs_pose_norm = None
    abs_pose_rho_norm = None
    abs_pose_theta_norm = None
    scene_scale_used = view.get('stageA_scene_scale')
    if hasattr(vp, 'R0') and hasattr(vp, 'T0') and vp.R0 is not None and vp.T0 is not None:
        w2c0 = torch.eye(4, device=vp.R0.device, dtype=vp.R0.dtype)
        w2c0[:3, :3] = vp.R0
        w2c0[:3, 3] = vp.T0
        w2c0_list = w2c0.detach().cpu().numpy().tolist()

        delta = current_w2c(vp) @ torch.linalg.inv(w2c0)
        tau = SE3_log(delta)
        abs_pose_rho = tau[:3].detach().cpu().tolist()
        abs_pose_theta = tau[3:].detach().cpu().tolist()
        abs_pose_rho_norm = float(torch.norm(tau[:3]).detach().cpu().item())
        abs_pose_theta_norm = float(torch.norm(tau[3:]).detach().cpu().item())
        abs_pose_norm = float(torch.norm(tau).detach().cpu().item())

    state = ExportedPseudoCameraState(
        sample_id=int(view['sample_id']),
        frame_id=int(view['frame_id']),
        pose_c2w=c2w.tolist(),
        pose_w2c=w2c.tolist(),
        pose_w2c_initial=w2c0_list,
        cam_rot_delta=vp.cam_rot_delta.detach().cpu().tolist(),
        cam_trans_delta=vp.cam_trans_delta.detach().cpu().tolist(),
        abs_pose_rho=abs_pose_rho,
        abs_pose_theta=abs_pose_theta,
        abs_pose_norm=abs_pose_norm,
        abs_pose_rho_norm=abs_pose_rho_norm,
        abs_pose_theta_norm=abs_pose_theta_norm,
        scene_scale_used=float(scene_scale_used) if scene_scale_used is not None else None,
        exposure_a=float(vp.exposure_a.detach().cpu().item()),
        exposure_b=float(vp.exposure_b.detach().cpu().item()),
        target_rgb_path=str(view.get('target_rgb_path', '')),
        target_depth_path=str(view.get('target_depth_for_refine_path', view.get('target_depth_path', ''))),
        confidence_path=view.get('confidence_path'),
        confidence_source_kind=view.get('confidence_source_kind'),
    )
    return state.to_dict()
