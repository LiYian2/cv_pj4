#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
S3PO_ROOT = "/home/bzhang512/CV_Project/third_party/S3PO-GS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, S3PO_ROOT)
sys.path.insert(0, f"{S3PO_ROOT}/gaussian_splatting")

from pseudo_branch.refine.pseudo_camera_state import (
    make_viewpoint_trainable,
    export_view_state,
    apply_pose_residual_,
    load_exported_view_states,
    apply_loaded_view_state_,
    summarize_true_pose_deltas,
)
from pseudo_branch.refine.pseudo_loss_v2 import build_stageA_loss, build_stageA_loss_source_aware, build_stageA_loss_exact_shared_cm
from pseudo_branch.refine.pseudo_refine_scheduler import StageAConfig, StageA5Config, build_stageA_optimizer, build_stageA5_optimizers
from pseudo_branch.gaussian_management.gaussian_param_groups import build_micro_gaussian_param_groups
from pseudo_branch.gaussian_management.local_gating import (
    PseudoLocalGatingConfig,
    evaluate_sampled_views_for_local_gating,
    build_visibility_weight_map,
    apply_gaussian_grad_mask,
    build_iteration_gating_summary,
)
from pseudo_branch.gaussian_management.spgm import (
    collect_spgm_stats,
    build_spgm_importance_score,
    build_spgm_update_policy,
    apply_spgm_state_management,
)


def load_v1_module():
    script = Path(__file__).parent / 'compat' / 'run_pseudo_refinement.py'
    spec = importlib.util.spec_from_file_location('run_pseudo_refinement_v1', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description='Part3 BRPO refine v2 (Stage A first)')
    p.add_argument('--ply_path', required=True)
    p.add_argument('--pseudo_cache', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--target_side', choices=['left', 'right', 'fused'], default='fused')
    p.add_argument('--confidence_mask_source', choices=['legacy', 'brpo'], default='brpo')
    p.add_argument('--brpo_mask_root', default=None)
    p.add_argument('--signal_pipeline', choices=['legacy', 'brpo_v2'], default='legacy')
    p.add_argument('--signal_v2_root', default=None, help='Optional root containing signal_v2/frame_<id> artifacts; defaults to <prepare_root>/signal_v2 or <prepare_root>/signal_v2_* when discoverable.')
    p.add_argument('--pseudo_observation_mode', choices=['off', 'brpo_joint_v1', 'brpo_verify_v1', 'brpo_style_v1', 'brpo_style_v2', 'brpo_direct_v1', 'hybrid_brpo_cm_geo_v1', 'exact_brpo_cm_old_target_v1', 'exact_brpo_full_target_v1', 'exact_brpo_cm_hybrid_target_v1', 'exact_brpo_cm_stable_target_v1', 'exact_brpo_upstream_target_v1'], default='off')
    p.add_argument('--stage_mode', choices=['stageA', 'stageA5', 'stageB'], default='stageA')
    p.add_argument('--joint_topology_mode', choices=['off', 'brpo_joint_v1'], default='off')
    p.add_argument('--stageA_iters', type=int, default=300)
    p.add_argument('--stageA_beta_rgb', type=float, default=0.7)
    p.add_argument('--stageA_lambda_pose', type=float, default=0.01)
    p.add_argument('--stageA_lambda_exp', type=float, default=0.001)
    p.add_argument('--stageA_lambda_abs_pose', type=float, default=0.0, help='Legacy single-scalar abs pose prior weight')
    p.add_argument('--stageA_lambda_abs_t', type=float, default=0.0)
    p.add_argument('--stageA_lambda_abs_r', type=float, default=0.0)
    p.add_argument('--stageA_abs_pose_robust', choices=['charbonnier', 'huber', 'l2'], default='charbonnier')
    p.add_argument('--stageA_abs_pose_scale_source', choices=['render_depth_trainmask_median', 'render_depth_valid_median', 'fixed'], default='render_depth_trainmask_median')
    p.add_argument('--stageA_abs_pose_fixed_scale', type=float, default=1.0)
    p.add_argument('--stageA_log_grad_contrib', action='store_true')
    p.add_argument('--stageA_log_grad_interval', type=int, default=20)
    p.add_argument('--stageA_trans_reg_weight', type=float, default=1.0)
    p.add_argument('--stageA_lr_rot', type=float, default=0.003)
    p.add_argument('--stageA_lr_trans', type=float, default=0.001)
    p.add_argument('--stageA_lr_exp', type=float, default=0.01)
    p.add_argument('--stageA_disable_depth', action='store_true')
    p.add_argument('--stageA_apply_pose_update', action='store_true', default=True)
    p.add_argument('--stageA_no_apply_pose_update', dest='stageA_apply_pose_update', action='store_false')
    p.add_argument(
        '--stageA_mask_mode',
        choices=['auto', 'train_mask', 'seed_support_only', 'legacy'],
        default='train_mask',
        help='Legacy shared mask selector; kept for backward compatibility when rgb/depth mask modes stay auto.',
    )
    p.add_argument('--stageA_rgb_mask_mode', choices=['auto', 'raw_confidence', 'seed_support_only', 'train_mask', 'legacy', 'brpo_v2_raw', 'brpo_v2_cont', 'joint_confidence_v2', 'joint_confidence_cont_v2', 'joint_confidence_expand_v1', 'joint_confidence_cont_expand_v1'], default='auto')
    p.add_argument('--stageA_depth_mask_mode', choices=['auto', 'seed_support_only', 'train_mask', 'legacy', 'brpo_v2_depth', 'joint_confidence_v2', 'joint_confidence_cont_v2', 'joint_confidence_expand_v1', 'joint_confidence_cont_expand_v1'], default='auto')
    p.add_argument('--stageA_confidence_variant', choices=['discrete', 'continuous'], default='discrete')
    p.add_argument(
        '--stageA_target_depth_mode',
        choices=['auto', 'blended_depth', 'blended_depth_m5', 'render_depth_only', 'target_depth_for_refine', 'target_depth_for_refine_v2', 'target_depth', 'render_depth', 'brpo_v2', 'target_depth_for_refine_v2_brpo', 'joint_depth_v2', 'joint_depth_expand_v1'],
        default='blended_depth',
        help='Which depth target Stage A should actually consume.',
    )
    p.add_argument('--stageA_depth_loss_mode', choices=['legacy', 'source_aware', 'exact_shared_cm_v1'], default='legacy')
    p.add_argument('--stageA_lambda_depth_seed', type=float, default=1.0)
    p.add_argument('--stageA_lambda_depth_dense', type=float, default=0.35)
    p.add_argument('--stageA_lambda_depth_fallback', type=float, default=0.0)
    p.add_argument('--num_pseudo_views', type=int, default=4)
    p.add_argument('--stageA5_trainable_params', choices=['xyz', 'xyz_opacity'], default='xyz')
    p.add_argument('--stageA5_lr_xyz', type=float, default=1e-4)
    p.add_argument('--stageA5_lr_opacity', type=float, default=5e-4)
    p.add_argument('--stageA5_disable_densify', action='store_true', default=True)
    p.add_argument('--stageA5_disable_prune', action='store_true', default=True)
    p.add_argument('--pseudo_local_gating', choices=['off', 'hard_visible_union_signal', 'soft_visible_union_signal', 'spgm_keep', 'spgm_soft'], default='off')
    p.add_argument('--pseudo_local_gating_params', choices=['xyz', 'xyz_opacity'], default='xyz')
    p.add_argument('--pseudo_local_gating_min_verified_ratio', type=float, default=0.01)
    p.add_argument('--pseudo_local_gating_min_rgb_mask_ratio', type=float, default=0.01)
    p.add_argument('--pseudo_local_gating_max_fallback_ratio', type=float, default=0.995)
    p.add_argument('--pseudo_local_gating_min_correction', type=float, default=0.0)
    p.add_argument('--pseudo_local_gating_soft_power', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_log_interval', type=int, default=20)
    # SPGM-specific CLI parameters
    p.add_argument('--pseudo_local_gating_spgm_num_clusters', type=int, default=3)
    p.add_argument('--pseudo_local_gating_spgm_alpha_depth', type=float, default=0.5)
    p.add_argument('--pseudo_local_gating_spgm_beta_entropy', type=float, default=0.5)
    p.add_argument('--pseudo_local_gating_spgm_gamma_entropy', type=float, default=0.5)
    p.add_argument('--pseudo_local_gating_spgm_support_eta', type=float, default=0.5)
    p.add_argument('--pseudo_local_gating_spgm_weight_floor', type=float, default=0.05)
    p.add_argument('--pseudo_local_gating_spgm_entropy_bins', type=int, default=32)
    p.add_argument('--pseudo_local_gating_spgm_density_mode', default='opacity_support')
    p.add_argument('--pseudo_local_gating_spgm_cluster_keep_near', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_cluster_keep_mid', type=float, default=0.8)
    p.add_argument('--pseudo_local_gating_spgm_cluster_keep_far', type=float, default=0.6)
    p.add_argument('--pseudo_local_gating_spgm_policy_mode', choices=['dense_keep', 'selector_quantile'], default='dense_keep')
    p.add_argument('--pseudo_local_gating_spgm_ranking_mode', choices=['v1', 'support_blend'], default='v1')
    p.add_argument('--pseudo_local_gating_spgm_lambda_support_rank', type=float, default=0.0)
    p.add_argument('--pseudo_local_gating_spgm_selector_keep_ratio_near', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_selector_keep_ratio_mid', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_selector_keep_ratio_far', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_selector_min_keep', type=int, default=1)
    p.add_argument('--pseudo_local_gating_spgm_manager_mode', choices=['summary_only', 'xyz_lr_scale', 'deterministic_participation', 'deterministic_opacity_participation'], default='summary_only')
    p.add_argument('--pseudo_local_gating_spgm_state_candidate_quantile', type=float, default=0.5)
    p.add_argument('--pseudo_local_gating_spgm_state_base_scale_near', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_state_base_scale_mid', type=float, default=0.95)
    p.add_argument('--pseudo_local_gating_spgm_state_base_scale_far', type=float, default=0.90)
    p.add_argument('--pseudo_local_gating_spgm_state_participation_keep_near', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_state_participation_keep_mid', type=float, default=0.9)
    p.add_argument('--pseudo_local_gating_spgm_state_participation_keep_far', type=float, default=0.75)
    p.add_argument('--pseudo_local_gating_spgm_state_opacity_floor_near', type=float, default=1.0)
    p.add_argument('--pseudo_local_gating_spgm_state_opacity_floor_mid', type=float, default=0.98)
    p.add_argument('--pseudo_local_gating_spgm_state_opacity_floor_far', type=float, default=0.92)
    p.add_argument('--stageB_iters', type=int, default=120)
    p.add_argument('--stageB_post_switch_iter', type=int, default=0, help='If >0, apply post-switch StageB settings starting from this iteration + 1')
    p.add_argument('--stageB_post_lr_scale_xyz', type=float, default=1.0, help='Scale xyz lr by this factor after --stageB_post_switch_iter')
    p.add_argument('--stageB_post_lr_scale_opacity', type=float, default=1.0, help='Scale opacity lr by this factor after --stageB_post_switch_iter')
    p.add_argument('--stageB_post_lambda_real', type=float, default=None, help='Optional late-stage lambda_real override after --stageB_post_switch_iter')
    p.add_argument('--stageB_post_lambda_pseudo', type=float, default=None, help='Optional late-stage lambda_pseudo override after --stageB_post_switch_iter')
    p.add_argument('--init_pseudo_camera_states_json', default=None, help='Optional pseudo_camera_states_*.json from a previous stage for sequential handoff')
    p.add_argument('--init_pseudo_reference_mode', choices=['keep', 'reset_to_loaded'], default='keep', help='Whether to keep original R0/T0 anchors or reset them to the loaded handoff state')
    p.add_argument('--train_manifest', default=None, help='split_manifest.json for sparse train views (real anchor)')
    p.add_argument('--train_rgb_dir', default=None, help='RGB dir for sparse train views; defaults to manifest files.rgb')
    p.add_argument('--lambda_real', type=float, default=1.0)
    p.add_argument('--lambda_pseudo', type=float, default=1.0)
    p.add_argument('--num_real_views', type=int, default=2)
    p.add_argument('--sh_degree', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def sample_indices(n_total: int, n_sample: int, rng: np.random.Generator):
    n_sample = max(0, min(n_total, n_sample))
    if n_sample == 0:
        return []
    return rng.choice(n_total, n_sample, replace=False).tolist()


def _apply_stageB_post_switch_if_needed(args, iteration: int, gaussian_optimizer: torch.optim.Optimizer, stageB_history: dict, state: dict):
    switch_iter = int(getattr(args, 'stageB_post_switch_iter', 0) or 0)
    if switch_iter <= 0 or iteration <= switch_iter or state.get('applied', False):
        return

    xyz_scale = float(getattr(args, 'stageB_post_lr_scale_xyz', 1.0) or 1.0)
    opacity_scale = float(getattr(args, 'stageB_post_lr_scale_opacity', 1.0) or 1.0)
    for group in gaussian_optimizer.param_groups:
        group_name = str(group.get('name') or '').lower()
        scale = xyz_scale if 'xyz' in group_name else opacity_scale
        group['lr'] = float(group['lr']) * scale

    state['applied'] = True
    state['applied_at_iter'] = int(iteration)
    state['applied_xyz_scale'] = xyz_scale
    state['applied_opacity_scale'] = opacity_scale

    stageB_history['post_switch_applied'] = True
    stageB_history['post_switch_applied_at_iter'] = int(iteration)
    stageB_history['post_switch_applied_xyz_scale'] = xyz_scale
    stageB_history['post_switch_applied_opacity_scale'] = opacity_scale


def _resolve_depth_mode_alias(mode: str):
    if mode == 'blended_depth':
        return 'target_depth_for_refine'
    if mode == 'blended_depth_m5':
        return 'target_depth_for_refine_v2'
    if mode == 'render_depth_only':
        return 'render_depth'
    return mode


def _discover_signal_v2_root(sample_dir: Path, explicit_root: str | None):
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    prepare_root = sample_dir.parent.parent.parent if sample_dir.parent.name == 'samples' else sample_dir.parent.parent
    candidates.extend([
        prepare_root / 'signal_v2',
        prepare_root / 'signal_v2_e15',
        prepare_root / 'signal_v2_smoke',
    ])
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _signal_v2_frame_dir(sample_dir: Path, frame_id: int, explicit_root: str | None):
    root = _discover_signal_v2_root(sample_dir, explicit_root)
    if root is None:
        return None
    frame_dir = root / f'frame_{int(frame_id):04d}'
    return frame_dir if frame_dir.exists() else None


def _mask_v2_candidates(requested_mode: str):
    if requested_mode == 'brpo_v2_raw':
        return [('raw_rgb_confidence_v2.npy', 'discrete')]
    if requested_mode == 'brpo_v2_cont':
        return [('raw_rgb_confidence_cont_v2.npy', 'continuous')]
    if requested_mode == 'brpo_v2_depth':
        return [('depth_supervision_mask_v2_brpo.npy', 'discrete')]
    if requested_mode == 'joint_confidence_v2':
        return [('joint_confidence_v2.npy', 'discrete')]
    if requested_mode == 'joint_confidence_cont_v2':
        return [('joint_confidence_cont_v2.npy', 'continuous')]
    if requested_mode == 'joint_confidence_expand_v1':
        return [('joint_confidence_expand_v1.npy', 'discrete')]
    if requested_mode == 'joint_confidence_cont_expand_v1':
        return [('joint_confidence_cont_expand_v1.npy', 'continuous')]
    return []


def _summarize_source_map(source_map: np.ndarray | None, kind: str | None = None):
    if source_map is None:
        return {'verified_ratio': None, 'render_fallback_ratio': None, 'seed_ratio': None, 'dense_ratio': None}
    source_map = np.asarray(source_map)
    total = float(source_map.size)
    no_none_as_fallback_kinds = {
        'target_depth_for_refine_v2_brpo',
        'joint_depth_target_v2',
        'joint_depth_target_expand_v1',
        'joint_depth_target_joint_v1',
        'joint_depth_target_brpo_style_v1',
        'joint_depth_target_brpo_style_v2',
        'joint_depth_target_brpo_direct_v1',
        'joint_depth_target_hybrid_brpo_cm_geo_v1',
        'joint_depth_target_exact_brpo_cm_old_target_v1',
        'joint_depth_target_exact_brpo_full_target_v1',
        'joint_depth_target_exact_brpo_cm_hybrid_target_v1',
        'joint_depth_target_exact_brpo_cm_stable_target_v1',
    }
    if kind in no_none_as_fallback_kinds:
        verified = np.isin(source_map, [1, 2, 3])
        fallback = source_map == 4
        return {
            'verified_ratio': float(verified.sum() / total),
            'render_fallback_ratio': float(fallback.sum() / total),
            'seed_ratio': float(verified.sum() / total),
            'dense_ratio': 0.0,
        }
    return {
        'verified_ratio': float((source_map != 0).sum() / total),
        'render_fallback_ratio': float((source_map == 0).sum() / total),
        'seed_ratio': float(np.isin(source_map, [1, 2, 3]).sum() / total),
        'dense_ratio': float((source_map == 4).sum() / total),
    }


def resolve_depth_target_path(sample_dir: Path, mode: str, frame_id: int | None = None, signal_v2_root: str | None = None):
    resolved_mode = _resolve_depth_mode_alias(mode)
    candidates = []
    frame_dir_v2 = None
    if frame_id is not None:
        frame_dir_v2 = _signal_v2_frame_dir(sample_dir, frame_id, signal_v2_root)
    if resolved_mode in {'brpo_v2', 'target_depth_for_refine_v2_brpo'}:
        if frame_dir_v2 is None:
            raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
        candidates = [('target_depth_for_refine_v2_brpo', frame_dir_v2 / 'target_depth_for_refine_v2_brpo.npy', frame_dir_v2 / 'target_depth_source_map_v2_brpo.npy')]
    elif resolved_mode == 'joint_depth_v2':
        if frame_dir_v2 is None:
            raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
        candidates = [('joint_depth_target_v2', frame_dir_v2 / 'joint_depth_target_v2.npy', frame_dir_v2 / 'target_depth_source_map_v2_brpo.npy')]
    elif resolved_mode == 'joint_depth_v1':
        if frame_dir_v2 is None:
            raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
        candidates = [('joint_depth_target_joint_v1', frame_dir_v2 / 'pseudo_depth_target_joint_v1.npy', frame_dir_v2 / 'pseudo_source_map_joint_v1.npy')]
    elif resolved_mode == 'joint_depth_expand_v1':
        if frame_dir_v2 is None:
            raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
        candidates = [('joint_depth_target_expand_v1', frame_dir_v2 / 'joint_depth_target_expand_v1.npy', frame_dir_v2 / 'joint_expand_source_map_v1.npy')]
    elif resolved_mode == 'target_depth_for_refine_v2':
        candidates = [('target_depth_for_refine_v2', sample_dir / 'target_depth_for_refine_v2.npy', sample_dir / 'target_depth_dense_source_map.npy')]
    elif resolved_mode == 'target_depth_for_refine':
        candidates = [('target_depth_for_refine', sample_dir / 'target_depth_for_refine.npy', sample_dir / 'target_depth_for_refine_source_map.npy')]
    elif resolved_mode == 'target_depth':
        candidates = [('target_depth', sample_dir / 'target_depth.npy', None)]
    elif resolved_mode == 'render_depth':
        candidates = [('render_depth', sample_dir / 'render_depth.npy', None)]
    else:
        candidates = [
            ('target_depth_for_refine_v2', sample_dir / 'target_depth_for_refine_v2.npy', sample_dir / 'target_depth_dense_source_map.npy'),
            ('target_depth_for_refine', sample_dir / 'target_depth_for_refine.npy', sample_dir / 'target_depth_for_refine_source_map.npy'),
            ('target_depth', sample_dir / 'target_depth.npy', None),
            ('render_depth', sample_dir / 'render_depth.npy', None),
        ]
        if frame_dir_v2 is not None:
            candidates.insert(0, ('target_depth_for_refine_v2_brpo', frame_dir_v2 / 'target_depth_for_refine_v2_brpo.npy', frame_dir_v2 / 'target_depth_source_map_v2_brpo.npy'))
            candidates.insert(0, ('joint_depth_target_v2', frame_dir_v2 / 'joint_depth_target_v2.npy', frame_dir_v2 / 'target_depth_source_map_v2_brpo.npy'))
            candidates.insert(0, ('joint_depth_target_joint_v1', frame_dir_v2 / 'pseudo_depth_target_joint_v1.npy', frame_dir_v2 / 'pseudo_source_map_joint_v1.npy'))
            candidates.insert(0, ('joint_depth_target_expand_v1', frame_dir_v2 / 'joint_depth_target_expand_v1.npy', frame_dir_v2 / 'joint_expand_source_map_v1.npy'))
    for kind, path, source_map_path in candidates:
        if path.exists():
            return kind, path, source_map_path
    raise FileNotFoundError(f'No depth target candidate found under {sample_dir} for mode={mode}')


def _mask_train_candidates(target_side: str):
    if target_side == 'fused':
        return ['train_confidence_mask_brpo_fused.npy', 'confidence_mask_brpo_fused.npy', 'confidence_mask_brpo.npy']
    if target_side == 'right':
        return ['train_confidence_mask_brpo_right.npy', 'confidence_mask_brpo_right.npy', 'confidence_mask_brpo.npy']
    return ['train_confidence_mask_brpo_left.npy', 'confidence_mask_brpo_left.npy', 'confidence_mask_brpo.npy']


def _compose_seed_support_mask(sample_dir: Path, target_side: str):
    seed_left = np.load(sample_dir / 'seed_support_left.npy').astype(np.float32)
    seed_right = np.load(sample_dir / 'seed_support_right.npy').astype(np.float32)
    seed_both = np.load(sample_dir / 'seed_support_both.npy').astype(np.float32)
    seed_single = np.load(sample_dir / 'seed_support_single.npy').astype(np.float32)

    if target_side == 'fused':
        conf = np.zeros_like(seed_both, dtype=np.float32)
        conf[seed_both > 0.5] = 1.0
        conf[seed_single > 0.5] = 0.5
        return conf, 'generated::seed_support_fused'

    if target_side == 'right':
        right_only = (seed_right > 0.5) & ~(seed_both > 0.5)
        conf = np.zeros_like(seed_both, dtype=np.float32)
        conf[seed_both > 0.5] = 1.0
        conf[right_only] = 0.5
        return conf, 'generated::seed_support_right'

    left_only = (seed_left > 0.5) & ~(seed_both > 0.5)
    conf = np.zeros_like(seed_both, dtype=np.float32)
    conf[seed_both > 0.5] = 1.0
    conf[left_only] = 0.5
    return conf, 'generated::seed_support_left'


def _mask_raw_candidates(target_side: str, confidence_variant: str):
    if confidence_variant == 'continuous':
        if target_side == 'fused':
            return ['raw_confidence_mask_brpo_cont_fused.npy']
        if target_side == 'right':
            return ['raw_confidence_mask_brpo_cont_right.npy']
        return ['raw_confidence_mask_brpo_cont_left.npy']
    if target_side == 'fused':
        return ['raw_confidence_mask_brpo_fused.npy', 'confidence_mask_brpo_fused.npy', 'confidence_mask_brpo.npy']
    if target_side == 'right':
        return ['raw_confidence_mask_brpo_right.npy', 'confidence_mask_brpo_right.npy', 'confidence_mask_brpo.npy']
    return ['raw_confidence_mask_brpo_left.npy', 'confidence_mask_brpo_left.npy', 'confidence_mask_brpo.npy']


def _resolve_effective_mask_mode(requested_mode: str, legacy_mode: str):
    if requested_mode != 'auto':
        return requested_mode
    if legacy_mode and legacy_mode != 'auto':
        return legacy_mode
    return 'train_mask'


def resolve_stageA_mask(sample_dir: Path, frame_id: int, target_side: str, requested_mode: str, confidence_variant: str, default_conf: np.ndarray, default_path: str | None, default_kind: str | None, legacy_mode: str = 'train_mask', signal_v2_root: str | None = None):
    requested_mode = _resolve_effective_mask_mode(requested_mode, legacy_mode)

    if requested_mode == 'legacy':
        if default_conf is None:
            raise FileNotFoundError(f'No legacy/default confidence mask available under {sample_dir}')
        return default_conf.astype(np.float32), default_path, default_kind or 'legacy_default', 'legacy', 'legacy'

    if requested_mode == 'train_mask':
        for name in _mask_train_candidates(target_side):
            path = sample_dir / name
            if path.exists():
                return np.load(path).astype(np.float32), str(path), f'sample_explicit_{name}', 'train_mask', 'discrete'
        raise FileNotFoundError(f'No train-mask candidate found under {sample_dir} for target_side={target_side}')

    if requested_mode == 'raw_confidence':
        for variant in ([confidence_variant, 'discrete'] if confidence_variant == 'continuous' else ['discrete']):
            for name in _mask_raw_candidates(target_side, variant):
                path = sample_dir / name
                if path.exists():
                    return np.load(path).astype(np.float32), str(path), f'sample_explicit_{name}', 'raw_confidence', variant
        raise FileNotFoundError(f'No raw-confidence candidate found under {sample_dir} for target_side={target_side}, variant={confidence_variant}')

    if requested_mode in {'brpo_v2_raw', 'brpo_v2_cont', 'brpo_v2_depth', 'joint_confidence_v2', 'joint_confidence_cont_v2', 'joint_confidence_expand_v1', 'joint_confidence_cont_expand_v1'}:
        frame_dir = _signal_v2_frame_dir(sample_dir, frame_id, signal_v2_root)
        if frame_dir is None:
            raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
        for name, variant in _mask_v2_candidates(requested_mode):
            path = frame_dir / name
            if path.exists():
                return np.load(path).astype(np.float32), str(path), f'signal_v2::{name}', requested_mode, variant
        raise FileNotFoundError(f'No signal_v2 mask candidate found under {frame_dir} for mode={requested_mode}')

    if requested_mode == 'seed_support_only':
        conf, kind = _compose_seed_support_mask(sample_dir, target_side)
        return conf, kind, kind, 'seed_support_only', 'discrete'

    raise ValueError(f'Unsupported stageA mask mode={requested_mode}')


def _depth_source_id_groups_for_view(view: dict):
    groups = view.get('depth_source_id_groups') or {}
    seed_ids = tuple(int(x) for x in groups.get('seed', [1, 2, 3]))
    dense_ids = tuple(int(x) for x in groups.get('dense', [4]))
    fallback_ids = tuple(int(x) for x in groups.get('fallback', [0]))
    return seed_ids, dense_ids, fallback_ids


def _load_joint_observation_bundle(frame_dir: Path):
    return {
        'confidence_joint': np.load(frame_dir / 'pseudo_confidence_joint_v1.npy').astype(np.float32),
        'confidence_rgb': np.load(frame_dir / 'pseudo_confidence_rgb_joint_v1.npy').astype(np.float32),
        'confidence_depth': np.load(frame_dir / 'pseudo_confidence_depth_joint_v1.npy').astype(np.float32),
        'depth_target': np.load(frame_dir / 'pseudo_depth_target_joint_v1.npy').astype(np.float32),
        'source_map': np.load(frame_dir / 'pseudo_source_map_joint_v1.npy'),
        'valid_mask': np.load(frame_dir / 'pseudo_valid_mask_joint_v1.npy').astype(np.float32),
        'meta_path': frame_dir / 'joint_observation_meta_v1.json',
    }


def _load_verify_observation_bundle(frame_dir: Path):
    return {
        'confidence_verify': np.load(frame_dir / 'pseudo_confidence_verify_v1.npy').astype(np.float32),
        'depth_target': np.load(frame_dir / 'pseudo_depth_target_joint_v1.npy').astype(np.float32),
        'source_map': np.load(frame_dir / 'pseudo_source_map_joint_v1.npy'),
        'valid_mask': np.load(frame_dir / 'pseudo_valid_mask_verify_v1.npy').astype(np.float32),
        'meta_path': frame_dir / 'pseudo_verify_meta_v1.json',
    }


def _load_brpo_style_observation_bundle(frame_dir: Path, version: str = 'v1'):
    if version not in {'v1', 'v2'}:
        raise ValueError(f'Unsupported BRPO-style observation bundle version={version}')
    suffix = f'brpo_style_{version}'
    return {
        'confidence_joint': np.load(frame_dir / f'pseudo_confidence_{suffix}.npy').astype(np.float32),
        'depth_target': np.load(frame_dir / f'pseudo_depth_target_{suffix}.npy').astype(np.float32),
        'source_map': np.load(frame_dir / f'pseudo_source_map_{suffix}.npy'),
        'valid_mask': np.load(frame_dir / f'pseudo_valid_mask_{suffix}.npy').astype(np.float32),
        'meta_path': frame_dir / f'brpo_style_observation_meta_{version}.json',
    }


def _load_brpo_direct_observation_bundle(frame_dir: Path):
    return {
        'confidence_joint': np.load(frame_dir / 'pseudo_confidence_brpo_direct_v1.npy').astype(np.float32),
        'depth_target': np.load(frame_dir / 'pseudo_depth_target_brpo_direct_v1.npy').astype(np.float32),
        'source_map': np.load(frame_dir / 'pseudo_source_map_brpo_direct_v1.npy'),
        'valid_mask': np.load(frame_dir / 'pseudo_valid_mask_brpo_direct_v1.npy').astype(np.float32),
        'meta_path': frame_dir / 'brpo_direct_observation_meta_v1.json',
    }


def _load_named_basic_observation_bundle(frame_dir: Path, prefix: str, meta_filename: str):
    return {
        'confidence_joint': np.load(frame_dir / f'pseudo_confidence_{prefix}.npy').astype(np.float32),
        'depth_target': np.load(frame_dir / f'pseudo_depth_target_{prefix}.npy').astype(np.float32),
        'source_map': np.load(frame_dir / f'pseudo_source_map_{prefix}.npy'),
        'valid_mask': np.load(frame_dir / f'pseudo_valid_mask_{prefix}.npy').astype(np.float32),
        'meta_path': frame_dir / meta_filename,
    }


def compute_stageA_scene_scale(sample_dir: Path, conf_arr: np.ndarray, scale_source: str, fixed_scale: float):
    if scale_source == 'fixed':
        return max(float(fixed_scale), 1e-6), 'fixed'

    render_depth_path = sample_dir / 'render_depth.npy'
    if not render_depth_path.exists():
        return max(float(fixed_scale), 1e-6), 'fixed_no_render_depth'

    render_depth = np.load(render_depth_path).astype(np.float32)
    valid = render_depth > 1e-6
    if scale_source == 'render_depth_trainmask_median':
        valid = valid & (conf_arr > 0)

    vals = render_depth[valid]
    if vals.size == 0:
        return max(float(fixed_scale), 1e-6), 'fixed_empty_valid'

    return max(float(np.median(vals)), 1e-6), scale_source


def _grad_group_norm(params):
    total = 0.0
    for p in params:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return float(total ** 0.5)


def _collect_view_param_groups(sampled_views):
    rot = []
    trans = []
    exp_a = []
    exp_b = []
    seen = set()
    for view in sampled_views:
        sid = int(view['sample_id'])
        if sid in seen:
            continue
        seen.add(sid)
        vp = view['vp']
        rot.append(vp.cam_rot_delta)
        trans.append(vp.cam_trans_delta)
        exp_a.append(vp.exposure_a)
        exp_b.append(vp.exposure_b)
    return {'rot': rot, 'trans': trans, 'exp_a': exp_a, 'exp_b': exp_b}


def diagnose_grad_contrib_for_sampled_views(args, cfg, sampled_views, gaussians, pipeline_params, bg):
    from gaussian_renderer import render as gs_render
    term_accum = {}
    for view in sampled_views:
        pkg = gs_render(view['vp'], gaussians, pipeline_params, bg)
        if args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
            _, _, terms = build_stageA_loss_source_aware(
                render_rgb=pkg['render'],
                render_depth=pkg['depth'],
                target_rgb=view['rgb'],
                target_depth=view['depth_for_refine'],
                confidence_mask=view['conf'],
                depth_source_map=view['target_depth_source_map'],
                viewpoint=view['vp'],
                beta_rgb=cfg.beta_rgb,
                lambda_pose=cfg.lambda_pose,
                lambda_exp=cfg.lambda_exp,
                trans_weight=cfg.trans_reg_weight,
                lambda_depth_seed=args.stageA_lambda_depth_seed,
                lambda_depth_dense=args.stageA_lambda_depth_dense,
                lambda_depth_fallback=args.stageA_lambda_depth_fallback,
                use_depth=not args.stageA_disable_depth,
                lambda_abs_pose=cfg.lambda_abs_pose,
                lambda_abs_t=cfg.lambda_abs_t,
                lambda_abs_r=cfg.lambda_abs_r,
                abs_pose_robust=cfg.abs_pose_robust,
                scene_scale=view.get('stageA_scene_scale', 1.0),
                depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0],
                depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1],
                depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                return_terms=True,
            )
        else:
            _, _, terms = build_stageA_loss(
                render_rgb=pkg['render'],
                render_depth=pkg['depth'],
                target_rgb=view['rgb'],
                target_depth=view['depth_for_refine'],
                confidence_mask=view['conf'],
                viewpoint=view['vp'],
                beta_rgb=cfg.beta_rgb,
                lambda_pose=cfg.lambda_pose,
                lambda_exp=cfg.lambda_exp,
                trans_weight=cfg.trans_reg_weight,
                use_depth=not args.stageA_disable_depth,
                lambda_abs_pose=cfg.lambda_abs_pose,
                lambda_abs_t=cfg.lambda_abs_t,
                lambda_abs_r=cfg.lambda_abs_r,
                abs_pose_robust=cfg.abs_pose_robust,
                scene_scale=view.get('stageA_scene_scale', 1.0),
                depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0],
                depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1],
                depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                return_terms=True,
            )
        for k, v in terms.items():
            term_accum.setdefault(k, []).append(v)

    term_mean = {k: sum(v) / float(len(v)) for k, v in term_accum.items()}
    groups = _collect_view_param_groups(sampled_views)
    out = {}
    for term_name, term_loss in term_mean.items():
        for plist in groups.values():
            for pp in plist:
                if pp.grad is not None:
                    pp.grad.zero_()
        if term_loss.requires_grad:
            term_loss.backward(retain_graph=True)
        out[term_name] = {
            'grad_norm_rot': _grad_group_norm(groups['rot']),
            'grad_norm_trans': _grad_group_norm(groups['trans']),
            'grad_norm_exp_a': _grad_group_norm(groups['exp_a']),
            'grad_norm_exp_b': _grad_group_norm(groups['exp_b']),
            'loss_value': float(term_loss.detach().item()),
        }
    for plist in groups.values():
        for pp in plist:
            if pp.grad is not None:
                pp.grad.zero_()
    return out


def load_stageA_pseudo_views(args):
    v1 = load_v1_module()
    pseudo_views, pseudo_manifest_info = v1.load_pseudo_viewpoints(args.pseudo_cache, args.target_side, args.confidence_mask_source, args.brpo_mask_root)
    for view in pseudo_views:
        sample_dir = Path(view['target_rgb_path']).parent
        frame_id = int(view.get('frame_id', view.get('sample_id')))
        source_meta_path = sample_dir / 'source_meta.json'
        source_meta = json.load(open(source_meta_path)) if source_meta_path.exists() else {}
        signal_v2_frame_dir = _signal_v2_frame_dir(sample_dir, frame_id, args.signal_v2_root)

        pseudo_observation_mode = str(getattr(args, 'pseudo_observation_mode', 'off') or 'off')
        if pseudo_observation_mode == 'exact_brpo_full_target_v1' and args.stageA_depth_loss_mode != 'legacy':
            raise ValueError('exact_brpo_full_target_v1 is reserved for full BRPO target-side semantics and must run with --stageA_depth_loss_mode legacy so RGB/depth share the same C_m without source-aware fallback tiers')
        if pseudo_observation_mode == 'exact_brpo_upstream_target_v1' and args.stageA_depth_loss_mode != 'exact_shared_cm_v1':
            raise ValueError('exact_brpo_upstream_target_v1 must run with --stageA_depth_loss_mode exact_shared_cm_v1 to enforce exact loss contract')
        rgb_mask_mode = args.stageA_rgb_mask_mode
        depth_mask_mode = args.stageA_depth_mask_mode
        depth_target_mode = args.stageA_target_depth_mode
        if args.signal_pipeline == 'brpo_v2':
            if rgb_mask_mode == 'auto':
                rgb_mask_mode = 'brpo_v2_raw' if args.stageA_confidence_variant != 'continuous' else 'brpo_v2_cont'
            if depth_mask_mode == 'auto':
                depth_mask_mode = 'brpo_v2_depth'
            if depth_target_mode in {'auto', 'blended_depth', 'blended_depth_m5'}:
                depth_target_mode = 'brpo_v2'

        if pseudo_observation_mode in {'brpo_joint_v1', 'brpo_verify_v1', 'brpo_style_v1', 'brpo_style_v2', 'brpo_direct_v1', 'hybrid_brpo_cm_geo_v1', 'exact_brpo_cm_old_target_v1', 'exact_brpo_full_target_v1', 'exact_brpo_cm_hybrid_target_v1', 'exact_brpo_cm_stable_target_v1', 'exact_brpo_upstream_target_v1'}:
            if signal_v2_frame_dir is None:
                raise FileNotFoundError(f'No signal_v2 frame dir found for frame_id={frame_id} under sample_dir={sample_dir}')
            if pseudo_observation_mode == 'brpo_joint_v1':
                bundle = _load_joint_observation_bundle(signal_v2_frame_dir)
                rgb_conf = bundle['confidence_joint']
                depth_conf = bundle['confidence_joint']
                rgb_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_joint_v1.npy')
                depth_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_joint_v1.npy')
                rgb_conf_kind = 'joint_observation::pseudo_confidence_joint_v1.npy'
                depth_conf_kind = 'joint_observation::pseudo_confidence_joint_v1.npy'
                rgb_conf_mode = 'pseudo_observation::brpo_joint_v1'
                depth_conf_mode = 'pseudo_observation::brpo_joint_v1'
                depth_meta_path = bundle['meta_path']
                depth_kind = 'joint_depth_target_joint_v1'
                depth_for_refine = signal_v2_frame_dir / 'pseudo_depth_target_joint_v1.npy'
                source_map_path = signal_v2_frame_dir / 'pseudo_source_map_joint_v1.npy'
            elif pseudo_observation_mode == 'brpo_verify_v1':
                bundle = _load_verify_observation_bundle(signal_v2_frame_dir)
                rgb_conf = bundle['confidence_verify']
                depth_conf = bundle['confidence_verify']
                rgb_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_verify_v1.npy')
                depth_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_verify_v1.npy')
                rgb_conf_kind = 'joint_observation::pseudo_confidence_verify_v1.npy'
                depth_conf_kind = 'joint_observation::pseudo_confidence_verify_v1.npy'
                rgb_conf_mode = 'pseudo_observation::brpo_verify_v1'
                depth_conf_mode = 'pseudo_observation::brpo_verify_v1'
                depth_meta_path = bundle['meta_path']
                depth_kind = 'joint_depth_target_joint_v1'
                depth_for_refine = signal_v2_frame_dir / 'pseudo_depth_target_joint_v1.npy'
                source_map_path = signal_v2_frame_dir / 'pseudo_source_map_joint_v1.npy'
            elif pseudo_observation_mode in {'brpo_style_v1', 'brpo_style_v2'}:
                brpo_style_version = 'v2' if pseudo_observation_mode == 'brpo_style_v2' else 'v1'
                bundle = _load_brpo_style_observation_bundle(signal_v2_frame_dir, version=brpo_style_version)
                suffix = f'brpo_style_{brpo_style_version}'
                rgb_conf = bundle['confidence_joint']
                depth_conf = bundle['confidence_joint']
                rgb_conf_path = str(signal_v2_frame_dir / f'pseudo_confidence_{suffix}.npy')
                depth_conf_path = str(signal_v2_frame_dir / f'pseudo_confidence_{suffix}.npy')
                rgb_conf_kind = f'joint_observation::pseudo_confidence_{suffix}.npy'
                depth_conf_kind = f'joint_observation::pseudo_confidence_{suffix}.npy'
                rgb_conf_mode = f'pseudo_observation::{suffix}'
                depth_conf_mode = f'pseudo_observation::{suffix}'
                depth_meta_path = bundle['meta_path']
                depth_kind = f'joint_depth_target_{suffix}'
                depth_for_refine = signal_v2_frame_dir / f'pseudo_depth_target_{suffix}.npy'
                source_map_path = signal_v2_frame_dir / f'pseudo_source_map_{suffix}.npy'
            elif pseudo_observation_mode in {'brpo_direct_v1', 'hybrid_brpo_cm_geo_v1'}:
                bundle = _load_brpo_direct_observation_bundle(signal_v2_frame_dir)
                rgb_conf = bundle['confidence_joint']
                depth_conf = bundle['confidence_joint']
                rgb_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_brpo_direct_v1.npy')
                depth_conf_path = str(signal_v2_frame_dir / 'pseudo_confidence_brpo_direct_v1.npy')
                rgb_conf_kind = 'joint_observation::pseudo_confidence_brpo_direct_v1.npy'
                depth_conf_kind = 'joint_observation::pseudo_confidence_brpo_direct_v1.npy'
                rgb_conf_mode = f'pseudo_observation::{pseudo_observation_mode}'
                depth_conf_mode = f'pseudo_observation::{pseudo_observation_mode}'
                depth_meta_path = bundle['meta_path']
                depth_kind = 'joint_depth_target_hybrid_brpo_cm_geo_v1' if pseudo_observation_mode == 'hybrid_brpo_cm_geo_v1' else 'joint_depth_target_brpo_direct_v1'
                depth_for_refine = signal_v2_frame_dir / 'pseudo_depth_target_brpo_direct_v1.npy'
                source_map_path = signal_v2_frame_dir / 'pseudo_source_map_brpo_direct_v1.npy'
            else:
                exact_prefix_map = {
                    'exact_brpo_cm_old_target_v1': ('exact_brpo_cm_old_target_v1', 'exact_brpo_cm_old_target_meta_v1.json'),
                    'exact_brpo_full_target_v1': ('exact_brpo_full_target_v1', 'exact_brpo_full_target_meta_v1.json'),
                    'exact_brpo_cm_hybrid_target_v1': ('exact_brpo_cm_hybrid_target_v1', 'exact_brpo_cm_hybrid_target_meta_v1.json'),
                    'exact_brpo_cm_stable_target_v1': ('exact_brpo_cm_stable_target_v1', 'exact_brpo_cm_stable_target_meta_v1.json'),
                    'exact_brpo_upstream_target_v1': ('exact_brpo_upstream_target_v1', 'exact_brpo_upstream_target_observation_meta_v1.json'),
                }
                prefix, meta_filename = exact_prefix_map[pseudo_observation_mode]
                bundle = _load_named_basic_observation_bundle(signal_v2_frame_dir, prefix=prefix, meta_filename=meta_filename)
                rgb_conf = bundle['confidence_joint']
                depth_conf = bundle['confidence_joint']
                rgb_conf_path = str(signal_v2_frame_dir / f'pseudo_confidence_{prefix}.npy')
                depth_conf_path = str(signal_v2_frame_dir / f'pseudo_confidence_{prefix}.npy')
                rgb_conf_kind = f'joint_observation::pseudo_confidence_{prefix}.npy'
                depth_conf_kind = f'joint_observation::pseudo_confidence_{prefix}.npy'
                rgb_conf_mode = f'pseudo_observation::{pseudo_observation_mode}'
                depth_conf_mode = f'pseudo_observation::{pseudo_observation_mode}'
                depth_meta_path = bundle['meta_path']
                depth_kind = f'joint_depth_target_{prefix}'
                depth_for_refine = signal_v2_frame_dir / f'pseudo_depth_target_{prefix}.npy'
                source_map_path = signal_v2_frame_dir / f'pseudo_source_map_{prefix}.npy'
                if pseudo_observation_mode == 'exact_brpo_upstream_target_v1':
                    target_confidence_path = signal_v2_frame_dir / f'pseudo_target_confidence_{prefix}.npy'
                    target_confidence = np.load(target_confidence_path).astype(np.float32) if target_confidence_path.exists() else bundle['confidence_joint']
                    view['exact_upstream_bundle'] = {'valid_mask': bundle['valid_mask'], 'target_confidence': target_confidence}
                else:
                    view.pop('exact_upstream_bundle', None)
            rgb_conf_variant = 'continuous'
            depth_conf_variant = 'continuous'
            depth_arr = bundle['depth_target']
            depth_meta = json.load(open(depth_meta_path)) if depth_meta_path.exists() else {}
            source_map = bundle['source_map']
            source_summary = _summarize_source_map(source_map, kind=depth_kind)
            view['pseudo_observation_mode_effective'] = pseudo_observation_mode
            view['pseudo_observation_meta_path'] = str(depth_meta_path)
            view['depth_source_id_groups'] = {'seed': [1, 2, 3], 'dense': [], 'fallback': [4]}
        else:
            rgb_conf, rgb_conf_path, rgb_conf_kind, rgb_conf_mode, rgb_conf_variant = resolve_stageA_mask(
                sample_dir, frame_id, args.target_side, rgb_mask_mode, args.stageA_confidence_variant,
                view.get('conf'), view.get('confidence_path'), view.get('confidence_source_kind'), legacy_mode=args.stageA_mask_mode, signal_v2_root=args.signal_v2_root,
            )
            depth_conf, depth_conf_path, depth_conf_kind, depth_conf_mode, depth_conf_variant = resolve_stageA_mask(
                sample_dir, frame_id, args.target_side, depth_mask_mode, 'discrete',
                view.get('conf'), view.get('confidence_path'), view.get('confidence_source_kind'), legacy_mode=args.stageA_mask_mode, signal_v2_root=args.signal_v2_root,
            )
            depth_kind, depth_for_refine, source_map_path = resolve_depth_target_path(sample_dir, depth_target_mode, frame_id=frame_id, signal_v2_root=args.signal_v2_root)
            depth_arr = np.load(depth_for_refine).astype(np.float32)
            if depth_kind == 'target_depth_for_refine_v2_brpo':
                depth_meta_path = (signal_v2_frame_dir / 'depth_meta_v2_brpo.json') if signal_v2_frame_dir is not None else None
            elif depth_kind == 'joint_depth_target_v2':
                depth_meta_path = (signal_v2_frame_dir / 'joint_meta_v2.json') if signal_v2_frame_dir is not None else None
            elif depth_kind == 'joint_depth_target_joint_v1':
                depth_meta_path = (signal_v2_frame_dir / 'joint_observation_meta_v1.json') if signal_v2_frame_dir is not None else None
            elif depth_kind == 'joint_depth_target_expand_v1':
                depth_meta_path = (signal_v2_frame_dir / 'joint_expand_meta_v1.json') if signal_v2_frame_dir is not None else None
            else:
                depth_meta_path = sample_dir / 'target_depth_for_refine_meta.json'
            depth_meta = json.load(open(depth_meta_path)) if depth_meta_path is not None and depth_meta_path.exists() else {}
            source_map = np.load(source_map_path) if source_map_path is not None and source_map_path.exists() else None
            source_summary = _summarize_source_map(source_map, kind=depth_kind)
            view['pseudo_observation_mode_effective'] = 'off'
            view['pseudo_observation_meta_path'] = None
            view['depth_source_id_groups'] = {'seed': [1, 2, 3], 'dense': [4], 'fallback': [0]}

        rgb_positive = rgb_conf[rgb_conf > 0]
        depth_positive = depth_conf[depth_conf > 0]
        view['conf'] = rgb_conf.astype(np.float32)
        view['conf_rgb'] = rgb_conf.astype(np.float32)
        view['conf_depth'] = depth_conf.astype(np.float32)
        view['confidence_path'] = rgb_conf_path
        view['confidence_source_kind'] = rgb_conf_kind
        view['stageA_mask_mode_effective'] = rgb_conf_mode
        view['stageA_rgb_mask_mode_effective'] = rgb_conf_mode
        view['stageA_depth_mask_mode_effective'] = depth_conf_mode
        view['stageA_rgb_confidence_variant_effective'] = rgb_conf_variant
        view['stageA_depth_confidence_variant_effective'] = depth_conf_variant
        view['rgb_confidence_path'] = rgb_conf_path
        view['depth_confidence_path'] = depth_conf_path
        view['rgb_confidence_source_kind'] = rgb_conf_kind
        view['depth_confidence_source_kind'] = depth_conf_kind
        view['confidence_nonzero_ratio'] = float((rgb_conf > 0).sum() / rgb_conf.size)
        view['confidence_mean_positive'] = float(rgb_positive.mean()) if rgb_positive.size > 0 else 0.0
        view['rgb_confidence_nonzero_ratio'] = float((rgb_conf > 0).sum() / rgb_conf.size)
        view['rgb_confidence_mean_positive'] = float(rgb_positive.mean()) if rgb_positive.size > 0 else 0.0
        view['depth_confidence_nonzero_ratio'] = float((depth_conf > 0).sum() / depth_conf.size)
        view['depth_confidence_mean_positive'] = float(depth_positive.mean()) if depth_positive.size > 0 else 0.0

        view['source_meta'] = source_meta
        view['depth_meta'] = depth_meta
        view['target_depth_for_refine_kind'] = depth_kind
        view['stageA_target_depth_mode_effective'] = ('joint_depth_v1' if pseudo_observation_mode in {'brpo_joint_v1', 'brpo_verify_v1'} else (pseudo_observation_mode if pseudo_observation_mode in {'brpo_style_v1', 'brpo_style_v2', 'brpo_direct_v1', 'hybrid_brpo_cm_geo_v1', 'exact_brpo_cm_old_target_v1', 'exact_brpo_cm_hybrid_target_v1', 'exact_brpo_cm_stable_target_v1'} else _resolve_depth_mode_alias(depth_target_mode)))
        view['target_depth_for_refine_path'] = str(depth_for_refine)
        view['target_depth_for_refine_source_map_path'] = str(source_map_path) if source_map_path is not None and Path(source_map_path).exists() else None
        view['target_depth_source_map'] = source_map
        view['signal_pipeline'] = args.signal_pipeline
        view['signal_v2_frame_dir'] = str(signal_v2_frame_dir) if signal_v2_frame_dir is not None else None
        view['depth_for_refine'] = depth_arr
        scene_scale, scene_scale_src = compute_stageA_scene_scale(
            sample_dir=sample_dir,
            conf_arr=depth_conf,
            scale_source=args.stageA_abs_pose_scale_source,
            fixed_scale=args.stageA_abs_pose_fixed_scale,
        )
        view['stageA_scene_scale'] = float(scene_scale)
        view['stageA_scene_scale_source_effective'] = scene_scale_src
        view['target_depth_nonzero_ratio'] = float((depth_arr > 1e-6).sum() / depth_arr.size)
        view['target_depth_verified_ratio'] = source_summary['verified_ratio'] if source_summary['verified_ratio'] is not None else depth_meta.get('verified_ratio')
        view['target_depth_render_fallback_ratio'] = source_summary['render_fallback_ratio'] if source_summary['render_fallback_ratio'] is not None else depth_meta.get('render_fallback_ratio')
        view['target_depth_seed_ratio'] = source_summary['seed_ratio']
        view['target_depth_dense_ratio'] = source_summary['dense_ratio']
        make_viewpoint_trainable(view['vp'])
    return pseudo_views, pseudo_manifest_info


def initialize_pseudo_views_from_saved_states(pseudo_views, init_states_json: str | None, reference_mode: str):
    summary = {
        'requested': bool(init_states_json),
        'source_json': init_states_json,
        'reference_mode': reference_mode,
        'loaded_count': 0,
        'missing_sample_ids': [],
        'frame_id_mismatches': [],
    }
    if not init_states_json:
        return summary
    loaded = load_exported_view_states(init_states_json)
    for view in pseudo_views:
        sid = int(view['sample_id'])
        if sid not in loaded:
            summary['missing_sample_ids'].append(sid)
            continue
        state = loaded[sid]
        if int(state.get('frame_id', sid)) != int(view.get('frame_id', sid)):
            summary['frame_id_mismatches'].append({
                'sample_id': sid,
                'loaded_frame_id': int(state.get('frame_id', sid)),
                'current_frame_id': int(view.get('frame_id', sid)),
            })
        apply_loaded_view_state_(view['vp'], state, reference_mode=reference_mode)
        summary['loaded_count'] += 1
    return summary


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def build_pseudo_local_gating_cfg(args) -> PseudoLocalGatingConfig:
    return PseudoLocalGatingConfig(
        mode=args.pseudo_local_gating,
        params=args.pseudo_local_gating_params,
        min_verified_ratio=float(args.pseudo_local_gating_min_verified_ratio),
        min_rgb_mask_ratio=float(args.pseudo_local_gating_min_rgb_mask_ratio),
        max_fallback_ratio=float(args.pseudo_local_gating_max_fallback_ratio),
        min_correction=float(args.pseudo_local_gating_min_correction),
        soft_power=float(args.pseudo_local_gating_soft_power),
        log_interval=int(args.pseudo_local_gating_log_interval),
        # SPGM-specific hyperparameters
        spgm_num_clusters=int(args.pseudo_local_gating_spgm_num_clusters),
        spgm_alpha_depth=float(args.pseudo_local_gating_spgm_alpha_depth),
        spgm_beta_entropy=float(args.pseudo_local_gating_spgm_beta_entropy),
        spgm_gamma_entropy=float(args.pseudo_local_gating_spgm_gamma_entropy),
        spgm_support_eta=float(args.pseudo_local_gating_spgm_support_eta),
        spgm_weight_floor=float(args.pseudo_local_gating_spgm_weight_floor),
        spgm_entropy_bins=int(args.pseudo_local_gating_spgm_entropy_bins),
        spgm_density_mode=args.pseudo_local_gating_spgm_density_mode,
        spgm_cluster_keep_near=float(args.pseudo_local_gating_spgm_cluster_keep_near),
        spgm_cluster_keep_mid=float(args.pseudo_local_gating_spgm_cluster_keep_mid),
        spgm_cluster_keep_far=float(args.pseudo_local_gating_spgm_cluster_keep_far),
        spgm_policy_mode=args.pseudo_local_gating_spgm_policy_mode,
        spgm_ranking_mode=args.pseudo_local_gating_spgm_ranking_mode,
        spgm_lambda_support_rank=float(args.pseudo_local_gating_spgm_lambda_support_rank),
        spgm_selector_keep_ratio_near=float(args.pseudo_local_gating_spgm_selector_keep_ratio_near),
        spgm_selector_keep_ratio_mid=float(args.pseudo_local_gating_spgm_selector_keep_ratio_mid),
        spgm_selector_keep_ratio_far=float(args.pseudo_local_gating_spgm_selector_keep_ratio_far),
        spgm_selector_min_keep=int(args.pseudo_local_gating_spgm_selector_min_keep),
        spgm_manager_mode=args.pseudo_local_gating_spgm_manager_mode,
        spgm_state_candidate_quantile=float(args.pseudo_local_gating_spgm_state_candidate_quantile),
        spgm_state_base_scale_near=float(args.pseudo_local_gating_spgm_state_base_scale_near),
        spgm_state_base_scale_mid=float(args.pseudo_local_gating_spgm_state_base_scale_mid),
        spgm_state_base_scale_far=float(args.pseudo_local_gating_spgm_state_base_scale_far),
        spgm_state_participation_keep_near=float(args.pseudo_local_gating_spgm_state_participation_keep_near),
        spgm_state_participation_keep_mid=float(args.pseudo_local_gating_spgm_state_participation_keep_mid),
        spgm_state_participation_keep_far=float(args.pseudo_local_gating_spgm_state_participation_keep_far),
        spgm_state_opacity_floor_near=float(args.pseudo_local_gating_spgm_state_opacity_floor_near),
        spgm_state_opacity_floor_mid=float(args.pseudo_local_gating_spgm_state_opacity_floor_mid),
        spgm_state_opacity_floor_far=float(args.pseudo_local_gating_spgm_state_opacity_floor_far),
    )


def _grad_norm_from_param(param) -> float:
    if param is None or getattr(param, 'grad', None) is None:
        return 0.0
    return float(torch.norm(param.grad).detach().cpu().item())


def _build_spgm_candidate_mask(cluster_id: torch.Tensor, control_mask: torch.Tensor, score: torch.Tensor | None, quantile: float) -> torch.Tensor:
    candidate_mask = torch.zeros_like(control_mask, dtype=torch.bool)
    if score is None or not bool(control_mask.any()):
        return candidate_mask
    q = float(min(max(quantile, 0.0), 1.0))
    active_cluster_ids = torch.unique(cluster_id[control_mask])
    for raw_cluster in active_cluster_ids.tolist():
        cid = int(raw_cluster)
        mask = control_mask & (cluster_id == cid)
        if not bool(mask.any()):
            continue
        values = score[mask]
        threshold = float(torch.quantile(values, q=q).item())
        cur = mask.clone()
        cur[mask] = values <= threshold
        candidate_mask |= cur
    return candidate_mask


def _masked_mean_tensor(values: torch.Tensor | None, mask: torch.Tensor) -> float:
    if values is None or not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().item())


def _masked_cluster_ratio(cluster_id: torch.Tensor, mask: torch.Tensor, cid: int) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((cluster_id[mask] == int(cid)).float().mean().item())


def _build_spgm_c0_partition_stats(spgm_score: dict, active_mask: torch.Tensor, candidate_quantile: float) -> dict:
    cluster_id = spgm_score['cluster_id']
    state_score = spgm_score['state_score']
    participation_score = spgm_score['participation_score']
    ranking_score = spgm_score['ranking_score']
    support_norm = spgm_score['support_norm']
    population_support_norm = spgm_score['population_support_norm']
    depth_score = spgm_score['depth_score']

    state_candidate = _build_spgm_candidate_mask(cluster_id, active_mask, state_score, candidate_quantile)
    part_candidate = _build_spgm_candidate_mask(cluster_id, active_mask, participation_score, candidate_quantile)

    partitions = {
        'state_only': state_candidate & ~part_candidate,
        'part_only': part_candidate & ~state_candidate,
        'both': state_candidate & part_candidate,
        'neither': active_mask & ~(state_candidate | part_candidate),
    }

    active_count = int(active_mask.sum().item())
    out = {}
    for name, mask in partitions.items():
        ratio = float(mask.sum().item() / active_count) if active_count > 0 else 0.0
        out[f'c0_{name}_ratio'] = ratio
        out[f'c0_{name}_ranking_mean'] = _masked_mean_tensor(ranking_score, mask)
        out[f'c0_{name}_state_mean'] = _masked_mean_tensor(state_score, mask)
        out[f'c0_{name}_participation_mean'] = _masked_mean_tensor(participation_score, mask)
        out[f'c0_{name}_support_norm_mean'] = _masked_mean_tensor(support_norm, mask)
        out[f'c0_{name}_population_support_norm_mean'] = _masked_mean_tensor(population_support_norm, mask)
        out[f'c0_{name}_depth_score_mean'] = _masked_mean_tensor(depth_score, mask)
        out[f'c0_{name}_near_ratio'] = _masked_cluster_ratio(cluster_id, mask, 0)
        out[f'c0_{name}_mid_ratio'] = _masked_cluster_ratio(cluster_id, mask, 1)
        out[f'c0_{name}_far_ratio'] = _masked_cluster_ratio(cluster_id, mask, 2)
    return out


def _init_gating_history_lists(container: dict) -> None:
    container.update(
        {
            'pseudo_local_gating_mode': [],
            'pseudo_local_gating_params': [],
            'accepted_pseudo_sample_ids': [],
            'rejected_pseudo_sample_ids': [],
            'rejected_reasons': [],
            'accepted_signal_weights': [],
            'sample_signal_metrics': [],
            # Legacy visibility-union fields
            'visible_union_ratio': [],
            'visible_union_weight_mean': [],
            'accepted_visibility_count': [],
            # Grad stats (common)
            'grad_keep_ratio_xyz': [],
            'grad_keep_ratio_opacity': [],
            'grad_weight_mean_xyz': [],
            'grad_weight_mean_opacity': [],
            'grad_norm_xyz_pre_mask': [],
            'grad_norm_xyz_post_mask': [],
            'grad_norm_opacity_pre_mask': [],
            'grad_norm_opacity_post_mask': [],
            # SPGM-specific fields
            'spgm_active_ratio': [],
            'spgm_selected_ratio': [],
            'spgm_policy_mode_effective': [],
            'spgm_ranking_mode_effective': [],
            'spgm_accepted_view_count': [],
            'spgm_support_mean': [],
            'spgm_support_p50': [],
            'spgm_support_max': [],
            'spgm_depth_entropy': [],
            'spgm_density_entropy': [],
            'spgm_importance_mean': [],
            'spgm_importance_p50': [],
            'spgm_ranking_score_mean': [],
            'spgm_ranking_score_p50': [],
            'spgm_support_norm_mean': [],
            'spgm_weight_mean': [],
            'spgm_weight_p10': [],
            'spgm_weight_p50': [],
            'spgm_weight_p90': [],
            'spgm_cluster_count_near': [],
            'spgm_cluster_count_mid': [],
            'spgm_cluster_count_far': [],
            'spgm_selected_count_near': [],
            'spgm_selected_count_mid': [],
            'spgm_selected_count_far': [],
            'spgm_density_mode_effective': [],
            'spgm_state_score_mean': [],
            'spgm_state_score_p50': [],
            'spgm_participation_score_mean': [],
            'spgm_participation_score_p50': [],
            'spgm_state_score_mean_near': [],
            'spgm_state_score_mean_mid': [],
            'spgm_state_score_mean_far': [],
            'spgm_state_score_p50_near': [],
            'spgm_state_score_p50_mid': [],
            'spgm_state_score_p50_far': [],
            'spgm_participation_score_mean_near': [],
            'spgm_participation_score_mean_mid': [],
            'spgm_participation_score_mean_far': [],
            'spgm_participation_score_p50_near': [],
            'spgm_participation_score_p50_mid': [],
            'spgm_participation_score_p50_far': [],
            'spgm_population_support_mean': [],
            'spgm_struct_density_mean': [],
            'spgm_manager_mode_effective': [],
            'spgm_state_action_applied': [],
            'spgm_state_xyz_scale_mean': [],
            'spgm_state_xyz_scale_mean_near': [],
            'spgm_state_xyz_scale_mean_mid': [],
            'spgm_state_xyz_scale_mean_far': [],
            'spgm_state_grad_norm_xyz_pre_state': [],
            'spgm_state_grad_norm_xyz_post_state': [],
            'spgm_state_lr_scale_near': [],
            'spgm_state_lr_scale_mid': [],
            'spgm_state_lr_scale_far': [],
            'spgm_state_opacity_decay_near': [],
            'spgm_state_opacity_decay_mid': [],
            'spgm_state_opacity_decay_far': [],
            'spgm_state_opacity_scale_mean': [],
            'spgm_state_opacity_scale_mean_near': [],
            'spgm_state_opacity_scale_mean_mid': [],
            'spgm_state_opacity_scale_mean_far': [],
            'spgm_state_candidate_count_near': [],
            'spgm_state_candidate_count_mid': [],
            'spgm_state_candidate_count_far': [],
            'spgm_state_participation_ratio': [],
            'spgm_state_participation_ratio_near': [],
            'spgm_state_participation_ratio_mid': [],
            'spgm_state_participation_ratio_far': [],
            'spgm_state_participation_drop_count_near': [],
            'spgm_state_participation_drop_count_mid': [],
            'spgm_state_participation_drop_count_far': [],
            'spgm_c0_state_only_ratio': [],
            'spgm_c0_state_only_ranking_mean': [],
            'spgm_c0_state_only_state_mean': [],
            'spgm_c0_state_only_participation_mean': [],
            'spgm_c0_state_only_support_norm_mean': [],
            'spgm_c0_state_only_population_support_norm_mean': [],
            'spgm_c0_state_only_depth_score_mean': [],
            'spgm_c0_state_only_near_ratio': [],
            'spgm_c0_state_only_mid_ratio': [],
            'spgm_c0_state_only_far_ratio': [],
            'spgm_c0_part_only_ratio': [],
            'spgm_c0_part_only_ranking_mean': [],
            'spgm_c0_part_only_state_mean': [],
            'spgm_c0_part_only_participation_mean': [],
            'spgm_c0_part_only_support_norm_mean': [],
            'spgm_c0_part_only_population_support_norm_mean': [],
            'spgm_c0_part_only_depth_score_mean': [],
            'spgm_c0_part_only_near_ratio': [],
            'spgm_c0_part_only_mid_ratio': [],
            'spgm_c0_part_only_far_ratio': [],
            'spgm_c0_both_ratio': [],
            'spgm_c0_both_ranking_mean': [],
            'spgm_c0_both_state_mean': [],
            'spgm_c0_both_participation_mean': [],
            'spgm_c0_both_support_norm_mean': [],
            'spgm_c0_both_population_support_norm_mean': [],
            'spgm_c0_both_depth_score_mean': [],
            'spgm_c0_both_near_ratio': [],
            'spgm_c0_both_mid_ratio': [],
            'spgm_c0_both_far_ratio': [],
            'spgm_c0_neither_ratio': [],
            'spgm_c0_neither_ranking_mean': [],
            'spgm_c0_neither_state_mean': [],
            'spgm_c0_neither_participation_mean': [],
            'spgm_c0_neither_support_norm_mean': [],
            'spgm_c0_neither_population_support_norm_mean': [],
            'spgm_c0_neither_depth_score_mean': [],
            'spgm_c0_neither_near_ratio': [],
            'spgm_c0_neither_mid_ratio': [],
            'spgm_c0_neither_far_ratio': [],
        }
    )


def _append_gating_history(container: dict, summary: dict) -> None:
    keys = [
        'pseudo_local_gating_mode',
        'pseudo_local_gating_params',
        'accepted_pseudo_sample_ids',
        'rejected_pseudo_sample_ids',
        'rejected_reasons',
        'accepted_signal_weights',
        'sample_signal_metrics',
        # Legacy visibility-union fields
        'visible_union_ratio',
        'visible_union_weight_mean',
        'accepted_visibility_count',
        # Grad stats (common)
        'grad_keep_ratio_xyz',
        'grad_keep_ratio_opacity',
        'grad_weight_mean_xyz',
        'grad_weight_mean_opacity',
        'grad_norm_xyz_pre_mask',
        'grad_norm_xyz_post_mask',
        'grad_norm_opacity_pre_mask',
        'grad_norm_opacity_post_mask',
        # SPGM-specific fields
        'spgm_active_ratio',
        'spgm_selected_ratio',
        'spgm_policy_mode_effective',
        'spgm_ranking_mode_effective',
        'spgm_accepted_view_count',
        'spgm_support_mean',
        'spgm_support_p50',
        'spgm_support_max',
        'spgm_depth_entropy',
        'spgm_density_entropy',
        'spgm_importance_mean',
        'spgm_importance_p50',
        'spgm_ranking_score_mean',
        'spgm_ranking_score_p50',
        'spgm_support_norm_mean',
        'spgm_weight_mean',
        'spgm_weight_p10',
        'spgm_weight_p50',
        'spgm_weight_p90',
        'spgm_cluster_count_near',
        'spgm_cluster_count_mid',
        'spgm_cluster_count_far',
        'spgm_selected_count_near',
        'spgm_selected_count_mid',
        'spgm_selected_count_far',
        'spgm_density_mode_effective',
        'spgm_state_score_mean',
        'spgm_state_score_p50',
        'spgm_participation_score_mean',
        'spgm_participation_score_p50',
        'spgm_state_score_mean_near',
        'spgm_state_score_mean_mid',
        'spgm_state_score_mean_far',
        'spgm_state_score_p50_near',
        'spgm_state_score_p50_mid',
        'spgm_state_score_p50_far',
        'spgm_participation_score_mean_near',
        'spgm_participation_score_mean_mid',
        'spgm_participation_score_mean_far',
        'spgm_participation_score_p50_near',
        'spgm_participation_score_p50_mid',
        'spgm_participation_score_p50_far',
        'spgm_population_support_mean',
        'spgm_struct_density_mean',
        'spgm_manager_mode_effective',
        'spgm_state_action_applied',
        'spgm_state_xyz_scale_mean',
        'spgm_state_xyz_scale_mean_near',
        'spgm_state_xyz_scale_mean_mid',
        'spgm_state_xyz_scale_mean_far',
        'spgm_state_grad_norm_xyz_pre_state',
        'spgm_state_grad_norm_xyz_post_state',
        'spgm_state_lr_scale_near',
        'spgm_state_lr_scale_mid',
        'spgm_state_lr_scale_far',
        'spgm_state_opacity_decay_near',
        'spgm_state_opacity_decay_mid',
        'spgm_state_opacity_decay_far',
        'spgm_state_opacity_scale_mean',
        'spgm_state_opacity_scale_mean_near',
        'spgm_state_opacity_scale_mean_mid',
        'spgm_state_opacity_scale_mean_far',
        'spgm_state_candidate_count_near',
        'spgm_state_candidate_count_mid',
        'spgm_state_candidate_count_far',
        'spgm_state_participation_ratio',
        'spgm_state_participation_ratio_near',
        'spgm_state_participation_ratio_mid',
        'spgm_state_participation_ratio_far',
        'spgm_state_participation_drop_count_near',
        'spgm_state_participation_drop_count_mid',
        'spgm_state_participation_drop_count_far',
        'spgm_c0_state_only_ratio',
        'spgm_c0_state_only_ranking_mean',
        'spgm_c0_state_only_state_mean',
        'spgm_c0_state_only_participation_mean',
        'spgm_c0_state_only_support_norm_mean',
        'spgm_c0_state_only_population_support_norm_mean',
        'spgm_c0_state_only_depth_score_mean',
        'spgm_c0_state_only_near_ratio',
        'spgm_c0_state_only_mid_ratio',
        'spgm_c0_state_only_far_ratio',
        'spgm_c0_part_only_ratio',
        'spgm_c0_part_only_ranking_mean',
        'spgm_c0_part_only_state_mean',
        'spgm_c0_part_only_participation_mean',
        'spgm_c0_part_only_support_norm_mean',
        'spgm_c0_part_only_population_support_norm_mean',
        'spgm_c0_part_only_depth_score_mean',
        'spgm_c0_part_only_near_ratio',
        'spgm_c0_part_only_mid_ratio',
        'spgm_c0_part_only_far_ratio',
        'spgm_c0_both_ratio',
        'spgm_c0_both_ranking_mean',
        'spgm_c0_both_state_mean',
        'spgm_c0_both_participation_mean',
        'spgm_c0_both_support_norm_mean',
        'spgm_c0_both_population_support_norm_mean',
        'spgm_c0_both_depth_score_mean',
        'spgm_c0_both_near_ratio',
        'spgm_c0_both_mid_ratio',
        'spgm_c0_both_far_ratio',
        'spgm_c0_neither_ratio',
        'spgm_c0_neither_ranking_mean',
        'spgm_c0_neither_state_mean',
        'spgm_c0_neither_participation_mean',
        'spgm_c0_neither_support_norm_mean',
        'spgm_c0_neither_population_support_norm_mean',
        'spgm_c0_neither_depth_score_mean',
        'spgm_c0_neither_near_ratio',
        'spgm_c0_neither_mid_ratio',
        'spgm_c0_neither_far_ratio',
    ]
    for key in keys:
        container[key].append(summary.get(key))


def _build_spgm_passthrough_update_policy(spgm_raw_stats, spgm_score, policy_mode: str) -> dict:
    weights = torch.ones_like(spgm_score['importance_score'])
    return {
        'weights': weights,
        'weight_mean': 1.0,
        'weight_p10': 1.0,
        'weight_p50': 1.0,
        'weight_p90': 1.0,
        'active_ratio': spgm_raw_stats['active_ratio'],
        'selected_ratio': spgm_raw_stats['active_ratio'],
        'selected_count_near': spgm_score['cluster_count_near'],
        'selected_count_mid': spgm_score['cluster_count_mid'],
        'selected_count_far': spgm_score['cluster_count_far'],
        'policy_mode_effective': f"{policy_mode}:passthrough",
    }


def _build_spgm_summary_payload(spgm_raw_stats, spgm_score, spgm_update_policy, spgm_manager, accepted_view_count: int) -> dict:
    return {
        'active_ratio': spgm_raw_stats['active_ratio'],
        'selected_ratio': spgm_update_policy['selected_ratio'],
        'policy_mode_effective': spgm_update_policy['policy_mode_effective'],
        'ranking_mode_effective': spgm_score['ranking_mode_effective'],
        'accepted_view_count': int(accepted_view_count),
        'support_mean': spgm_raw_stats['support_mean'],
        'support_p50': spgm_raw_stats['support_p50'],
        'support_max': spgm_raw_stats['support_max'],
        'depth_entropy': spgm_score['depth_entropy'],
        'density_entropy': spgm_score['density_entropy'],
        'importance_mean': spgm_score['importance_mean'],
        'importance_p50': spgm_score['importance_p50'],
        'ranking_score_mean': spgm_score['ranking_score_mean'],
        'ranking_score_p50': spgm_score['ranking_score_p50'],
        'support_norm_mean': spgm_score['support_norm_mean'],
        'weight_mean': spgm_update_policy['weight_mean'],
        'weight_p10': spgm_update_policy['weight_p10'],
        'weight_p50': spgm_update_policy['weight_p50'],
        'weight_p90': spgm_update_policy['weight_p90'],
        'cluster_count_near': spgm_score['cluster_count_near'],
        'cluster_count_mid': spgm_score['cluster_count_mid'],
        'cluster_count_far': spgm_score['cluster_count_far'],
        'selected_count_near': spgm_update_policy['selected_count_near'],
        'selected_count_mid': spgm_update_policy['selected_count_mid'],
        'selected_count_far': spgm_update_policy['selected_count_far'],
        'density_mode_effective': spgm_score['density_mode_effective'],
        'state_score_mean': spgm_score['state_score_mean'],
        'state_score_p50': spgm_score['state_score_p50'],
        'participation_score_mean': spgm_score['participation_score_mean'],
        'participation_score_p50': spgm_score['participation_score_p50'],
        'state_score_mean_near': spgm_manager['state_score_mean_near'],
        'state_score_mean_mid': spgm_manager['state_score_mean_mid'],
        'state_score_mean_far': spgm_manager['state_score_mean_far'],
        'state_score_p50_near': spgm_manager['state_score_p50_near'],
        'state_score_p50_mid': spgm_manager['state_score_p50_mid'],
        'state_score_p50_far': spgm_manager['state_score_p50_far'],
        'participation_score_mean_near': spgm_manager['participation_score_mean_near'],
        'participation_score_mean_mid': spgm_manager['participation_score_mean_mid'],
        'participation_score_mean_far': spgm_manager['participation_score_mean_far'],
        'participation_score_p50_near': spgm_manager['participation_score_p50_near'],
        'participation_score_p50_mid': spgm_manager['participation_score_p50_mid'],
        'participation_score_p50_far': spgm_manager['participation_score_p50_far'],
        'population_support_mean': spgm_score['population_support_mean'],
        'struct_density_mean': spgm_score['struct_density_mean'],
        'manager_mode_effective': spgm_manager['manager_mode_effective'],
        'state_action_applied': spgm_manager['state_action_applied'],
        'state_xyz_scale_mean': spgm_manager['state_xyz_scale_mean'],
        'state_xyz_scale_mean_near': spgm_manager['state_xyz_scale_mean_near'],
        'state_xyz_scale_mean_mid': spgm_manager['state_xyz_scale_mean_mid'],
        'state_xyz_scale_mean_far': spgm_manager['state_xyz_scale_mean_far'],
        'state_grad_norm_xyz_pre_state': spgm_manager['state_grad_norm_xyz_pre_state'],
        'state_grad_norm_xyz_post_state': spgm_manager['state_grad_norm_xyz_post_state'],
        'state_lr_scale_near': spgm_manager['state_lr_scale_near'],
        'state_lr_scale_mid': spgm_manager['state_lr_scale_mid'],
        'state_lr_scale_far': spgm_manager['state_lr_scale_far'],
        'state_opacity_decay_near': spgm_manager['state_opacity_decay_near'],
        'state_opacity_decay_mid': spgm_manager['state_opacity_decay_mid'],
        'state_opacity_decay_far': spgm_manager['state_opacity_decay_far'],
        'state_opacity_scale_mean': spgm_manager['state_opacity_scale_mean'],
        'state_opacity_scale_mean_near': spgm_manager['state_opacity_scale_mean_near'],
        'state_opacity_scale_mean_mid': spgm_manager['state_opacity_scale_mean_mid'],
        'state_opacity_scale_mean_far': spgm_manager['state_opacity_scale_mean_far'],
        'state_candidate_count_near': spgm_manager['state_candidate_count_near'],
        'state_candidate_count_mid': spgm_manager['state_candidate_count_mid'],
        'state_candidate_count_far': spgm_manager['state_candidate_count_far'],
        'state_participation_ratio': spgm_manager['state_participation_ratio'],
        'state_participation_ratio_near': spgm_manager['state_participation_ratio_near'],
        'state_participation_ratio_mid': spgm_manager['state_participation_ratio_mid'],
        'state_participation_ratio_far': spgm_manager['state_participation_ratio_far'],
        'state_participation_drop_count_near': spgm_manager['state_participation_drop_count_near'],
        'state_participation_drop_count_mid': spgm_manager['state_participation_drop_count_mid'],
        'state_participation_drop_count_far': spgm_manager['state_participation_drop_count_far'],
    }


def maybe_apply_pseudo_local_gating(
    gaussians,
    sampled_views,
    render_packages,
    gating_cfg: PseudoLocalGatingConfig,
    extra_window_views=None,
    extra_window_render_packages=None,
    extra_window_weights=None,
) -> dict:
    sampled_ids = [int(v['sample_id']) for v in sampled_views]
    gate_results = evaluate_sampled_views_for_local_gating(sampled_views, gating_cfg)

    spgm_stats = None
    spgm_render_mask = None
    spgm_opacity_scale = None

    if not gating_cfg.enabled():
        grad_xyz = _grad_norm_from_param(getattr(gaussians, '_xyz', None))
        grad_opacity = _grad_norm_from_param(getattr(gaussians, '_opacity', None))
        visibility_stats = {
            'visible_union_ratio': None,
            'visible_union_weight_mean': None,
            'accepted_count': len(sampled_views),
        }
        grad_stats = {
            'grad_keep_ratio_xyz': 1.0 if 'xyz' in gating_cfg.params else None,
            'grad_keep_ratio_opacity': 1.0 if 'opacity' in gating_cfg.params else None,
            'grad_weight_mean_xyz': 1.0 if 'xyz' in gating_cfg.params else None,
            'grad_weight_mean_opacity': 1.0 if 'opacity' in gating_cfg.params else None,
            'grad_norm_xyz_pre_mask': grad_xyz,
            'grad_norm_xyz_post_mask': grad_xyz,
            'grad_norm_opacity_pre_mask': grad_opacity,
            'grad_norm_opacity_post_mask': grad_opacity,
        }

    elif gating_cfg.uses_visibility_union():
        visibility_stats = build_visibility_weight_map(
            render_packages=render_packages,
            gate_results=gate_results,
            num_gaussians=int(gaussians._xyz.shape[0]),
            device=gaussians._xyz.device,
        )
        grad_stats = apply_gaussian_grad_mask(
            gaussians=gaussians,
            weights=visibility_stats['weights'],
            params_mode=gating_cfg.params,
        )

    elif gating_cfg.uses_spgm():
        accepted_view_count = sum(1 for item in gate_results if float(item.get('weight', 0.0)) > 0.0)
        spgm_raw_stats = collect_spgm_stats(
            sampled_views=sampled_views,
            gate_results=gate_results,
            render_packages=render_packages,
            gaussians=gaussians,
            device=gaussians._xyz.device,
            extra_window_views=extra_window_views,
            extra_window_render_packages=extra_window_render_packages,
            extra_window_weights=extra_window_weights,
        )
        spgm_score = build_spgm_importance_score(
            depth_value=spgm_raw_stats['depth_value'],
            density_proxy=spgm_raw_stats['density_proxy'],
            support_count=spgm_raw_stats['support_count'],
            population_support_count=spgm_raw_stats['population_support_count'],
            struct_density_proxy=spgm_raw_stats['struct_density_proxy'],
            active_mask=spgm_raw_stats['active_mask'],
            num_clusters=gating_cfg.spgm_num_clusters,
            alpha_depth=gating_cfg.spgm_alpha_depth,
            beta_entropy=gating_cfg.spgm_beta_entropy,
            gamma_entropy=gating_cfg.spgm_gamma_entropy,
            support_eta=gating_cfg.spgm_support_eta,
            entropy_bins=gating_cfg.spgm_entropy_bins,
            density_mode=gating_cfg.spgm_density_mode,
            ranking_mode=gating_cfg.spgm_ranking_mode,
            lambda_support_rank=gating_cfg.spgm_lambda_support_rank,
        )
        manager_mode_requested = str(gating_cfg.spgm_manager_mode or 'summary_only').strip().lower()
        no_action_summary_only = manager_mode_requested in {'summary_only', 'off', 'disabled'}
        if no_action_summary_only:
            spgm_update_policy = _build_spgm_passthrough_update_policy(
                spgm_raw_stats=spgm_raw_stats,
                spgm_score=spgm_score,
                policy_mode=gating_cfg.spgm_policy_mode,
            )
            grad_stats = apply_gaussian_grad_mask(
                gaussians=gaussians,
                weights=spgm_update_policy['weights'],
                params_mode=gating_cfg.params,
            )
        else:
            spgm_update_policy = build_spgm_update_policy(
                weight_score=spgm_score['importance_score'],
                ranking_score=spgm_score['ranking_score'],
                cluster_id=spgm_score['cluster_id'],
                active_mask=spgm_raw_stats['active_mask'],
                weight_floor=gating_cfg.spgm_weight_floor,
                cluster_keep=(gating_cfg.spgm_cluster_keep_near, gating_cfg.spgm_cluster_keep_mid, gating_cfg.spgm_cluster_keep_far),
                policy_mode=gating_cfg.spgm_policy_mode,
                selector_keep_ratio=(
                    gating_cfg.spgm_selector_keep_ratio_near,
                    gating_cfg.spgm_selector_keep_ratio_mid,
                    gating_cfg.spgm_selector_keep_ratio_far,
                ),
                selector_min_keep=gating_cfg.spgm_selector_min_keep,
            )
            grad_stats = apply_gaussian_grad_mask(
                gaussians=gaussians,
                weights=spgm_update_policy['weights'],
                params_mode=gating_cfg.params,
            )
        spgm_manager = apply_spgm_state_management(
            gaussians=gaussians,
            cluster_id=spgm_score['cluster_id'],
            control_mask=spgm_raw_stats['active_mask'],
            update_policy=spgm_update_policy,
            manager_mode=gating_cfg.spgm_manager_mode,
            state_score=spgm_score['state_score'],
            participation_score=spgm_score['participation_score'],
            population_support_count=spgm_raw_stats['population_support_count'],
            state_candidate_quantile=gating_cfg.spgm_state_candidate_quantile,
            state_base_scale_near=gating_cfg.spgm_state_base_scale_near,
            state_base_scale_mid=gating_cfg.spgm_state_base_scale_mid,
            state_base_scale_far=gating_cfg.spgm_state_base_scale_far,
            state_participation_keep_near=gating_cfg.spgm_state_participation_keep_near,
            state_participation_keep_mid=gating_cfg.spgm_state_participation_keep_mid,
            state_participation_keep_far=gating_cfg.spgm_state_participation_keep_far,
            state_opacity_floor_near=gating_cfg.spgm_state_opacity_floor_near,
            state_opacity_floor_mid=gating_cfg.spgm_state_opacity_floor_mid,
            state_opacity_floor_far=gating_cfg.spgm_state_opacity_floor_far,
        )
        visibility_stats = {
            'visible_union_ratio': None,
            'visible_union_weight_mean': None,
            'accepted_count': int(accepted_view_count),
        }
        spgm_stats = _build_spgm_summary_payload(
            spgm_raw_stats=spgm_raw_stats,
            spgm_score=spgm_score,
            spgm_update_policy=spgm_update_policy,
            spgm_manager=spgm_manager,
            accepted_view_count=int(accepted_view_count),
        )
        spgm_partition_stats = _build_spgm_c0_partition_stats(
            spgm_score=spgm_score,
            active_mask=spgm_raw_stats['active_mask'],
            candidate_quantile=gating_cfg.spgm_state_candidate_quantile,
        )
        spgm_stats.update(spgm_partition_stats)
        spgm_render_mask = spgm_manager.get('participation_render_mask')
        spgm_opacity_scale = spgm_manager.get('participation_opacity_scale')

    else:
        raise ValueError(f'Unknown gating mode: {gating_cfg.mode}')

    summary = build_iteration_gating_summary(
        mode=gating_cfg.mode,
        params=gating_cfg.params,
        sampled_ids=sampled_ids,
        gate_results=gate_results,
        visibility_stats=visibility_stats,
        grad_stats=grad_stats,
        spgm_stats=spgm_stats,
    )
    if spgm_stats is not None:
        for key, value in spgm_stats.items():
            if key.startswith('c0_'):
                summary[f'spgm_{key}'] = value
    if spgm_render_mask is not None:
        summary['_spgm_participation_render_mask'] = spgm_render_mask.detach()
    if spgm_opacity_scale is not None:
        summary['_spgm_participation_opacity_scale'] = spgm_opacity_scale.detach()
    return summary


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render

    pipeline_params = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')

    print(f'[Part3 StageA] Loading PLY: {args.ply_path}')
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(args.ply_path)
    print(f'  Loaded {len(gaussians.get_xyz)} Gaussians (sh_degree={args.sh_degree})')

    pseudo_views, pseudo_manifest_info = load_stageA_pseudo_views(args)
    init_handoff_summary = initialize_pseudo_views_from_saved_states(
        pseudo_views,
        init_states_json=args.init_pseudo_camera_states_json,
        reference_mode=args.init_pseudo_reference_mode,
    )
    if not pseudo_views:
        raise RuntimeError(f'No pseudo viewpoints found under {args.pseudo_cache}')
    if init_handoff_summary['requested']:
        print(f"  Loaded prior pseudo states: {init_handoff_summary['loaded_count']}/{len(pseudo_views)} from {args.init_pseudo_camera_states_json} (reference_mode={args.init_pseudo_reference_mode})")
        if init_handoff_summary['missing_sample_ids']:
            print(f"  Missing prior states for sample_ids: {init_handoff_summary['missing_sample_ids']}")
        if init_handoff_summary['frame_id_mismatches']:
            print(f"  Frame-id mismatches in prior states: {init_handoff_summary['frame_id_mismatches']}")
    print(f'  Loaded {len(pseudo_views)} pseudo viewpoints (target_side={args.target_side}, confidence_mask_source={args.confidence_mask_source})')
    print(
        '  StageA effective sources: '
        f"signal_pipeline={args.signal_pipeline}, signal_v2_root={args.signal_v2_root}, mask_mode(legacy)={args.stageA_mask_mode}, rgb_mask_mode={args.stageA_rgb_mask_mode}, depth_mask_mode={args.stageA_depth_mask_mode}, conf_variant={args.stageA_confidence_variant}, depth_mode={args.stageA_target_depth_mode}, depth_loss_mode={args.stageA_depth_loss_mode}, apply_pose_update={args.stageA_apply_pose_update}, lambda_abs_pose(legacy)={args.stageA_lambda_abs_pose}, lambda_abs_t={args.stageA_lambda_abs_t}, lambda_abs_r={args.stageA_lambda_abs_r}, abs_robust={args.stageA_abs_pose_robust}, "
        f"mean_mask_cov={np.mean([v['confidence_nonzero_ratio'] for v in pseudo_views]):.4f}, "
        f"mean_depth_verified={np.mean([v.get('target_depth_verified_ratio') or 0.0 for v in pseudo_views]):.4f}, "
        f"mean_depth_dense={np.mean([v.get('target_depth_dense_ratio') or 0.0 for v in pseudo_views]):.4f}"
    )

    init_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_init.json', init_states)

    cfg_cls = StageA5Config if args.stage_mode == "stageA5" else StageAConfig
    cfg = cfg_cls(
        num_iterations=args.stageA_iters,
        beta_rgb=args.stageA_beta_rgb,
        lambda_pose=args.stageA_lambda_pose,
        lambda_exp=args.stageA_lambda_exp,
        lambda_abs_pose=args.stageA_lambda_abs_pose,
        lambda_abs_t=args.stageA_lambda_abs_t,
        lambda_abs_r=args.stageA_lambda_abs_r,
        abs_pose_robust=args.stageA_abs_pose_robust,
        trans_reg_weight=args.stageA_trans_reg_weight,
        lr_rot=args.stageA_lr_rot,
        lr_trans=args.stageA_lr_trans,
        lr_exp=args.stageA_lr_exp,
        num_pseudo_views=args.num_pseudo_views,
        **({
            "trainable_params": args.stageA5_trainable_params,
            "lr_xyz": args.stageA5_lr_xyz,
            "lr_opacity": args.stageA5_lr_opacity,
            "disable_densify": args.stageA5_disable_densify,
            "disable_prune": args.stageA5_disable_prune,
        } if args.stage_mode == "stageA5" else {}),
    )
    pseudo_local_gating_cfg = build_pseudo_local_gating_cfg(args)
    if args.stage_mode == "stageA5":
        optimizer, gaussian_optimizer = build_stageA5_optimizers(pseudo_views, gaussians, cfg)
    else:
        optimizer = build_stageA_optimizer(pseudo_views, cfg)
        gaussian_optimizer = None

    history = {
        'iterations': [],
        'loss_total': [], 'loss_rgb': [], 'loss_depth': [],
        'loss_depth_seed': [], 'loss_depth_dense': [], 'loss_depth_fallback': [],
        'loss_pose_reg': [], 'loss_abs_pose_reg': [],
        'loss_abs_pose_trans': [], 'loss_abs_pose_rot': [],
        'abs_pose_rho_norm': [], 'abs_pose_theta_norm': [], 'scene_scale_used': [],
        'loss_exp_reg': [],
        'sampled_sample_ids': [], 'num_pose_updates_applied': [],
        'grad_contrib': [],
        'grad_norm_xyz': []
    }
    _init_gating_history_lists(history)

    for it in tqdm(range(1, cfg.num_iterations + 1), desc='StageA pseudo pose+exposure'):
        optimizer.zero_grad(set_to_none=True)
        if gaussian_optimizer is not None:
            gaussian_optimizer.zero_grad(set_to_none=True)
        pseudo_indices = sample_indices(len(pseudo_views), cfg.num_pseudo_views, rng)
        if not pseudo_indices:
            raise RuntimeError('No pseudo viewpoints sampled in Stage A')

        per_view_losses = []
        per_view_stats = []
        sampled_ids = []
        sampled_views = []
        render_packages = []
        for pidx in pseudo_indices:
            view = pseudo_views[pidx]
            sampled_ids.append(int(view['sample_id']))
            sampled_views.append(view)
            pkg = render(view['vp'], gaussians, pipeline_params, bg)
            render_packages.append(pkg)
            if args.stageA_depth_loss_mode == 'exact_shared_cm_v1':
                exact_upstream_bundle = view.get('exact_upstream_bundle', {})
                loss, stats = build_stageA_loss_exact_shared_cm(
                    render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth=args.stageA_lambda_depth_seed, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), valid_mask=exact_upstream_bundle.get('valid_mask'), target_confidence=exact_upstream_bundle.get('target_confidence'),
                )
            elif args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
                loss, stats = build_stageA_loss_source_aware(
                    render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], depth_source_map=view['target_depth_source_map'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth_seed=args.stageA_lambda_depth_seed, lambda_depth_dense=args.stageA_lambda_depth_dense, lambda_depth_fallback=args.stageA_lambda_depth_fallback, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0], depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1], depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                )
            else:
                loss, stats = build_stageA_loss(
                    render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0], depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1], depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                )
            per_view_losses.append(loss)
            per_view_stats.append(stats)

        total_loss = torch.stack(per_view_losses).mean()
        total_loss.backward()
        gating_summary = maybe_apply_pseudo_local_gating(
            gaussians=gaussians,
            sampled_views=sampled_views,
            render_packages=render_packages,
            gating_cfg=pseudo_local_gating_cfg if gaussian_optimizer is not None else PseudoLocalGatingConfig(mode='off', params=args.pseudo_local_gating_params),
        )
        _append_gating_history(history, gating_summary)
        optimizer.step()
        if gaussian_optimizer is not None:
            gaussian_optimizer.step()

        num_pose_updates = 0
        if args.stageA_apply_pose_update:
            seen = set()
            for pidx in pseudo_indices:
                view = pseudo_views[pidx]
                sid = int(view['sample_id'])
                if sid in seen:
                    continue
                apply_pose_residual_(view['vp'])
                seen.add(sid)
            num_pose_updates = len(seen)

        avg_stats = {key: float(np.mean([s[key] for s in per_view_stats])) for key in ['loss_total', 'loss_rgb', 'loss_depth', 'loss_depth_seed', 'loss_depth_dense', 'loss_depth_fallback', 'loss_pose_reg', 'loss_abs_pose_reg', 'loss_abs_pose_trans', 'loss_abs_pose_rot', 'abs_pose_rho_norm', 'abs_pose_theta_norm', 'scene_scale_used', 'loss_exp_reg']}
        for key in avg_stats:
            history[key].append(avg_stats[key])
        history['iterations'].append(it)
        history['sampled_sample_ids'].append(sampled_ids)
        history['num_pose_updates_applied'].append(int(num_pose_updates))
        history['grad_norm_xyz'].append(float(torch.norm(gaussians._xyz.grad).detach().cpu().item()) if gaussians._xyz.grad is not None else 0.0)

        if args.stageA_log_grad_contrib and (it == 1 or it % max(1, int(args.stageA_log_grad_interval)) == 0):
            sampled_views = [pseudo_views[i] for i in pseudo_indices]
            grad_contrib = diagnose_grad_contrib_for_sampled_views(args, cfg, sampled_views, gaussians, pipeline_params, bg)
            history['grad_contrib'].append({'iter': int(it), 'terms': grad_contrib})
            rgb_rot = grad_contrib.get('rgb', {}).get('grad_norm_rot', 0.0)
            depth_rot = grad_contrib.get('depth_total', {}).get('grad_norm_rot', 0.0)
            abs_rot = grad_contrib.get('abs_pose_rot', {}).get('grad_norm_rot', 0.0)
            print(f"    [grad] iter {it}: rot rgb={rgb_rot:.3e}, depth={depth_rot:.3e}, abs_rot={abs_rot:.3e}")

        if it == 1 or it % 20 == 0:
            print(
                f"  iter {it}: total={avg_stats['loss_total']:.4f}, rgb={avg_stats['loss_rgb']:.4f}, depth={avg_stats['loss_depth']:.4f}, "
                f"depth_seed={avg_stats['loss_depth_seed']:.4f}, depth_dense={avg_stats['loss_depth_dense']:.4f}, depth_fb={avg_stats['loss_depth_fallback']:.4f}, "
                f"pose_reg={avg_stats['loss_pose_reg']:.4f}, abs_pose={avg_stats['loss_abs_pose_reg']:.4f} (t={avg_stats['loss_abs_pose_trans']:.4f}, r={avg_stats['loss_abs_pose_rot']:.4f}), "
                f"abs_norm(rho={avg_stats['abs_pose_rho_norm']:.4f}, theta={avg_stats['abs_pose_theta_norm']:.4f}), scale={avg_stats['scene_scale_used']:.4f}, exp_reg={avg_stats['loss_exp_reg']:.4f}, pose_updates={num_pose_updates}"
            )
        if gaussian_optimizer is not None and pseudo_local_gating_cfg.enabled() and (it == 1 or it % max(1, int(pseudo_local_gating_cfg.log_interval)) == 0):
            if pseudo_local_gating_cfg.uses_spgm():
                print(
                    f"    [StageA local-gating] iter {it}: accepted={gating_summary.get('accepted_visibility_count')}/{len(sampled_views)}, "
                    f"policy={gating_summary.get('spgm_policy_mode_effective')}, "
                    f"ranking={gating_summary.get('spgm_ranking_mode_effective')}, "
                    f"active={float(gating_summary.get('spgm_active_ratio') or 0.0):.4f}, "
                    f"selected={float(gating_summary.get('spgm_selected_ratio') or 0.0):.4f}, "
                    f"rank_p50={float(gating_summary.get('spgm_ranking_score_p50') or 0.0):.4f}, "
                    f"state_p50={float(gating_summary.get('spgm_state_score_p50') or 0.0):.4f}, "
                    f"state_far={float(gating_summary.get('spgm_state_score_p50_far') or 0.0):.4f}, "
                    f"w_p50={float(gating_summary.get('spgm_weight_p50') or 0.0):.4f}, "
                    f"manager={gating_summary.get('spgm_manager_mode_effective')}, "
                    f"lr_far={float(gating_summary.get('spgm_state_lr_scale_far') or 0.0):.3f}, "
                    f"cand_far={int(gating_summary.get('spgm_state_candidate_count_far') or 0)}, "
                    f"state_xyz={float(gating_summary.get('spgm_state_grad_norm_xyz_pre_state') or 0.0):.3e}->{float(gating_summary.get('spgm_state_grad_norm_xyz_post_state') or 0.0):.3e}, "
                    f"xyz_grad={float(gating_summary.get('grad_norm_xyz_pre_mask') or 0.0):.3e}->{float(gating_summary.get('grad_norm_xyz_post_mask') or 0.0):.3e}"
                )
            else:
                print(
                    f"    [local-gating] iter {it}: accepted={len(gating_summary['accepted_pseudo_sample_ids'])}/{len(sampled_ids)}, "
                    f"visible_union={float(gating_summary.get('visible_union_ratio') or 0.0):.4f}, "
                    f"keep_xyz={float(gating_summary.get('grad_keep_ratio_xyz') or 0.0):.4f}, "
                    f"xyz_grad={float(gating_summary.get('grad_norm_xyz_pre_mask') or 0.0):.3e}->{float(gating_summary.get('grad_norm_xyz_post_mask') or 0.0):.3e}"
                )

    stageA_states = [export_view_state(v) for v in pseudo_views]
    save_json(output_dir / 'pseudo_camera_states_stageA.json', stageA_states)

    true_pose_delta_summary = summarize_true_pose_deltas(init_states, stageA_states)

    stageA_history = {
        'args': vars(args), 'stage_mode': args.stage_mode, 'pseudo_manifest_info': pseudo_manifest_info, 'num_pseudo_viewpoints_loaded': len(pseudo_views), 'init_handoff_summary': init_handoff_summary,
        'effective_source_summary': {
            'signal_pipeline': args.signal_pipeline,
            'signal_v2_root': args.signal_v2_root,
            'pseudo_observation_mode_requested': args.pseudo_observation_mode,
            'stageA_mask_mode_requested': args.stageA_mask_mode,
            'stageA_rgb_mask_mode_requested': args.stageA_rgb_mask_mode,
            'stageA_depth_mask_mode_requested': args.stageA_depth_mask_mode,
            'stageA_confidence_variant_requested': args.stageA_confidence_variant,
            'stageA_target_depth_mode_requested': args.stageA_target_depth_mode,
            'stageA_depth_loss_mode': args.stageA_depth_loss_mode,
            'stageA_apply_pose_update': bool(args.stageA_apply_pose_update),
            'stageA_disable_depth': bool(args.stageA_disable_depth),
            'stageA_lambda_abs_pose': float(args.stageA_lambda_abs_pose),
            'stageA_lambda_abs_t': float(args.stageA_lambda_abs_t),
            'stageA_lambda_abs_r': float(args.stageA_lambda_abs_r),
            'stageA_abs_pose_robust': args.stageA_abs_pose_robust,
            'stageA_abs_pose_scale_source': args.stageA_abs_pose_scale_source,
            'stageA_abs_pose_fixed_scale': float(args.stageA_abs_pose_fixed_scale),
            'pseudo_local_gating_mode': pseudo_local_gating_cfg.mode,
            'pseudo_local_gating_params': pseudo_local_gating_cfg.params,
            'pseudo_local_gating_min_verified_ratio': float(pseudo_local_gating_cfg.min_verified_ratio),
            'pseudo_local_gating_min_rgb_mask_ratio': float(pseudo_local_gating_cfg.min_rgb_mask_ratio),
            'pseudo_local_gating_max_fallback_ratio': float(pseudo_local_gating_cfg.max_fallback_ratio),
            'pseudo_local_gating_min_correction': float(pseudo_local_gating_cfg.min_correction),
            'pseudo_local_gating_soft_power': float(pseudo_local_gating_cfg.soft_power),
            'mean_stageA_scene_scale': float(np.mean([v.get('stageA_scene_scale', 1.0) for v in pseudo_views])),
            'mean_confidence_nonzero_ratio': float(np.mean([v.get('confidence_nonzero_ratio', 0.0) for v in pseudo_views])),
            'mean_target_depth_verified_ratio': float(np.mean([(v.get('target_depth_verified_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_render_fallback_ratio': float(np.mean([(v.get('target_depth_render_fallback_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_seed_ratio': float(np.mean([(v.get('target_depth_seed_ratio') or 0.0) for v in pseudo_views])),
            'mean_target_depth_dense_ratio': float(np.mean([(v.get('target_depth_dense_ratio') or 0.0) for v in pseudo_views])),
        },
        'pseudo_sample_meta': [
            {
                'sample_id': int(v['sample_id']), 'frame_id': int(v['frame_id']), 'target_rgb_path': v.get('target_rgb_path'),
                'target_depth_for_refine_path': v.get('target_depth_for_refine_path'), 'target_depth_for_refine_kind': v.get('target_depth_for_refine_kind'), 'stageA_target_depth_mode_effective': v.get('stageA_target_depth_mode_effective'),
                'target_depth_nonzero_ratio': v.get('target_depth_nonzero_ratio'), 'target_depth_verified_ratio': v.get('target_depth_verified_ratio'), 'target_depth_render_fallback_ratio': v.get('target_depth_render_fallback_ratio'), 'target_depth_seed_ratio': v.get('target_depth_seed_ratio'), 'target_depth_dense_ratio': v.get('target_depth_dense_ratio'),
                'target_depth_for_refine_source': v.get('source_meta', {}).get('target_depth_for_refine_source'), 'target_depth_for_refine_source_map_path': v.get('target_depth_for_refine_source_map_path'), 'confidence_path': v.get('confidence_path'), 'confidence_source_kind': v.get('confidence_source_kind'), 'stageA_mask_mode_effective': v.get('stageA_mask_mode_effective'), 'stageA_rgb_mask_mode_effective': v.get('stageA_rgb_mask_mode_effective'), 'stageA_depth_mask_mode_effective': v.get('stageA_depth_mask_mode_effective'), 'stageA_rgb_confidence_variant_effective': v.get('stageA_rgb_confidence_variant_effective'), 'stageA_depth_confidence_variant_effective': v.get('stageA_depth_confidence_variant_effective'), 'pseudo_observation_mode_effective': v.get('pseudo_observation_mode_effective'), 'pseudo_observation_meta_path': v.get('pseudo_observation_meta_path'), 'rgb_confidence_path': v.get('rgb_confidence_path'), 'depth_confidence_path': v.get('depth_confidence_path'), 'rgb_confidence_nonzero_ratio': v.get('rgb_confidence_nonzero_ratio'), 'depth_confidence_nonzero_ratio': v.get('depth_confidence_nonzero_ratio'), 'confidence_nonzero_ratio': v.get('confidence_nonzero_ratio'), 'confidence_mean_positive': v.get('confidence_mean_positive'), 'stageA_scene_scale': v.get('stageA_scene_scale'), 'stageA_scene_scale_source_effective': v.get('stageA_scene_scale_source_effective'),
            }
            for v in pseudo_views
        ],
        'history': history, 'stageA_disable_depth': bool(args.stageA_disable_depth), 'stageA_mask_mode': args.stageA_mask_mode, 'stageA_rgb_mask_mode': args.stageA_rgb_mask_mode, 'stageA_depth_mask_mode': args.stageA_depth_mask_mode, 'stageA_confidence_variant': args.stageA_confidence_variant, 'stageA_target_depth_mode': args.stageA_target_depth_mode, 'stageA_depth_loss_mode': args.stageA_depth_loss_mode, 'stageA_apply_pose_update': bool(args.stageA_apply_pose_update), 'stageA_lambda_depth_seed': float(args.stageA_lambda_depth_seed), 'stageA_lambda_depth_dense': float(args.stageA_lambda_depth_dense), 'stageA_lambda_depth_fallback': float(args.stageA_lambda_depth_fallback), 'stageA_lambda_abs_pose': float(args.stageA_lambda_abs_pose), 'stageA_lambda_abs_t': float(args.stageA_lambda_abs_t), 'stageA_lambda_abs_r': float(args.stageA_lambda_abs_r), 'stageA_abs_pose_robust': args.stageA_abs_pose_robust, 'stageA_abs_pose_scale_source': args.stageA_abs_pose_scale_source, 'stageA_abs_pose_fixed_scale': float(args.stageA_abs_pose_fixed_scale), 'stageA_log_grad_contrib': bool(args.stageA_log_grad_contrib), 'stageA_log_grad_interval': int(args.stageA_log_grad_interval), 'pseudo_local_gating': pseudo_local_gating_cfg.as_dict(), 'final_states': stageA_states,
        'residual_delta_summary_post_foldback': [
            {'sample_id': int(v['sample_id']), 'frame_id': int(v['frame_id']), 'rot_norm': float(torch.norm(v['vp'].cam_rot_delta).detach().cpu().item()), 'trans_norm': float(torch.norm(v['vp'].cam_trans_delta).detach().cpu().item()), 'exposure_a': float(v['vp'].exposure_a.detach().cpu().item()), 'exposure_b': float(v['vp'].exposure_b.detach().cpu().item())}
            for v in pseudo_views
        ],
        'true_pose_delta_summary': true_pose_delta_summary['per_view'],
        'true_pose_delta_aggregate': true_pose_delta_summary['aggregate'],
    }
    save_json(output_dir / 'stageA_history.json', stageA_history)

    stageB_history = None
    final_states = stageA_states
    if args.stage_mode == 'stageB' and args.stageB_iters > 0:
        v1 = load_v1_module()
        real_views, resolved_train_rgb_dir = [], None
        if args.train_manifest:
            real_views, resolved_train_rgb_dir = v1.load_real_viewpoints(args.train_manifest, args.train_rgb_dir)
            print(f"  [StageB] loaded real sparse-train views: {len(real_views)} from {args.train_manifest}")
        has_real_branch = bool(real_views and args.num_real_views > 0 and args.lambda_real > 0)
        if args.lambda_real > 0 and args.train_manifest and not real_views:
            raise RuntimeError('--train_manifest was given but no real viewpoints were loaded')

        gaussian_groups = build_micro_gaussian_param_groups(
            gaussians=gaussians,
            mode=args.stageA5_trainable_params,
            lr_xyz=args.stageA5_lr_xyz,
            lr_opacity=args.stageA5_lr_opacity,
        )
        stageB_gaussian_optimizer = torch.optim.Adam(gaussian_groups)
        stageB_pseudo_optimizer = build_stageA_optimizer(pseudo_views, cfg)

        stageB_history = {
            'iters': int(args.stageB_iters),
            'has_real_branch': bool(has_real_branch),
            'resolved_train_rgb_dir': resolved_train_rgb_dir,
            'joint_topology_mode_requested': str(getattr(args, 'joint_topology_mode', 'off') or 'off'),
            'joint_topology_mode_effective': 'brpo_joint_v1' if str(getattr(args, 'joint_topology_mode', 'off') or 'off') == 'brpo_joint_v1' else 'off',
            'pseudo_local_gating': pseudo_local_gating_cfg.as_dict(),
            'stageB_post_switch_iter': int(args.stageB_post_switch_iter),
            'stageB_post_lr_scale_xyz': float(args.stageB_post_lr_scale_xyz),
            'stageB_post_lr_scale_opacity': float(args.stageB_post_lr_scale_opacity),
            'stageB_post_lambda_real': None if args.stageB_post_lambda_real is None else float(args.stageB_post_lambda_real),
            'stageB_post_lambda_pseudo': None if args.stageB_post_lambda_pseudo is None else float(args.stageB_post_lambda_pseudo),
            'post_switch_applied': False,
            'iterations': [],
            'lambda_real_effective': [],
            'lambda_pseudo_effective': [],
            'gaussian_lr_xyz_effective': [],
            'loss_total': [],
            'loss_real': [],
            'loss_pseudo': [],
            'loss_rgb': [],
            'loss_depth': [],
            'loss_pose_reg': [],
            'loss_exp_reg': [],
            'grad_norm_xyz': [],
            'sampled_real_frame_ids': [],
            'sampled_pseudo_sample_ids': [],
        }
        _init_gating_history_lists(stageB_history)
        cfg_real = {'Training': {'lambda_pseudo': 1.0, 'rgb_boundary_threshold': 0.01, 'monocular': False}}
        post_switch_state = {'applied': False}
        stageB_spgm_participation_render_mask = None
        stageB_spgm_participation_opacity_scale = None

        stageB_loop_desc = 'StageB BRPO joint topology' if str(getattr(args, 'joint_topology_mode', 'off') or 'off') == 'brpo_joint_v1' else 'StageB conservative joint'
        for it in tqdm(range(1, int(args.stageB_iters) + 1), desc=stageB_loop_desc):
            _apply_stageB_post_switch_if_needed(args, it, stageB_gaussian_optimizer, stageB_history, post_switch_state)
            cur_lambda_real = float(args.lambda_real)
            cur_lambda_pseudo = float(args.lambda_pseudo)
            if int(args.stageB_post_switch_iter or 0) > 0 and it > int(args.stageB_post_switch_iter):
                if args.stageB_post_lambda_real is not None:
                    cur_lambda_real = float(args.stageB_post_lambda_real)
                if args.stageB_post_lambda_pseudo is not None:
                    cur_lambda_pseudo = float(args.stageB_post_lambda_pseudo)
            stageB_pseudo_optimizer.zero_grad(set_to_none=True)
            stageB_gaussian_optimizer.zero_grad(set_to_none=True)

            real_loss_tensor = torch.zeros((), device='cuda')
            sampled_real_ids = []
            sampled_real_views = []
            real_render_packages = []
            if has_real_branch:
                real_indices = sample_indices(len(real_views), args.num_real_views, rng)
                real_losses = []
                for ridx in real_indices:
                    rv = real_views[ridx]
                    sampled_real_ids.append(int(rv.get('frame_id', ridx)))
                    sampled_real_views.append(rv)
                    pkg = render(rv['vp'], gaussians, pipeline_params, bg)
                    real_render_packages.append(pkg)
                    real_losses.append(v1.get_loss_mapping_rgb(cfg_real, pkg['render'], rv['vp']))
                if real_losses:
                    real_loss_tensor = torch.stack(real_losses).mean()

            pseudo_indices = sample_indices(len(pseudo_views), cfg.num_pseudo_views, rng)
            if not pseudo_indices:
                raise RuntimeError('No pseudo viewpoints sampled in Stage B')

            pseudo_losses = []
            per_view_stats = []
            sampled_pseudo_ids = []
            sampled_pseudo_views = []
            pseudo_render_packages = []
            current_pseudo_render_mask = stageB_spgm_participation_render_mask if (pseudo_local_gating_cfg.uses_spgm() and pseudo_local_gating_cfg.spgm_manager_mode == 'deterministic_participation') else None
            current_pseudo_opacity_scale = stageB_spgm_participation_opacity_scale if (pseudo_local_gating_cfg.uses_spgm() and pseudo_local_gating_cfg.spgm_manager_mode == 'deterministic_opacity_participation') else None
            for pidx in pseudo_indices:
                view = pseudo_views[pidx]
                sampled_pseudo_ids.append(int(view['sample_id']))
                sampled_pseudo_views.append(view)
                if current_pseudo_render_mask is not None or current_pseudo_opacity_scale is not None:
                    pkg = render(view['vp'], gaussians, pipeline_params, bg, mask=current_pseudo_render_mask, opacity_scale=current_pseudo_opacity_scale)
                else:
                    pkg = render(view['vp'], gaussians, pipeline_params, bg)
                pseudo_render_packages.append(pkg)
                if args.stageA_depth_loss_mode == 'exact_shared_cm_v1':
                    exact_upstream_bundle = view.get('exact_upstream_bundle', {})
                    loss, stats = build_stageA_loss_exact_shared_cm(
                        render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth=args.stageA_lambda_depth_seed, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), valid_mask=exact_upstream_bundle.get('valid_mask'), target_confidence=exact_upstream_bundle.get('target_confidence'),
                    )
                elif args.stageA_depth_loss_mode == 'source_aware' and view.get('target_depth_source_map') is not None:
                    loss, stats = build_stageA_loss_source_aware(
                        render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], depth_source_map=view['target_depth_source_map'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, lambda_depth_seed=args.stageA_lambda_depth_seed, lambda_depth_dense=args.stageA_lambda_depth_dense, lambda_depth_fallback=args.stageA_lambda_depth_fallback, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0], depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1], depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                    )
                else:
                    loss, stats = build_stageA_loss(
                        render_rgb=pkg['render'], render_depth=pkg['depth'], target_rgb=view['rgb'], target_depth=view['depth_for_refine'], confidence_mask=view['conf'], viewpoint=view['vp'], rgb_confidence_mask=view.get('conf_rgb'), depth_confidence_mask=view.get('conf_depth'), beta_rgb=cfg.beta_rgb, lambda_pose=cfg.lambda_pose, lambda_exp=cfg.lambda_exp, trans_weight=cfg.trans_reg_weight, use_depth=not args.stageA_disable_depth, lambda_abs_pose=cfg.lambda_abs_pose, lambda_abs_t=cfg.lambda_abs_t, lambda_abs_r=cfg.lambda_abs_r, abs_pose_robust=cfg.abs_pose_robust, scene_scale=view.get('stageA_scene_scale', 1.0), depth_seed_source_ids=_depth_source_id_groups_for_view(view)[0], depth_dense_source_ids=_depth_source_id_groups_for_view(view)[1], depth_fallback_source_ids=_depth_source_id_groups_for_view(view)[2],
                    )
                pseudo_losses.append(loss)
                per_view_stats.append(stats)

            pseudo_loss_tensor = torch.stack(pseudo_losses).mean()
            total_loss = cur_lambda_real * real_loss_tensor + cur_lambda_pseudo * pseudo_loss_tensor
            joint_topology_mode_effective = 'brpo_joint_v1' if str(getattr(args, 'joint_topology_mode', 'off') or 'off') == 'brpo_joint_v1' else 'off'
            if joint_topology_mode_effective == 'brpo_joint_v1':
                total_loss.backward()
            else:
                pseudo_backward = cur_lambda_pseudo * pseudo_loss_tensor
                pseudo_backward.backward(retain_graph=bool(has_real_branch))
            gating_summary = maybe_apply_pseudo_local_gating(
                gaussians=gaussians,
                sampled_views=sampled_pseudo_views,
                render_packages=pseudo_render_packages,
                gating_cfg=pseudo_local_gating_cfg,
                extra_window_views=sampled_real_views,
                extra_window_render_packages=real_render_packages,
                extra_window_weights=[1.0] * len(real_render_packages),
            )
            gating_summary['joint_topology_mode_effective'] = joint_topology_mode_effective
            if pseudo_local_gating_cfg.uses_spgm() and pseudo_local_gating_cfg.spgm_manager_mode == 'deterministic_participation':
                next_render_mask = gating_summary.pop('_spgm_participation_render_mask', None)
                if next_render_mask is not None:
                    stageB_spgm_participation_render_mask = next_render_mask
            elif pseudo_local_gating_cfg.uses_spgm() and pseudo_local_gating_cfg.spgm_manager_mode == 'deterministic_opacity_participation':
                next_opacity_scale = gating_summary.pop('_spgm_participation_opacity_scale', None)
                if next_opacity_scale is not None:
                    stageB_spgm_participation_opacity_scale = next_opacity_scale
            _append_gating_history(stageB_history, gating_summary)
            if joint_topology_mode_effective != 'brpo_joint_v1' and has_real_branch:
                real_backward = cur_lambda_real * real_loss_tensor
                real_backward.backward()
            stageB_pseudo_optimizer.step()
            stageB_gaussian_optimizer.step()

            if args.stageA_apply_pose_update:
                seen = set()
                for pidx in pseudo_indices:
                    sid = int(pseudo_views[pidx]['sample_id'])
                    if sid in seen:
                        continue
                    apply_pose_residual_(pseudo_views[pidx]['vp'])
                    seen.add(sid)

            avg_stats = {key: float(np.mean([s[key] for s in per_view_stats])) for key in ['loss_rgb', 'loss_depth', 'loss_pose_reg', 'loss_exp_reg']}
            stageB_history['iterations'].append(int(it))
            stageB_history['lambda_real_effective'].append(float(cur_lambda_real))
            stageB_history['lambda_pseudo_effective'].append(float(cur_lambda_pseudo))
            xyz_lr_current = None
            for group in stageB_gaussian_optimizer.param_groups:
                if 'xyz' in str(group.get('name') or '').lower():
                    xyz_lr_current = float(group['lr'])
                    break
            stageB_history['gaussian_lr_xyz_effective'].append(xyz_lr_current)
            stageB_history['loss_total'].append(float(total_loss.detach().item()))
            stageB_history['loss_real'].append(float(real_loss_tensor.detach().item()))
            stageB_history['loss_pseudo'].append(float(pseudo_loss_tensor.detach().item()))
            stageB_history['loss_rgb'].append(avg_stats['loss_rgb'])
            stageB_history['loss_depth'].append(avg_stats['loss_depth'])
            stageB_history['loss_pose_reg'].append(avg_stats['loss_pose_reg'])
            stageB_history['loss_exp_reg'].append(avg_stats['loss_exp_reg'])
            stageB_history['grad_norm_xyz'].append(float(torch.norm(gaussians._xyz.grad).detach().cpu().item()) if gaussians._xyz.grad is not None else 0.0)
            stageB_history['sampled_real_frame_ids'].append(sampled_real_ids)
            stageB_history['sampled_pseudo_sample_ids'].append(sampled_pseudo_ids)

            if it == 1 or it % 20 == 0:
                print(f"  [StageB] iter {it}: total={stageB_history['loss_total'][-1]:.4f}, real={stageB_history['loss_real'][-1]:.4f}, pseudo={stageB_history['loss_pseudo'][-1]:.4f}, rgb={stageB_history['loss_rgb'][-1]:.4f}, depth={stageB_history['loss_depth'][-1]:.4f}")
            if pseudo_local_gating_cfg.enabled() and (it == 1 or it % max(1, int(pseudo_local_gating_cfg.log_interval)) == 0):
                if pseudo_local_gating_cfg.uses_spgm():
                    manager_mode = gating_summary.get('spgm_manager_mode_effective')
                    if manager_mode == 'deterministic_opacity_participation':
                        print(
                            f"    [StageB local-gating] iter {it}: accepted={gating_summary.get('accepted_visibility_count')}/{len(sampled_pseudo_ids)}, "
                            f"policy={gating_summary.get('spgm_policy_mode_effective')}, "
                            f"ranking={gating_summary.get('spgm_ranking_mode_effective')}, "
                            f"active={float(gating_summary.get('spgm_active_ratio') or 0.0):.4f}, "
                            f"selected={float(gating_summary.get('spgm_selected_ratio') or 0.0):.4f}, "
                            f"part_p50={float(gating_summary.get('spgm_participation_score_p50') or 0.0):.4f}, "
                            f"part_far={float(gating_summary.get('spgm_participation_score_p50_far') or 0.0):.4f}, "
                            f"opacity_far={float(gating_summary.get('spgm_state_opacity_scale_mean_far') or 1.0):.4f}, "
                            f"opacity_all={float(gating_summary.get('spgm_state_opacity_scale_mean') or 1.0):.4f}, "
                            f"cand_far={int(gating_summary.get('spgm_state_candidate_count_far') or 0)}, "
                            f"xyz_grad={float(gating_summary.get('grad_norm_xyz_pre_mask') or 0.0):.3e}->{float(gating_summary.get('grad_norm_xyz_post_mask') or 0.0):.3e}"
                        )
                    else:
                        print(
                            f"    [StageB local-gating] iter {it}: accepted={gating_summary.get('accepted_visibility_count')}/{len(sampled_pseudo_ids)}, "
                            f"policy={gating_summary.get('spgm_policy_mode_effective')}, "
                            f"ranking={gating_summary.get('spgm_ranking_mode_effective')}, "
                            f"active={float(gating_summary.get('spgm_active_ratio') or 0.0):.4f}, "
                            f"selected={float(gating_summary.get('spgm_selected_ratio') or 0.0):.4f}, "
                            f"rank_p50={float(gating_summary.get('spgm_ranking_score_p50') or 0.0):.4f}, "
                            f"state_p50={float(gating_summary.get('spgm_state_score_p50') or 0.0):.4f}, "
                            f"state_far={float(gating_summary.get('spgm_state_score_p50_far') or 0.0):.4f}, "
                            f"w_p50={float(gating_summary.get('spgm_weight_p50') or 0.0):.4f}, "
                            f"manager={gating_summary.get('spgm_manager_mode_effective')}, "
                            f"part_far={float(gating_summary.get('spgm_state_participation_ratio_far') or 1.0):.3f}, "
                            f"drop_far={int(gating_summary.get('spgm_state_participation_drop_count_far') or 0)}, "
                            f"cand_far={int(gating_summary.get('spgm_state_candidate_count_far') or 0)}, "
                            f"xyz_grad={float(gating_summary.get('grad_norm_xyz_pre_mask') or 0.0):.3e}->{float(gating_summary.get('grad_norm_xyz_post_mask') or 0.0):.3e}"
                        )
                else:
                    print(
                        f"    [StageB local-gating] iter {it}: accepted={len(gating_summary['accepted_pseudo_sample_ids'])}/{len(sampled_pseudo_ids)}, "
                        f"visible_union={float(gating_summary.get('visible_union_ratio') or 0.0):.4f}, "
                        f"keep_xyz={float(gating_summary.get('grad_keep_ratio_xyz') or 0.0):.4f}, "
                        f"xyz_grad={float(gating_summary.get('grad_norm_xyz_pre_mask') or 0.0):.3e}->{float(gating_summary.get('grad_norm_xyz_post_mask') or 0.0):.3e}"
                    )

        save_json(output_dir / 'stageB_history.json', stageB_history)
        final_states = [export_view_state(v) for v in pseudo_views]

    save_json(output_dir / 'pseudo_camera_states_final.json', final_states)
    final_true_pose_delta_summary = summarize_true_pose_deltas(init_states, final_states)
    if stageB_history is None:
        stageA_history['final_true_pose_delta_summary'] = final_true_pose_delta_summary['per_view']
        stageA_history['final_true_pose_delta_aggregate'] = final_true_pose_delta_summary['aggregate']
        save_json(output_dir / 'stageA_history.json', stageA_history)
    else:
        stageB_history['final_true_pose_delta_summary'] = final_true_pose_delta_summary['per_view']
        stageB_history['final_true_pose_delta_aggregate'] = final_true_pose_delta_summary['aggregate']
        save_json(output_dir / 'stageB_history.json', stageB_history)

    if stageB_history is None:
        save_json(output_dir / 'refinement_history.json', stageA_history)
    else:
        save_json(output_dir / 'refinement_history.json', {'stageA': stageA_history, 'stageB': stageB_history})

    refined_ply_path = output_dir / 'refined_gaussians.ply'
    gaussians.save_ply(str(refined_ply_path))

    print(f'[Part3 StageA] Saved: {output_dir}')
    print(f'  Init states: {output_dir / "pseudo_camera_states_init.json"}')
    print(f'  StageA states: {output_dir / "pseudo_camera_states_stageA.json"}')
    print(f'  Final states: {output_dir / "pseudo_camera_states_final.json"}')
    agg = final_true_pose_delta_summary['aggregate']
    print(f"  True pose delta summary: mean_trans={agg['mean_trans_norm']:.6f}, max_trans={agg['max_trans_norm']:.6f}, mean_rotF={agg['mean_rot_fro_norm']:.6f}, max_rotF={agg['max_rot_fro_norm']:.6f}")
    print(f'  StageA history: {output_dir / "stageA_history.json"}')
    if stageB_history is not None:
        print(f'  StageB history: {output_dir / "stageB_history.json"}')


if __name__ == '__main__':
    main()

