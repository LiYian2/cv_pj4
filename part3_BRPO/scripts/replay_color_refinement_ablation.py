#!/usr/bin/env python3
"""Replay color refinement with optional pseudo exclusion.

Usage:
    python replay_color_refinement_ablation.py         --experiment_dir /data3/bzhang512/.../2026-05-08-18-06-08         --output_suffix no_pseudo         --iterations 26000         --gpu 0

This script loads before_opt ply and camera states, then runs color refinement
using only real KF views (no pseudo) to compare the effect.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import yaml

# Set paths before imports
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.sh_utils import SH2RGB
from utils.slam_utils import get_loss_mapping
from munch import munchify
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.camera_utils import Camera
from utils.eval_utils import eval_rendering
from utils.dataset import dl3dvDataset


def load_gaussians_from_ply(ply_path: Path, device: str = 'cuda'):
    """Load Gaussians from PLY file using GaussianModel's built-in loader."""
    gaussians = GaussianModel(0)  # sh_degree=0
    gaussians.load_ply(str(ply_path))
    
    # Ensure tensors are on device and require grad
    gaussians._xyz = gaussians._xyz.to(device).requires_grad_(True)
    gaussians._opacity = gaussians._opacity.to(device).requires_grad_(True)
    gaussians._scaling = gaussians._scaling.to(device).requires_grad_(True)
    gaussians._rotation = gaussians._rotation.to(device).requires_grad_(True)
    gaussians._features_dc = gaussians._features_dc.to(device).requires_grad_(True)
    gaussians.active_sh_degree = 0
    
    return gaussians


def load_cameras_from_json(json_path: Path):
    """Load cameras from camera_states.json."""
    with open(json_path, 'r') as f:
        camera_states = json.load(f)
    
    cameras_info = {}
    for state in camera_states:
        frame_id = state['frame_id']
        uid = state['uid']
        is_kf = state.get('is_keyframe', False)
        
        R = np.array(state['R'])
        T = np.array(state['T'])
        
        cameras_info[frame_id] = {
            'uid': uid,
            'is_keyframe': is_kf,
            'R': R,
            'T': T,
        }
    
    return cameras_info


def build_camera_with_pose(dataset, frame_id: int, R: np.ndarray, T: np.ndarray, projection_matrix: torch.Tensor, device: str):
    """Build Camera object with specified pose."""
    gt_color, gt_depth, gt_pose, mono_depth = dataset[frame_id]
    
    # Create Camera with GT pose first
    cam = Camera.init_from_dataset(dataset, frame_id, projection_matrix)
    
    # Update to optimized pose
    R_tensor = torch.tensor(R, dtype=torch.float32, device=device)
    T_tensor = torch.tensor(T, dtype=torch.float32, device=device)
    cam.update_RT(R_tensor, T_tensor)
    
    return cam


def run_color_refinement(
    gaussians: GaussianModel,
    cameras_list: list,
    iterations: int,
    lambda_dssim: float,
    background: torch.Tensor,
    device: str,
):
    """Run color refinement loop using only real KF views."""
    
    # Setup optimizer (matching original S3PO color refinement)
    # gaussians.training_setup() # Not needed for color refinement only
    
    # Re-create optimizer with typical color refinement learning rates
    gaussians.optimizer = torch.optim.Adam([
        {'params': [gaussians._xyz], 'lr': 0.00016, 'name': 'xyz'},
        {'params': [gaussians._features_dc], 'lr': 0.0025, 'name': 'f_dc'},
        {'params': [gaussians._opacity], 'lr': 0.05, 'name': 'opacity'},
        {'params': [gaussians._scaling], 'lr': 0.001, 'name': 'scaling'},
        {'params': [gaussians._rotation], 'lr': 0.001, 'name': 'rotation'},
    ])
    
    # Statistics
    num_real_steps = 0
    real_loss_sum = 0.0
    
    print(f"Starting color refinement: iterations={iterations}, cameras={len(cameras_list)}")
    
    for iteration in tqdm(range(1, iterations + 1)):
        gaussians.optimizer.zero_grad(set_to_none=True)
        
        # Sample random camera
        viewpoint_cam = random.choice(cameras_list)
        
        # Render
        pipeline_params = munchify({"compute_cov3D_python": False, "convert_SHs_python": False})
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
        image = render_pkg['render']
        
        # GT image
        gt_image = viewpoint_cam.original_image
        
        # Full-image L1 + SSIM loss (same as original S3PO)
        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
        
        num_real_steps += 1
        real_loss_sum += float(loss.detach().item())
        
        loss.backward()
        gaussians.optimizer.step()
    
    # Summary
    summary = {
        "use_pseudo": False,
        "pseudo_pool_size": 0,
        "num_real_steps": num_real_steps,
        "num_pseudo_steps": 0,
        "mean_real_loss": real_loss_sum / max(num_real_steps, 1),
        "mean_pseudo_loss": 0.0,
        "color_refinement_updates_pose": False,
        "total_iterations": iterations,
    }
    
    return gaussians, summary


class SimpleModelParams:
    """Minimal model params for dataset loading."""
    def __init__(self, config):
        ds_cfg = config['Dataset']
        cal_cfg = ds_cfg.get('Calibration', {})
        
        self.fx = cal_cfg.get('fx', 408.0)
        self.fy = cal_cfg.get('fy', 408.0)
        self.cx = cal_cfg.get('cx', 256.0)
        self.cy = cal_cfg.get('cy', 256.0)
        self.width = cal_cfg.get('width', 512)
        self.height = cal_cfg.get('height', 512)
        self.fovx = 2 * np.arctan(self.width / (2 * self.fx))
        self.fovy = 2 * np.arctan(self.height / (2 * self.fy))
        
        self.source_path = ds_cfg['dataset_path']
        self.sh_degree = 0
        self.resolution = -1
        self.white_background = False
        self.data_device = 'cuda'
        
        # Compute projection matrix
        znear = 0.01
        zfar = 100.0
        P = getProjectionMatrix2(znear, zfar, self.fx, self.fy, self.cx, self.cy, self.width, self.height)
        self.projection_matrix = P.transpose(0, 1)


def main():
    parser = argparse.ArgumentParser(description='Replay color refinement ablation')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory (e.g., .../2026-05-08-18-06-08)')
    parser.add_argument('--output_suffix', type=str, default='no_pseudo',
                        help='Suffix for output directory')
    parser.add_argument('--iterations', type=int, default=26000,
                        help='Number of color refinement iterations')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'
    
    experiment_dir = Path(args.experiment_dir)
    before_opt_dir = experiment_dir / 'internal_eval_cache' / 'before_opt'
    
    # Load before_opt ply
    ply_path = before_opt_dir / 'point_cloud' / 'point_cloud.ply'
    if not ply_path.exists():
        print(f"PLY not found: {ply_path}")
        sys.exit(1)
    
    print(f"Loading Gaussians from {ply_path}")
    gaussians = load_gaussians_from_ply(ply_path, device)
    print(f"Loaded {gaussians._xyz.shape[0]} Gaussians")
    
    # Load camera states
    camera_json = experiment_dir / 'internal_eval_cache' / 'camera_states.json'
    if not camera_json.exists():
        print(f"Camera states not found: {camera_json}")
        sys.exit(1)
    
    print(f"Loading camera states from {camera_json}")
    cameras_info = load_cameras_from_json(camera_json)
    kf_frame_ids = [fid for fid, cam in cameras_info.items() if cam['is_keyframe']]
    print(f"Keyframe frame IDs: {kf_frame_ids}")
    
    # Load config
    config_path = experiment_dir / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = config['Dataset']['dataset_path']
    print(f"Dataset path: {dataset_path}")
    
    # Load dataset using dl3dvDataset
    print("Loading dataset...")
    model_params = SimpleModelParams(config)
    dataset = dl3dvDataset(model_params, model_params.source_path, config)
    print(f"Dataset size: {len(dataset)}")
    
    # Build Camera objects with optimized poses
    print("Building Camera objects...")
    cameras_list = []
    for fid in kf_frame_ids:
        cam_info = cameras_info[fid]
        cam = build_camera_with_pose(dataset, fid, cam_info['R'], cam_info['T'], model_params.projection_matrix, device)
        cameras_list.append(cam)
    print(f"Built {len(cameras_list)} Camera objects")
    
    # Background color
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    # Run color refinement
    lambda_dssim = config.get('opt_params', {}).get('lambda_dssim', 0.2)
    gaussians, summary = run_color_refinement(
        gaussians=gaussians,
        cameras_list=cameras_list,
        iterations=args.iterations,
        lambda_dssim=lambda_dssim,
        background=background,
        device=device,
    )
    
    # Save results
    output_dir = experiment_dir / f'replay_color_refinement_{args.output_suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ply
    ply_output = output_dir / 'point_cloud.ply'
    gaussians.save_ply(str(ply_output))
    print(f"Saved ply to {ply_output}")
    
    # Save summary
    summary_path = output_dir / 'color_refinement_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    print(f"Summary: mean_real_loss={summary['mean_real_loss']:.6f}")
    
    # Evaluate - need all frames for evaluation
    print("Building all Camera objects for evaluation...")
    all_cameras = []
    for fid in range(len(dataset)):
        cam_info = cameras_info.get(fid, None)
        if cam_info is not None:
            # Use optimized pose
            cam = build_camera_with_pose(dataset, fid, cam_info['R'], cam_info['T'], model_params.projection_matrix, device)
        else:
            # Use GT pose (for non-KF frames)
            gt_color, gt_depth, gt_pose, mono_depth = dataset[fid]
            cam = Camera.init_from_dataset(dataset, fid, model_params.projection_matrix)
        all_cameras.append(cam)
    
    # Run evaluation
    print("Running evaluation...")
    pipeline_params = munchify({"compute_cov3D_python": False, "convert_SHs_python": False})
    results = eval_rendering(
        all_cameras,
        gaussians,
        dataset,
        str(output_dir),
        pipeline_params,
        background,
        datatype='dl3dv',
        kf_indices=kf_frame_ids,
        iteration='final',
    )
    
    # Save final results
    final_results = {
        'mean_psnr': results.get('mean_psnr', 0),
        'mean_ssim': results.get('mean_ssim', 0),
        'mean_lpips': results.get('mean_lpips', 0),
    }
    with open(output_dir / 'final_result.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("=" * 60)
    print("Done! Results:")
    print(f"  PSNR: {final_results['mean_psnr']:.4f}")
    print(f"  SSIM: {final_results['mean_ssim']:.4f}")
    print(f"  LPIPS: {final_results['mean_lpips']:.4f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
