#!/usr/bin/env python3
"""Round 2: Verify pose gradient in actual S3PO render context.

Tests that the fix works with the real GaussianRasterizer.
"""

import torch
import sys
import os

# Add paths
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS/gaussian_splatting')

print("=" * 60)
print("Pose Gradient Fix - Round 2: Real Render Test")
print("=" * 60)

# Import S3PO components
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.gaussian_renderer import GaussianRasterizationSettings
from utils.camera_utils import Camera

print("\n[Step 1] Creating mock Gaussian model...")
gaussians = GaussianModel(0)
gaussians._xyz = torch.randn(100, 3, device='cuda') * 0.1
gaussians._features_dc = torch.zeros(100, 1, 3, device='cuda')
gaussians._features_rest = torch.zeros(100, 0, 3, device='cuda')
gaussians._scaling = torch.ones(100, 3, device='cuda') * 0.05
gaussians._rotation = torch.zeros(100, 4, device='cuda')
gaussians._rotation[:, 0] = 1.0  # unit quaternions
gaussians._opacity = torch.ones(100, 1, device='cuda') * 0.5
gaussians.active_sh_degree = 0

# Setup optimizer params for gradient tracking
gaussians._xyz = torch.nn.Parameter(gaussians._xyz, requires_grad=True)
gaussians._opacity = torch.nn.Parameter(gaussians._opacity, requires_grad=True)
gaussians._scaling = torch.nn.Parameter(gaussians._scaling, requires_grad=True)
gaussians._rotation = torch.nn.Parameter(gaussians._rotation, requires_grad=True)
gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc, requires_grad=True)

print(f"  Gaussians created: {gaussians.get_xyz.shape[0]} points")

# Create camera with pose delta
print("\n[Step 2] Creating camera with pose delta...")

from utils.pose_utils import SE3_exp

class MockCamera:
    def __init__(self):
        self.uid = 0
        self.R = torch.eye(3, device='cuda')
        self.T = torch.zeros(3, device='cuda')
        self.R0 = self.R.clone()
        self.T0 = self.T.clone()
        self.cam_rot_delta = torch.nn.Parameter(torch.tensor([0.05, 0.0, 0.0], device='cuda'), requires_grad=True)
        self.cam_trans_delta = torch.nn.Parameter(torch.tensor([0.1, 0.0, 0.0], device='cuda'), requires_grad=True)
        self.exposure_a = torch.nn.Parameter(torch.zeros(1, device='cuda'), requires_grad=True)
        self.exposure_b = torch.nn.Parameter(torch.zeros(1, device='cuda'), requires_grad=True)

        # Camera intrinsics
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 256.0
        self.cy = 256.0
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.image_width = 512
        self.image_height = 512

        # Projection matrix
        self.projection_matrix = torch.eye(4, device='cuda')

        # Update transforms
        self._update_transforms()

    def _update_transforms(self):
        from gaussian_splatting.utils.graphics_utils import getWorld2View2
        self.world_view_transform = getWorld2View2(self.R, self.T).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def apply_pose_delta(self):
        """Apply pose delta - the FIX."""
        from utils.pose_utils import SE3_exp
        base = torch.eye(4, device='cuda')
        base[:3, :3] = self.R
        base[:3, 3] = self.T
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], dim=0)
        w2c_current = SE3_exp(tau) @ base
        self.world_view_transform = w2c_current.transpose(0, 1).contiguous()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0).contiguous()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

camera = MockCamera()
print(f"  Camera created with pose delta: rot={camera.cam_rot_delta.data.tolist()}, trans={camera.cam_trans_delta.data.tolist()}")

# Test render WITHOUT pose delta (baseline)
print("\n[Step 3] Render WITHOUT pose delta (baseline - S3PO bug)...")

camera._update_transforms()  # Use R/T only

bg_color = torch.zeros(3, device='cuda')
pipe = type('obj', (object,), {
    'compute_cov3D_python': False,
    'convert_SHs_python': False,
})()

raster_settings = GaussianRasterizationSettings(
    image_height=512,
    image_width=512,
    tanfovx=math.tan(camera.FoVx * 0.5) if hasattr(camera, 'FoVx') else 0.5,
    tanfovy=math.tan(camera.FoVy * 0.5) if hasattr(camera, 'FoVy') else 0.5,
    bg=bg_color,
    scale_modifier=1.0,
    viewmatrix=camera.world_view_transform,
    projmatrix=camera.full_proj_transform,
    projmatrix_raw=camera.projection_matrix,
    sh_degree=0,
    campos=camera.camera_center,
    prefiltered=False,
    debug=False,
)

import math
raster_settings.tanfovx = 0.5
raster_settings.tanfovy = 0.5

# Zero gradients
if camera.cam_rot_delta.grad is not None:
    camera.cam_rot_delta.grad.zero_()
if camera.cam_trans_delta.grad is not None:
    camera.cam_trans_delta.grad.zero_()

# Render baseline
render_pkg_baseline = render(camera, gaussians, pipe, bg_color)
rgb_baseline = render_pkg_baseline['render']

print(f"  Baseline render shape: {rgb_baseline.shape}")
print(f"  Baseline render mean: {rgb_baseline.mean().item():.4f}")

# Compute loss and backward
target = torch.zeros(3, 512, 512, device='cuda')
loss_baseline = torch.abs(rgb_baseline - target).mean()
loss_baseline.backward()

baseline_rot_grad = camera.cam_rot_delta.grad.clone() if camera.cam_rot_delta.grad is not None else torch.zeros(3, device='cuda')
baseline_trans_grad = camera.cam_trans_delta.grad.clone() if camera.cam_trans_delta.grad is not None else torch.zeros(3, device='cuda')

print(f"  Baseline cam_rot_delta.grad norm: {baseline_rot_grad.norm().item():.6f}")
print(f"  Baseline cam_trans_delta.grad norm: {baseline_trans_grad.norm().item():.6f}")

# Test render WITH pose delta (fixed)
print("\n[Step 4] Render WITH pose delta (fixed)...")

# Reset gradients
camera.cam_rot_delta.grad = None
camera.cam_trans_delta.grad = None
gaussians._xyz.grad = None

# Apply pose delta
camera.apply_pose_delta()

print(f"  world_view_transform after fix:\n{camera.world_view_transform[:2, :2]}")

# Render with pose delta
render_pkg_fixed = render(camera, gaussians, pipe, bg_color)
rgb_fixed = render_pkg_fixed['render']

print(f"  Fixed render shape: {rgb_fixed.shape}")
print(f"  Fixed render mean: {rgb_fixed.mean().item():.4f}")

# Compute loss and backward
loss_fixed = torch.abs(rgb_fixed - target).mean()
loss_fixed.backward()

fixed_rot_grad = camera.cam_rot_delta.grad.clone() if camera.cam_rot_delta.grad is not None else torch.zeros(3, device='cuda')
fixed_trans_grad = camera.cam_trans_delta.grad.clone() if camera.cam_trans_delta.grad is not None else torch.zeros(3, device='cuda')

print(f"  Fixed cam_rot_delta.grad norm: {fixed_rot_grad.norm().item():.6f}")
print(f"  Fixed cam_trans_delta.grad norm: {fixed_trans_grad.norm().item():.6f}")

# Compare
print("\n[Step 5] Comparing baseline vs fixed...")

render_differs = not torch.allclose(rgb_baseline, rgb_fixed, atol=1e-4)
print(f"  ✓ Render outputs differ: {render_differs}")

grad_increased = fixed_rot_grad.norm() > baseline_rot_grad.norm() or fixed_trans_grad.norm() > baseline_trans_grad.norm()
print(f"  ✓ Fixed has more pose gradient: {grad_increased}")

baseline_has_zero_grad = baseline_rot_grad.norm() < 1e-6 and baseline_trans_grad.norm() < 1e-6
print(f"  ✓ Baseline has near-zero pose gradient (expected bug): {baseline_has_zero_grad}")

fixed_has_grad = fixed_rot_grad.norm() > 1e-6 or fixed_trans_grad.norm() > 1e-6
print(f"  ✓ Fixed has non-zero pose gradient: {fixed_has_grad}")

print("\n" + "=" * 60)
if render_differs and fixed_has_grad:
    print("SUCCESS! Pose gradient fix works with real render!")
    print("=" * 60)
    print("\nKey findings:")
    print(f"  1. Baseline render (R/T only): pose grad ~ {baseline_rot_grad.norm().item():.6f}")
    print(f"  2. Fixed render (R/T+delta): pose grad ~ {fixed_rot_grad.norm().item():.6f}")
    print(f"  3. Pose delta affects render output: {render_differs}")
    print("\nThe fix enables pose optimization in S3PO online mapping!")
else:
    print("FAILURE! Pose gradient fix may not work correctly.")
    print("=" * 60)
    sys.exit(1)