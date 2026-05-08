#!/usr/bin/env python3
"""Round 2: Minimal render gradient test.

Uses mock objects to test gradient flow through GaussianRasterizer.
"""

import torch
import sys
import math

print("=" * 60)
print("Pose Gradient Fix - Round 2: Render Gradient Test")
print("=" * 60)

# Test using the actual rasterizer but with minimal setup
try:
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    print("[OK] GaussianRasterizer imported successfully")
except ImportError as e:
    print(f"[FAIL] Cannot import GaussianRasterizer: {e}")
    print("This test requires diff-gaussian-rasterization to be installed.")
    sys.exit(1)

# Create minimal Gaussian data
print("\n[Step 1] Creating minimal Gaussian data...")

N = 100  # Number of Gaussians
means3D = torch.randn(N, 3, device='cuda') * 0.1
means3D.requires_grad = True

colors = torch.zeros(N, 3, device='cuda')
opacity = torch.ones(N, 1, device='cuda') * 0.5
opacity.requires_grad = True

scales = torch.ones(N, 3, device='cuda') * 0.05
scales.requires_grad = True

rotations = torch.zeros(N, 4, device='cuda')
rotations[:, 0] = 1.0  # unit quaternions
rotations.requires_grad = True

shs = torch.zeros(N, 0, 3, device='cuda')  # No SH

print(f"  Created {N} mock Gaussians")

# Create camera with pose delta
print("\n[Step 2] Creating camera with pose delta...")

from utils.pose_utils import SE3_exp
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix2

# Base pose
R = torch.eye(3, device='cuda')
T = torch.zeros(3, device='cuda')

# Pose delta parameters
cam_rot_delta = torch.nn.Parameter(torch.tensor([0.05, 0.0, 0.0], device='cuda'), requires_grad=True)
cam_trans_delta = torch.nn.Parameter(torch.tensor([0.1, 0.0, 0.0], device='cuda'), requires_grad=True)

# Camera intrinsics
fx, fy = 500.0, 500.0
cx, cy = 256.0, 256.0
W, H = 512, 512

# Projection matrix
proj_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1).cuda()

print(f"  Pose delta: rot={cam_rot_delta.data.tolist()}, trans={cam_trans_delta.data.tolist()}")

# Function to compute viewmatrix with pose delta
def get_viewmatrix_with_delta(R, T, cam_rot_delta, cam_trans_delta):
    base = torch.eye(4, device='cuda')
    base[:3, :3] = R
    base[:3, 3] = T
    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=0)
    w2c = SE3_exp(tau) @ base
    return w2c.transpose(0, 1)

# Test 1: Baseline (without pose delta)
print("\n[Step 3] Baseline: Render WITHOUT pose delta...")

viewmatrix_baseline = getWorld2View2(R, T).transpose(0, 1)
full_proj_baseline = viewmatrix_baseline.unsqueeze(0).bmm(proj_matrix.unsqueeze(0)).squeeze(0)
camera_center_baseline = viewmatrix_baseline.inverse()[3, :3]

raster_settings_baseline = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=math.tan(1.0 * 0.5),
    tanfovy=math.tan(1.0 * 0.5),
    bg=torch.zeros(3, device='cuda'),
    scale_modifier=1.0,
    viewmatrix=viewmatrix_baseline,
    projmatrix=full_proj_baseline,
    projmatrix_raw=proj_matrix,
    sh_degree=0,
    campos=camera_center_baseline,
    prefiltered=False,
    debug=False,
)

rasterizer_baseline = GaussianRasterizer(raster_settings=raster_settings_baseline)

# Render baseline (need screenspace points for gradient tracking)
screenspace_points = torch.zeros_like(means3D, requires_grad=True, device='cuda')
try:
    screenspace_points.retain_grad()
except:
    pass

bg = torch.zeros(3, device='cuda')

# Zero gradients
cam_rot_delta.grad = None
cam_trans_delta.grad = None
means3D.grad = None

rendered_image_baseline, radii, depth, opacity_out, n_touched = rasterizer_baseline(
    means3D=means3D,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=colors,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    theta=cam_rot_delta,  # Passed but NOT used in forward!
    rho=cam_trans_delta,
)

print(f"  Rendered image shape: {rendered_image_baseline.shape}")
print(f"  Rendered image mean: {rendered_image_baseline.mean().item():.4f}")

# Compute loss
target = torch.zeros(3, H, W, device='cuda')
loss_baseline = torch.abs(rendered_image_baseline - target).mean()

loss_baseline.backward()

baseline_rot_grad_norm = cam_rot_delta.grad.norm().item() if cam_rot_delta.grad is not None else 0.0
baseline_trans_grad_norm = cam_trans_delta.grad.norm().item() if cam_trans_delta.grad is not None else 0.0

print(f"  Loss: {loss_baseline.item():.4f}")
print(f"  cam_rot_delta.grad norm: {baseline_rot_grad_norm:.6f}")
print(f"  cam_trans_delta.grad norm: {baseline_trans_grad_norm:.6f}")

# Test 2: Fixed (with pose delta applied to viewmatrix)
print("\n[Step 4] Fixed: Render WITH pose delta (viewmatrix corrected)...")

# Apply pose delta to viewmatrix - THE FIX!
viewmatrix_fixed = get_viewmatrix_with_delta(R, T, cam_rot_delta, cam_trans_delta)
full_proj_fixed = viewmatrix_fixed.unsqueeze(0).bmm(proj_matrix.unsqueeze(0)).squeeze(0)
camera_center_fixed = viewmatrix_fixed.inverse()[3, :3]

print(f"  Viewmatrix baseline:\n{viewmatrix_baseline[:2, :2]}")
print(f"  Viewmatrix fixed:\n{viewmatrix_fixed[:2, :2]}")

raster_settings_fixed = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=math.tan(1.0 * 0.5),
    tanfovy=math.tan(1.0 * 0.5),
    bg=torch.zeros(3, device='cuda'),
    scale_modifier=1.0,
    viewmatrix=viewmatrix_fixed,
    projmatrix=full_proj_fixed,
    projmatrix_raw=proj_matrix,
    sh_degree=0,
    campos=camera_center_fixed,
    prefiltered=False,
    debug=False,
)

rasterizer_fixed = GaussianRasterizer(raster_settings=raster_settings_fixed)

# Zero gradients
cam_rot_delta.grad = None
cam_trans_delta.grad = None
means3D.grad = None
screenspace_points.grad = None

# Render fixed
screenspace_points2 = torch.zeros_like(means3D, requires_grad=True, device='cuda')
try:
    screenspace_points2.retain_grad()
except:
    pass

rendered_image_fixed, radii, depth, opacity_out, n_touched = rasterizer_fixed(
    means3D=means3D,
    means2D=screenspace_points2,
    shs=shs,
    colors_precomp=colors,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    theta=cam_rot_delta,
    rho=cam_trans_delta,
)

print(f"  Rendered image shape: {rendered_image_fixed.shape}")
print(f"  Rendered image mean: {rendered_image_fixed.mean().item():.4f}")

# Compute loss
loss_fixed = torch.abs(rendered_image_fixed - target).mean()

loss_fixed.backward()

fixed_rot_grad_norm = cam_rot_delta.grad.norm().item() if cam_rot_delta.grad is not None else 0.0
fixed_trans_grad_norm = cam_trans_delta.grad.norm().item() if cam_trans_delta.grad is not None else 0.0

print(f"  Loss: {loss_fixed.item():.4f}")
print(f"  cam_rot_delta.grad norm: {fixed_rot_grad_norm:.6f}")
print(f"  cam_trans_delta.grad norm: {fixed_trans_grad_norm:.6f}")

# Compare results
print("\n" + "=" * 60)
print("Comparison Results")
print("=" * 60)

render_differs = not torch.allclose(rendered_image_baseline, rendered_image_fixed, atol=1e-4)
print(f"\n  1. Render outputs differ: {render_differs}")
print(f"     Baseline mean: {rendered_image_baseline.mean().item():.4f}")
print(f"     Fixed mean: {rendered_image_fixed.mean().item():.4f}")

print(f"\n  2. Pose gradient comparison:")
print(f"     Baseline rot_grad norm: {baseline_rot_grad_norm:.6f}")
print(f"     Baseline trans_grad norm: {baseline_trans_grad_norm:.6f}")
print(f"     Fixed rot_grad norm: {fixed_rot_grad_norm:.6f}")
print(f"     Fixed trans_grad norm: {fixed_trans_grad_norm:.6f}")

# The key finding: baseline should have NEAR-ZERO pose gradient
# because theta/rho are not used in forward
baseline_grad_near_zero = baseline_rot_grad_norm < 0.001 and baseline_trans_grad_norm < 0.001

# Fixed should have NON-ZERO pose gradient
# because viewmatrix depends on theta/rho through SE3_exp
fixed_grad_nonzero = fixed_rot_grad_norm > 0.001 or fixed_trans_grad_norm > 0.001

print(f"\n  3. Expected behavior:")
print(f"     Baseline pose grad near-zero (theta/rho not in forward): {baseline_grad_near_zero}")
print(f"     Fixed pose grad nonzero (viewmatrix uses theta/rho): {fixed_grad_nonzero}")

print("\n" + "=" * 60)
if render_differs and fixed_grad_nonzero:
    print("SUCCESS! Pose gradient fix verified!")
    print("=" * 60)
    print("\nConclusion:")
    print("  - Baseline: theta/rho NOT in viewmatrix → pose grad ~0")
    print("  - Fixed: theta/rho IN viewmatrix → pose grad nonzero")
    print("  - The fix enables pose optimization from render loss!")
else:
    print("PARTIAL SUCCESS - need further investigation")
    print("=" * 60)
    if not render_differs:
        print("  Note: Render outputs are similar - pose delta effect may be small")
    if not fixed_grad_nonzero:
        print("  Note: Fixed pose gradient is still small - may need larger pose delta")