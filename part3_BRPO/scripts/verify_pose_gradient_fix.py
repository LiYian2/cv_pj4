#!/usr/bin/env python3
"""Verification test for pose gradient fix and Gauss-Newton module.

Tests:
1. Pose gradient: theta/rho should receive gradient from RGB/depth loss
2. Gauss-Newton: pose update should reduce loss
"""

import sys
import os

# Add paths
sys.path.insert(0, '/home/bzhang512/CV_Project/part3_BRPO')
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS')
sys.path.insert(0, '/home/bzhang512/CV_Project/third_party/S3PO-GS/gaussian_splatting')

import torch
import numpy as np

print("=" * 60)
print("Pose Gradient Fix Verification Test")
print("=" * 60)

# Test 1: Check apply_pose_delta_before_render_ changes world_view_transform
print("\n[Test 1] apply_pose_delta_before_render_ changes transforms...")

from pseudo_branch.refine.pseudo_camera_state import (
    apply_pose_delta_before_render_,
    current_w2c,
    make_viewpoint_trainable,
)

# Create mock viewpoint
class MockViewpoint:
    def __init__(self):
        self.R = torch.eye(3, device='cuda')
        self.T = torch.zeros(3, device='cuda')
        self.cam_rot_delta = torch.zeros(3, device='cuda', requires_grad=True)
        self.cam_trans_delta = torch.zeros(3, device='cuda', requires_grad=True)
        self.projection_matrix = torch.eye(4, device='cuda')
        self.world_view_transform = torch.eye(4, device='cuda')
        self.full_proj_transform = torch.eye(4, device='cuda')
        self.camera_center = torch.zeros(3, device='cuda')

vp = MockViewpoint()
make_viewpoint_trainable(vp)

# Initial world_view_transform
initial_wvt = vp.world_view_transform.clone()

# Set non-zero pose delta
vp.cam_rot_delta.data = torch.tensor([0.01, 0.02, 0.03], device='cuda')
vp.cam_trans_delta.data = torch.tensor([0.1, 0.2, 0.3], device='cuda')

# Apply pose delta
apply_pose_delta_before_render_(vp)

# Check that world_view_transform changed
wvt_changed = not torch.allclose(initial_wvt, vp.world_view_transform)
print(f"  Initial wvt: {initial_wvt[:3, :3].flatten()[:3].tolist()}")
print(f"  After apply: {vp.world_view_transform[:3, :3].flatten()[:3].tolist()}")
print(f"  ✓ world_view_transform changed: {wvt_changed}")

# Check that current_w2c matches world_view_transform
expected_w2c = current_w2c(vp)
expected_wvt = expected_w2c.transpose(0, 1)
matches = torch.allclose(vp.world_view_transform, expected_wvt, atol=1e-5)
print(f"  ✓ world_view_transform matches current_w2c: {matches}")

if wvt_changed and matches:
    print("  [PASS] Test 1 passed!")
else:
    print("  [FAIL] Test 1 failed!")
    sys.exit(1)

# Test 2: Check gradient flows through render to theta/rho
print("\n[Test 2] Gradient flows from loss to theta/rho...")

# Mock render function that uses world_view_transform
def mock_render_loss(viewpoint):
    """Simulates render + loss, using world_view_transform."""
    # This simulates what forward.cu does: use world_view_transform
    # Loss = sum of world_view_transform elements (simplified)
    loss = viewpoint.world_view_transform.sum()
    return loss

# Reset pose delta
vp.cam_rot_delta.data = torch.tensor([0.01, 0.0, 0.0], device='cuda')
vp.cam_trans_delta.data = torch.tensor([0.1, 0.0, 0.0], device='cuda')
apply_pose_delta_before_render_(vp)

# Compute loss and backward
loss = mock_render_loss(vp)
loss.backward()

# Check gradient exists on theta/rho
rot_grad_exists = vp.cam_rot_delta.grad is not None and vp.cam_rot_delta.grad.abs().sum() > 0
trans_grad_exists = vp.cam_trans_delta.grad is not None and vp.cam_trans_delta.grad.abs().sum() > 0

print(f"  Loss value: {loss.item():.4f}")
print(f"  cam_rot_delta.grad: {vp.cam_rot_delta.grad.tolist() if vp.cam_rot_delta.grad is not None else 'None'}")
print(f"  cam_trans_delta.grad: {vp.cam_trans_delta.grad.tolist() if vp.cam_trans_delta.grad is not None else 'None'}")
print(f"  ✓ Rotation gradient exists: {rot_grad_exists}")
print(f"  ✓ Translation gradient exists: {trans_grad_exists}")

if rot_grad_exists and trans_grad_exists:
    print("  [PASS] Test 2 passed!")
else:
    print("  [FAIL] Test 2 failed!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Gauss-Newton Module Verification Test")
print("=" * 60)

# Test 3: Gauss-Newton reduces loss
print("\n[Test 3] Gauss-Newton reduces loss...")

from pseudo_branch.refine.pose_gauss_newton import (
    gauss_newton_pose_update,
    GaussNewtonPoseOptimizer,
)

# Create new viewpoint with large pose delta
vp2 = MockViewpoint()
make_viewpoint_trainable(vp2)
vp2.cam_rot_delta.data = torch.tensor([0.1, 0.1, 0.1], device='cuda')
vp2.cam_trans_delta.data = torch.tensor([0.5, 0.5, 0.5], device='cuda')

def simple_loss_fn(viewpoint):
    """Simple loss: minimize pose delta magnitude."""
    apply_pose_delta_before_render_(viewpoint)
    return (viewpoint.cam_rot_delta.norm() + viewpoint.cam_trans_delta.norm())

initial_loss = simple_loss_fn(vp2).item()
initial_tau_norm = (vp2.cam_rot_delta.norm() + vp2.cam_trans_delta.norm()).item()

print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Initial tau norm: {initial_tau_norm:.4f}")

# Run Gauss-Newton
optimizer = GaussNewtonPoseOptimizer(max_iters=5, damping=0.1)
converged, stats = optimizer.optimize(vp2, simple_loss_fn, verbose=False)

final_loss = simple_loss_fn(vp2).item()
final_tau_norm = (vp2.cam_rot_delta.norm() + vp2.cam_trans_delta.norm()).item()

print(f"  Final loss: {final_loss:.4f}")
print(f"  Final tau norm: {final_tau_norm:.4f}")
print(f"  Iterations: {stats['iterations']}")
print(f"  Converged: {converged}")

loss_reduced = final_loss < initial_loss
print(f"  ✓ Loss reduced: {loss_reduced}")

if loss_reduced:
    print("  [PASS] Test 3 passed!")
else:
    print("  [FAIL] Test 3 failed!")
    sys.exit(1)

# Test 4: Scale regularization
print("\n[Test 4] Scale regularization loss works...")

from pseudo_branch.refine.pseudo_loss_v2 import scale_reg_loss

# Mock Gaussians
class MockGaussians:
    def __init__(self):
        self._scaling = torch.ones(10, 3, device='cuda') * 0.1

    @property
    def get_scaling(self):
        return self._scaling

gaussians = MockGaussians()
initial_scale_loss = scale_reg_loss(gaussians).item()
print(f"  Initial scale loss (isotropic): {initial_scale_loss:.4f}")

# Make scales non-isotropic
gaussians._scaling[:, 0] = 0.5  # Large scale
non_iso_loss = scale_reg_loss(gaussians).item()
print(f"  Non-isotropic scale loss: {non_iso_loss:.4f}")

scale_loss_increased = non_iso_loss > initial_scale_loss
print(f"  ✓ Scale loss increased for non-isotropic: {scale_loss_increased}")

# Test max_scale
gaussians._scaling[:, 0] = 2.0  # Exceed max_scale
max_scale_loss = scale_reg_loss(gaussians, max_scale=1.0).item()
print(f"  Scale loss with max_scale=1.0: {max_scale_loss:.4f}")

max_scale_penalty_works = max_scale_loss > non_iso_loss
print(f"  ✓ Max_scale penalty works: {max_scale_penalty_works}")

if scale_loss_increased and max_scale_penalty_works:
    print("  [PASS] Test 4 passed!")
else:
    print("  [FAIL] Test 4 failed!")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nSummary:")
print("  1. apply_pose_delta_before_render_ correctly changes world_view_transform")
print("  2. Gradient flows from loss to theta/rho")
print("  3. Gauss-Newton reduces pose delta")
print("  4. Scale regularization works correctly")
print("\nNext: Run full online mapping experiment to verify real-world performance.")