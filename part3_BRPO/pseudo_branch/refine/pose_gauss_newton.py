"""Gauss-Newton pose optimization for pseudo view refinement.

This module implements the Gauss-Newton optimization algorithm for pose refinement,
following the BRPO paper section 3.2 approach.

Key difference from Adam optimizer:
- Gauss-Newton directly computes pose Jacobian via finite difference
- Updates pose in closed form (H^-1 @ J.T @ residual)
- More efficient for pose optimization than Adam's gradient descent
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any, Callable, Tuple

from utils.pose_utils import SE3_exp, SE3_log


def compute_pose_jacobian_fd(
    viewpoint: Any,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute pose Jacobian via finite difference.

    Args:
        viewpoint: Camera/viewpoint with cam_rot_delta, cam_trans_delta
        loss_fn: Function that takes (render_rgb, render_depth) and returns loss
        epsilon: Finite difference step size

    Returns:
        Jacobian matrix J of shape (6, loss_dim) where:
        - rows 0-2: translation (rho)
        - rows 3-5: rotation (theta)
    """
    device = viewpoint.cam_rot_delta.device
    dtype = viewpoint.cam_rot_delta.dtype

    # Current pose delta
    rho = viewpoint.cam_trans_delta.detach().clone()
    theta = viewpoint.cam_rot_delta.detach().clone()
    tau = torch.cat([rho, theta], dim=0)  # shape (6,)

    # Compute loss at current pose
    loss_current = loss_fn(viewpoint)

    # Handle scalar vs vector loss
    if loss_current.ndim == 0:
        loss_dim = 1
    else:
        loss_dim = loss_current.shape[0]

    J = torch.zeros(6, loss_dim, device=device, dtype=dtype)

    # Compute finite difference for each component
    for i in range(6):
        tau_plus = tau.clone()
        tau_plus[i] += epsilon

        # Apply perturbation
        with torch.no_grad():
            tau_plus_rho = tau_plus[:3]
            tau_plus_theta = tau_plus[3:]

        # Temporarily modify viewpoint
        old_rho = viewpoint.cam_trans_delta.detach().clone()
        old_theta = viewpoint.cam_rot_delta.detach().clone()

        viewpoint.cam_rot_delta.data = tau_plus_theta
        viewpoint.cam_trans_delta.data = tau_plus_rho

        # Compute loss at perturbed pose
        loss_plus = loss_fn(viewpoint)

        # Restore original pose
        viewpoint.cam_rot_delta.data = old_theta
        viewpoint.cam_trans_delta.data = old_rho

        # Compute derivative
        if loss_plus.ndim == 0:
            J[i, 0] = (loss_plus - loss_current) / epsilon
        else:
            J[i, :] = (loss_plus - loss_current) / epsilon

    return J


def gauss_newton_pose_update(
    viewpoint: Any,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_iters: int = 5,
    damping: float = 0.01,
    epsilon: float = 1e-4,
    convergence_threshold: float = 1e-6,
    verbose: bool = False,
) -> Tuple[bool, dict[str, Any]]:
    """Perform Gauss-Newton pose optimization.

    Args:
        viewpoint: Camera/viewpoint with cam_rot_delta, cam_trans_delta
        loss_fn: Function that takes viewpoint and returns loss
        max_iters: Maximum iterations
        damping: Levenberg-Marquardt damping factor
        epsilon: Finite difference step size
        convergence_threshold: Convergence threshold for tau norm
        verbose: Whether to print debug info

    Returns:
        (converged, stats) where:
        - converged: Whether optimization converged
        - stats: Dict with optimization statistics
    """
    device = viewpoint.cam_rot_delta.device
    dtype = viewpoint.cam_rot_delta.dtype

    stats = {
        'iterations': 0,
        'final_tau_norm': 0.0,
        'initial_loss': 0.0,
        'final_loss': 0.0,
        'loss_history': [],
        'tau_norm_history': [],
    }

    # Initial loss
    initial_loss = loss_fn(viewpoint)
    stats['initial_loss'] = float(initial_loss.detach().item())

    for iter_idx in range(max_iters):
        # Current tau
        rho = viewpoint.cam_trans_delta.detach().clone()
        theta = viewpoint.cam_rot_delta.detach().clone()
        tau = torch.cat([rho, theta], dim=0)
        tau_norm = float(torch.norm(tau).detach().item())

        stats['tau_norm_history'].append(tau_norm)

        # Check convergence
        if tau_norm < convergence_threshold:
            stats['iterations'] = iter_idx
            stats['final_tau_norm'] = tau_norm
            if verbose:
                print(f"[GN] Converged at iter {iter_idx}, tau_norm={tau_norm:.6f}")
            return True, stats

        # Compute Jacobian
        J = compute_pose_jacobian_fd(viewpoint, loss_fn, epsilon)

        # Compute current loss
        loss_current = loss_fn(viewpoint)
        loss_val = float(loss_current.detach().item())
        stats['loss_history'].append(loss_val)

        if verbose:
            print(f"[GN] iter {iter_idx}: loss={loss_val:.6f}, tau_norm={tau_norm:.6f}")

        # Handle scalar loss
        if loss_current.ndim == 0:
            residual = loss_current.unsqueeze(0)
        else:
            residual = loss_current

        # Levenberg-Marquardt: H = J.T @ J + damping * I
        H = J.T @ J + damping * torch.eye(6, device=device, dtype=dtype)

        # delta = H^-1 @ J.T @ residual
        # Note: We want to minimize loss, so we move in negative direction
        Jt_r = J @ residual  # shape (6,)
        try:
            delta = torch.linalg.solve(H, Jt_r)
        except RuntimeError:
            # Fallback to least squares if solve fails
            delta = torch.linalg.lstsq(H, Jt_r.unsqueeze(1)).solution.squeeze(1)

        # Apply update (move in negative direction to minimize)
        new_tau = tau - delta

        # Update viewpoint
        with torch.no_grad():
            viewpoint.cam_rot_delta.data = new_tau[3:6].to(dtype)
            viewpoint.cam_trans_delta.data = new_tau[:3].to(dtype)

    # Final stats
    final_loss = loss_fn(viewpoint)
    final_tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], dim=0)
    stats['iterations'] = max_iters
    stats['final_tau_norm'] = float(torch.norm(final_tau).detach().item())
    stats['final_loss'] = float(final_loss.detach().item())

    converged = stats['final_tau_norm'] < convergence_threshold
    return converged, stats


def gauss_newton_batch_update(
    viewpoints: list[Any],
    loss_fns: list[Callable],
    max_iters: int = 5,
    damping: float = 0.01,
    epsilon: float = 1e-4,
    convergence_threshold: float = 1e-6,
    verbose: bool = False,
) -> dict[str, Any]:
    """Perform batched Gauss-Newton optimization for multiple viewpoints.

    Args:
        viewpoints: List of cameras/viewpoints
        loss_fns: List of loss functions (one per viewpoint)
        max_iters: Maximum iterations per viewpoint
        damping: Levenberg-Marquardt damping
        epsilon: Finite difference step size
        convergence_threshold: Convergence threshold
        verbose: Whether to print debug info

    Returns:
        Dict with batch statistics
    """
    results = []
    converged_count = 0

    for i, (vp, loss_fn) in enumerate(zip(viewpoints, loss_fns)):
        converged, stats = gauss_newton_pose_update(
            viewpoint=vp,
            loss_fn=loss_fn,
            max_iters=max_iters,
            damping=damping,
            epsilon=epsilon,
            convergence_threshold=convergence_threshold,
            verbose=verbose,
        )
        results.append({
            'viewpoint_idx': i,
            'converged': converged,
            'stats': stats,
        })
        if converged:
            converged_count += 1

    return {
        'num_viewpoints': len(viewpoints),
        'converged_count': converged_count,
        'per_viewpoint_results': results,
    }


class GaussNewtonPoseOptimizer:
    """Stateful Gauss-Newton optimizer for pose refinement.

    Usage:
        optimizer = GaussNewtonPoseOptimizer(damping=0.01, max_iters=5)
        converged, stats = optimizer.optimize(viewpoint, loss_fn)
    """

    def __init__(
        self,
        max_iters: int = 5,
        damping: float = 0.01,
        epsilon: float = 1e-4,
        convergence_threshold: float = 1e-6,
        damping_decay: float = 0.9,
        damping_min: float = 1e-6,
    ):
        self.max_iters = max_iters
        self.damping = damping
        self.epsilon = epsilon
        self.convergence_threshold = convergence_threshold
        self.damping_decay = damping_decay
        self.damping_min = damping_min

    def optimize(
        self,
        viewpoint: Any,
        loss_fn: Callable,
        verbose: bool = False,
    ) -> Tuple[bool, dict[str, Any]]:
        """Run Gauss-Newton optimization with adaptive damping."""
        current_damping = self.damping

        for iter_idx in range(self.max_iters):
            converged, stats = gauss_newton_pose_update(
                viewpoint=viewpoint,
                loss_fn=loss_fn,
                max_iters=1,  # Single iteration
                damping=current_damping,
                epsilon=self.epsilon,
                convergence_threshold=self.convergence_threshold,
                verbose=verbose,
            )

            if converged:
                return True, stats

            # Adaptive damping: decay if loss decreased
            if len(stats['loss_history']) >= 2:
                if stats['loss_history'][-1] < stats['loss_history'][-2]:
                    current_damping = max(current_damping * self.damping_decay, self.damping_min)

        # Final check
        tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], dim=0)
        final_norm = float(torch.norm(tau).detach().item())
        return final_norm < self.convergence_threshold, stats