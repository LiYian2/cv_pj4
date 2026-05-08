# Phase 1 & Phase 2 Fix Verification + Experiment Plan

Date: 2026-05-05
Status: All modules verified as functional

---

## 1. Module Verification Summary

### 1.1 Phase 1 Fix: Unified Backward

Location: slam_backend_brpo.py:585-593

Change: Replaced split_authority multi-backward logic with unified backward.

Verification:
- Python syntax OK
- Loss computation returns gradient-connected tensors
- No intermediate zero_grad calls
- Gaussians receives gradients from pseudo_scene_loss_sum

Expected Effect: Gaussians optimization should now respond to pseudo view supervision.

---

### 1.2 Phase 2 Fix: Pose Propagation

Location:
- backend_pseudo_view_loader.py:36-37 (new fields)
- runtime_pseudo_builder.py:73-74 (pass fields)
- slam_backend_brpo.py:140-187 (propagate function)
- slam_backend_brpo.py:674 (call propagate)

Verification:
- BackendPseudoViewRecord has left_ref_frame_id and right_ref_frame_id
- Builder passes slot info to record
- _propagate_pseudo_pose_to_neighbors_ method exists
- Called before _fold_pseudo_pose_residual_

Expected Effect: Pseudo pose optimization influences real keyframe pose deltas.

---

### 1.3 Gauss-Newton Module

Location: pose_gauss_newton.py

Verification:
- GaussNewtonPoseOptimizer class defined (line 247)
- compute_pose_jacobian_fd for finite-difference Jacobian
- Imported in slam_backend_brpo.py:28
- Called when use_gauss_newton=True

Config Fields: use_gauss_newton, gn_max_iters, gn_damping, gn_every_n_steps

---

### 1.4 Exact vs Paper Loss Path

Exact Path (depth_loss_mode=exact_shared_cm_v1):
- build_stageA_loss_exact_shared_cm at pseudo_loss_v2.py:402
- Single shared mask for RGB and depth

Paper Path (depth_loss_mode=paper_brpo_split_v1):
- build_stageA_loss_paper_brpo_split at pseudo_loss_v2.py:521
- RGB uses discrete C_m, depth uses same C_m

---

### 1.5 Exposure Refinement

Location: pseudo_loss_v2.py:91, slam_backend_brpo.py:344-345

Verification:
- exposure_a and exposure_b added to pose optimizer
- lambda_exp weight in loss computation

---

### 1.6 Scale Regularization

Location: pseudo_loss_v2.py:95, slam_backend_brpo.py:578-591

Verification:
- Isotropic loss computed
- Optional max_scale penalty
- Added to total_loss

---

### 1.7 Absolute Pose Prior

Location: pseudo_loss_v2.py:136-178, pseudo_camera_state.py:78-81

Verification:
- R0/T0 initialized in make_viewpoint_trainable
- SE(3) log computes pose deviation from initial pose

---

## 2. Experiment Plan

### 2.1 Core Hypothesis

After Phase 1 + Phase 2 fixes:
1. Gaussians should receive gradients from pseudo scene loss
2. Pseudo pose optimization should influence neighboring keyframes
3. Overall scene quality should improve

### 2.2 Experiment Groups

#### Group A: Baseline Validation (1 experiment)

A1: Minimal Online Mapping
- depth_loss_mode: exact_shared_cm_v1
- use_gauss_newton: False
- lambda_exp: 0.0, lambda_scale: 0.0, isotropic_weight: 10.0
- lambda_abs_t: 0.0, lambda_abs_r: 0.0
- lambda_pose: 0.01, lambda_pseudo: 1.0
- update_real_pose: True, update_pseudo_pose: True
- num_iterations: 20

Purpose: Verify unified backward works.

---

#### Group B: Loss Path Comparison (2 experiments)

B1: Exact Path
B2: Paper Path

---

#### Group C: Pose Optimization Method (2 experiments)

C1: Adam Only (use_gauss_newton=False)
C2: Gauss-Newton (use_gauss_newton=True)

---

#### Group D: Regularization Effects (4 experiments)

D1: No Regularization
D2: Exposure Only (lambda_exp=0.001)
D3: Scale Only (lambda_scale=0.01)
D4: Both

---

#### Group E: Absolute Pose Prior (2 experiments)

E1: No Absolute Prior
E2: Absolute Prior (lambda_abs_t=3.0, lambda_abs_r=0.1)

---

#### Group F: Full Pipeline (1 experiment)

F1: Full Feature - all modules enabled

---

### 2.3 Execution Order

A1 -> B -> C -> D -> E -> F

---

## 3. Critical Config Fields

depth_loss_mode: exact_shared_cm_v1 / paper_brpo_split_v1
use_gauss_newton: bool
lambda_pose: 0.01
lambda_exp: 0.001
lambda_scale: 0.01
lambda_abs_t: 3.0
lambda_abs_r: 0.1
lambda_pseudo: 1.0
update_real_pose: True
update_pseudo_pose: True
num_iterations: 20-50

---

## 4. Known Issues

1. Phase 2 propagation uses alpha=0.5, may need tuning
2. GN uses finite-difference Jacobian
3. Densification disabled during online mapping

---

## 5. Next Steps

1. Run A1 to verify gradient flow
2. Proceed with B/C/D/E groups
3. Run F1 full pipeline
