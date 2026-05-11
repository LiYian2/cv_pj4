# Part3 BRPO Online Mapping Pipeline

Last updated: 2026-05-11
Status: current source-of-truth for the live Part3 BRPO online-mapping route
Scope: document the actual online-mapping mainline now used in the live backend, plus the optional branches that have been added around it

## 1. Purpose and scope

This document records the current live Part3 BRPO online-mapping pipeline.
It is intended for:
- report writing
- method section drafting
- future code closeout / simplification
- identifying which components are mainline versus optional ablations

This document is intentionally centered on the online-mapping route, not on the old standalone route.
Standalone and offline-preparation code still exists, but it is no longer the main engineering landing path.

## 2. Top-level system view

The live route is:

1. S3PO frontend tracks frames and decides keyframes.
2. When a new keyframe closes a gap, the backend selects one or more runtime pseudo slots inside that newly closed interval.
3. For each pseudo slot, the backend renders a coarse pseudo RGB/depth view from the current Gaussian scene.
4. Optional Difix restoration produces left-fixed and right-fixed pseudo RGBs using the left/right reference keyframes.
5. A BRPO-style overlap-confidence fusion produces a fused pseudo RGB target for matching and RGB supervision.
6. MASt3R matching builds pseudo-to-left and pseudo-to-right correspondences.
7. Exact branch verification converts correspondences into strict support masks, projected depth, confidence, provenance, and diagnostics.
8. Exact-upstream signal building converts those branch results into strict discrete `C_m`, depth target, valid mask, source map, and target confidence.
9. The runtime pseudo bundle is packed into an in-memory pseudo record.
10. The backend runs a joint pseudo mapping loop, where pseudo supervision updates the live Gaussian scene and pseudo pose variables, and may optionally update real poses too.
11. Optional masked pseudo color refinement can reuse the runtime pseudo record pool after mapping.

Important semantic boundary:
- pseudo views do not become persistent S3PO keyframes in `self.viewpoints` or `self.current_window`
- but they do become runtime supervision members whose losses can directly update the current Gaussian scene and pose residual variables

## 3. Live module map

### 3.1 Primary execution files

- `third_party/S3PO-GS/slam.py`
- `third_party/S3PO-GS/utils/slam_frontend.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`

### 3.2 Runtime builder chain

- slot selection: `pseudo_branch/integration/runtime_slot_selector.py`
- runtime exact backend build: `pseudo_branch/integration/runtime_exact_backend.py`
- runtime signal build: `pseudo_branch/integration/runtime_signal_builder.py`
- runtime pseudo record build: `pseudo_branch/integration/runtime_pseudo_builder.py`
- runtime debug export: `pseudo_branch/integration/runtime_debug_export.py`

### 3.3 Shared algorithm kernels used by the runtime

- matching: `pseudo_branch/common/flow_matcher.py`, `pseudo_branch/common/mast3r_matchers.py`, `pseudo_branch/common/mast3r_pair_forward.py`
- verification: `pseudo_branch/observation/brpo_reprojection_verify.py`
- fusion: `pseudo_branch/observation/pseudo_fusion.py`
- observation/signal semantics: `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- target build: `pseudo_branch/target/depth_supervision_v2.py`
- runtime loss contract: `pseudo_branch/refine/backend_pseudo_loss.py`, `pseudo_branch/refine/pseudo_loss_v2.py`
- pose math: `pseudo_branch/refine/pseudo_camera_state.py`, `pseudo_branch/refine/pose_gauss_newton.py`

## 4. Frontend stage: runtime state export and gap creation

### 4.1 Frontend responsibility

The frontend still runs the normal S3PO tracking/keyframe pipeline.
The online pseudo branch does not replace frontend tracking.
Instead, it augments the backend once a keyframe event happens.

Relevant code:
- `slam_frontend.py`

The frontend maintains:
- `self.runtime_camera_state_cache`
- one exported camera-state record per frame
- keyframe message payloads that include the runtime-state cache

### 4.2 Exported runtime camera state

For each frame, the frontend exports:
- frame id / uid
- image path / image name
- width / height
- intrinsics `(fx, fy, cx, cy)`
- FoV values
- `pose_c2w`
- keyframe flag

This gives the backend enough information to create pseudo/reference cameras later without going back to the dataset loader.

### 4.3 Trigger semantics

The pseudo branch is triggered at keyframe events.
More precisely:
- when a new keyframe arrives, the backend receives the current real-keyframe window and the runtime-state cache
- the pseudo branch then looks for newly closed gaps inside the current real-keyframe window
- pseudo insertion therefore happens on the arrival of the right keyframe of a gap, not continuously on every frame

This means the common user-level description “trigger at `kf2` for the gap between `kf1` and `kf2`” is correct.

## 5. Slot-selection stage: where pseudo frames are placed

Relevant code:
- `pseudo_branch/integration/runtime_slot_selector.py`
- `slam_backend.py::_maybe_prepare_brpo_runtime_slots()`

### 5.1 Basic rule

Given an ordered real-keyframe window, the runtime selects non-keyframe frame ids inside newly closed intervals.
A slot is represented by:
- `frame_id`
- `left_ref_frame_id`
- `right_ref_frame_id`
- placement label
- gap metadata

### 5.2 Default and optional placement modes

The live selector supports:
- `midpoint_only`: one pseudo at the middle of the gap
- `quartile`: three pseudos at roughly 25%, 50%, 75%
- `quintile`: four pseudos at roughly 20%, 40%, 60%, 80%
- `uniform`: more general placement controlled by `max_pseudo_per_gap`

Therefore, “render frame 1.5” is only the midpoint special case, not the universal definition.

### 5.3 Mathematical interpretation

If the left and right keyframes are indexed by `k_l` and `k_r`, then slot placement corresponds to sampling intermediate frame ids near

`k_l + r * (k_r - k_l)`

where `r` is a placement ratio such as `0.5`, `0.25`, `0.75`, etc., then snapping to available non-keyframe ids.

## 6. Runtime pseudo input stage: coarse render, optional Difix, and fused pseudo RGB

Relevant code:
- `runtime_exact_backend.py`
- `pseudo_fusion.py`

### 6.1 Coarse pseudo render

For each selected pseudo slot, the backend renders:
- `pseudo_render_rgb`
- `pseudo_render_depth`

from the current Gaussian scene under the pseudo camera pose.
It also renders left and right reference depths from the same live scene.

These renders are the online analogue of the old standalone pseudo-cache stage.

### 6.2 Optional pseudo RGB source override

Mainline default:
- `pseudo_rgb_source=render`

Optional upper-bound route:
- `pseudo_rgb_source=gt`

Meaning:
- default online mapping uses the coarse rendered pseudo RGB as the pseudo image source
- the upper-bound route can use the dataset GT image directly for matching / C_m / pseudo RGB supervision while still using rendered depth for geometric verification

### 6.3 Optional Difix restoration

When `use_difix_restoration=true`, the runtime performs two branchwise restorations:
- left-fixed pseudo RGB using the left reference image
- right-fixed pseudo RGB using the right reference image

This restores the missing Difix stage that earlier online versions lacked.

### 6.4 Geometry-guided branch confidence

The fusion code computes one overlap-confidence map per branch.
For a pseudo-depth point projected into a reference view, the branch confidence is built from two factors.

1. Depth consistency term

If `z_ref` is the projected pseudo point depth in the reference view and `d_ref` is the sampled reference depth at that projection, the relative depth error is

`e_rel = |z_ref - d_ref| / (0.5 * (|z_ref| + |d_ref|) + eps)`

The depth-consistency score is

`c_depth = exp(- e_rel / tau_depth)`

where `tau_depth = depth_consistency_tau`.

2. Translation consistency term

Let `t_p` and `t_r` be the pseudo and reference camera centers in world coordinates. Then

`c_trans = exp(- ||t_p - t_r|| / tau_trans)`

where `tau_trans = translation_scale_tau`.

3. Branch overlap confidence

`c_overlap = c_depth * c_trans`

This produces one map for the left branch and one for the right branch.

### 6.5 Residual fusion

Let
- `I_render` = coarse pseudo render
- `I_L` = left-fixed Difix result
- `I_R` = right-fixed Difix result
- `W_L`, `W_R` = normalized left/right overlap weights

The weights are normalized by

`W_L = c_L / (c_L + c_R)`
`W_R = c_R / (c_L + c_R)`

and the fused confidence is clipped as

`C_fused = clip(c_L + c_R, 0, 1)`

The fused pseudo RGB is built by residual fusion:

`R_L = I_L - I_render`
`R_R = I_R - I_render`
`I_fused = I_render + W_L * R_L + W_R * R_R`

This fused pseudo RGB becomes the matching input and the RGB target used downstream.

### 6.6 Mainline meaning

So the mainline pseudo RGB stage is not simply “render then match”.
It is more precisely:
- render coarse pseudo RGB/depth
- optionally restore left/right pseudo RGB with Difix
- geometry-weight those two restored branches
- fuse them into a single pseudo RGB target
- use that fused pseudo RGB for matching and RGB supervision

## 7. Matching stage: MASt3R reciprocal correspondences

Relevant code:
- `pseudo_branch/common/flow_matcher.py`
- matcher factory in `pseudo_branch/common/mast3r_matchers.py`

### 7.1 Default matching route

The current mainline uses dense 3D point matching:
- `matcher_mode=dense_pts3d_3d` (current default across E5c, E7a, E9, R5c, W5c)
- MASt3R inference extracts dense 3D point maps for pseudo and reference views
- dense correspondence is built by projecting 3D points and finding nearest neighbors
- produces higher-density support maps compared to sparse reciprocal matching

Output per branch:
- pseudo points `pts_pseudo` (dense)
- reference points `pts_ref` (dense)
- per-point confidence from MASt3R output

### 7.2 Optional matcher family

The runtime supports two matcher modes:
- `matcher_mode=dense_pts3d_3d` (current mainline default)
- `matcher_mode=sparse_desc_2d` (legacy sparse reciprocal route)

Current mainline understanding:
- the strict BRPO semantics are downstream of the matcher
- matcher choice changes support density and composition, but not the discrete `C_m` definition itself

## 8. Exact branch verification stage: support, projected depth, confidence, provenance

Relevant code:
- `pseudo_branch/observation/brpo_reprojection_verify.py`

This stage is one of the most important algorithmic stages in the pipeline.
It turns sparse correspondences into geometrically filtered support and projected depth.

### 8.1 Geometry of branch verification

For each matched pair `(p_pseudo, p_ref)`:
1. Backproject the reference point through reference depth to world coordinates.
2. Project that world point into the pseudo camera.
3. Compare the reprojection with the original pseudo matched point.
4. Compare the projected pseudo depth with the pseudo rendered depth at the pseudo point.

### 8.2 Reprojection error

For pseudo pixel `u_p` and reprojected pseudo pixel `u_hat`, the reprojection error is

`e_reproj = ||u_hat - u_p||_2`

### 8.3 Relative depth error

If `z_hat` is the reprojected pseudo depth and `d_p` is the pseudo-render depth sampled at the pseudo point, then

`e_depth = |z_hat - d_p| / max(d_p, eps)`

### 8.4 Binary branch support

A correspondence becomes supported if it satisfies all of:
- valid reference depth
- valid pseudo depth
- projected point lies in bounds
- `e_reproj < tau_reproj_px`
- `e_depth < tau_rel_depth`

This defines the branch support mask.

### 8.5 Continuous branch confidence

The exact backend also computes a continuous confidence:

`c_reproj = exp(- e_reproj / tau_reproj_px)`
`c_depth = exp(- e_depth / tau_rel_depth)`
`c_branch = c_reproj * c_depth`

For pixels with multiple valid hits, the backend keeps the projected depth with the highest confidence.

### 8.6 Additional exact-backend diagnostics

The exact route also records:
- `provenance_map` (left/right origin)
- `hit_count`
- `occlusion_reason_map`
- `depth_variance_map`
- projected depth valid mask

This is why the live route is more than just a binary support mask builder.
It is an exact backend bundle carrying the information needed for strict `C_m`, target-depth composition, and later audits.

## 9. Strict `C_m` stage: discrete BRPO confidence semantics

Relevant code:
- `pseudo_observation_brpo_style.py`

### 9.1 Raw strict discrete semantics

Let
- `S_L` = left branch support set
- `S_R` = right branch support set

Then
- `verify_both = S_L ∩ S_R`
- `verify_left_only = S_L \ S_R`
- `verify_right_only = S_R \ S_L`
- `verify_xor = verify_left_only ∪ verify_right_only`
- `verify_union = S_L ∪ S_R`

The strict BRPO discrete confidence map is

`C_m(x) = 1.0` if `x in verify_both`
`C_m(x) = 0.5` if `x in verify_xor`
`C_m(x) = 0.0` otherwise

This is the core three-level BRPO confidence semantics now used by the live exact-upstream route.

### 9.2 Important semantic clarification

The pipeline does not define BRPO confidence by a soft fusion formula first.
The strict mainline definition is still discrete:
- both support -> 1.0
- single support -> 0.5
- unsupported -> 0.0

Optional modules may override or expand it, but the mainline default is this strict discrete rule.

## 10. Exact-upstream depth target stage

Relevant code:
- `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`

This stage defines what pseudo depth supervision actually means in the current mainline.

### 10.1 Supported region logic

The exact-upstream builder first forms verified support regions:
- both-supported region
- left-only region
- right-only region
- union region

Then it checks whether projected depth is available on each branch.
No render-depth fallback is allowed in the exact-upstream default route.
Unsupported pixels stay invalid / zero.

This is a major semantic distinction from older routes.

### 10.2 Both-supported target depth

When both left and right branch projected depths are available, the current default is confidence-weighted composition.

Let
- `d_L`, `d_R` be the left/right target depths
- `c_L`, `c_R` be the left/right exact branch confidences

Then on the both-supported region:

`d_target = (c_L * d_L + c_R * d_R) / (c_L + c_R)`

and the target confidence becomes approximately the average branch confidence:

`c_target = (c_L + c_R) / 2`

If confidence-weighted composition is disabled, a legacy fusion-weight route can be used instead, but current exact-upstream mainline uses branch confidence weighting.

### 10.3 Single-supported target depth

If only one branch is valid, the target depth is copied from that branch directly:

`d_target = d_L` on left-only valid region
`d_target = d_R` on right-only valid region

The source map records where the depth came from.

### 10.4 No-render-fallback rule

This is critical for the report:
- the current exact-upstream target builder does not silently fill unsupported pixels with render depth
- unsupported within verified union stays unsupported
- outside support stays zero/invalid

So the exact-upstream route is a “verified projected depth only” target, not a dense completion route.

### 10.5 Output fields

The runtime target builder emits:
- pseudo depth target
- pseudo source map
- pseudo valid mask
- pseudo target confidence
- `verify_both`, `verify_left_only`, `verify_right_only`, `verify_union`
- exact-upstream summary metadata

## 11. Runtime pseudo record stage

Relevant code:
- `pseudo_branch/integration/runtime_pseudo_builder.py`

The runtime packs all pseudo supervision into an in-memory `BackendPseudoViewRecord`.
Each record contains:
- pseudo viewpoint (trainable pose/exposure variables)
- target RGB
- target depth
- discrete confidence mask `C_m`
- source map
- valid mask
- target confidence
- both-support mask
- left/right reference frame ids
- paths to saved diagnostics

This is the handoff object from the signal-building side to the optimizer side.

## 12. Runtime optimization stage: joint pseudo mapping

Relevant code:
- `slam_backend.py::_run_brpo_runtime_pseudo_mapping()`
- `slam_backend_brpo.py`
- `backend_pseudo_loss.py`
- `pseudo_loss_v2.py`

This stage is where the online route truly differs from the old standalone narrative.

### 12.1 What is optimized

The live joint pseudo mapping loop may optimize:
- Gaussian scene parameters through the existing Gaussian optimizer
- pseudo pose residuals (`cam_rot_delta`, `cam_trans_delta`)
- pseudo exposure variables
- optionally real keyframe pose/exposure variables

Pseudo views are runtime supervision members, not persistent mapping members, but their losses still backpropagate into the same live Gaussian scene.

### 12.2 Real branch loss

Real keyframes continue to use the normal S3PO mapping loss via rendered RGB/depth against the real viewpoints.
This remains the real-branch anchor.

### 12.3 Paper-aligned split loss contract

Current mainline default depth loss mode is `paper_brpo_split_v1`.

**Critical semantics**: RGB and depth losses use **different mask scopes**.

RGB loss uses only the discrete `C_m`:
- No valid_mask gating
- No target_confidence weighting
- Pure three-level BRPO confidence mask

Depth loss can optionally use additional weighting:
- Uses `C_m` as base mask
- Optionally applies `depth_confidence` (target_confidence) for continuous weighting
- `depth_confidence` never gates RGB loss ("never gates RGB" rule)

Formally, for `paper_brpo_split_v1`:

```
M_rgb = C_m          (RGB uses pure discrete C_m)
M_depth = C_m * depth_confidence   (depth may use additional continuous weighting)
```

The loss terms are:

```
L_rgb = masked_rgb_loss(render_rgb, target_rgb, M_rgb, viewpoint)
L_depth = masked_depth_loss(render_depth, target_depth, M_depth)
```

This split contract ensures depth-side weighting does not flow back to affect RGB supervision.

**Historical note**: The older `exact_shared_cm_v1` mode uses `M_eff = C_m * valid_mask * target_confidence` for both RGB and depth, but this is no longer the mainline default.

### 12.4 Depth loss and RGB loss form

The masked RGB loss is a confidence-weighted masked L1 over RGB channels after exposure adjustment.
The masked depth loss is a confidence-weighted masked L1 over valid target-depth pixels.

At a high level, the pseudo objective for `paper_brpo_split_v1` is:

```
L_pseudo = beta_rgb * L_rgb(C_m) + lambda_depth * L_depth(C_m * depth_conf) + lambda_pose * L_pose + L_abs_pose + lambda_exp * L_exp
```

where:
- `L_pose` is pose residual regularization
- `L_abs_pose` is optional absolute pose prior
- `L_exp` is exposure regularization

### 12.5 Pose regularization and absolute pose prior

The base pose regularizer penalizes residual pose variables:

`L_pose = ||theta||_2 + w_t * ||rho||_2`

where `theta` is rotational residual and `rho` is translational residual.

The optional absolute pose prior compares the current pose against a reference pose and can separately weight translation and rotation components, with robust penalties such as Charbonnier or Huber.

### 12.6 Pose-gradient fix

This is a key online-mapping implementation detail.
The renderer itself only uses the camera transform matrices, not raw pose residual tensors.
Therefore the pipeline explicitly folds pose residuals into the viewpoint transform before rendering.

Conceptually:
- start from base world-to-camera transform
- apply `SE(3)` exponential of the pose residual
- use the pose-corrected transform for render

This ensures gradients from rendering flow back to `cam_rot_delta` and `cam_trans_delta`.
Without this step, pseudo pose optimization would be structurally broken.

### 12.7 Gauss-Newton pose option

Optional pose refinement can replace or augment Adam-based pseudo pose updates.
The Gauss-Newton module uses finite-difference Jacobians and solves a damped normal equation.

Given residual/Jacobian matrix `J`, the update approximates:

`H = J^T J + lambda I`
`delta = H^{-1} J^T r`
`tau_new = tau - delta`

with adaptive damping.

This gives a second-order pose update option in the online pseudo loop.

### 12.8 Scale regularization

The runtime also adds Gaussian scale regularization to avoid scale explosion.
Current implementation penalizes anisotropy by measuring deviation from each Gaussian's mean scale, and can optionally penalize scales above a maximum threshold.

At a high level:

`L_scale = mean(|s - mean(s)|)`

plus an optional overflow penalty when `s > s_max`.

### 12.9 Gaussian maintenance semantics

Current runtime loop supports optional Gaussian maintenance toggles:
- `enable_densify`
- `enable_prune`
- `enable_opacity_reset`

But these are configuration-controlled and route-dependent.
The important semantics are:
- pseudo supervision can update the Gaussian scene directly
- pseudo does not automatically become a persistent keyframe/fusion member
- Gaussian maintenance source can be restricted, e.g. `real_only`, so pseudo affects the scene through loss gradients without necessarily becoming the maintenance source

## 13. Joint-primary versus side-branch semantics

Relevant code:
- `slam_backend.py`

The runtime supports two major topology styles.

### 13.1 Side-branch mode

Order:
1. run legacy real-keyframe mapping first
2. then prepare pseudo runtime slots
3. then run pseudo mapping as a side branch

Meaning:
- the pseudo path augments the real branch after the main real update

### 13.2 Joint-primary mode

Order:
1. prepare pseudo runtime slots first
2. run pseudo mapping together with real-window losses as the primary mapping step
3. optionally run a legacy prune-only cleanup pass afterward

Meaning:
- pseudo is not just an afterthought; it participates in the primary backend mapping event
- this is the current more important online route

### 13.3 Important semantic clarification

Even in joint-primary mode, pseudo does not become a normal persistent S3PO keyframe.
The correct description is:
- pseudo is a runtime equal-member supervision view inside the mapping event
- not a persistent real-window/keyframe object in the SLAM state graph

### 13.4 Pseudo scene mask mode behavior

When `pseudo_window_equivalence=true` (joint-primary default), the `pseudo_scene_mask_mode` config parameter is **overridden**.

Code behavior (`slam_backend_brpo.py:531`):
```python
scene_mask_mode = "none" if pseudo_window_equivalence else str(cfg.pseudo_scene_mask_mode or "all_valid")
```

This means:
- When `pseudo_window_equivalence=true`: `scene_mask_mode="none"` (forced)
- The configured `pseudo_scene_mask_mode: both_only` is **not applied** in joint-primary mode
- Pseudo loss uses the full `C_m` domain without additional `both_only` restriction

The `pseudo_scene_mask_mode` config only applies when `pseudo_window_equivalence=false` (side-branch mode).
In that case, `both_only` would restrict supervision to `verify_both` regions only.

**Report implication**: When documenting joint-primary experiments, the effective supervision domain is the full `C_m`, not `both_only` restricted.

## 14. Optional post-map masked pseudo color refinement

Relevant code:
- `slam_backend.py::color_refinement()`

The final color refinement can optionally mix:
- real keyframe refinement steps
- pseudo-view masked refinement steps

Pseudo color refinement uses the runtime pseudo record pool and a masked RGB loss using one chosen pseudo mask source.
This is an optional stage and should be documented as optional, not as the core online pseudo mapping stage itself.

## 15. Current mainline defaults versus optional modules

### 15.1 Current mainline defaults worth reporting as the canonical route

The most stable current online interpretation is:
- keyframe-triggered runtime pseudo slot insertion
- midpoint or other configured placement inside newly closed gaps
- coarse pseudo render
- optional Difix restoration and BRPO-style residual fusion (default: `difix_fusion_mode=brpo_overlap_confidence`)
- MASt3R dense 3D matching (`matcher_mode=dense_pts3d_3d`)
- exact branch verification with RGB-only support (`rgb_only_verification=true`, `rgb_only_support_mode=reciprocal_seed`)
- strict discrete three-level `C_m`
- exact-upstream projected-depth target (`depth_generation_mode=projected` for baseline experiments)
- paper-aligned split loss (`depth_loss_mode=paper_brpo_split_v1`) with RGB using pure `C_m`
- joint backend mapping (`topology_mode=joint_primary`, `pseudo_window_equivalence=true`)
- pose-gradient fix for proper gradient flow
- optional Gauss-Newton and scale regularization

### 15.2 Optional module family A: placement and event structure

- `placement_mode = midpoint_only | quartile | quintile | uniform`
- `max_pseudo_per_gap`
- `num_pseudo_views_per_step`
- `topology_mode = side_branch | joint_primary`

### 15.3 Optional module family B: pseudo RGB construction

- `pseudo_rgb_source = render | gt`
- `use_difix_restoration`
- `difix_*` parameters
- `difix_fusion_mode`
- `depth_consistency_tau`
- `translation_scale_tau`

### 15.4 Optional module family C: matching and support generation

- `matcher_mode = sparse_desc_2d | dense_pts3d_3d` (mainline default: `dense_pts3d_3d`)
- `rgb_only_verification = true | false` (mainline default: `true`)
- `rgb_only_support_mode = reciprocal_seed | dense_match_v1` (mainline default: `reciprocal_seed`)

When `rgb_only_verification=true`:
- `C_m` is computed from RGB matching correspondences only
- No depth-based geometric verification is used for `C_m` generation
- Depth targets still come from projected depth from matched correspondences

`dense_match_v1` (E9 experiment) is not the default path.
It is an optional densified RGB-only support route that does:
- reciprocal seed rasterization
- Gaussian blur
- normalization
- thresholding

### 15.5 Optional module family D: local support expansion

- `cm_expansion_mode = none | local_soft_v1`
- `cm_expansion_apply_to_depth_scope`

This is also optional and should be documented as a side branch over the raw strict discrete support, not as the default definition of `C_m`.

### 15.6 Optional module family E: depth-generation overrides

Baseline experiments (E5c, R5c, W5c):
- `depth_generation_mode = projected` (exact upstream projected depth)

E7a/E9 experiments:
- `depth_generation_mode = twoimg_pair_proxy_cm_capped_v1` (2IMG pair proxy depth)

Optional alternatives not currently in mainline:
- `mast3r_direct_exact_anchor_v1`

The important closeout message:
- baseline experiments use projected depth from left/right branch correspondences
- E7a/E9 use the `twoimg_pair_proxy_cm_capped_v1` variant which computes depth differently
- report should distinguish which depth generation mode is used per experiment

### 15.7 Optional module family F: optimizer and control switches

Loss weight modes:
- `match_real_loss_weights = true | false`
  - `true` (R5c, W5c): `beta_rgb` and `lambda_depth` auto-synced with `Training.alpha`
  - `false` (E7a, E9): explicit `beta_rgb` and `lambda_depth` from config

Pose optimization:
- `update_real_pose`
- `update_pseudo_pose` (default: `true`)
- `update_real_exposure`
- `use_gauss_newton`

Scale regularization:
- `lambda_scale`, `max_scale`
- `isotropic_weight` (fallback when `lambda_scale=0`)

Absolute pose prior:
- `lambda_abs_pose`, `lambda_abs_t`, `lambda_abs_r`
- `abs_pose_robust = charbonnier | huber`

Joint-primary topology:
- `pseudo_window_equivalence = true` (default for joint-primary)
  - When `true`, `pseudo_scene_mask_mode` is forced to `"none"` (see Section 13.4)
- `extra_real_views`
- `propagate_pseudo_delta_to_neighbors`
- `gaussian_maintenance_source = real_only` (default)
- `joint_primary_run_legacy_prune`

Side-branch only (not used in joint-primary):
- `pseudo_scene_mask_mode = all_valid | both_only`
  - Only applies when `pseudo_window_equivalence=false`
  - `both_only` would restrict supervision to verify_both regions

## 16. What is not the core mainline and should be described carefully in a report

1. The old standalone `run_pseudo_refinement_v2.py` route is not the main engineering landing route anymore.
2. `dense_match_v1` is an optional RGB-only support densify branch, not the default support rule.
3. `cm_expansion_mode=local_soft_v1` is an optional soft expansion branch, not the default `C_m` definition.
4. `pseudo_rgb_source=gt` is an upper-bound route, not the deployment/default route.
5. `twoimg_pair_proxy_cm_capped_v1` is an optional depth-generation branch, not the default target-depth route.
6. pseudo views are not persistent keyframes, even when they act like runtime equal-members in joint-primary mapping.

## 17. Concise report-ready statement

A compact but faithful description of the current online mainline is:

The live Part3 BRPO system inserts runtime pseudo supervision into the S3PO backend at keyframe-triggered gap-closure events. For each selected pseudo slot, it renders a coarse pseudo RGB/depth view, optionally performs left/right Difix restoration and geometry-guided residual RGB fusion using overlap confidence, matches the fused pseudo image to the left/right real keyframes with MASt3R dense 3D correspondences, builds RGB-only support masks to obtain strict three-level BRPO confidence `C_m` (both=1.0, single=0.5, none=0), computes projected-depth targets from verified correspondences, then optimizes the live Gaussian scene and pose residuals inside the joint-primary backend mapping loop. The pseudo loss uses `paper_brpo_split_v1` contract where RGB supervision uses pure discrete `C_m` without additional gating, while depth supervision may optionally apply continuous confidence weighting. In joint-primary mode with `pseudo_window_equivalence=true`, the configured `pseudo_scene_mask_mode` is overridden to `"none"`, meaning the full `C_m` domain receives supervision rather than being restricted to `both_only` regions.

## 18. Bottom line

For future report writing, the safest mainline interpretation is:
- online pseudo supervision now lives inside the S3PO backend mapping event
- its core semantics are strict discrete `C_m` plus exact-upstream projected depth
- pseudo affects the live Gaussian scene through backend RGB-D refinement, while remaining a runtime supervision member rather than a persistent SLAM keyframe
