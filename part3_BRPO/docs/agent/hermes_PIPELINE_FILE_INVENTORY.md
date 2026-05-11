# Hermes Pipeline File Inventory and Refactor Labels

Last updated: 2026-05-11
Status: grounded inventory for future refactor; no live Python file moves/renames executed here
Scope: label which pseudo_branch/ and scripts/ files are used by the current online pipeline, which are standalone/control/historical, and what should be split later.

## 1. Live-route conclusion

Current judgment after direct remote inspection:
1. The real live mainline is the S3PO online-mapping route, not the old standalone `signal_v2 -> run_pseudo_refinement_v2.py` route.
2. The current live executor chain is `slam_frontend.py -> slam_backend.py -> slam_backend_brpo.py`.
3. The active online configs checked on 2026-05-11 (`e9`, `w5c`, `r5c`) all have `Results.brpo_online_mapping.enabled: true` and `topology_mode: joint_primary`.
4. Therefore, files should be labeled by whether they feed this online route, overlap with it as shared kernels, or only serve standalone / diagnostics / archive roles.

Important safety note:
- At inspection time there were active `slam.py` runs (`e9_dense_match_v1_depthoff_crpseudo`, `w5c_depthoff_crpseudo`, `r5c_depthoff_crpseudo`).
- For that reason, this pass only removed backup files and archived shell launchers; it did not rename or move any live Python execution file.
- Refactor scope update: `third_party/S3PO-GS/*` remains execution boundary only and must stay untouched; all planned restructuring is inside `part3_BRPO/`.

## 2. Label scheme used in this note

- `ONLINE_LIVE_CORE`: directly used by the current online mapping pipeline.
- `ONLINE_OPTIONAL_BRANCH`: online-only or online-reachable branch/module, but not part of the smallest default route.
- `SHARED_KERNEL_CANDIDATE`: algorithmic code duplicated or logically shared between online and standalone; future extraction target.
- `STANDALONE_CONTROL`: standalone/offline/reference pipeline code, still useful for control or replay, but not current live route.
- `DIAGNOSTIC_OR_TEST`: analysis, verification, test, or one-off support code.
- `LEGACY_OR_ARCHIVE`: historical wrapper, launcher, or code kept only for provenance/compat.
- `MIXED_FILE`: a file currently mixes multiple route families and must be split later.

## 3. Current online pipeline files inside pseudo_branch/

### 3.1 Runtime integration layer

These are the clearest `ONLINE_LIVE_CORE` files because `slam_backend.py` calls into them directly:
- `pseudo_branch/integration/runtime_slot_selector.py`
  - label: `ONLINE_LIVE_CORE`
  - role: newly-closed-gap pseudo slot placement (`midpoint_only`, `quartile`, `quintile`, `uniform`).
- `pseudo_branch/integration/runtime_exact_backend.py`
  - label: `MIXED_FILE`, `ONLINE_LIVE_CORE`
  - role: coarse render, optional GT pseudo RGB, optional Difix, matching, exact verification, optional RGB-only support branches, optional C_m expansion, optional direct-depth branches.
  - future split seam: exact default runtime builder vs optional RGB/support/depth branches.
- `pseudo_branch/integration/runtime_signal_builder.py`
  - label: `MIXED_FILE`, `ONLINE_LIVE_CORE`, `SHARED_KERNEL_CANDIDATE`
  - role: exact-upstream signal/target assembly from exact backend outputs.
  - future split seam: projected default target path vs `twoimg_pair_proxy_cm_capped_v1` override family.
- `pseudo_branch/integration/runtime_pseudo_builder.py`
  - label: `ONLINE_LIVE_CORE`
  - role: pack runtime pseudo supervision into `BackendPseudoViewRecord`.
- `pseudo_branch/integration/runtime_debug_export.py`
  - label: `ONLINE_LIVE_CORE`
  - role: structured runtime debug output writer; support module for online route.

### 3.2 Online/shared matcher + verification + fusion + target modules

These are either direct online dependencies or immediate shared-kernel candidates:
- `pseudo_branch/common/flow_matcher.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/common/mast3r_matchers.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/common/mast3r_pair_forward.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_OPTIONAL_BRANCH`
- `pseudo_branch/common/twoimg_pair_proxy_depth.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_OPTIONAL_BRANCH`
- `pseudo_branch/observation/brpo_reprojection_verify.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/observation/pseudo_fusion.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py` — `MIXED_FILE`, `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
  - reason: carries exact-upstream mainline semantics but also multiple historical target/observation families used by standalone scripts.
- `pseudo_branch/target/depth_supervision_v2.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`

### 3.3 Online/shared mask support modules

- `pseudo_branch/mask/rgb_mask_inference.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/mask/dense_match_densify.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL_CANDIDATE`
  - current role: online `rgb_only_support_mode=dense_match_v1` branch; active in checked `e9` config.
- `pseudo_branch/mask/cm_local_expansion.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL_CANDIDATE`
- `pseudo_branch/mask/brpo_confidence_mask.py` — `STANDALONE_CONTROL`, `SHARED_KERNEL_CANDIDATE`
- `pseudo_branch/mask/brpo_train_mask.py` — `STANDALONE_CONTROL`, `SHARED_KERNEL_CANDIDATE`
- `pseudo_branch/mask/confidence_builder.py` — `STANDALONE_CONTROL`
- `pseudo_branch/mask/joint_confidence.py` — `STANDALONE_CONTROL`

### 3.4 Online optimizer/refine layer

These are consumed directly by `slam_backend_brpo.py` and are the highest-priority shared-core extraction targets:
- `pseudo_branch/refine/__init__.py` — `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/backend_pseudo_bundle.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/backend_pseudo_loss.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/backend_pseudo_view_loader.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/pseudo_camera_state.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/pose_gauss_newton.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL_CANDIDATE`
- `pseudo_branch/refine/pseudo_loss_v2.py` — `SHARED_KERNEL_CANDIDATE`, `ONLINE_LIVE_CORE`
- `pseudo_branch/refine/pseudo_refine_scheduler.py` — `STANDALONE_CONTROL`

### 3.5 Not current online mainline inside pseudo_branch/

These are not the current online executor surface and should not define the future repo shape:
- `pseudo_branch/gaussian_management/**` — `LEGACY_OR_ARCHIVE` or at most `STANDALONE_CONTROL`
  - reason: tied to earlier local-gating / SPGM / standalone history; not part of current live online mainline call chain.
- `pseudo_branch/brpo_v2_signal/__init__.py` — `LEGACY_OR_ARCHIVE`
- `pseudo_branch/common/build_pseudo_cache.py` — `STANDALONE_CONTROL`
- `pseudo_branch/common/align_depth_scale.py` — `STANDALONE_CONTROL`
- `pseudo_branch/common/epipolar_depth.py` — `STANDALONE_CONTROL`
- `pseudo_branch/common/diag_writer.py` — `DIAGNOSTIC_OR_TEST`
- `pseudo_branch/observation/joint_observation.py` — `STANDALONE_CONTROL`
- `pseudo_branch/observation/pseudo_observation_verifier.py` — `STANDALONE_CONTROL`
- `pseudo_branch/target/brpo_depth_densify.py` — `STANDALONE_CONTROL`
- `pseudo_branch/target/brpo_depth_target.py` — `STANDALONE_CONTROL`
- `pseudo_branch/target/depth_target_builder.py` — `STANDALONE_CONTROL`
- `pseudo_branch/target/support_expand.py` — `STANDALONE_CONTROL`

## 4. scripts/ inventory and labels

### 4.1 Current online live route

Direct conclusion:
- the current online mainline does not depend on top-level `scripts/*.py` for execution.
- the live route is inside S3PO `slam.py` + `slam_frontend.py` + `slam_backend.py` + `slam_backend_brpo.py`, with pseudo_branch modules imported from there.

So scripts should be treated as support/control/reference layers unless explicitly proven otherwise.

### 4.2 Standalone/reference/control scripts worth keeping visible in refactor planning

- `scripts/build_brpo_v2_signal_from_internal_cache.py` — `MIXED_FILE`, `STANDALONE_CONTROL`, `SHARED_KERNEL_CANDIDATE`
  - overlaps with online signal-building semantics.
- `scripts/brpo_build_mask_from_internal_cache.py` — `MIXED_FILE`, `STANDALONE_CONTROL`, `SHARED_KERNEL_CANDIDATE`
  - overlaps with online exact backend / verifier semantics.
- `scripts/brpo_verify_single_branch.py` — `STANDALONE_CONTROL`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py` — `STANDALONE_CONTROL`
- `scripts/materialize_m5_depth_targets.py` — `STANDALONE_CONTROL`
- `scripts/materialize_twoimg_pair_proxy_depth.py` — `STANDALONE_CONTROL`
- `scripts/replay_internal_eval.py` — `STANDALONE_CONTROL`
- `scripts/replay_color_refinement_ablation.py` — `STANDALONE_CONTROL`
- `scripts/select_signal_aware_pseudos.py` — `STANDALONE_CONTROL`
- `scripts/compute_full_ate_from_camera_states.py` — `DIAGNOSTIC_OR_TEST`
- `scripts/generate_d_series_configs.py` — `LEGACY_OR_ARCHIVE` (experiment launcher/config helper, not live route)

### 4.3 Compatibility / historical wrappers

- `scripts/run_pseudo_refinement.py` — `LEGACY_OR_ARCHIVE`
  - top-level CLI compatibility wrapper only.
- `scripts/compat/run_pseudo_refinement.py` — `LEGACY_OR_ARCHIVE`
  - internal compatibility entry to archived legacy refine runner.
- `scripts/archive_experiments/legacy_entry/run_pseudo_refinement.py` — `LEGACY_OR_ARCHIVE`
  - archived historical entry.
- `scripts/run_pseudo_refinement_v2.py` — `LEGACY_OR_ARCHIVE`, `BROKEN_EXTERNAL_SYMLINK`
  - current state at inspection: symlink to `/data/bzhang512/tmp/run_pseudo_refinement_v2_pose_fix.py`.
  - remote check showed this target currently does not exist.
  - therefore this path must not be treated as a stable in-repo authority file during refactor planning.
  - later action should be either: restore an in-repo source file, replace with a clear wrapper, or retire it.

### 4.4 Diagnostics, tests, and experiment support

- `scripts/diagnostics/**` — `DIAGNOSTIC_OR_TEST`
- `scripts/test_*.py` — `DIAGNOSTIC_OR_TEST`
- `scripts/verify_pose_gradient_*.py` — `DIAGNOSTIC_OR_TEST`
- `scripts/archive_experiments/**` — `LEGACY_OR_ARCHIVE`
- `scripts/legacy_prepare/**` — `LEGACY_OR_ARCHIVE` or `STANDALONE_CONTROL` depending on whether a specific prepare path is still needed

## 5. Cleanup executed in this pass

Executed on 2026-05-11 after confirming active runs and avoiding live Python renames.

### 5.1 Deleted backup files

All `*.bak*` files under `pseudo_branch/` and `scripts/` were removed.
Reason:
- user explicitly approved cleanup of backup files
- they cluttered the live tree
- they are not the right mechanism for future provenance compared with git history / archive records

### 5.2 Archived shell launchers

All top-level `scripts/*.sh` launchers were moved to:
- `scripts/archive_experiments/shell_launchers_20260511/`

This keeps experiment launcher history available without leaving top-level `scripts/` dominated by historical shell launchers.
Already-archived shell files under `scripts/archive_experiments/stageA/` were left in place.

## 6. Practical refactor guidance derived from this inventory

### 6.1 First extraction target

Do not start from cleaning the whole `scripts/` tree.
Start from the online live core inside `pseudo_branch/integration/` and `pseudo_branch/refine/`, because this is the code directly consumed by the current S3PO online route.

### 6.2 First split seams

Highest-priority split seams:
1. `pseudo_branch/integration/runtime_exact_backend.py`
   - split default exact runtime builder from optional support/depth/RGB-source variants.
2. `pseudo_branch/integration/runtime_signal_builder.py`
   - split default projected exact-upstream path from `twoimg_pair_proxy_cm_capped_v1` override family.
3. `pseudo_branch/observation/pseudo_observation_brpo_style.py`
   - split current exact-upstream mainline semantics from historical target/observation builders.
4. `scripts/build_brpo_v2_signal_from_internal_cache.py` and `scripts/brpo_build_mask_from_internal_cache.py`
   - stop treating them as authorities; instead extract shared kernels they duplicate with the online runtime.

### 6.3 First shared-kernel extraction set

When actual refactor starts, the first candidates to extract into a shared core are:
- matching factory / pair forward
- exact reprojection verification
- pseudo fusion
- exact-upstream target build
- pseudo loss contract
- pseudo record / bundle data model
- pose residual application and Gauss-Newton utilities

### 6.4 What should not drive the future repo structure

These should not define the public/simple GitHub shape:
- `pseudo_branch/gaussian_management/**`
- `scripts/archive_experiments/**`
- `scripts/diagnostics/**`
- `scripts/test_*.py`
- `scripts/verify_pose_gradient_*.py`
- top-level historical `.sh` launchers
- backup files
- broken external symlink `scripts/run_pseudo_refinement_v2.py`

## 7. Immediate takeaway

The current live repository should now be read as:
1. `third_party/S3PO-GS/*` = online executor/orchestration boundary
2. `pseudo_branch/integration + refine + selected common/observation/mask/target files` = current online payload
3. `scripts/*` = mostly standalone/reference/diagnostic/archive surface

So the next refactor step should be: freeze the online payload set, then split mixed online files, then extract shared kernels, and only after that simplify standalone/history.
