# Hermes Pipeline File Inventory and Refactor Labels

Last updated: 2026-05-12
Status: grounded inventory updated after first repo-side extraction pass
Scope: label which `pseudo_branch/` and related repo files belong to the current online route, which were already extracted into new roots, and which should move out of `pseudo_branch/` next.

## 1. Live-route conclusion

Current judgment after direct remote inspection and post-refactor import verification:
1. The real live mainline is still the S3PO online-mapping route, not the old standalone `signal_v2 -> run_pseudo_refinement_v2.py` route.
2. The live executor chain is still `slam_frontend.py -> slam_backend.py -> slam_backend_brpo.py`.
3. The first repo-side extraction pass has already landed successfully, but the live route still reaches `pseudo_branch.*` facades and selected non-facade modules.
4. The user’s target end state is now explicit: `pseudo_branch/` should contain only current online-mapping modules and facades; everything else should move out.

Important safety note:
- At update time there was still an active `slam.py` run on the remote machine, so this note records grounded route boundaries and next-step move targets, not broad destructive cleanup.

## 2. Label scheme used in this note

- `ONLINE_BRIDGE_FACADE`: package/file kept mainly to preserve current `third_party` import surfaces.
- `ONLINE_LIVE_CORE`: directly used by the current online mapping pipeline.
- `ONLINE_OPTIONAL_BRANCH`: online-reachable branch/module, but not part of the smallest default route.
- `SHARED_KERNEL`: logic that should live in `core_shared/` or continue moving there.
- `STANDALONE_CONTROL`: standalone/offline/reference pipeline code.
- `DIAGNOSTIC_OR_TEST`: analysis, verification, test, or one-off support code.
- `MOVE_OUT_OF_PSEUDO_BRANCH`: not desired in final online-only `pseudo_branch/`; should migrate to `standalone_*` or `legacy_or_archive/` once imports are decoupled.
- `MIXED_FILE`: currently mixes live online logic with optional or historical families and needs further split.

## 3. What already moved out of old authority files

The following old paths now exist primarily as compatibility wrappers over extracted modules:
- `pseudo_branch/integration/runtime_slot_selector.py` -> `online_mapping/runtime/slot_selector.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py` -> `online_mapping/records/runtime_record_builder.py`
- `pseudo_branch/observation/brpo_reprojection_verify.py` -> `core_shared/verification/brpo_reprojection_verify.py`
- `pseudo_branch/observation/pseudo_fusion.py` -> `core_shared/fusion/pseudo_fusion.py`
- `pseudo_branch/refine/backend_pseudo_bundle.py` -> `core_shared/records/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py` -> `core_shared/records/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py` -> `core_shared/losses/backend_pseudo_loss.py`
- `pseudo_branch/refine/pseudo_camera_state.py` -> `core_shared/pose/pseudo_camera_state.py`
- `pseudo_branch/refine/pose_gauss_newton.py` -> `core_shared/pose/pose_gauss_newton.py`

These should stay import-stable for now, but they should not be treated as long-term implementation authority files anymore.

## 4. Current online-route files inside `pseudo_branch/`

### 4.1 Runtime integration layer

- `pseudo_branch/integration/__init__.py`
  - label: `ONLINE_BRIDGE_FACADE`
  - role: stable surface imported by `slam_backend.py`
  - next action: keep path stable, continue redirecting exports behind it

- `pseudo_branch/integration/runtime_slot_selector.py`
  - label: `ONLINE_BRIDGE_FACADE`, `ONLINE_LIVE_CORE`
  - role: stable wrapper for runtime slot selection

- `pseudo_branch/integration/runtime_exact_backend.py`
  - label: `MIXED_FILE`, `ONLINE_LIVE_CORE`
  - role: still main online runtime builder authority
  - next action: split into exact core / Difix / support variants / depth variants under `online_mapping/runtime/`

- `pseudo_branch/integration/runtime_signal_builder.py`
  - label: `MIXED_FILE`, `ONLINE_LIVE_CORE`
  - role: still main online signal-builder authority
  - next action: split default exact-upstream path from override families

- `pseudo_branch/integration/runtime_pseudo_builder.py`
  - label: `ONLINE_BRIDGE_FACADE`, `ONLINE_LIVE_CORE`
  - role: stable wrapper for runtime pseudo-record pack

- `pseudo_branch/integration/runtime_debug_export.py`
  - label: `ONLINE_LIVE_CORE`
  - role: runtime debug writer still used by online record/export path
  - next action: likely migrate into `online_mapping/records/` while preserving wrapper

### 4.2 Common / matching layer

- `pseudo_branch/common/__init__.py`
  - label: `ONLINE_BRIDGE_FACADE`, `MIXED_FILE`
  - role: current online matcher surface, but still bundles non-online helpers
  - next action: trim exports to online-only matcher/pair-forward surface

- `pseudo_branch/common/flow_matcher.py`
  - label: `ONLINE_LIVE_CORE`, `SHARED_KERNEL`
- `pseudo_branch/common/mast3r_matchers.py`
  - label: `ONLINE_LIVE_CORE`, `SHARED_KERNEL`
- `pseudo_branch/common/mast3r_pair_forward.py`
  - label: `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL`
- `pseudo_branch/common/twoimg_pair_proxy_depth.py`
  - label: `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL`

Files not desired in final online-only `pseudo_branch/`:
- `pseudo_branch/common/build_pseudo_cache.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/common/align_depth_scale.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/common/epipolar_depth.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/common/diag_writer.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`

### 4.3 Observation / target / fusion layer

- `pseudo_branch/observation/brpo_reprojection_verify.py`
  - label: `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/observation/pseudo_fusion.py`
  - label: `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`
  - label: `MIXED_FILE`, `ONLINE_LIVE_CORE`, `SHARED_KERNEL`
  - reason: still mixes exact-upstream online semantics with historical standalone observation families
  - next action: isolate exact-upstream online subset; move old families out
- `pseudo_branch/target/depth_supervision_v2.py`
  - label: `ONLINE_LIVE_CORE`, `SHARED_KERNEL`

Files not desired in final online-only `pseudo_branch/`:
- `pseudo_branch/observation/joint_observation.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/observation/pseudo_observation_verifier.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `legacy_or_archive/pseudo_branch_legacy/target/brpo_depth_target.py` — `ARCHIVED_FROM_PSEUDO_BRANCH`
- `legacy_or_archive/pseudo_branch_legacy/target/brpo_depth_densify.py` — `ARCHIVED_FROM_PSEUDO_BRANCH`
- `pseudo_branch/target/depth_target_builder.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/target/support_expand.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`

### 4.4 Mask support layer

Keep for online route:
- `pseudo_branch/mask/rgb_mask_inference.py` — `ONLINE_LIVE_CORE`, `SHARED_KERNEL`
- `pseudo_branch/mask/dense_match_densify.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL`
- `pseudo_branch/mask/cm_local_expansion.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL`

Move out of final online-only `pseudo_branch/`:
- `legacy_or_archive/pseudo_branch_legacy/mask/brpo_confidence_mask.py` — `ARCHIVED_FROM_PSEUDO_BRANCH`
- `legacy_or_archive/pseudo_branch_legacy/mask/brpo_train_mask.py` — `ARCHIVED_FROM_PSEUDO_BRANCH`
- `legacy_or_archive/pseudo_branch_legacy/mask/confidence_builder.py` — `ARCHIVED_FROM_PSEUDO_BRANCH`
- `pseudo_branch/mask/joint_confidence.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`

### 4.5 Refine layer

- `pseudo_branch/refine/__init__.py`
  - label: `ONLINE_BRIDGE_FACADE`, `MIXED_FILE`
  - role: stable surface imported by `slam_backend_brpo.py`
  - problem: still re-exports standalone-oriented scheduler/loss surface together with online runtime surface
  - next action: reduce to the exact online exports consumed by `slam_backend_brpo.py`

Current online-needed surface:
- `pseudo_branch/refine/backend_pseudo_bundle.py` — `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/refine/backend_pseudo_view_loader.py` — `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/refine/backend_pseudo_loss.py` — `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/refine/pseudo_camera_state.py` — `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/refine/pose_gauss_newton.py` — `ONLINE_BRIDGE_FACADE`, `SHARED_KERNEL`
- `pseudo_branch/refine/pseudo_loss_v2.py` — `ONLINE_OPTIONAL_BRANCH`, `SHARED_KERNEL`

Move out of final online-only `pseudo_branch/` if not needed by the live bridge after facade slimming:
- `pseudo_branch/refine/pseudo_refine_scheduler.py` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- any remaining StageA/StageA.5 standalone scheduling-only exports

### 4.6 Historical / legacy blocks

These should not remain in final online-only `pseudo_branch/`:
- `pseudo_branch/gaussian_management/**` — `MOVE_OUT_OF_PSEUDO_BRANCH`
- `pseudo_branch/brpo_v2_signal/**` — `MOVE_OUT_OF_PSEUDO_BRANCH`

## 5. Current standalone and archive roots after phase 1

New non-online homes now in place:
- `standalone_mask_signal/`
- `standalone_prepare/`
- `standalone_replay/`
- `legacy_or_archive/`

These should continue absorbing non-online content as `pseudo_branch/` is slimmed.

## 6. GitHub-facing interpretation after this pass

For an online-mainline-oriented GitHub tree, the code should now be read as:
1. `third_party/S3PO-GS/*` = bridge/orchestration shell
2. `online_mapping/` + `core_shared/` = extracted online payload roots
3. `pseudo_branch/` = temporary online-facing compatibility package that still needs shrinking
4. `standalone_*` + `legacy_or_archive/` = destinations for non-online pseudo/standalone history

So the next cleanup objective is not “delete pseudo_branch”; it is “shrink pseudo_branch until only online route modules remain.”

## 7. Immediate next-step move order

1. extract `runtime_exact_backend.py`
2. extract `runtime_signal_builder.py`
3. isolate exact-upstream online subset from `pseudo_observation_brpo_style.py`
4. slim `pseudo_branch/refine/__init__.py`
5. slim `pseudo_branch/common/__init__.py`
6. once indirect imports are gone, move non-online pseudo_branch modules into `standalone_*` or `legacy_or_archive/`

## 8. Immediate takeaway

The current most important inventory fact is:
- the final desired public shape is not “huge pseudo_branch + some new folders”
- it is “`online_mapping/` + `core_shared/` as payload, plus a much smaller `pseudo_branch/` containing only the online bridge facade/live subset.”
