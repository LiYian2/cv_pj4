# Hermes Refactor Plan for Part3 BRPO

Last updated: 2026-05-12
Status: phase 1 executed; first repo-side extraction landed and import-verified
Scope: record the current online/standalone route boundaries, the now-implemented first extraction pass, and the next-step plan for shrinking `pseudo_branch/` down to online-only content while keeping the S3PO bridge stable.

## 1. Executive summary

Current judgment:
1. The real engineering mainline is the S3PO online-mapping route, not the old standalone `pseudo_cache -> signal_v2 -> run_pseudo_refinement_v2.py` route.
2. The user wants `pseudo_branch/` to end in an online-only state: keep only modules still used by the current online mapping route, and move everything else out to standalone/archive areas.
3. `third_party/S3PO-GS/*` is still the live executor boundary, but it should be treated as a thin bridge shell rather than the place where BRPO implementation complexity lives.
4. The first additive extraction pass has already landed inside `part3_BRPO/`: new `online_mapping/`, `core_shared/`, `standalone_*`, and `legacy_or_archive/` roots now exist, and import-preserving wrappers were verified.

Important current constraint:
- Do not break the live online route.
- Keep `third_party/S3PO-GS/*` behavior stable while repo-side modules are cleaned up.
- Prefer wrapper-first moves and import-surface preservation over rename-first cleanup.

## 2. What was actually implemented in phase 1

### 2.1 New repo-side roots created

The following new roots now exist and are import-verified:
- `online_mapping/`
- `core_shared/`
- `standalone_mask_signal/`
- `standalone_prepare/`
- `standalone_replay/`
- `legacy_or_archive/`

### 2.2 Implemented extractions already landed

Moved into new internal locations, with old import paths preserved as wrappers/facades:
- `pseudo_branch/integration/runtime_slot_selector.py` -> `online_mapping/runtime/slot_selector.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py` -> `online_mapping/records/runtime_record_builder.py`
- `pseudo_branch/observation/brpo_reprojection_verify.py` -> `core_shared/verification/brpo_reprojection_verify.py`
- `pseudo_branch/observation/pseudo_fusion.py` -> `core_shared/fusion/pseudo_fusion.py`
- `pseudo_branch/refine/backend_pseudo_bundle.py` -> `core_shared/records/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py` -> `core_shared/records/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py` -> `core_shared/losses/backend_pseudo_loss.py`
- `pseudo_branch/refine/pseudo_camera_state.py` -> `core_shared/pose/pseudo_camera_state.py`
- `pseudo_branch/refine/pose_gauss_newton.py` -> `core_shared/pose/pose_gauss_newton.py`
- online-facing portion of `pseudo_branch/refine/pseudo_loss_v2.py` -> `core_shared/losses/pseudo_loss_v2.py`

Standalone/control scripts were also moved behind compatibility wrappers into:
- `standalone_mask_signal/`
- `standalone_prepare/`
- `standalone_replay/`

Historical broken path handled explicitly:
- `scripts/run_pseudo_refinement_v2.py` is no longer a broken external symlink; it is now a deliberate retired wrapper pointing into `legacy_or_archive/retired_entrypoints/`.

### 2.3 Verification completed in phase 1

Verified with real remote checks:
- `py_compile` passed for new modules and preserved wrappers.
- direct import smoke passed for:
  - new `online_mapping/*`
  - new `core_shared/*`
  - preserved `pseudo_branch/*` facades
  - preserved `scripts/*` wrappers
- result: `ALL_IMPORTS_OK`

## 3. Current live route after phase 1

### 3.1 Stable bridge boundary

Current live executor chain remains:
- `third_party/S3PO-GS/slam.py`
- `third_party/S3PO-GS/utils/slam_frontend.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`

This boundary is still authoritative for runtime orchestration.

### 3.2 What `pseudo_branch/` still needs to provide today

Directly or indirectly, the current live online route still depends on:
- `pseudo_branch.integration`
- `pseudo_branch.common`
- `pseudo_branch.refine`
- selected `pseudo_branch.observation/*`
- selected `pseudo_branch.mask/*`
- selected `pseudo_branch.target/*`

So after phase 1, `pseudo_branch/` is smaller in responsibility, but it is not yet online-only in content.

## 4. End-state goal for `pseudo_branch/`

The desired end state is now explicit:
- `pseudo_branch/` should contain only the current online-mapping route’s compatibility facade and any still-live online modules not yet extracted.
- Anything standalone-only, historical, diagnostic, or archive-only should move out.

Target end-state by subarea:
- keep as online facade/core: `integration/`, `refine/`, selected `common/`, selected `observation/`, selected `mask/`, selected `target/`
- move out: standalone signal/prepare/replay logic, old StageA/StageB scheduling helpers, historical gaussian-management/SPGM/local-gating code, old target builders, old joint-observation families, diagnostics/cache-build helpers

## 5. Immediate next extraction priorities

### 5.1 Highest-priority online files still too mixed

1. `pseudo_branch/integration/runtime_exact_backend.py`
   - still the biggest online mixed file
   - next split should produce repo-side helpers for:
     - exact default core
     - Difix branch
     - support variants (`reciprocal_seed`, `dense_match_v1`, `cm_local_expansion`)
     - depth override branches

2. `pseudo_branch/integration/runtime_signal_builder.py`
   - still mixes default exact-upstream path with depth override families
   - next split should isolate:
     - exact-upstream default builder
     - `twoimg_pair_proxy_cm_capped_v1` and any future depth-override family

3. `pseudo_branch/observation/pseudo_observation_brpo_style.py`
   - still contains both current exact-upstream semantics and historical standalone observation families
   - next split should isolate exact-upstream online semantics from old standalone modes

### 5.2 Package facades that still need slimming

1. `pseudo_branch/refine/__init__.py`
   - currently still re-exports standalone-oriented scheduler/loss surface together with online runtime surface
   - next step: trim it to online-only exports required by `slam_backend_brpo.py`

2. `pseudo_branch/common/__init__.py`
   - currently still bundles online matcher surface together with non-online geometry/cache helpers
   - next step: trim it to only the matcher/pair-forward symbols used by the online route

3. `pseudo_branch/integration/__init__.py`
   - currently still points runtime exact/signal/debug exports at old locations
   - next step: keep the facade stable, but move underlying implementation deeper into `online_mapping/`

## 6. Planned `pseudo_branch/` shrink order

Do this in order, not by broad deletion:
1. keep facade imports stable for `third_party` callers
2. continue extracting mixed online implementations into `online_mapping/` and `core_shared/`
3. slim `pseudo_branch/common/__init__.py` and `pseudo_branch/refine/__init__.py` to online-only exports
4. once package import chains no longer reach old modules, move the non-online modules out of `pseudo_branch/`
5. only then archive/remove the old pseudo_branch content

This order matters because many currently non-online files are still dragged in indirectly through package `__init__` files.

## 7. Planned `slam` bridge cleanup strategy

The four S3PO bridge files should not be the place where BRPO implementation keeps growing.
They should converge to thin shells with stable contracts.

### 7.1 `slam.py`
- keep as launch/runtime entry shell
- avoid adding BRPO implementation detail here

### 7.2 `slam_frontend.py`
- keep as frontend state / keyframe-event shell
- only light cleanup unless a clear online-mapping responsibility leak is found

### 7.3 `slam_backend.py`
Target it as a thin orchestration shell around these blocks:
- config resolve
- runtime camera-state cache
- gap closure / slot selection
- runtime exact backend prepare
- runtime signal build
- runtime pseudo record pack
- topology dispatch
- optional masked pseudo color refinement

The file can remain in place, but its internals should become clearer and thinner by delegating repo-side work to `online_mapping/` and `core_shared/`.

### 7.4 `slam_backend_brpo.py`
Target it as a thin optimization shell around these blocks:
- runtime pseudo record normalization
- optimizer-group assembly
- loss-call boundary
- pose-update helpers
- topology dispatch
- stats/export

Keep external names stable:
- `BRPOMappingConfig`
- `run_brpo_pseudo_mapping(...)`

## 8. What should move out of `pseudo_branch/` once facades are slim enough

Priority move-out candidates:
- `pseudo_branch/gaussian_management/**`
- `pseudo_branch/brpo_v2_signal/**`
- `pseudo_branch/common/build_pseudo_cache.py`
- `pseudo_branch/common/align_depth_scale.py`
- `pseudo_branch/common/epipolar_depth.py`
- `pseudo_branch/common/diag_writer.py`
- `pseudo_branch/observation/joint_observation.py`
- `pseudo_branch/observation/pseudo_observation_verifier.py`
- `legacy_or_archive/pseudo_branch_legacy/mask/brpo_confidence_mask.py` — archived from pseudo_branch
- `legacy_or_archive/pseudo_branch_legacy/mask/brpo_train_mask.py` — archived from pseudo_branch
- `legacy_or_archive/pseudo_branch_legacy/mask/confidence_builder.py` — archived from pseudo_branch
- `pseudo_branch/mask/joint_confidence.py`
- `legacy_or_archive/pseudo_branch_legacy/target/brpo_depth_target.py` — archived from pseudo_branch
- `legacy_or_archive/pseudo_branch_legacy/target/brpo_depth_densify.py` — archived from pseudo_branch
- `pseudo_branch/target/depth_target_builder.py`
- `pseudo_branch/target/support_expand.py`
- standalone-oriented pieces of `pseudo_branch/refine/` not needed by the online route

## 9. Immediate next-step plan

Next implementation wave should be:
1. extract `runtime_exact_backend.py`
2. extract `runtime_signal_builder.py`
3. split online-only semantics out of `pseudo_observation_brpo_style.py`
4. slim `pseudo_branch/refine/__init__.py` to only what `slam_backend_brpo.py` uses
5. slim `pseudo_branch/common/__init__.py` to only what `slam_backend.py` and live runtime code use
6. then move non-online pseudo_branch content into `standalone_*` or `legacy_or_archive/`
7. after repo-side payload stabilizes, do a light shell-cleanup pass over the four bridge `slam*` files

## 10. Immediate takeaway

The current state should be read as:
1. `third_party/S3PO-GS/*` = stable online bridge shell
2. `online_mapping/` + `core_shared/` = new extracted payload roots already landed
3. `pseudo_branch/` = temporary compatibility + partially extracted online payload; target is online-only
4. `standalone_*` + `legacy_or_archive/` = place where non-online content should continue to move

So the next refactor objective is no longer just “extract online pieces somewhere”; it is specifically:
- finish extracting online authority out of mixed pseudo_branch files
- slim pseudo_branch down to online-only content
- leave the four `slam*` files as stable, clearer shells over that payload

Update note (2026-05-12, phase 1.5): exact-upstream observation semantics moved behind core_shared.targets; runtime_exact_backend/runtime_signal_builder authority moved behind online_mapping.runtime wrappers; standalone legacy observation consumers redirected to standalone_mask_signal.

Archived safe subset completed on 2026-05-10:
- moved `brpo_confidence_mask.py`, `brpo_train_mask.py`, `confidence_builder.py` out of `pseudo_branch/mask/`
- moved `brpo_depth_target.py`, `brpo_depth_densify.py` out of `pseudo_branch/target/`
- remaining move-out candidates (`joint_confidence.py`, `depth_target_builder.py`, `support_expand.py`, legacy observation helpers) still require bridge-aware follow-up
