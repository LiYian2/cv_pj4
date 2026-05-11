# Hermes Refactor Plan for Part3 BRPO

Last updated: 2026-05-11
Status: planning only; no refactor executed in this document
Scope: record the current route boundaries, target repository structure, module split strategy, and recommended execution order for the future Part3 BRPO closeout refactor.

## 1. Executive summary

Current judgment:
1. The real engineering mainline is no longer the old standalone `pseudo_cache -> signal_v2 -> run_pseudo_refinement_v2.py` route.
2. The real live mainline is the S3PO online-mapping route, where pseudo supervision is built and consumed inside the backend keyframe event.
3. A future GitHub-facing simplification should therefore be organized around route boundaries, not around the current historical directory pile (`pseudo_branch/` and `scripts/`).
4. Refactor should not begin by moving files blindly. The first step is to freeze the route map, then extract the online mainline skeleton, then separate mixed online/legacy code.

Important current constraint:
- Do not execute code reorganization yet.
- First keep this document as the planning authority; actual file moves/splits come later after the online mainline and documentation are frozen.
- User constraint: do not refactor or rename `third_party/S3PO-GS/*`; keep third-party code untouched and perform only `part3_BRPO`-side restructuring.

## 2. Current route boundary: what is actually live

### 2.1 Live online-mapping mainline

The current online mainline is driven by:
- `third_party/S3PO-GS/slam.py`
- `third_party/S3PO-GS/utils/slam_frontend.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`

Its runtime pseudo branch is built through:
- `pseudo_branch/integration/runtime_slot_selector.py`
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/integration/runtime_signal_builder.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py`
- `pseudo_branch/integration/runtime_debug_export.py`

Its shared algorithmic dependencies currently include:
- `pseudo_branch/common/*`
- `pseudo_branch/observation/brpo_reprojection_verify.py`
- `pseudo_branch/observation/pseudo_fusion.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`
- `pseudo_branch/refine/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/pseudo_camera_state.py`
- `pseudo_branch/refine/pose_gauss_newton.py`
- optional support/densify modules under `pseudo_branch/mask/`

### 2.2 Standalone / offline control route

The old standalone and offline-preparation route still exists and is still useful as control/reference, but it is no longer the primary engineering landing route.

Its main entrypoints are:
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/run_pseudo_refinement.py`
- `scripts/compat/run_pseudo_refinement.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/select_signal_aware_pseudos.py`
- `scripts/materialize_m5_depth_targets.py`
- `scripts/materialize_twoimg_pair_proxy_depth.py`
- `scripts/replay_internal_eval.py`
- `scripts/brpo_verify_single_branch.py`
- `scripts/legacy_prepare/*`

### 2.3 Historical / experimental / non-mainline support area

These are not the online mainline skeleton and should not define future repo shape:
- `pseudo_branch/gaussian_management/*` (currently much more tied to standalone/history than live online mapping)
- `scripts/diagnostics/*`
- `scripts/test_*.py`
- `scripts/verify_*.py`
- `scripts/run_*.sh`
- `scripts/archive_experiments/*`
- many `.bak*` files
- one-off config generators and temporary wrappers

## 3. What is currently mixed and needs later separation

The future refactor must explicitly separate mixed files instead of pretending every file belongs to only one route.

### 3.1 Mixed executor files

1. `third_party/S3PO-GS/utils/slam_backend.py`
   - Contains original S3PO real-keyframe mapping.
   - Also contains online pseudo slot preparation.
   - Also contains runtime exact backend / runtime signal / runtime pseudo record preparation calls.
   - Also contains joint-primary and side-branch route switching.
   - Also contains optional masked pseudo color refinement.
   - Therefore this file is both the live mainline entry and the biggest mixed boundary.

2. `third_party/S3PO-GS/utils/slam_backend_brpo.py`
   - Contains online mapping pseudo loop.
   - Also contains after-opt continuation logic.
   - Also contains shared joint pseudo engine reused by both continuation and online mapping.
   - Future split should make “shared pseudo optimizer core” explicit, then keep thin online/continuation wrappers around it.

### 3.2 Mixed runtime builder files

1. `pseudo_branch/integration/runtime_exact_backend.py`
   - Core online runtime builder.
   - But it also contains multiple optional branches:
     - default projected-depth exact route
     - GT pseudo RGB upper-bound route
     - Difix restoration + residual fusion route
     - RGB-only verification route
     - `dense_match_v1` optional support densify route
     - `cm_expansion_mode=local_soft_v1` optional soft expansion route
     - direct-depth override route
   - Future split should separate:
     - exact runtime builder core
     - optional RGB-source variants
     - optional support-generation variants
     - optional depth-generation variants

2. `pseudo_branch/integration/runtime_signal_builder.py`
   - Default exact-upstream target/signal builder.
   - Also contains optional depth override branches, especially `twoimg_pair_proxy_cm_capped_v1`.
   - Future split should isolate the depth-override family from the exact-upstream default path.

### 3.3 Mixed standalone files

1. `scripts/run_pseudo_refinement_v2.py`
   - Still a large historical standalone executor.
   - Mixes StageA / StageA.5 / StageB.
   - Mixes old/legacy observation modes with exact-upstream modes.
   - Mixes old G~/SPGM/local-gating experiments with main refine logic.
   - Future split should not move this wholesale into the GitHub simple version; it should be decomposed or retained only as historical control.

2. `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - Offline builder, but algorithmically overlaps with online runtime signal building.
   - Future split should extract shared signal/target assembly kernels rather than keeping separate near-duplicate routes forever.

3. `scripts/brpo_build_mask_from_internal_cache.py`
   - Offline exact backend / mask build path.
   - Algorithmically overlaps with runtime exact backend.
   - Future split should extract shared verifier/support core.

## 4. Refactor goal: desired conceptual repository structure

This is the target logical structure, not an immediate file-move instruction.

```text
part3_brpo/
  online_mapping/
    bridge/
    runtime/
    config/
  standalone_pipeline/
    prepare/
    signal/
    refine/
    replay/
  core_shared/
    matching/
    verification/
    fusion/
    target/
    loss/
    pose/
    io/
  legacy_or_experiments/
    standalone_history/
    diagnostics/
    launchers/
    ablations/
```

### 4.1 `online_mapping/`

Purpose:
- Keep only the live S3PO-online route.
- Expose the runtime pseudo supervision chain clearly.

Expected contents:
- S3PO bridge glue
- runtime slot selection
- runtime exact backend build
- runtime signal build
- runtime pseudo record build
- online mapping config resolver and route toggles

### 4.2 `standalone_pipeline/`

Purpose:
- Preserve the standalone/reference route as a separate route family.
- Keep it explicit that this is not the primary live path.

Expected contents:
- offline prepare / difix dataset build
- offline exact backend / signal build
- standalone refine executor or decomposed refine stages
- replay evaluation

### 4.3 `core_shared/`

Purpose:
- Hold algorithm kernels shared between online and standalone.
- This is the most important future extraction boundary.

Expected contents:
- pair matching / reciprocal matching / dense3d matching
- reprojection verification
- pseudo fusion
- exact-upstream target builder
- pseudo loss contracts
- pose-delta application and Gauss-Newton utilities
- pseudo record bundle definitions and loaders

### 4.4 `legacy_or_experiments/`

Purpose:
- Move non-mainline experiment scaffolding out of the live surface.
- Prevent the future GitHub simple version from being dominated by historical artifacts.

Expected contents:
- old runner scripts
- one-off launchers
- diagnostics and verification scripts
- archival ablation code
- backup files if they must remain in-repo temporarily

## 5. First extraction priority: the online mainline skeleton

Before any big cleanup, the first thing to extract conceptually is the smallest continuous online-mapping chain:

1. frontend runtime state cache export
2. gap closure + pseudo slot selection
3. coarse pseudo render
4. optional Difix + left/right restoration
5. overlap-guided residual fusion
6. MASt3R matching + reciprocal correspondence
7. exact branch verification
8. strict discrete `C_m`
9. exact-upstream depth target build
10. runtime pseudo record build
11. backend joint pseudo mapping loop
12. optional masked pseudo color refinement

This skeleton should become the primary route map for future simplification.

## 6. Later split strategy by boundary, not by current filename

### 6.1 Split 1: shared algorithm cores from both online and standalone

Extract and stabilize reusable kernels first:
- matching
- verification
- fusion
- exact target builder
- loss contract
- pose math
- pseudo record data model

Reason:
- this gives a real common substrate and reduces duplicate logic
- lower risk than moving top-level entrypoints first

### 6.2 Split 2: online runtime glue from S3PO backend control flow

After shared cores are explicit:
- thin the runtime builder pieces in `slam_backend.py`
- make backend event order easier to read
- keep one obvious online execution spine

Reason:
- current live mainline truth is hidden inside large mixed backend files
- future GitHub users must be able to see online route without reading the whole historical backend

### 6.3 Split 3: standalone route into clean reference pipeline

Only after online mainline is stabilized:
- isolate prepare
- isolate signal build
- isolate refine
- isolate replay
- retain only the minimal reference/control implementation needed

Reason:
- standalone is now mainly reference/control
- it should be simplified after the online route is made legible

### 6.4 Split 4: archive experiment scaffolding

Last step:
- move diagnostics, test harnesses, runner shells, and stale variants
- do not mix this with core route extraction

Reason:
- archive work is low-value but noisy
- doing it too early creates churn without clarifying the main system

## 7. Files that should define future GitHub simple version

The future simple GitHub should not attempt to include every historical experiment.
It should primarily expose:

1. the live online mainline route
2. the exact-upstream pseudo supervision components
3. one explicit standalone reference/control route
4. minimal replay/evaluation support
5. a concise pipeline doc and code map

It should not center around:
- old compare launchers
- every D/E-series wrapper
- every temporary verification script
- every historical docs artifact

## 8. Recommended phased execution order when refactor actually starts

### Phase 0 — freeze and document
Output:
- final online mainline `PIPELINE.md`
- this refactor plan
- explicit route inventory

### Phase 1 — inventory and tag
Tasks:
- label files as `online`, `standalone`, `shared`, `legacy`, or `mixed`
- do not move files yet

### Phase 2 — extract shared kernels
Tasks:
- create stable shared modules for verifier/fusion/target/loss/pose/data model
- switch both online and standalone callers to the shared kernels

### Phase 3 — expose online skeleton cleanly
Tasks:
- reduce online runtime path to a readable chain
- separate route control from algorithm payload where possible
- keep S3PO patch boundary explicit

### Phase 4 — simplify standalone reference route
Tasks:
- split prepare/signal/refine/replay boundaries
- keep only minimal reference entrypoints

### Phase 5 — archive or drop non-mainline artifacts
Tasks:
- move launchers, diagnostics, ad-hoc tests, and stale backups out of live surface

## 9. Explicit non-goals for the current step

This document does not authorize immediate:
- file moves
- import rewrites
- path renaming
- launcher cleanup
- deletion of historical modules
- GitHub packaging

Those come later, after the online pipeline doc is finalized and accepted.

## 10. Practical notes for the next session

When refactor work begins later, the first two concrete questions should be:
1. Which exact online-mapping chain do we want to preserve as the canonical minimal route?
2. Which mixed files must be split first because they currently hide both online and historical logic in the same executor?

Current best answer:
- preserve the online runtime chain rooted in `slam_frontend.py -> slam_backend.py -> slam_backend_brpo.py`
- first split shared algorithm kernels from the mixed executors, not vice versa

## 11. Immediate takeaway

Do not start from “cleaning `scripts/`” or “cleaning `pseudo_branch/`”.
Start from “extract the online-mapping skeleton, then separate shared kernels, then shrink the historical surface.”

That order matches the real current project state and will produce a much cleaner final repository than a directory-only cleanup.