# Hermes Online Mapping Extraction Design

Last updated: 2026-05-12
Status: design updated after phase 1 extraction landed and passed real import verification
Goal: finish converging the live online route toward `online_mapping/` + `core_shared/` payload roots while reducing `pseudo_branch/` to online-only content and keeping the four S3PO bridge files stable.

## 1. Immediate conclusion

Current grounded conclusion:
1. The live online mainline still runs through `slam_frontend.py -> slam_backend.py -> slam_backend_brpo.py`.
2. The first repo-side extraction pass succeeded: several online kernels and record/pose/loss modules already moved under `online_mapping/` and `core_shared/`.
3. However, the bridge still imports `pseudo_branch.integration`, `pseudo_branch.common`, and `pseudo_branch.refine`, so `pseudo_branch/` is not yet online-only.
4. The user’s desired end state is now sharper than before: `pseudo_branch/` should eventually contain only current online-mapping modules and facades; everything else should move out.

## 2. Non-breakage principle

The non-breakage rules remain:
1. keep `third_party/S3PO-GS/*` import/call surfaces stable first
2. continue additive-first extraction behind existing facades
3. do not broad-delete pseudo_branch content before package facades stop importing it
4. verify imports after each structural step

This means:
- `slam_backend.py` may continue importing `pseudo_branch.integration`
- `slam_backend_brpo.py` may continue importing `pseudo_branch.refine`
- `pseudo_branch/` should shrink by facade slimming plus internal extraction, not by early deletion

## 3. Current live bridge contracts

### 3.1 `slam_backend.py`

Current repo-side contract it consumes:
- from `pseudo_branch.integration`:
  - `RuntimeExactBackendConfig`
  - `RuntimeSlotSelectorConfig`
  - `build_runtime_exact_backend_bundle`
  - `build_runtime_exact_signal_bundle`
  - `build_runtime_pseudo_record_bundle`
  - `select_runtime_pseudo_slots`
- from `pseudo_branch.common`:
  - `build_pair_matcher`

Implication:
- `pseudo_branch.integration` and the online matcher surface in `pseudo_branch.common` must remain stable until a later bridge cleanup pass.

### 3.2 `slam_backend_brpo.py`

Current repo-side contract it consumes:
- from `pseudo_branch.refine`:
  - `BackendPseudoLossConfig`
  - `BackendPseudoViewRecord`
  - `build_records_from_pseudo_bundle`
  - `compute_backend_pseudo_exact_loss`
  - `current_w2c`
  - `load_pseudo_bundle_from_stageA_history`
  - `refresh_viewpoint_transforms_`
  - `apply_pose_delta_before_render_`
  - `viewpoint_optimizer_groups`
  - `scale_reg_loss`
  - `GaussNewtonPoseOptimizer`

Implication:
- `pseudo_branch.refine` must become an online-only facade, not a mixed online/standalone export bag.

## 4. Phase 1 extraction already complete

### 4.1 Online/runtime and shared-core moves landed

Already extracted behind compatibility wrappers:
- slot selection -> `online_mapping/runtime/slot_selector.py`
- runtime pseudo record build -> `online_mapping/records/runtime_record_builder.py`
- reprojection verification -> `core_shared/verification/brpo_reprojection_verify.py`
- pseudo fusion -> `core_shared/fusion/pseudo_fusion.py`
- pseudo bundle/view loader -> `core_shared/records/*`
- backend pseudo loss -> `core_shared/losses/*`
- pose state / GN -> `core_shared/pose/*`

### 4.2 Standalone move-outs already landed

Standalone/control entrypoints now have dedicated non-online homes:
- `standalone_mask_signal/`
- `standalone_prepare/`
- `standalone_replay/`
- `legacy_or_archive/`

This is important because future pseudo_branch slimming no longer needs to preserve those as top-level `scripts/*` authorities.

## 5. New target structure

The target structure is now more specific than before:

```text
part3_BRPO/
  online_mapping/
    runtime/
    records/
  core_shared/
    matching/
    verification/
    fusion/
    targets/
    losses/
    pose/
    records/
  pseudo_branch/
    integration/   # online bridge facade only
    refine/        # online bridge facade only
    common/        # only online matcher surface still needed by bridge/runtime
    observation/   # only still-live online observation helpers
    mask/          # only still-live online support helpers
    target/        # only still-live online target helpers
  standalone_mask_signal/
  standalone_prepare/
  standalone_replay/
  legacy_or_archive/
```

Critical revision:
- `pseudo_branch/` is no longer a long-term “everything BRPO” namespace.
- It is only a temporary compatibility package plus the still-unextracted online subset.

## 6. Safe extraction strategy from here

### Phase A — finish extracting remaining online authorities

#### A1. `runtime_exact_backend.py`

Next target split under `online_mapping/runtime/`:
- `exact_core.py`
- `difix_rgb.py`
- `support_variants.py`
- `depth_variants.py`

Safe rule:
- keep `build_runtime_exact_backend_bundle(...)` and `RuntimeExactBackendConfig` stable at `pseudo_branch.integration`
- move implementation behind that surface

#### A2. `runtime_signal_builder.py`

Next target split under `online_mapping/runtime/`:
- `signal_exact_upstream.py`
- `signal_depth_overrides.py`

Safe rule:
- keep `build_runtime_exact_signal_bundle(...)` stable at `pseudo_branch.integration`
- move implementation behind that surface

#### A3. `pseudo_observation_brpo_style.py`

Next target split under `core_shared/targets/` or `online_mapping/runtime/targets/`:
- exact-upstream online semantics
- old standalone observation families moved elsewhere

Safe rule:
- do not delete the file early; first isolate the online authority subset

### Phase B — slim package facades to online-only

#### B1. `pseudo_branch/refine/__init__.py`

Goal:
- export only the symbols actually used by `slam_backend_brpo.py`
- stop re-exporting StageA/StageA.5 standalone scheduler surface from the online bridge package

Expected result:
- once facade is slim, standalone-only refine helpers can move out of `pseudo_branch/refine/`

#### B2. `pseudo_branch/common/__init__.py`

Goal:
- export only the online matcher/pair-forward surface used by live online runtime code
- stop dragging non-online geometry/cache helpers through the bridge package

Expected result:
- `build_pseudo_cache.py`, `epipolar_depth.py`, `align_depth_scale.py`, `diag_writer.py` become movable

#### B3. `pseudo_branch/integration/__init__.py`

Goal:
- keep stable public names
- make every export resolve to extracted `online_mapping/` implementation where possible

### Phase C — move non-online pseudo content out

Once phases A and B are done, move out of `pseudo_branch/`:
- `gaussian_management/**`
- `brpo_v2_signal/**`
- old target-builder family
- old joint-observation / verifier family
- standalone confidence/mask family
- standalone scheduler and similar refine-only helpers
- cache-build/diagnostic utilities not needed by the online path

## 7. How the four `slam*` bridge files should be cleaned up

These files should be treated as thin shells over the cleaned repo-side payload.

### 7.1 `slam.py`
- keep as launch/runtime shell
- no major BRPO logic growth here

### 7.2 `slam_frontend.py`
- keep as frontend state shell
- only light structural cleanup unless clear online-mapping logic leakage is found

### 7.3 `slam_backend.py`
Reorganize internally around explicit sections:
- config resolve
- runtime matcher/state setup
- gap closure and slot selection
- exact backend prepare
- signal build
- pseudo record pack
- mapping dispatch
- optional masked color refinement

Target result:
- same path, same external behavior, much thinner and easier to read
- heavy BRPO logic delegated to `online_mapping/` and `core_shared/`

### 7.4 `slam_backend_brpo.py`
Reorganize internally around explicit sections:
- pseudo record normalization
- optimizer-group assembly
- loss-call boundary
- pose-update helper calls
- topology dispatch
- stats/export

Target result:
- keep `BRPOMappingConfig` and `run_brpo_pseudo_mapping(...)` stable
- make the file a thin optimization shell over repo-side payload helpers

## 8. What not to do next

Do not do these first:
1. broad-delete non-online pseudo_branch modules before facades stop importing them
2. edit all four `slam*` files heavily before repo-side payload extraction is more complete
3. treat `pseudo_branch/` as the permanent public home for standalone/history code
4. move standalone/history code back into top-level `scripts/*`

## 9. Immediate next-step implementation plan

1. extract `runtime_exact_backend.py`
2. extract `runtime_signal_builder.py`
3. isolate exact-upstream online semantics from `pseudo_observation_brpo_style.py`
4. slim `pseudo_branch/refine/__init__.py` to online-only bridge exports
5. slim `pseudo_branch/common/__init__.py` to online-only matcher exports
6. move the now-detached non-online pseudo_branch modules into `standalone_*` or `legacy_or_archive/`
7. only then do a shell-cleanup pass on `slam_backend.py` and `slam_backend_brpo.py`

## 10. Immediate takeaway

The refactor target is now:
- not “keep a large mixed `pseudo_branch/` forever”
- but “finish extracting payload into `online_mapping/` and `core_shared/`, then leave `pseudo_branch/` as a small online-only bridge compatibility layer, with the four `slam*` files cleaned into thin orchestration shells.”

Update note (2026-05-12, phase 1.5): exact-upstream observation semantics moved behind core_shared.targets; runtime_exact_backend/runtime_signal_builder authority moved behind online_mapping.runtime wrappers; standalone legacy observation consumers redirected to standalone_mask_signal.


## 2026-05-12 phase 2 mask/target extraction
- live online mask helpers moved to `online_mapping/mask/` (`rgb_mask_inference.py`, `dense_match_densify.py`, `cm_local_expansion.py`) and `online_mapping/runtime/runtime_exact_backend.py` now imports them directly.
- shared exact/signal target builders moved to `core_shared/targets/` (`depth_supervision_v2.py`, `joint_confidence.py`, `joint_observation.py`, `exact_upstream_observation.py`).
- standalone BRPO signal/depth helpers moved out of `pseudo_branch` into `standalone_mask_signal/` (`brpo_confidence_mask.py`, `brpo_train_mask.py`, `brpo_depth_densify.py`, `brpo_depth_target.py`, `support_expand.py`).
- `pseudo_branch/mask/*`, `pseudo_branch/target/*`, and `pseudo_branch/observation/joint_observation.py` are now compatibility facades only; `confidence_builder.py` and `depth_target_builder.py` are retired into `legacy_or_archive/retired_pseudo_branch_mask_target/`.
