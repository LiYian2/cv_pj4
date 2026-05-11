# Hermes Online Mapping Extraction Design

Last updated: 2026-05-11
Status: design-only; no Python refactor executed here
Goal: design how to extract the current online mapping path into a cleaner module structure without breaking the live system.

## 1. Immediate conclusion

### 1.1 About `scripts/run_pseudo_refinement_v2.py`

Grounded conclusion after direct remote inspection:
1. It is not part of the current live online-mapping mainline.
2. Current active runs are all `slam.py --config ...` online-mapping jobs; none use `run_pseudo_refinement_v2.py`.
3. Current mainline route is `slam_frontend.py -> slam_backend.py -> slam_backend_brpo.py`.
4. `scripts/run_pseudo_refinement_v2.py` is currently a broken external symlink to `/data/bzhang512/tmp/run_pseudo_refinement_v2_pose_fix.py`, and the target does not exist.
5. Therefore: deleting it would not break the current online mainline, but it still belongs to legacy/standalone/control history and should be retired deliberately, not confused with live online code.

Important nuance:
- `scripts/README.md` still lists `run_pseudo_refinement_v2.py` as top-level live core, but that is stale relative to the current online mainline.
- For refactor planning, treat that README statement as historical, not authoritative.

## 2. Non-breakage principle

The top design requirement is exactly what the user asked: do not break the system.

So the extraction order must be additive-first, not move-first:
1. add new clean modules first
2. keep old import surface alive through thin wrappers / facades
3. switch one caller boundary at a time
4. verify behavior after each switch
5. only archive or remove old paths after the new path has already been exercised by the real online route

This means:
- no changes to `third_party/S3PO-GS/*` in this refactor plan
- no deletion of mixed online files before repo-side wrappers exist
- no path surgery inside active runtime entrypoints
- keep current `pseudo_branch.integration` and `pseudo_branch.refine` import surfaces stable for the untouched third-party caller

## 3. What the current online system really looks like

### 3.1 Executor boundary (must stay stable at first)

Current live executor chain:
- `third_party/S3PO-GS/slam.py`
- `third_party/S3PO-GS/utils/slam_frontend.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`

Current live payload imported from `part3_BRPO`:
- `pseudo_branch/integration/runtime_slot_selector.py`
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/integration/runtime_signal_builder.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py`
- `pseudo_branch/integration/runtime_debug_export.py`
- `pseudo_branch/refine/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`
- `pseudo_branch/refine/pseudo_camera_state.py`
- `pseudo_branch/refine/pseudo_loss_v2.py`
- `pseudo_branch/refine/pose_gauss_newton.py`
- selected `common/`, `observation/`, `mask/`, and `target/` helpers

### 3.2 What is mixed today

The main mixed boundaries are:
1. `third_party/S3PO-GS/utils/slam_backend.py`
   - config resolving
   - slot selection / preparation orchestration
   - runtime exact backend building
   - runtime signal building
   - runtime pseudo record packing
   - pseudo mapping dispatch
   - topology dispatch (`joint_primary` vs `side_branch`)
   - optional masked pseudo color refinement
   - plus legacy real-keyframe mapping in the same file

2. `third_party/S3PO-GS/utils/slam_backend_brpo.py`
   - runtime pseudo mapping entry
   - online pseudo optimizer implementation
   - continuation-oriented code paths in the same file

3. `pseudo_branch/integration/runtime_exact_backend.py`
   - default projected exact path
   - optional GT pseudo RGB path
   - optional Difix branch
   - optional `rgb_only_support_mode=dense_match_v1`
   - optional `cm_expansion_mode=local_soft_v1`
   - optional direct-depth branch

4. `pseudo_branch/integration/runtime_signal_builder.py`
   - default projected exact-upstream path
   - optional `twoimg_pair_proxy_cm_capped_v1` override path

5. `pseudo_branch/observation/pseudo_observation_brpo_style.py`
   - current exact-upstream mainline builder
   - multiple historical target/observation families for standalone experiments

## 4. Safe target structure

The refactor should converge to a structure like this conceptually:

```text
part3_BRPO/
  online_mapping/
    runtime/
    records/
    config/
  core_shared/
    matching/
    verification/
    fusion/
    targets/
    losses/
    pose/
    records/
  standalone_pipeline/
    prepare/
    mask_signal/
    refine/
    replay/
  legacy_or_archive/
    old_runners/
    diagnostics/
    experiments/
  pseudo_branch/
    integration/   # compatibility facade kept for untouched third-party imports
    refine/        # compatibility facade kept for untouched third-party imports
```

This is a design target, not a first-step file move instruction.
Important revision: there is no plan to create a new repo-side bridge by editing third-party code. The bridge remains the current untouched third-party caller; `part3_BRPO` only provides cleaner modules behind the existing `pseudo_branch` import surface.

## 5. Safe extraction strategy

### Phase A — keep third-party untouched, extract behind `pseudo_branch` facades

Do not touch the external execution shape.
Keep these import/call sites stable because the third-party caller is not allowed to change:
- untouched `slam_backend.py` continues importing from `pseudo_branch.integration`
- untouched `slam_backend_brpo.py` continues importing from `pseudo_branch.refine`

So the first safe refactor move is entirely repo-side:
- keep `pseudo_branch/integration/__init__.py` as the stable facade
- keep `pseudo_branch/refine/__init__.py` as the stable facade
- gradually redirect those exports to cleaner internal modules under `online_mapping/` and `core_shared/`

This avoids breaking the untouched third-party caller while still letting us refactor internals.

### Phase B — split online runtime by semantic layer, not by filename age

#### B1. Split `runtime_exact_backend.py`

Current file should be decomposed conceptually into:
1. `online_mapping/runtime/exact_core.py`
   - render pseudo/reference RGB/depth
   - matcher dispatch
   - exact left/right verification
   - default exact bundle assembly
2. `online_mapping/runtime/difix_rgb.py`
   - Difix load/runtime helpers
   - left/right branch restoration
   - residual fusion entry
3. `online_mapping/runtime/support_variants.py`
   - `reciprocal_seed`
   - `dense_match_v1`
   - `cm_local_expansion`
4. `online_mapping/runtime/depth_variants.py`
   - projected default
   - `mast3r_direct_exact_anchor_v1`
   - any direct-depth side branches

Safe first implementation rule:
- do not replace the old file at once
- first create helpers and make `build_runtime_exact_backend_bundle(...)` delegate internally to them
- preserve the top-level function signature unchanged

#### B2. Split `runtime_signal_builder.py`

Target substructure:
1. `online_mapping/runtime/signal_exact_upstream.py`
   - default exact-upstream projected target path
2. `online_mapping/runtime/signal_depth_overrides.py`
   - `twoimg_pair_proxy_cm_capped_v1`
   - any future override families

Safe rule:
- keep `build_runtime_exact_signal_bundle(...)` name and signature stable at first
- route to default/override implementation internally

#### B3. Split `runtime_pseudo_builder.py`

This file is already relatively clean.
Refactor goal is smaller:
- move record schema/data-model logic into a shared record module
- keep runtime wrapper thin

Good target:
1. `core_shared/records/backend_pseudo_record.py`
2. `online_mapping/records/runtime_record_builder.py`

### Phase C — split optimizer core from runtime shell

#### C1. `slam_backend_brpo.py`

Because third-party is frozen, do not plan any implementation step that edits this file.
Instead, repo-side refactor must work under the existing untouched call boundary:
1. keep `run_brpo_pseudo_mapping(...)` as an external fixed boundary
2. restructure only the repo-side payload objects and helpers that this function already imports from `pseudo_branch.refine`
3. treat optimizer-core extraction from this file as out of scope unless the user later relaxes the third-party constraint

Safe rule:
- assume `run_brpo_pseudo_mapping(...)` is fixed
- improve readability and separation only on the `part3_BRPO` side behind the current import boundary

#### C2. `pseudo_branch/refine/__init__.py`

Current problem:
- online runtime imports and standalone StageA/StageB utilities are re-exported from the same package facade

Safe design:
- keep `pseudo_branch.refine` import path stable for current callers
- internally split exports into two groups:
  1. online-needed exports
  2. standalone-only exports

Suggested internal layout:
- `core_shared/losses/backend_pseudo_loss.py`
- `core_shared/records/backend_pseudo_bundle.py`
- `core_shared/records/backend_pseudo_view_loader.py`
- `core_shared/pose/pseudo_camera_state.py`
- `core_shared/pose/pose_gauss_newton.py`
- `standalone_pipeline/refine/stage_scheduler.py`
- `standalone_pipeline/refine/stage_losses.py`

### Phase D — extract shared algorithm kernels from mixed online/standalone code

The first shared-kernel extraction set should be:
1. matching
   - `flow_matcher.py`
   - `mast3r_matchers.py`
   - `mast3r_pair_forward.py`
2. verification
   - `brpo_reprojection_verify.py`
3. fusion
   - `pseudo_fusion.py`
4. exact-upstream target semantics
   - current mainline subset from `pseudo_observation_brpo_style.py`
5. pose math
   - `pseudo_camera_state.py`
   - `pose_gauss_newton.py`
6. record schema
   - `backend_pseudo_bundle.py`
   - `backend_pseudo_view_loader.py`
7. loss contract
   - `backend_pseudo_loss.py`
   - only the online-relevant subset of `pseudo_loss_v2.py`

Important non-breakage rule:
- extract only the exact-upstream mainline subset first
- do not drag every historical observation/target mode into `core_shared` on day one

## 6. What not to do first

These would be high-risk and are explicitly out of scope or not the first move:
1. editing `third_party/S3PO-GS/utils/slam_backend.py`
2. editing `third_party/S3PO-GS/utils/slam_backend_brpo.py`
3. deleting `pseudo_branch/integration/runtime_exact_backend.py` before a delegated wrapper exists
4. deleting `pseudo_branch/observation/pseudo_observation_brpo_style.py` before isolating exact-upstream mainline helpers
5. treating `scripts/run_pseudo_refinement_v2.py` as an authority path for new work
6. starting from `scripts/` cleanup instead of online payload extraction

## 7. First concrete engineering steps when implementation starts

### Step 1
Create a new repo-side internal namespace without changing callers yet.

Recommended first directories:
- `online_mapping/runtime/`
- `online_mapping/records/`
- `core_shared/matching/`
- `core_shared/verification/`
- `core_shared/fusion/`
- `core_shared/targets/`
- `core_shared/losses/`
- `core_shared/pose/`
- `core_shared/records/`

### Step 2
Copy/extract the smallest online-stable helpers first, not the giant mixed files.

Best first candidates:
- exact reprojection verification
- pseudo fusion
- pseudo camera state / pose delta application
- backend pseudo loss
- backend pseudo record schema

### Step 3
Refactor `runtime_signal_builder.py` and `runtime_exact_backend.py` into delegating wrappers.

Target state:
- old filenames still exist
- public function names still exist
- logic moved into smaller helpers behind them

### Step 4
Refactor `slam_backend_brpo.py` into shell + core.

Target state:
- `run_brpo_pseudo_mapping(...)` still exists
- implementation body becomes a thin wrapper around extracted runtime mapping core

### Step 5
Only after online extraction is stable, reclassify standalone.

That is when to decide:
- which parts of `build_brpo_v2_signal_from_internal_cache.py` become wrappers over shared kernels
- whether `brpo_build_mask_from_internal_cache.py` should remain a standalone control script
- whether `run_pseudo_refinement_v2.py` should be restored as a real archived control runner or fully retired

## 8. Validation ladder for a non-breaking refactor

Each split phase should be validated with the smallest possible ladder.

### Level 0 — static import check
Verify that these still import after each extraction step:
- `from pseudo_branch.integration import build_runtime_exact_backend_bundle`
- `from pseudo_branch.integration import build_runtime_exact_signal_bundle`
- `from pseudo_branch.integration import build_runtime_pseudo_record_bundle`
- `from pseudo_branch.refine import compute_backend_pseudo_exact_loss`
- `from pseudo_branch.refine import apply_pose_delta_before_render_`

### Level 1 — event preparation smoke
On one saved runtime event or one tiny debug config, verify:
- slot selection still works
- exact bundle still writes outputs
- signal bundle still writes outputs
- runtime pseudo record still builds

### Level 2 — one-keyframe online smoke
Verify one short `slam.py` online run with:
- `topology_mode=joint_primary`
- `placement_mode=midpoint_only`
- `use_difix_restoration=true`
- projected depth path

### Level 3 — option-branch smoke
Separately verify the option branches that are currently known-live in recent configs:
- `depth_generation_mode=twoimg_pair_proxy_cm_capped_v1` (`e9`-family)
- `rgb_only_support_mode=dense_match_v1` (`e9`-family)
- `use_difix_restoration=true` (all checked live configs)

### Level 4 — topology regression
Verify both:
- `joint_primary`
- `side_branch`

Even if `joint_primary` is the current focus, do not silently break `side_branch` while refactoring shared code.

## 7. Interim design verdict

The safest way to “split out the online mapping module” under the current constraint is:
1. leave third-party entry files completely untouched
2. preserve current `pseudo_branch.integration` / `pseudo_branch.refine` import surfaces
3. extract cleaner repo-side online/core modules behind those surfaces
4. turn mixed current repo files into thin delegating wrappers
5. only then decide which standalone/history paths to keep

That approach matches the current live system and best satisfies the “do not break the system” requirement.


## 8. When execution can safely start

Short answer: almost yes, but not on the basis of “runs finished” alone.

After the current cloud runs finish, the project should be considered ready to start execution only if these gates are satisfied:
1. no active `slam.py` / `part3_BRPO` jobs are still importing the live repo
2. the first implementation step is additive-only (new modules + wrapper delegation), not path-moving
3. we have one tiny online smoke config prepared for immediate regression checking after each step
4. we do not batch naming cleanup and structural extraction into the same first patch

So the correct readiness statement is:
- once the current runs finish, we can start the refactor implementation phase,
- but only with the wrapper-first extraction plan already frozen,
- and with a smoke-validation ladder ready before the first code move.

### 8.1 First implementation patch that should be allowed after runs finish

The first execution patch should be limited to:
1. create new internal directories for `online_mapping/` and `core_shared/`
2. extract one or two low-risk helper modules into the new structure
3. keep old import surfaces in `pseudo_branch.integration` and `pseudo_branch.refine`
4. make one existing mixed file delegate to the new helper without changing its public API

This is safe enough to begin once the active runs are done.

### 8.2 What should still wait even after runs finish

Even after active runs end, these are still not first-step operations:
- any edit to `third_party/S3PO-GS/*`
- renaming many files in one batch
- deleting legacy standalone code before deciding what is retained as control/reference
- deleting `scripts/run_pseudo_refinement_v2.py` before recording an explicit retirement decision

## 9. Naming strategy for readability

Short answer to the user’s naming question:
- yes, the current naming is not good enough
- and yes, renaming should happen after the online structural extraction begins, not before

Reason:
- if we rename first and split later, we will mix two axes of change at once: semantic relocation and lexical rename
- that makes code review, blame, smoke debugging, and fallback much harder
- for a fragile live system, the lower-risk order is: extract -> stabilize -> rename surviving modules

### 9.1 Naming policy

Use role-based engineering names, not paper/history labels.

Good naming principles:
1. name by runtime role (`target_builder`, `record_builder`, `support_verify`, `mapping_loop`)
2. reserve `legacy_`, `archive_`, or `standalone_` prefixes for retained historical routes
3. avoid paper-specific labels in live module names when the file’s real role is broader than one paper variant
4. keep variant names at function/config level when possible instead of baking them into file names

### 9.2 Recommended rename direction for the online mainline

These are design targets, not immediate renames:

- `Results.brpo_online_mapping` -> `Results.online_pseudo_mapping`
  - reason: this is the live system role, not the paper provenance.

- `pseudo_branch/integration/runtime_exact_backend.py` -> future internal role equivalent:
  - `online_mapping/runtime/runtime_supervision_prepare.py`
  - or split into `exact_core.py`, `difix_rgb.py`, `support_variants.py`, `depth_variants.py`
  - reason: this file prepares runtime supervision bundles; “backend” is misleading here.

- `pseudo_branch/integration/runtime_signal_builder.py` -> future internal role equivalent:
  - `online_mapping/runtime/runtime_target_builder.py`
  - reason: it builds supervision targets/signals, not generic “signals”.

- `pseudo_branch/integration/runtime_pseudo_builder.py` -> future internal role equivalent:
  - `online_mapping/records/runtime_record_builder.py`
  - reason: it packs runtime pseudo records.

- `pseudo_branch/observation/brpo_reprojection_verify.py` -> future internal role equivalent:
  - `core_shared/verification/reprojection_support_verify.py`
  - reason: describes what it does without paper naming.

- `pseudo_branch/observation/pseudo_observation_brpo_style.py` -> split first, then rename surviving mainline subset to:
  - `core_shared/targets/exact_upstream_targets.py`
  - and move leftover historical builders to something like `legacy_target_variants.py`

- `pseudo_branch/refine/backend_pseudo_loss.py` -> future internal role equivalent:
  - `core_shared/losses/runtime_pseudo_supervision_loss.py`

- `pseudo_branch/refine/backend_pseudo_bundle.py` / `backend_pseudo_view_loader.py` -> future internal role equivalent:
  - `core_shared/records/runtime_pseudo_bundle.py`
  - `core_shared/records/runtime_pseudo_record_loader.py`

### 9.3 Rename order

Safe rename order:
1. first split live online modules behind stable wrappers
2. then rename newly extracted internal modules to clean engineering names
3. keep old module paths as wrappers/facades for one transition phase
4. only after smoke verification, update callers to the cleaner names
5. finally archive/remove obsolete wrappers

This keeps runtime risk low while still improving readability.

## 10. Policy for standalone naming and deletion

For standalone/history code, use a different rule from online mainline.

### 10.1 If a standalone module is already proven useless

If it is truly proven to be:
- not used by current online mainline
- not needed as control/reference
- not needed for provenance or replay
- and not the only remaining implementation of a shared algorithm kernel

then deletion is better than renaming.

Do not spend rename effort on dead code.

### 10.2 If a standalone module is still kept as control/reference

Then rename only after deciding it survives.
Use names like:
- `standalone_signal_build.py`
- `standalone_mask_build.py`
- `standalone_refine_legacy.py`
- `standalone_replay_eval.py`

This makes the route identity obvious and avoids pretending those files are part of the live online mainline.

### 10.3 Specific implication for `run_pseudo_refinement_v2.py`

Current recommendation:
- do not treat it as a live path
- do not rename it first
- later make an explicit decision:
  1. restore it as a real archived standalone control runner with a stable in-repo file, or
  2. retire it completely

Given its current broken external-symlink state, it is a bad candidate for early rename work.

## 11. Updated design verdict

Yes: after the current runs finish, we should be able to start executing the repo-side refactor plan.
But the first execution step must be a wrapper-first extraction patch entirely inside `part3_BRPO`, not a broad rename or any third-party edit.

And yes: naming cleanup is important, but for the current live online mainline it should happen after the first repo-side structural extraction boundary is in place, not before.
That order gives the best chance of improving readability without destabilizing the running system.
