# MIDPOINT8_M5_PIPELINE_REPAIR_PLAN_20260413

## 0. Purpose

This document converts the findings from `MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md` into explicit repair work items.

Scope:
1. fix the structural disconnects in the current Part3 pipeline;
2. repair the evaluation logic so StageA is not judged by a metric it cannot change;
3. strengthen later-stage geometry anchoring and paper-faithful mechanisms;
4. preserve a phase-by-phase execution style with rollback points.

## 1. Root causes to address

Current highest-priority issues are:
1. StageA replay gate is structurally invalid because StageA does not modify Gaussian PLY.
2. Stage handoff is incomplete: `StageA -> A.5 -> B` does not pass refined pseudo camera states forward.
3. Current midpoint8 M5 pseudo set has weak non-fallback geometric support inside train-mask.
4. A.5 / StageB can reduce pseudo-side loss while worsening downstream replay.
5. Current StageB real branch is too appearance-heavy and too weak geometrically.
6. Current refine loop is still materially narrower than the BRPO paper design.

## 2. Repair priorities

### P0. Fix stage handoff

Goal: make `StageA -> A.5 -> B` a true sequential pipeline in camera-state space.

Required changes:
1. Add a CLI argument to `run_pseudo_refinement_v2.py`, e.g.:
   - `--init_pseudo_camera_states_json`
2. Implement loading logic that, after `load_stageA_pseudo_views(...)`, overwrites each pseudo viewpoint's:
   - `R`
   - `T`
   - optionally `R0/T0` policy (must be chosen explicitly)
   - `exposure_a`
   - `exposure_b`
3. Match by stable key:
   - primary: `sample_id`
   - sanity check: `frame_id`
4. Add explicit reporting in saved history:
   - whether prior states were loaded
   - source json path
   - loaded sample count / mismatched sample count

Acceptance criteria:
- A.5 can start from StageA pseudo camera states.
- StageB can start from A.5 pseudo camera states.
- histories clearly record the handoff source.

### P0. Fix StageA evaluation logic

Goal: stop treating 270 replay as the primary gate for a stage that cannot change the PLY.

Required changes:
1. Keep StageA replay only as a “sanity no-change check”, not as the main success metric.
2. Add StageA-focused metrics:
   - true pose delta from `pseudo_camera_states_init.json -> final.json`
   - pseudo-side loss change
   - per-view pseudo reprojection / render change summary if practical
3. Update docs so future runs do not misread “StageA replay unchanged” as “StageA failed to optimize”.
4. Fix misleading summary naming:
   - current `final_delta_summary` actually records residual tensors after fold-back
   - rename or augment with true pose-change summary

Acceptance criteria:
- future StageA reports contain true camera-state deltas;
- docs explicitly say StageA replay is not the main gate.

### P1. Re-check midpoint pseudo signal quality before more large scans

Goal: decide whether midpoint8 M5 itself is too weak to support meaningful pose improvement.

Required checks:
1. For each midpoint sample, summarize:
   - train-mask ratio
   - seed ratio
   - dense ratio
   - non-fallback-in-trainmask ratio
2. Identify weakest frames, especially tails like sample 260.
3. Compare midpoint8 against alternative pseudo selection schemes:
   - two 2/3-position frames per gap
   - three 3/4-position frames per gap
   - or mixed midpoint + tertile coverage
4. Decide whether upstream pseudo placement is the limiting factor.

Acceptance criteria:
- one short comparison memo that answers whether midpoint8 is underpowered geometrically.

### P1. Strengthen StageB geometry anchoring

Goal: make StageB capable of correcting pseudo-side local overfitting instead of reinforcing it.

Possible directions:
1. Increase real-view coverage per iteration.
2. Introduce stronger real-view geometric constraints if available.
3. Rebalance `lambda_real / lambda_pseudo` only after structural handoff is fixed.
4. Consider delaying or limiting appearance-heavy updates until geometry stabilizes.

Acceptance criteria:
- StageB no longer regresses immediately on short replay gate after handoff repair.

### P2. Add paper-faithful missing mechanisms

Goal: close the gap between current implementation and BRPO method design where it matters.

Candidate missing mechanisms:
1. scene perception Gaussian management during refine;
2. broader joint Gaussian parameter space beyond `xyz/opacity` only;
3. paper-style appearance refinement stage;
4. better alignment with the paper’s joint RGB-D optimization semantics.

Acceptance criteria:
- each mechanism gets a “worth implementing / not worth implementing yet” decision with rationale.

## 3. Recommended execution order

### Phase R1. Handoff + evaluation repair
1. implement pseudo camera state loading;
2. add true pose-change summaries;
3. update StageA reporting logic.

### Phase R2. Re-run minimal sequential pipeline
1. StageA with true state export;
2. A.5 initialized from StageA states;
3. StageB initialized from A.5 states;
4. compare against current non-handoff baseline.

### Phase R3. Upstream pseudo-placement diagnosis
1. quantify midpoint8 support weakness;
2. decide whether to switch pseudo placement.

### Phase R4. StageB geometry-anchor strengthening
1. adjust real-branch design after handoff is fixed;
2. repeat conservative short-gate replay.

### Phase R5. Paper-faithful mechanism additions
1. selectively implement missing BRPO mechanisms with highest expected value.

## 4. What not to do next

Avoid these before P0 is repaired:
1. large hyperparameter scans on the current broken handoff design;
2. treating StageA replay as evidence for or against pose optimization quality;
3. long StageB runs before short-gate non-regression is restored.

## 5. Deliverables for the next iteration

Minimum next-step deliverables:
1. code patch for pseudo camera state handoff;
2. code patch/report fix for true pose-change summary;
3. one short rerun showing whether sequential handoff changes A.5 / StageB behavior.

## 6. Canonical references

- main execution/analysis report:
  `docs/MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md`
- runbook:
  `docs/M5_STAGEA-A.5-B_RUNBOOK.md`
- experiment root:
  `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_midpoint8_m5_fullpipeline`

This repair plan should be updated as soon as any P0 item is completed or re-scoped.
