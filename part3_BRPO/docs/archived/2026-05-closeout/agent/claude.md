# Claude Agent Context for BRPO C_m Expansion

> This document helps future Claude sessions quickly understand the project context and current work status.

---

## 1. Project Background

### 1.1 Core Problem

BRPO paper uses **reciprocal matching** (bidirectional nearest neighbor) as the confidence seed for pseudo view supervision. The issue:

- MASt3R dense candidate covers full image (candidate_ratio ≈ 1.0)
- But consumption is: `dense → reciprocal NN → point-like seed`
- Result: C_m (confidence mask) is too sparse

| Metric | E5c Measurement | Issue |
|--------|-----------------|-------|
| C_m union | ~26.2% | — |
| C_m both | ~4.0% | **Too low** |
| C_m single | ~22.2% | — |

This leads to insufficient effective supervision region for pseudo RGB/depth.

### 1.2 Solution: C_m Controlled Local Expansion

**Key principle**: Keep reciprocal seed as trusted anchor (paper-aligned), expand slightly in local neighborhood.

Expansion rules:
1. **Radius**: r=1 (3×3 neighborhood)
2. **Gates**: RGB continuity (tau=0.08) + depth continuity (tau=0.05)
3. **Weights**:
   - raw_both = 1.0 (unchanged)
   - raw_single = 0.5 (unchanged)
   - expanded_both = 0.6
   - expanded_single = 0.25

**Target**: weight gain 1.2x–1.8x, NOT full-image dense.

---

## 2. Technical Details

### 2.1 Expansion Algorithm

```
For each seed pixel p:
  For each neighbor q in radius-r:
    Check gates:
      1. seed_conf[p] >= min_seed_conf
      2. RGB: mean_abs(rgb[q] - rgb[p]) <= tau_rgb_l1
      3. Depth: abs(depth[q] - depth[p]) / max(depth) <= tau_depth_rel
      4. depth[p] > 1e-6 and depth[q] > 1e-6
    
    Compute confidence:
      spatial = exp(-dist(p,q) / radius)
      rgb_score = max(0, 1 - rgb_l1 / tau_rgb_l1)
      depth_score = max(0, 1 - rel_depth / tau_depth_rel)
      expanded_conf[q] = seed_conf[p] * spatial * rgb_score * depth_score * expansion_weight
```

### 2.2 Soft C_m Composition

```
Raw BRPO weights:
  raw_both → 1.0
  raw_single → 0.5

Expanded weights:
  expanded_both → 0.6      (both expanded, not raw both)
  raw_plus_exp_single → 0.5  (raw + expanded from other side)
  expanded_single → 0.25    (only one side expanded)
```

---

## 3. Code Changes (5 files)

### 3.1 Core Module

**File**: `pseudo_branch/mask/cm_local_expansion.py`

Functions:
- `expand_branch_support_local()` — single branch expansion with RGB/depth gates
- `compose_soft_cm_from_expanded_branches()` — soft weight composition
- `apply_cm_local_expansion()` — end-to-end wrapper
- `write_cm_expansion_outputs()` — save to disk with provenance tracking

Provenance labels:
```python
PROVENANCE_RAW_BOTH = 0
PROVENANCE_RAW_SINGLE = 1
PROVENANCE_EXPANDED_BOTH = 2
PROVENANCE_RAW_PLUS_EXP_SINGLE = 3
PROVENANCE_EXPANDED_SINGLE = 4
PROVENANCE_NONE = 5
```

### 3.2 Sidecar Diagnostic Script

**File**: `scripts/diagnostics/materialize_cm_local_expansion.py`

Usage:
```bash
python scripts/diagnostics/materialize_cm_local_expansion.py \
    --debug-root /path/to/brpo_debug \
    --radius 1 \
    --tau-rgb-l1 0.08 \
    --tau-depth-rel 0.05 \
    --out-name cm_local_expand_r1_v1
```

Outputs:
- Per-frame: `frame_xxxx/cm_local_expand_r1_v1/`
- Global: `brpo_debug/cm_local_expand_r1_v1_summary.json`

### 3.3 RuntimeExactBackendConfig

**File**: `pseudo_branch/integration/runtime_exact_backend.py`

Added 11 fields:
```python
cm_expansion_mode: str = "none"  # none | local_soft_v1
cm_expansion_radius: int = 1
cm_expansion_weight: float = 0.5
cm_expansion_tau_rgb_l1: float = 0.08
cm_expansion_tau_depth_rel: float = 0.05
cm_expansion_min_seed_conf: float = 0.0
cm_expansion_min_expanded_conf: float = 0.05
cm_expanded_both_weight: float = 0.6
cm_raw_exp_agree_weight: float = 0.5
cm_expanded_single_weight: float = 0.25
cm_expansion_apply_to_depth_scope: bool = False
```

Expansion code inserted after line 621 (after `fusion_weight_right`).

### 3.4 Runtime Signal Builder

**File**: `pseudo_branch/integration/runtime_signal_builder.py`

Added:
```python
confidence_cm_override = None
if "confidence_cm_override" in exact_bundle.left_result:
    confidence_cm_override = np.asarray(exact_bundle.left_result["confidence_cm_override"], dtype=np.float32)

result = build_exact_brpo_upstream_target_observation(
    ...
    confidence_cm_override=confidence_cm_override,
    ...
)
```

### 3.5 Pseudo Observation Builder

**File**: `pseudo_branch/observation/pseudo_observation_brpo_style.py`

Added parameter `confidence_cm_override: np.ndarray | None = None`

Logic:
```python
if confidence_cm_override is not None:
    confidence_cm = np.asarray(confidence_cm_override, dtype=np.float32)
    cm_source = "cm_expansion_override"
else:
    # Original discrete both/xor logic
    confidence_cm[verify_both] = 1.0
    confidence_cm[verify_xor] = 0.5
    cm_source = "exact_backend_support"
```

### 3.6 Slam Backend Config Parsing

**File**: `utils/slam_backend.py` (S3PO-GS repo)

Added parsing in `_resolve_brpo_online_mapping_cfg()` and `RuntimeExactBackendConfig()` instantiation.

---

## 4. Experiment Status

### 4.1 Completed Experiments

| Exp | Dataset | Config | Status |
|-----|---------|--------|--------|
| E5c | DL3DV-2 | No expansion | Completed, baseline |
| R5c | Re10k-1 | No expansion | Completed |
| W5c | Waymo-405841 | No expansion | Completed |

### 4.2 Running Experiment

| Exp | Dataset | Config | Status |
|-----|---------|--------|--------|
| E8 | DL3DV-2 | cm_expansion enabled | **Running on GPU 1** |

E8 config location:
```
/data3/bzhang512/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft/config.yml
```

Key settings:
```yaml
cm_expansion_mode: local_soft_v1
cm_expansion_radius: 1
cm_expansion_weight: 0.5
cm_expansion_tau_rgb_l1: 0.08
cm_expansion_tau_depth_rel: 0.05
cm_expanded_both_weight: 0.6
cm_expanded_single_weight: 0.25
```

Monitor:
```bash
tail -f /data3/bzhang512/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft/run_log.txt
```

### 4.3 First Frame Validation (E8, frame_0015)

| Metric | Value |
|--------|-------|
| Weight gain | **1.16x** ✓ |
| Raw C_m union | 40.69% |
| Expanded C_m nonzero | 52.53% |
| Expanded both | 11.95% |
| Expanded single | 8.7% |
| RGB gate reject | 20695 |
| Depth gate reject | 22116 |

---

## 5. Key File Paths

### 5.1 Design Document

```
/home/bzhang512/CV_Project/part3_BRPO/docs/CM_CONTROLLED_LOCAL_EXPANSION_PLAN_20260508.md
```

### 5.2 Core Module

```
/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/mask/cm_local_expansion.py
```

### 5.3 Sidecar Script

```
/home/bzhang512/CV_Project/part3_BRPO/scripts/diagnostics/materialize_cm_local_expansion.py
```

### 5.4 Experiment Outputs

```
/data3/bzhang512/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft/
├── config.yml
├── run_log.txt
└── brpo_debug/
    └── event_kf_0030/
        └── frame_0015/
            └── exact_backend_v1/
                ├── cm_expansion_v1/
                │   ├── cm_expanded_soft.npy
                │   ├── expansion_provenance.npy
                │   └── summary.json
                └── exact_backend_meta.json
```

---

## 6. Validation Checklist (from doc §9)

When reviewing expansion results, verify:

1. **Raw C_m union/both/single** — baseline statistics
2. **Expanded C_m nonzero ratio** — coverage gain
3. **Effective mask weight increase** — should be 1.2x–1.8x
4. **Expanded-only pixel ratio** — should not dominate
5. **Projected depth filled ratio unchanged** — depth scope not affected
6. **Final loss reads expanded soft C_m** — check `confidence_cm_override` in meta
7. **Expanded pixels use lower weights** — check provenance distribution

---

## 7. Next Steps

1. **Monitor E8 completion** — wait for all keyframes processed
2. **Run sidecar on E8 brpo_debug** — compare runtime vs sidecar consistency
3. **Analyze results** — compare E8 vs E5c metrics
4. **Consider E9 (r=2)** — if r=1 results are promising but conservative
5. **Depth scope expansion** — if RGB expansion works, consider `cm_expansion_apply_to_depth_scope=true`

---

## 8. Quick Start for Next Session

To check current status:
```bash
# Check E8 log
ssh Group8DDY "tail -50 /data3/bzhang512/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft/run_log.txt"

# Check GPU status
ssh Group8DDY "nvidia-smi"

# Check brpo_debug output
ssh Group8DDY "ls -la /data3/bzhang512/part3_online_mapping_experiments/E8_cm_local_expand_r1_soft/brpo_debug/"
```

Key questions to answer:
- Is E8 still running?
- Has cm_expansion been triggered for all keyframes?
- Are weight gains consistent across frames?
- Any errors in logs?

---

## 9. Related Memory Files

For additional context, check memory files:
- `project_part3_brpo.md` — project full context
- `part3_brpo_analysis_20260505.md` — pose gradient analysis
- `feedback_style.md` — communication preferences

---

*Last updated: 2026-05-09*
*Author: Claude (GLM-5)*

---

## 10. TODO for Next Session

### 10.1 Multi-Pseudo Pipeline Check

**Context**: E5c_3 experiment (quartile, max_pseudo_per_gap=3) showed degradation. Suspected cause: pseudo loss占比过大。

**Question to investigate**:

When `max_pseudo_per_gap > 1`, which pipeline is used?

**Option A**: 
```
kf0, kf1 → generate pseudo 0.25, 0.5, 0.75 simultaneously → refine all at once
```

**Option B**:
```
kf0, kf1 → generate pseudo 0.25 → refine → generate pseudo 0.5 → refine → generate pseudo 0.75 → refine
```

**Files to check**:
- `pseudo_branch/integration/runtime_slot_selector.py` — slot generation logic
- `utils/slam_backend.py` — `_run_brpo_online_mapping_event()` — pseudo processing loop
- Look for `max_pseudo_per_gap`, `placement_mode`, `num_pseudo_views_per_step` usage

**Why important**: 
- Option A → pseudo loss占比取决于同时refine的pseudo数量
- Option B → 逐步refine可能导致不同行为

**Experiment to review**:
```
/home/bzhang512/my_storage2_1T/part3_online_mapping_experiments/E5c_3_jointprimary_maskedcolor_rgbonly_cm_difix_quartile
```

**Priority**: High — understanding this is crucial before running E9/E10 with more pseudo frames.

---

*TODO added: 2026-05-09*
