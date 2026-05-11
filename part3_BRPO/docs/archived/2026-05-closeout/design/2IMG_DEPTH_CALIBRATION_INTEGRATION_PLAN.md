# 2IMG + PAIR-proxy Adaptive Scale Calibration Integration Plan

## Background

Current online mapping depth generation:
- PAIR route: MASt3R(pseudo RGB, ref KF RGB) -> matching -> projected depth
- Problem: Coverage only 3-20% due to cross-view matching failure

New approach validated:
- 2img + PAIR-proxy adaptive: MASt3R(pseudo, pseudo) + PAIR depth as scale proxy
- Coverage: 100%
- MRE: median 18.3%, std 5.7% (stable across 9 frames)

---

## Integration Design

### Option A: Replace PAIR route with 2img route (full replacement)

Pros: 100% coverage, simpler pipeline
Cons: Higher MRE (18%) vs PAIR's matching regions (3-4%), loss of high-quality matching info

Recommendation: Not recommended - loses PAIR's accuracy advantage

### Option B: Hybrid: 2img depth + PAIR anchor calibration (recommended)

Core idea:
1. Run MASt3R(pseudo, pseudo) -> 100% coverage depth (raw)
2. Run MASt3R(pseudo, ref KF) -> provides scale anchor per depth range
3. Apply PAIR-proxy adaptive scale calibration
4. Output: calibrated depth with 100% coverage + ~18% MRE

---

## Integration Points

### 1. runtime_exact_backend.py modification

Current flow:
```
build_runtime_exact_backend_bundle()
  -> matcher.match_pair(pseudo, ref)  -> pts_pseudo, pts_ref
  -> verify_single_branch_exact()     -> projected_depth_map (low coverage)
```

New flow (Option B hybrid):
```
build_runtime_exact_backend_bundle()
  -> matcher.match_pair(pseudo, pseudo)  -> depth_2img (100% coverage)
  -> matcher.match_pair(pseudo, ref)     -> depth_pair (scale anchor)
  -> apply_pair_proxy_adaptive_scale()   -> calibrated_depth
  -> (optional) verify_single_branch_exact() -> use calibrated_depth as depth_target
```

### 2. Configuration addition

```yaml
depth_generation:
  mode: "2img_pair_proxy"
  ranges: [[0, 2], [2, 5], [5, 10], [10, 20], [20, 100]]
  conf_threshold: 0.1
  fallback_scale: "global"
```

### 3. Debug output structure

```
frame_root/
  runtime_inputs/
    pseudo_render_rgb_runtime.png
    pseudo_render_depth_runtime.npy
  2img_backend/
    depth_2img_raw.npy
    depth_pair_raw.npy
    confidence_pair.npy
    depth_calibrated.npy
    scale_by_range.json
```

---

## Trade-offs

| Aspect | Current PAIR | New 2img+PAIR-proxy |
|--------|-------------|---------------------|
| Coverage | 3-20% | 100% (better) |
| MRE (matching region) | 3-4% (better) | N/A |
| MRE (overall) | N/A (sparse) | 18% |
| Depth consistency | High (geometric) | Medium (learned prior) |
| Computational cost | 1 match | 2 matches |

Recommendation: Use hybrid approach for online mapping where coverage matters more than per-pixel precision. For pose optimization, continue using PAIR matching regions (high precision).

---

## Next Steps

1. Implement apply_pair_proxy_adaptive_scale() in pseudo_branch/common/mast3r_pair_forward.py
2. Modify build_runtime_exact_backend_bundle() to support new mode
3. Add config flag depth_generation_mode: "2img_pair_proxy"
4. Validate with D-series experiments
5. Document in STATUS.md and CHANGELOG.md

---

## Open Questions

1. Should we use both depth sources (PAIR projected + 2img calibrated) in a blended fashion?
2. How to handle cases where PAIR depth itself is unreliable (very low coverage)?
3. Should scale calibration be per-frame or scene-level (across all frames)?