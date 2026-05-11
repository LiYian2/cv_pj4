# 2IMG + PAIR-proxy Adaptive Scale Depth Generation Analysis and Integration Plan

## Executive Summary

Validated a new depth generation approach for pseudo view supervision:
- **Coverage**: 100% (vs PAIR's 3-20%)
- **MRE**: median 18.3%, std 5.7% (stable across 9 frames)
- **Use case**: Depth loss supervision only (does NOT affect C_m / support_mask)

**Recommendation**: Proceed with integration for depth supervision, keep PAIR route for C_m generation.

---

## 1. Problem Analysis

### 1.1 Current PAIR Route Failure

Current online mapping depth generation:
```
MASt3R(pseudo RGB, ref KF RGB) -> pair matching -> projected depth
```

**Failure mode**: Cross-view matching has extremely low success rate.

Quantitative evidence from D4 experiments:
- frame_0016: coverage = 3.33%
- Multi-frame median: 9.79%
- Confidence > 0.3: only 0.73% of pixels

Root cause:
- Pseudo RGB is a rendered intermediate view between two KFs
- Ref KF RGB is from a different viewpoint (30+ frames apart)
- MASt3R pair matching assumes similar viewpoints
- Cross-view matching fails when viewpoint gap is large

### 1.2 Impact on Optimization

Depth supervision is critical for pseudo view optimization:
- Depth loss guides Gaussian depth positioning
- Sparse coverage means only 3-20% of pixels receive depth guidance
- Remaining 80-97% pixels rely only on RGB loss (no geometry constraint)

This explains why "pseudo view enhancement" has no visible effect:
- Added pseudo views but depth supervision is ineffective
- RGB loss alone cannot establish correct geometry
- Gaussians at uncovered pixels have no depth constraint

---

## 2. Proposed Solution: 2IMG + PAIR-proxy Adaptive Scale

### 2.1 Core Idea

```
Step 1: MASt3R(pseudo, pseudo) -> depth_2img (100% coverage, unknown scale)
Step 2: MASt3R(pseudo, ref KF) -> depth_pair (scale anchor, 3-20% coverage)
Step 3: PAIR-proxy adaptive scale calibration -> calibrated depth
```

Key insight:
- Same-image matching (img+img) always succeeds -> 100% coverage
- PAIR depth provides accurate scale in matching regions -> scale anchor
- Use PAIR depth range to determine depth-specific scale -> adaptive calibration

### 2.2 Validation Results

Multi-frame statistics (9 frames from D4 brpo_debug):

| Method | Coverage | Median MRE | Mean MRE | Std |
|--------|----------|-----------|----------|-----|
| Global scale | 100% | 31.8% | 32.0% | 10.5% |
| PAIR anchor | 100% | 28.6% | 29.9% | 5.9% |
| **PAIR-proxy adaptive** | **100%** | **18.3%** | **20.2%** | **5.7%** |

Per-depth-range analysis (frame_0016):

| Depth Range | Pixels | MRE (adaptive) |
|-------------|--------|----------------|
| 0-2m | 29.7% | 7.4% |
| 2-5m | 32.7% | 11.7% |
| 5-10m | 9.3% | 54.9% |
| 10-20m | 6.8% | 71.9% |
| 20-100m | 1.5% | 5.7% |

Note: High MRE in 5-20m range but fewer pixels there. Overall MRE dominated by near-range performance.

### 2.3 Stability Assessment

Criteria:
- Median MRE < 25%: PASS (18.3%)
- Std < 10%: PASS (5.7%)

**Conclusion**: Method is stable and suitable for integration.

---

## 3. Scope of Use

### 3.1 Depth Usage in Pipeline

Depth in Part3 BRPO has two independent paths:

1. **Depth loss supervision** (pseudo view optimization)
   - Used in pseudo_loss_v2.py: masked_depth_loss(render_depth, target_depth, mask)
   - Guides Gaussian depth positioning
   - **This is the target for 2IMG+PAIR-proxy approach**

2. **C_m / support_mask generation** (signal for pseudo selection)
   - RGB-only in current design (see runtime_exact_backend.py line 464)
   - support_mask comes from RGB matching confidence
   - **NOT affected by depth generation method**

### 3.2 Design Decision

Keep dual-path architecture:
- **C_m generation**: Continue using PAIR RGB matching route
  - High precision (3-4% MRE) where matching succeeds
  - RGB-only, no depth dependency
- **Depth supervision**: Use 2IMG+PAIR-proxy route
  - 100% coverage
  - ~18% MRE (acceptable for dense supervision)

This decouples:
- Signal quality (C_m) from supervision coverage (depth)
- High precision where available vs broad coverage everywhere

---

## 4. Integration Plan

### 4.1 Code Structure

**New module**: pseudo_branch/common/2img_pair_proxy_depth.py

Functions:
- build_2img_pair_proxy_depth(): Main entry point
- apply_pair_proxy_adaptive_scale(): Core calibration logic

### 4.2 Integration Points

**Entry point**: runtime_exact_backend.py

Add mode flag:
```
depth_generation_mode: "pair" or "2img_pair_proxy"
```

### 4.3 Configuration

```yaml
pseudo_depth:
  generation_mode: "2img_pair_proxy"
  2img_pair_proxy:
    ranges: [[0, 2], [2, 5], [5, 10], [10, 20], [20, 100]]
    conf_threshold: 0.1
    fallback_scale: "global"
```

### 4.4 Output Structure

```
frame_root/
  2img_backend_v1/
    depth_2img_raw.npy
    depth_pair_raw.npy
    confidence_pair.npy
    depth_calibrated.npy
    scale_by_range.json
```

---

## 5. Trade-offs

| Aspect | Current PAIR | New 2IMG+PAIR-proxy |
|--------|-------------|---------------------|
| Coverage | 3-20% | 100% |
| MRE (where valid) | 3-4% | 18% |
| Depth loss coverage | Sparse | Dense |
| Computational cost | 1 forward | 2 forwards |

---

## 6. Validation Plan

1. Unit test: scripts/test_2img_pair_proxy_unit.py
2. Integration test: D-series rerun with new mode
3. Fallback test: PAIR coverage < 5% cases

---

## 7. Timeline

1. Implement core module (1 day)
2. Integrate into runtime_exact_backend (1 day)
3. Run validation experiment (1 day)
4. Document and finalize (0.5 day)

---

## 8. Open Questions

1. Per-frame vs scene-level scale?
2. Blended depth (PAIR where available + 2img elsewhere)?
3. Depth variance for confidence weighting?

---

## Appendix: Experimental Evidence

### A. Coverage Analysis (frame_0016)

projected_depth coverage: 3.33%
confidence > 0.3: 0.73%

### B. Multi-frame Stability

PAIR-proxy adaptive across 9 frames:
- MRE median: 18.3%
- MRE std: 5.7%
- Best: 12.8%, Worst: 29.1%


重要提醒:给 Claude 的提醒我觉得有必要，规划文档还不够细，尤其要避免几个坑：

1. 不要做成全图 depth loss。2img+PAIR raw depth 可以 100%，但 safe mode 应该只填充/启用 C_m 内的 depth，RGB C_m 不变，depth effective mask 也不要越过 C_m，除非另开 diagnostic arm。

2. 生产校准不能用 pseudo_render_depth_runtime.npy 当 scale target。它只能作为诊断/评估。正式 scale anchor 应该来自 exact_backend 的 projected_depth_left/right_exact + valid/support/confidence，即 PAIR projected depth anchor。

3. 不要复用现有 mast3r_direct_exact_anchor_v1 当成这个方案。那个是 MASt3R(pseudo, ref) direct depth + global anchor，不是 MASt3R(pseudo, pseudo) + PAIR-proxy adaptive scale。

4. 新模块名不要叫 2img_pair_proxy_depth.py，Python 模块数字开头不合适。建议 twoimg_pair_proxy_depth.py。

5. 默认必须保持 projected，不影响现有实验。新增开关例如 depth_generation_mode: twoimg_pair_proxy_cm_capped_v1；新 config、新 save_dir、新 debug root，不能覆盖 E5c 当前目录。

6. metadata 必须写清楚三组比例：cm_nonzero_ratio、projected_depth_union_ratio、twoimg_depth_effective_ratio_after_cm_cap。否则后面很难判断到底是 depth 没进来，还是进来了但没效果。

7. 先做 sidecar materialize 脚本，在已有 brpo_debug frame 上只生成新 depth 和 summary，不改 signal_v2；确认比例和尺度后再接 runtime。

一句话给 Claude：目标不是 “100% dense depth supervision”，而是 “在 C_m 安全区域内，用 2img+PAIR calibrated depth 补齐 PAIR projected depth 缺口”。当前数据看这个缺口大约是一半 C_m，所以这个实验值得做。