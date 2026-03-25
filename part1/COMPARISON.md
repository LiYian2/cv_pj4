# Plan A vs. Plan B Comparison on Re10k-1

## Overview

We now compare three relevant settings for Part 1:

- **Plan A-279:** COLMAP initialization + 3DGS using all `279` views
- **Plan A-96:** COLMAP initialization + 3DGS using the exact same `96` image subset used by Plan B
- **Plan B-96:** VGGT initialization + 3DGS using the `96`-view subset

This updated comparison is much fairer than the original `Plan A-279 vs Plan B-96` comparison, because it separates the effect of **view count** from the effect of **initialization quality**.

---

## Final Results

| Setting | Views | Init | Final Test PSNR @40k | Final Test L1 @40k |
|---|---:|---|---:|---:|
| Plan A-279 | 279 | COLMAP | 35.1689 | 0.0106019 |
| Plan A-96 | 96 | COLMAP | 33.4120 | 0.0130271 |
| Plan B-96 | 96 | VGGT | 32.1220 | 0.0140688 |

---

## Key Comparisons

### 1. Effect of view count inside Plan A

Comparing Plan A-279 and Plan A-96:

- PSNR drop: `35.1689 - 33.4120 = 1.7569 dB`

This shows that using the full 279-view sequence gives a substantial advantage over the 96-view subset, even with the same COLMAP-based initialization pipeline.

### 2. Fair initialization comparison under the same 96-view budget

Comparing Plan A-96 and Plan B-96:

- PSNR gap: `33.4120 - 32.1220 = 1.2900 dB`

This is the most important result for Part 1. Once the input view count is matched, **Plan A still clearly outperforms Plan B**. Therefore, the lower Plan B performance cannot be attributed only to the fact that it used fewer images.

### 3. Original end-to-end practical gap

Comparing Plan A-279 and Plan B-96:

- PSNR gap: `35.1689 - 32.1220 = 3.0469 dB`

This is still useful as a practical comparison, but it should be interpreted together with the fact that the two systems had different input coverage.

---

## Convergence Trend

The convergence behavior is also informative:

- **Plan A-279** converges to the best final quality and is consistently strongest across the whole training curve.
- **Plan A-96** converges more slowly and to a lower ceiling than Plan A-279, showing the effect of reduced view coverage.
- **Plan B-96** improves steadily through 40k iterations, but remains below Plan A-96 at every major checkpoint.

This indicates that the current VGGT-based initialization does not provide either a faster convergence trajectory or a better final rendering quality than the matched COLMAP baseline.

---

## Main Interpretation

The updated experiments support the following conclusion:

1. **COLMAP initialization is the strongest solution overall on Re10k-1 under the current setup.**
2. **View count matters a lot:** reducing the scene from 279 to 96 views hurts Plan A noticeably.
3. **Initialization quality also matters independently:** even under the exact same 96-view subset, COLMAP initialization still outperforms VGGT initialization.
4. **Plan B is feasible but currently weaker**, likely due to a combination of:
   - lower-quality initialization,
   - reduced view coverage,
   - disabled fine tracking,
   - and memory-driven implementation constraints.

---

## Current Takeaway

For Part 1, the fairest and strongest conclusion is now:

- **Plan A-279 is the best-performing setting overall**
- **Plan A-96 vs Plan B-96 is the key controlled comparison**
- **Under the matched 96-view setting, COLMAP initialization still wins**

This gives a much more defensible answer to the project requirement of analyzing how different initializations affect 3DGS convergence and final reconstruction quality.

