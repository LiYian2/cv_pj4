# E5/E6 Series Experiment Summary

## Overview

- **Dataset**: DL3DV-2 part2_s3po
- **Iterations**: 26000 color refinement
- **Keyframes**: 10 (indices: 0, 33, 67, 101, 135, 169, 203, 237, 271, 305)

## Key Configuration Differences

| Experiment | Difix | GT-pseudo | update_real_pose | color_refinement_use_pseudo |
|------------|-------|-----------|------------------|----------------------------|
| E5a | yes | no | false | true |
| E5b | yes | no | true | true |
| E5c | yes | no | false | false |
| E6a | no | yes | false | true |
| E6b | no | yes | true | true |

## Results Summary

| Experiment | Method | before_opt PSNR | before_opt SSIM | before_opt LPIPS | after_opt PSNR | after_opt SSIM | after_opt LPIPS |
|------------|--------|-----------------|-----------------|------------------|----------------|----------------|-----------------|
| C1 (Baseline) | Baseline (S3PO, no BRPO) | 17.46 | 0.621 | 0.377 | 19.44 | 0.680 | 0.294 |
| E6a (Upper Bound) | GT-pseudo, no Difix, best run | 19.11 | 0.650 | 0.408 | **24.62** | **0.728** | **0.243** |
| E6b | GT-pseudo, no Difix, update_real_pose=true | 18.75 | 0.630 | 0.415 | 23.38 | 0.757 | 0.212 |
| E5a | Difix, pseudo in refinement | 18.37 | 0.598 | 0.439 | 19.21 | 0.618 | 0.329 |
| E5b | Difix, pseudo, update_real_pose=true | 17.95 | 0.580 | 0.446 | 18.70 | 0.596 | 0.331 |
| E5c | Difix, **no pseudo in refinement** | 18.60 | 0.624 | 0.417 | **21.20** | **0.698** | **0.243** |

## Key Findings

### 1. Pseudo in color refinement hurts E5 series
- **E5a** (with pseudo): after_opt = 19.21
- **E5c** (no pseudo): after_opt = **21.20** (+1.99 PSNR improvement)
- Difix-generated pseudo views have poor MASt3R matching quality (confidence mask coverage: 26.8% vs 49.5%)

### 2. GT-pseudo significantly better than Difix pseudo
- **E6a** (GT-pseudo, no Difix): after_opt = **24.62**
- **E5a** (Difix pseudo): after_opt = 19.21
- Gap: 5.4 PSNR

### 3. update_real_pose effect unclear
- E5a vs E5b: update_real_pose=false better (+0.51 PSNR)
- E6a vs E6b: update_real_pose=false potentially better (best E6a run = 24.62 vs E6b = 23.38)
- High variance across runs suggests other factors dominate

### 4. Color refinement only uses keyframes
- Original S3PO color refinement samples from viewpoints dict (keyframes only)
- Not all 306 frames, just 10 KFs

## Experiment Dates

- E5a: 2026-05-08
- E5b: 2026-05-08  
- E5c: 2026-05-08
- E6a (best): 2026-05-08-15-07-39
- E6b: 2026-05-08-18-43-31
- C1: baseline from earlier runs
