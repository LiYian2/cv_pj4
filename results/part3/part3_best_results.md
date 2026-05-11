# Part3 BRPO Best Results by Dataset

**Date:** 2026-05-12
**Selection Criteria:** Highest PSNR (after_opt) per dataset

---

## Summary Table

| Dataset | Best Experiment | Run | PSNR (dB) | SSIM | LPIPS | ATE RMSE (m) |
|---------|-----------------|-----|-----------|------|-------|--------------|
| Re10k-1 | r5c_depthoff | run_08 | 24.7544 | 0.8857 | 0.0743 | 0.0149 |
| Waymo-405841 | w5c_depthoff | run_05 | 24.8661 | 0.7854 | 0.2371 | 1.783 |
| DL3DV-2 | e7a_binarycap | run_06 | 21.9339 | 0.7205 | 0.2280 | 0.0693 |

---

## Detailed Results

### Re10k-1

**Best Experiment:** `r5c_depthoff`
**Best Run:** `run_08`

**Rendering Metrics (after_opt):**
```json
{"mean_psnr": 24.754436372827602, "mean_ssim": 0.8856818746637415, "mean_lpips": 0.07427067601432404}
```



---

### Waymo-405841

**Best Experiment:** `w5c_depthoff`
**Best Run:** `run_05`

**Rendering Metrics (after_opt):**
```json
{"mean_psnr": 24.866053320176107, "mean_ssim": 0.7854498501596504, "mean_lpips": 0.2370953668096212}
```



---

### DL3DV-2

**Best Experiment:** `e7a_binarycap`
**Best Run:** `run_06`

**Rendering Metrics (after_opt):**
```json
{"mean_psnr": 21.933881076606543, "mean_ssim": 0.7205473532950556, "mean_lpips": 0.22800369169311346}
```



---

