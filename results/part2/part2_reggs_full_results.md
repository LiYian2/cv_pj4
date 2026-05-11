# Part2 RegGS Baseline Results (All Non-Train Views)

**Data Source:** `/home/bzhang512/CV_Project/output/part2/`
**Test Protocol:** `all_non_train_subset_v1`
**Date:** 2026-05-11

## Metrics Summary

| Dataset | PSNR (dB) | SSIM | LPIPS | ATE RMSE (aligned) | Test Views |
|---------|-----------|------|-------|-------------------|------------|
| Waymo-405841 | 16.68 | 0.481 | 0.524 | 19.29 m | 179 |
| DL3DV-2 | 15.79 | 0.443 | 0.482 | 1.87 m | 296 |
| Re10k-1 | 26.02 | 0.895 | 0.128 | 0.011 m | 270 |

---

## Detailed Results

### 1. Waymo-405841

**Path:** `/home/bzhang512/CV_Project/output/part2/405841/reggs_405841_scene_full_dl3dv-ckpt_sr30_nv20_sm2_stable_v1`

**Config:** sr30, nv20, sm2, dl3dv-ckpt

**Rendering Metrics:**
```json
{
  "n_train_frames": 20,
  "n_test_frames_selected": 179,
  "avg_psnr": 16.680513056962848,
  "avg_ssim": 0.4812841236424846,
  "avg_lpips": 0.5235091968122141
}
```

**ATE Statistics:**
```json
{
  "aligned_test_ate": {
    "rmse": 19.29405975341797,
    "mean": 16.7898006439209,
    "median": 15.761176109313965,
    "std": 9.505967140197754
  }
}
```

---

### 2. DL3DV-2

**Path:** `/home/bzhang512/CV_Project/output/part2/dl3dv_2/reggs_dl3dv2_dl3dv-ckpt_sr30_nv10`

**Config:** sr30, nv10, dl3dv-ckpt

**Rendering Metrics:**
```json
{
  "n_train_frames": 10,
  "n_test_frames_selected": 296,
  "avg_psnr": 15.792936102764026,
  "avg_ssim": 0.4430072599568883,
  "avg_lpips": 0.48163133650716095
}
```

**ATE Statistics:**
```json
{
  "aligned_test_ate": {
    "rmse": 1.8730742931365967,
    "mean": 1.442458152770996,
    "median": 0.8742092847824097,
    "std": 1.1948732137680054
  }
}
```

---

### 3. Re10k-1

**Path:** `/home/bzhang512/CV_Project/output/part2/re10k_1/reggs_re10k1_re10k-ckpt_sr50_nv9_sm2_comparison_check`

**Config:** sr50, nv9, sm2, re10k-ckpt

**Rendering Metrics:**
```json
{
  "n_train_frames": 9,
  "n_test_frames_selected": 270,
  "avg_psnr": 26.024956123917192,
  "avg_ssim": 0.8950996368019669,
  "avg_lpips": 0.1276927331522866
}
```

**ATE Statistics:**
```json
{
  "aligned_test_ate": {
    "rmse": 0.010797359049320221,
    "mean": 0.010117258876562119,
    "median": 0.009847135283052921,
    "std": 0.0037714794743806124
  }
}
```

---

## Notes

1. **ATE Differences:** Waymo-405841 shows extremely high ATE (19.29m), indicating severe pose drift or misalignment. Re10k-1 has exceptional pose accuracy (1.1 cm).

2. **Rendering Quality:** RegGS baseline shows lower rendering quality compared to S3PO baseline on all datasets, especially on Waymo-405841 and DL3DV-2.

3. **Test Protocol:** All metrics computed on `all_non_train` protocol - all frames not used as training views, evaluated with pose optimization enabled.