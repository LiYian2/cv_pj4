# Part2 S3PO Baseline Results (Full Dataset)

**Data Source:** `/home/bzhang512/CV_Project/output/part2_s3po/`
**Date:** 2026-05-11

## Metrics Summary

| Dataset | PSNR (dB) | SSIM | LPIPS | ATE RMSE (m) | ATE Mean (m) |
|---------|-----------|------|-------|--------------|--------------|
| Waymo-405841 | 24.02 | 0.766 | 0.280 | 0.613 | 0.550 |
| DL3DV-2 | 17.48 | 0.615 | 0.354 | 0.463 | 0.394 |
| Re10k-1 | 23.95 | 0.873 | 0.079 | 0.007 | 0.006 |

---

## Detailed Results

### 1. Waymo-405841

**Path:** `/home/bzhang512/CV_Project/output/part2_s3po/405841/s3po_405841_full/405841_part2_s3po/2026-04-04-00-43-20`

**Rendering Metrics (after_opt):**
```json
{
    "mean_psnr": 24.016064148375442,
    "mean_ssim": 0.7663323013476153,
    "mean_lpips": 0.2795942471513535
}
```

**ATE Statistics:**
```json
{
    "rmse": 0.6126763947784072,
    "mean": 0.550351252777797,
    "median": 0.5600227464960688,
    "std": 0.26923198785541075,
    "min": 0.08335534193042048,
    "max": 1.0839439211699555
}
```

---

### 2. DL3DV-2

**Path:** `/home/bzhang512/CV_Project/output/part2_s3po/dl3dv-2/s3po_dl3dv-2_full/DL3DV-2_part2_s3po/2026-04-04-02-11-11`

**Rendering Metrics (after_opt):**
```json
{
    "mean_psnr": 17.47825269441347,
    "mean_ssim": 0.6147324532471798,
    "mean_lpips": 0.35376550060873097
}
```

**ATE Statistics:**
```json
{
    "rmse": 0.46345585531270633,
    "mean": 0.3941263134931234,
    "median": 0.3296489219633054,
    "std": 0.2438355569558149,
    "min": 0.08497083264559732,
    "max": 0.8380454888426708
}
```

---

### 3. Re10k-1

**Path:** `/home/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full/Re10k-1_part2_s3po/2026-04-04-02-11-09`

**Rendering Metrics (after_opt):**
```json
{
    "mean_psnr": 23.94890858685529,
    "mean_ssim": 0.8734854221343994,
    "mean_lpips": 0.0787798319839769
}
```

**ATE Statistics:**
```json
{
    "rmse": 0.006913714862604829,
    "mean": 0.005992591950096004,
    "median": 0.005263036106412593,
    "std": 0.003447940678295882,
    "min": 0.0017774014752364837,
    "max": 0.011652455390136563
}
```

---

## Notes

- These are Part2 baseline results using S3PO pipeline with COLMAP initialization
- "Full" refers to using the complete COLMAP reconstruction (not sparse subset)
- ATE (Absolute Trajectory Error) computed via evo trajectory alignment
- Re10k-1 shows exceptional pose accuracy (ATE < 1cm), likely due to well-conditioned COLMAP initialization