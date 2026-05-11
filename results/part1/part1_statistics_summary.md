# Part1 Experiment Statistics Summary

**Data Source:** `final_metrics_24.csv` (24 configurations at 40k iterations)
**Date:** 2026-05-11

---

## 1. Raw Unit Table M(d,i,b)

For appendix: average metrics over test views within same scene/initialization/backbone.

| dataset | initialization | backbone | PSNR | SSIM | LPIPS |
|---------|---------------|----------|------|------|-------|
| Waymo-405841 | COLMAP-full | 3DGS | 32.7703 | 0.9377 | 0.1258 |
| Waymo-405841 | COLMAP-full | Scaffold-GS | 32.0614 | 0.9339 | 0.1354 |
| Waymo-405841 | COLMAP-108 | 3DGS | 31.3506 | 0.9213 | 0.1403 |
| Waymo-405841 | COLMAP-108 | Scaffold-GS | 30.8637 | 0.9185 | 0.1488 |
| Waymo-405841 | VGGT-BA-108 | 3DGS | 29.7185 | 0.9063 | 0.1538 |
| Waymo-405841 | VGGT-BA-108 | Scaffold-GS | 30.0604 | 0.9129 | 0.1564 |
| DL3DV-2 | COLMAP-full | 3DGS | 34.7606 | 0.9657 | 0.0678 |
| DL3DV-2 | COLMAP-full | Scaffold-GS | 36.1819 | 0.9719 | 0.0564 |
| DL3DV-2 | COLMAP-108 | 3DGS | 32.2879 | 0.9507 | 0.0845 |
| DL3DV-2 | COLMAP-108 | Scaffold-GS | 34.4993 | 0.9653 | 0.0648 |
| DL3DV-2 | VGGT-BA-108 | 3DGS | 27.6671 | 0.8746 | 0.1530 |
| DL3DV-2 | VGGT-BA-108 | Scaffold-GS | 25.0696 | 0.8321 | 0.2094 |
| Re10k-1 | COLMAP-full | 3DGS | 35.1649 | 0.9776 | 0.0424 |
| Re10k-1 | COLMAP-full | Scaffold-GS | 34.8621 | 0.9769 | 0.0438 |
| Re10k-1 | COLMAP-108 | 3DGS | 33.5035 | 0.9694 | 0.0515 |
| Re10k-1 | COLMAP-108 | Scaffold-GS | 33.7991 | 0.9712 | 0.0491 |
| Re10k-1 | VGGT-BA-108 | 3DGS | 32.0530 | 0.9628 | 0.0548 |
| Re10k-1 | VGGT-BA-108 | Scaffold-GS | 32.7645 | 0.9670 | 0.0527 |

---

## 2. Initialization Effect (Main Result)

$$\Delta_{\mathrm{init}}(d,b) = M(d, \text{COLMAP-108}, b) - M(d, \text{VGGT-BA-108}, b)$$

| dataset | backbone | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|----------|-------|-------|--------|
| Waymo-405841 | 3DGS | +1.6321 | +0.0150 | -0.0135 |
| Waymo-405841 | Scaffold-GS | +0.8032 | +0.0056 | -0.0076 |
| DL3DV-2 | 3DGS | +4.6208 | +0.0761 | -0.0685 |
| DL3DV-2 | Scaffold-GS | +9.4298 | +0.1333 | -0.1446 |
| Re10k-1 | 3DGS | +1.4505 | +0.0066 | -0.0033 |
| Re10k-1 | Scaffold-GS | +1.0346 | +0.0041 | -0.0036 |

**Summary (macro mean over 6 paired differences):**

| metric | mean | std |
|--------|------|-----|
| ΔPSNR | **+3.16 dB** | 3.08 |
| ΔSSIM | **+0.040** | 0.049 |
| ΔLPIPS | **-0.040** | 0.052 |

---

## 3. Coverage Effect

$$\Delta_{\mathrm{cov}}(d,b) = M(d, \text{COLMAP-full}, b) - M(d, \text{COLMAP-108}, b)$$

| dataset | backbone | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|----------|-------|-------|--------|
| Waymo-405841 | 3DGS | +1.4197 | +0.0164 | -0.0145 |
| Waymo-405841 | Scaffold-GS | +1.1977 | +0.0154 | -0.0135 |
| DL3DV-2 | 3DGS | +2.4727 | +0.0151 | -0.0167 |
| DL3DV-2 | Scaffold-GS | +1.6826 | +0.0066 | -0.0084 |
| Re10k-1 | 3DGS | +1.6615 | +0.0082 | -0.0091 |
| Re10k-1 | Scaffold-GS | +1.0629 | +0.0058 | -0.0053 |

**Summary (macro mean over 6 paired differences):**

| metric | mean | std |
|--------|------|-----|
| ΔPSNR | **+1.58 dB** | 0.46 |
| ΔSSIM | **+0.011** | 0.0045 |
| ΔLPIPS | **-0.011** | 0.0039 |

---

## 4. Backbone Effect

$$\Delta_{\mathrm{backbone}}(d,i) = M(d, i, \text{Scaffold-GS}) - M(d, i, \text{3DGS})$$

### Under COLMAP-full initialization

| dataset | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|-------|-------|--------|
| Waymo-405841 | -0.7089 | -0.0038 | +0.0095 |
| DL3DV-2 | +1.4214 | +0.0062 | -0.0114 |
| Re10k-1 | -0.3029 | -0.0007 | +0.0014 |
| **Mean** | **+0.14** | +0.0006 | -0.0002 |

### Under COLMAP-108 initialization

| dataset | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|-------|-------|--------|
| Waymo-405841 | -0.4869 | -0.0028 | +0.0085 |
| DL3DV-2 | +2.2115 | +0.0147 | -0.0197 |
| Re10k-1 | +0.2956 | +0.0018 | -0.0024 |
| **Mean** | **+0.67** | +0.0046 | -0.0045 |

### Under VGGT-BA-108 initialization

| dataset | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|-------|-------|--------|
| Waymo-405841 | +0.3419 | +0.0066 | +0.0027 |
| DL3DV-2 | -2.5975 | -0.0425 | +0.0565 |
| Re10k-1 | +0.7116 | +0.0042 | -0.0021 |
| **Mean** | **-0.51** | -0.0106 | +0.0190 |

**Observation:** Scaffold-GS shows strong benefit on DL3DV-2 under COLMAP, but fails catastrophically under VGGT-BA-108 on DL3DV-2 (ΔPSNR = -2.60 dB, ΔLPIPS = +0.0565).

---

## 5. VGGT-BA Ablation

$$\Delta_{\mathrm{BA}}(d,b) = M(d, \text{VGGT-BA-108}, b) - M(d, \text{VGGT-w/o-BA-108}, b)$$

| dataset | backbone | ΔPSNR | ΔSSIM | ΔLPIPS |
|---------|----------|-------|-------|--------|
| Waymo-405841 | 3DGS | +2.2147 | +0.0568 | -0.0482 |
| Waymo-405841 | Scaffold-GS | +2.1108 | +0.0557 | -0.0310 |
| DL3DV-2 | 3DGS | -0.8213 | -0.0273 | +0.0298 |
| DL3DV-2 | Scaffold-GS | -4.5685 | -0.0790 | +0.1048 |
| Re10k-1 | 3DGS | +5.4557 | +0.0517 | -0.0503 |
| Re10k-1 | Scaffold-GS | +5.3416 | +0.0458 | -0.0347 |

**Available units:** 6 (all scene-backbone pairs)

**Summary:**

| metric | mean | std |
|--------|------|-----|
| ΔPSNR | **+1.62 dB** | 3.50 |
| ΔSSIM | **+0.017** | 0.052 |
| ΔLPIPS | **-0.005** | 0.056 |

**Note:** Large variance driven by DL3DV-2 failure case. For Re10k-1, BA brings ~5 dB gain; for Waymo-405841, ~2 dB gain; but for DL3DV-2, BA actually hurts (especially Scaffold-GS).

---

## 6. At 40k Iterations PSNR Gap

COLMAP-108 vs VGGT-BA-108:

| dataset | backbone | PSNR Gap |
|---------|----------|----------|
| Waymo-405841 | 3DGS | 1.63 dB |
| Waymo-405841 | Scaffold-GS | 0.80 dB |
| DL3DV-2 | 3DGS | 4.62 dB |
| DL3DV-2 | Scaffold-GS | 9.43 dB |
| Re10k-1 | 3DGS | 1.45 dB |
| Re10k-1 | Scaffold-GS | 1.03 dB |

- **Mean PSNR gap (all):** 3.16 dB
- **For 3DGS only:** 2.57 dB
- **For Scaffold-GS only:** 3.76 dB

---

## Key Findings Summary

1. **Initialization effect dominates**: COLMAP initialization consistently outperforms VGGT-BA by ~3 dB PSNR on average. The gap is most severe on DL3DV-2 (up to 9.4 dB for Scaffold-GS).

2. **Coverage matters**: Using full COLMAP (vs 108-frame subset) brings consistent +1.6 dB gain.

3. **Backbone effect is initialization-dependent**: Scaffold-GS shines under COLMAP (+0.67 dB average) but fails under VGGT-BA-108 (-0.51 dB average), with catastrophic collapse on DL3DV-2.

4. **BA refinement helps overall but with high variance**: +1.6 dB average gain, but DL3DV-2 shows negative effect while Re10k-1 gains >5 dB.

5. **Scene-dependent behavior**: DL3DV-2 is the most challenging for VGGT-based initialization; Waymo-405841 and Re10k-1 show more consistent patterns.