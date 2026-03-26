# Plan A Summary: COLMAP Initialization + 3D Gaussian Splatting on Re10k-1

## Goal

The goal of Plan A is to use **COLMAP** to estimate camera poses and a sparse point cloud from the input image sequence, then use this reconstruction as the initialization for **3D Gaussian Splatting (3DGS)** training on the `Re10k-1` scene.

This serves as the **classical geometry-based baseline** for Part 1, which will later be compared against **Plan B: Foundation Model Initialization**.

---

## Dataset and Preprocessing

We used the `Re10k-1` scene from the project dataset.

Original dataset format:

```text
Re10k-1/
├── images/
├── cameras.json
└── intrinsics.json
```

The original images were `256 x 256`. Initial COLMAP reconstruction at this resolution failed to register the full sequence reliably.

To improve reconstruction robustness, the input images were resized to `512 x 512` for COLMAP.

Key observations:

- At `256 x 256`, COLMAP only reconstructed a very small partial model.
- After resizing to `512 x 512`, reconstruction quality improved substantially.
- This suggests that low image resolution was a major bottleneck for feature extraction and matching.

---

## COLMAP Reconstruction Settings

For the final successful reconstruction, the following settings were used:

- Image resolution for COLMAP: `512 x 512`
- Camera model: `SIMPLE_PINHOLE`
- Single shared camera: enabled
- Maximum SIFT features: `4096`
- Matcher: `sequential_matcher`
- Sequential overlap: `10`
- Quadratic overlap: enabled

---

## Final COLMAP Result

The final full-sequence reconstruction result was:

- Registered images: `279 / 279`
- Sparse points: `9389`
- Observations: `444610`
- Mean track length: `47.354351`
- Mean observations per image: `1593.584229`
- Mean reprojection error: `0.672096 px`

This indicates that COLMAP successfully reconstructed the full sequence and produced a strong initialization for 3DGS training.

---

## 3DGS Training Setup

We ran two Plan A variants for analysis:

1. **Plan A-279**: train 3DGS on the full COLMAP reconstruction using all `279` views.
2. **Plan A-96**: construct a matched-subset COLMAP scene using the exact same `96` image names used by Plan B, then train 3DGS under the same optimization budget.

The goal of Plan A-96 is to remove view-count mismatch and make the comparison against Plan B much fairer.

Training command pattern:

```bash
python train.py   -s <scene_path>   -m <output_path>   --eval   --iterations 40000   --test_iterations 2000 4000 6000 ... 40000   --save_iterations 2000 4000 6000 ... 40000
```

---

## Final 3DGS Results

### Plan A-279 (full 279-view COLMAP initialization)

Final test metrics at `40000` iterations:

- Test L1: `0.0106019`
- Test PSNR: `35.1689`

### Plan A-96 (matched 96-view COLMAP initialization)

Final test metrics at `40000` iterations:

- Test L1: `0.0130271`
- Test PSNR: `33.4120`

---

## Convergence Snapshot

### Plan A-279 test-set results

| Iteration | Test L1 | Test PSNR |
|---|---:|---:|
| 2000 | 0.0181820 | 30.2679 |
| 4000 | 0.0151588 | 32.0054 |
| 6000 | 0.0134242 | 32.9823 |
| 8000 | 0.0128308 | 33.4142 |
| 10000 | 0.0131852 | 33.3984 |
| 12000 | 0.0122125 | 33.8900 |
| 14000 | 0.0121746 | 33.9803 |
| 16000 | 0.0117897 | 34.1633 |
| 18000 | 0.0113641 | 34.5437 |
| 20000 | 0.0113251 | 34.6676 |
| 22000 | 0.0111894 | 34.6980 |
| 24000 | 0.0110631 | 34.8166 |
| 26000 | 0.0109824 | 34.8539 |
| 28000 | 0.0109350 | 34.9391 |
| 30000 | 0.0108177 | 34.9682 |
| 32000 | 0.0108590 | 34.9591 |
| 34000 | 0.0108864 | 35.0009 |
| 36000 | 0.0108614 | 34.9795 |
| 38000 | 0.0106749 | 35.1119 |
| 40000 | 0.0106019 | 35.1689 |

### Plan A-96 test-set results

| Iteration | Test L1 | Test PSNR |
|---|---:|---:|
| 2000 | 0.0211584 | 29.3469 |
| 4000 | 0.0176317 | 30.8854 |
| 6000 | 0.0157483 | 31.8593 |
| 8000 | 0.0151744 | 32.1357 |
| 10000 | 0.0148966 | 32.3550 |
| 12000 | 0.0143635 | 32.6241 |
| 14000 | 0.0144091 | 32.4124 |
| 16000 | 0.0141225 | 32.5446 |
| 18000 | 0.0140980 | 32.6150 |
| 20000 | 0.0137941 | 33.0022 |
| 22000 | 0.0135438 | 33.1303 |
| 24000 | 0.0134292 | 33.1405 |
| 26000 | 0.0134823 | 33.2119 |
| 28000 | 0.0133425 | 33.2253 |
| 30000 | 0.0132879 | 33.2829 |
| 32000 | 0.0132415 | 33.2653 |
| 34000 | 0.0131702 | 33.3127 |
| 36000 | 0.0130211 | 33.3966 |
| 38000 | 0.0131955 | 33.3455 |
| 40000 | 0.0130271 | 33.4120 |

---

## Interpretation

The updated Plan A analysis gives two important conclusions:

1. **Plan A remains the strongest pipeline overall when all 279 views are available.**
2. **Reducing Plan A from 279 views to the matched 96-view subset causes a clear quality drop** (`35.1689 -> 33.4120` PSNR), showing that view coverage matters substantially.
3. **However, even under the matched 96-view setting, COLMAP initialization still remains strong enough to outperform Plan B.**

This makes Plan A not only the best full-data baseline, but also a stronger matched-subset initialization than Plan B under the current implementation setting.

