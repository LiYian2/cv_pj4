# Plan B Summary: VGGT Initialization + 3D Gaussian Splatting on Re10k-1

## Goal

The goal of Plan B is to use **VGGT** as a **foundation-model-based initialization pipeline** for 3D Gaussian Splatting (3DGS) on the same `Re10k-1` scene used in Plan A.

---

## Dataset and Input Preparation

We used the same `Re10k-1` scene as in Plan A and reused the resized `512 x 512` image set.

Because VGGT could not process the full `279` images under the available `48 GB` single-GPU memory budget, we tested progressively smaller uniformly sampled subsets.

Observed behavior:

- Full `279` images: out-of-memory
- `128` images: out-of-memory
- `96` images: successful
- `64` images: successful
- `32` images: successful

The final selected Plan B configuration used a **uniformly sampled 96-image subset**.

---

## Final VGGT Reconstruction Settings

The final successful reconstruction used:

- scene: `Re10k-1`
- input images: `96` uniformly sampled images
- image resolution: `512 x 512`
- `shared_camera = True`
- `camera_type = SIMPLE_PINHOLE`
- `use_ba = True`
- `fine_tracking = False`
- `query_frame_num = 4`
- `max_query_pts = 1024`

---

## Final VGGT Sparse Reconstruction Result

The final reconstruction result was:

- Registered images: `96 / 96`
- Sparse points: `5753`
- Observations: `184281`
- Mean track length: `32.032157`
- Mean observations per image: `1919.593750`
- Mean reprojection error: `0.000000 px`

As noted before, this reprojection number should not be over-interpreted as directly comparable to the COLMAP result.

---

## 3DGS Training Setup

We reran Plan B under a longer and denser evaluation schedule for direct comparison against the updated Plan A experiments:

```bash
python train.py   -s ~/CV_Project/part1/part1_data/re10k_1/planB_vggt/gs_scene   -m ~/CV_Project/part1/part1_data/re10k_1/planB_vggt/output_3dgs_40k_2k   --eval   --iterations 40000   --test_iterations 2000 4000 6000 ... 40000   --save_iterations 2000 4000 6000 ... 40000
```

---

## Convergence Snapshot

### Plan B-96 test-set results

| Iteration | Test L1 | Test PSNR |
|---|---:|---:|
| 2000 | 0.0259148 | 27.3231 |
| 4000 | 0.0211746 | 28.8345 |
| 6000 | 0.0185482 | 30.0227 |
| 8000 | 0.0173378 | 30.5353 |
| 10000 | 0.0167726 | 30.7774 |
| 12000 | 0.0158281 | 31.1705 |
| 14000 | 0.0158037 | 31.3410 |
| 16000 | 0.0153246 | 31.3829 |
| 18000 | 0.0151917 | 31.5657 |
| 20000 | 0.0149817 | 31.6930 |
| 22000 | 0.0146998 | 31.8402 |
| 24000 | 0.0145585 | 31.8398 |
| 26000 | 0.0144959 | 31.9995 |
| 28000 | 0.0144870 | 31.9336 |
| 30000 | 0.0143106 | 32.0377 |
| 32000 | 0.0143083 | 32.0062 |
| 34000 | 0.0142990 | 31.9661 |
| 36000 | 0.0142167 | 31.9568 |
| 38000 | 0.0142126 | 32.0555 |
| 40000 | 0.0140688 | 32.1220 |

---

## Final 3DGS Result

Final test metrics at `40000` iterations:

- Test L1: `0.0140688`
- Test PSNR: `32.1220`

---

## Interpretation

The longer 40k training run confirms the earlier trend:

1. **Plan B continues improving beyond 30k iterations**, so it was not fully saturated at 30k.
2. **However, the gain from 30k to 40k is modest** (`32.0377 -> 32.1220` PSNR).
3. **Plan B still remains clearly below both Plan A-279 and Plan A-96**, which means the gap cannot be explained only by shorter optimization.

Therefore, under the current hardware and implementation constraints, Plan B is feasible and stable, but still weaker than the COLMAP-based alternative.

