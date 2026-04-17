# P2-L：SPGM canonical StageB formal compare（2026-04-17）

> 目标：把 `P2-K` 审查中提出的“先对齐 canonical baseline 再做正式 compare”真正补齐，并回答：在完全对齐 `post40_lr03_120 / gated_rgb0192` 后，当前 deterministic `SPGM v1` 是否还能站住。

---

## 1. 一句话结论

正式对齐到 canonical `RGB-only v2 + gated_rgb0192 + post40_lr03_120` 之后，当前 deterministic `SPGM v1` 仍低于 canonical baseline。

因此，`protocol drift` 这一层现在已经不是主解释；当前更准确的结论是：
- SPGM wiring 是通的；
- compare protocol 现在也已经对齐；
- 但当前 policy 仍然更像“对同一 active set 做强连续压制”的 suppressor，而不是能改进 active set 的 selector。

所以这一步的产出不是“SPGM 被彻底否掉”，而是把下一步明确收束成：先做 conservative deterministic / selector-first repair，而不是继续把当前 policy 直接放大到 stochastic、xyz+opacity 或更长 iter。

---

## 2. 运行身份

### 2.1 代码版本
- repo commit：`6ad7bfd`

### 2.2 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2k_canonical_stageB_compare_e1`

### 2.3 对照臂
1. `canonical_baseline_post40_lr03_120`
   - `pseudo_local_gating=hard_visible_union_signal`
2. `canonical_spgm_keep_post40_lr03_120`
   - `pseudo_local_gating=spgm_keep`
   - 保持当前 deterministic v1 默认超参：
     - `num_clusters=3`
     - `alpha_depth=0.5`
     - `beta_entropy=0.5`
     - `gamma_entropy=0.5`
     - `support_eta=0.5`
     - `weight_floor=0.05`
     - `density_mode=opacity_support`
     - `cluster_keep=(1.0, 0.8, 0.6)`

---

## 3. protocol 对齐内容

这一轮和 `P2-J` canonical bounded baseline 对齐到以下四层：

1. 同 StageA.5 handoff
- `stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply`
- `init_pseudo_camera_states_json = stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json`
- `init_pseudo_reference_mode = keep`

2. 同 StageB bounded schedule
- `stageB_iters=120`
- `stageB_post_switch_iter=40`
- `stageB_post_lr_scale_xyz=0.3`

3. 同 upstream view gate / signal protocol
- `signal_pipeline=brpo_v2`
- `signal_v2_root` 与 `pseudo_cache` 与 canonical baseline 一致
- `pseudo_local_gating_min_rgb_mask_ratio=0.0192`
- `pseudo_local_gating_min_verified_ratio=0.01`
- `pseudo_local_gating_max_fallback_ratio=0.995`

4. 同 real branch / replay evaluator
- `num_real_views=2`
- `num_pseudo_views=4`
- `lambda_real=lambda_pseudo=1.0`
- `seed=0`
- replay 继续使用同一个 `replay_internal_eval.py` evaluator

也就是说，这一轮只保留一个 study variable：Gaussian-side pseudo grad manager。

---

## 4. 结果

### 4.1 主要 replay 指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔSSIM vs baseline | ΔLPIPS vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical baseline | `24.029824` | `0.872286` | `0.081774` | `0` | `0` | `0` |
| canonical SPGM | `23.942296` | `0.869559` | `0.083590` | `-0.087528` | `-0.002727` | `+0.001817` |

### 4.2 相对关键锚点

- `after_opt baseline`：`23.948911 / 0.873485 / 0.078780`
- `StageA.5 start`：`23.924801 / 0.872320 / 0.079931`
- `P2-J post40_lr03_120`：`24.030190 / 0.872288 / 0.081773`

对应关系：
- canonical baseline 相对 `P2-J post40_lr03_120` 基本重现（PSNR 仅 `-0.000366`）
- canonical SPGM 相对 `P2-J post40_lr03_120` 仍为 `-0.087894 PSNR / -0.002728 SSIM / +0.001818 LPIPS`
- canonical SPGM 仅略高于 `StageA.5 start` 的 PSNR（`+0.0175`），但明显低于 canonical baseline

这说明：baseline 本身是可复现的，当前负结果不是 reference 漂掉造成的。

---

## 5. 机制层观察

### 5.1 rejection / active set 没变
两臂完全一致：
- `iters_with_rejection = 96 / 120`
- `total_rejected_sample_evals = 116`
- `unique_rejected_ids = [225, 260]`

同时：
- baseline `grad_keep_ratio_xyz_mean = 0.729005`
- SPGM `grad_keep_ratio_xyz_mean = 0.729020`
- SPGM `spgm_active_ratio_mean = 0.729020`

这说明 current deterministic SPGM 并没有真正改变 active set。

### 5.2 真正变化仍是权重幅度
- baseline `grad_weight_mean_xyz_mean = 0.729005`
- SPGM `grad_weight_mean_xyz_mean = 0.358640`

也就是说，当前 SPGM 的主要作用仍然是：对几乎同一 active set 做更强的连续衰减，而不是重新挑出更值得更新的 Gaussian 子集。

### 5.3 real 更低、pseudo 更高的模式保留
末尾 loss：
- baseline：`loss_real_last = 0.121538`，`loss_pseudo_last = 0.027058`
- SPGM：`loss_real_last = 0.119203`，`loss_pseudo_last = 0.029216`

这和 `P2-K` 审查时的原始观察方向一致：real anchor 看起来更容易维持，但 pseudo-side 的有效更新被压住，最终 replay 没有受益。

---

## 6. 当前最重要的判断

1. 这一步已经把“是不是 protocol 漂了”这个问题正式回答掉了：
   - 是，原始 `P2-K` compare 的 protocol 确实漂了；
   - 但现在 protocol 已对齐后，当前 SPGM 仍是负结果。

2. 因而当前更准确的工程判断不再是“先去补 formal compare”，而是：
   - current deterministic `SPGM v1` policy 本身就过强；
   - 它更像 suppressor，不像 selector。

3. 这并不等于“SPGM 方向整体失败”：
   - 当前 wiring、stats、score、policy 主链路都已经接通；
   - 失败的是当前这版 deterministic keep 的默认行为方式，而不是“SPGM 这个插点本身不存在作用”。

---

## 7. 下一步建议

### 7.1 先做 conservative deterministic SPGM
优先目标不是“更聪明地压”，而是“先别压得这么狠”。建议先测至少两条更保守的臂：
- A：`cluster_keep=(1.0, 1.0, 1.0)`，`support_eta=0.0`，`weight_floor=0.25`
- B：`cluster_keep=(1.0, 1.0, 0.9)`，`support_eta=0.25`，`weight_floor=0.20`

验收标准先放低：
- replay 至少不要再明显低于 canonical baseline；
- `grad_weight_mean_xyz_mean` 不要继续掉到当前这种近乎腰斩的区间。

### 7.2 如果 conservative 仍不行，再改成 selector-first
下一层才是结构改写：
- cluster 内 quantile / top-k keep
- support ratio 先参与 hard select，再参与 soft weight
- 让 policy 先真正改变 active set，再谈 soft attenuation

### 7.3 当前明确先不做
在上述两步之前，先不做：
- stochastic drop
- `xyz+opacity`
- 更长 iter 放大
- raw RGB densify

---

## 8. 当前可以写进项目主判断的一句话

> `P2-L` 已确认：即便 compare protocol 对齐到 canonical `post40_lr03_120 / gated_rgb0192`，当前 deterministic `SPGM v1` 仍低于 canonical baseline；因此问题不再主要是 protocol drift，而是 current policy 仍更像过强 suppressor。下一步应转向 conservative / selector-first repair，而不是直接继续长程放大。
