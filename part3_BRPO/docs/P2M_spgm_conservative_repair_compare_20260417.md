# P2-M：SPGM conservative deterministic repair compare（2026-04-17）

> 目标：沿用 `P2-L` 已对齐的 canonical StageB protocol，不改动上游 gate / handoff / schedule，只通过减弱 deterministic SPGM 的 suppressive policy，验证它能否把 replay 拉回到 canonical baseline 附近。

---

## 1. 一句话结论

conservative deterministic repair 确实有效，但还不够。

两条 repair arm 都明显优于 `P2-L` 的原始 deterministic `spgm_keep`，其中 A 臂最好：
- A：`24.00212 / 0.87132 / 0.08244`
- B：`23.99595 / 0.87115 / 0.08259`
- 原始 SPGM：`23.94230 / 0.86956 / 0.08359`
- canonical baseline：`24.02982 / 0.87229 / 0.08177`

因此当前最准确的判断是：
- `P2-L` 暴露出的“压得太狠”问题是真问题；
- 通过去掉固定 cluster decay、抬高 `weight_floor`、减弱 `support_eta`，replay 的确能明显恢复；
- 但 conservative deterministic 还没有真正追平 canonical baseline，所以下一步应从 A 臂出发做 selector-first 改写，而不是继续只靠更温和的 soft suppress。

---

## 2. 运行身份

### 2.1 代码版本
- repo commit：`6ad7bfd`

### 2.2 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_p2m_spgm_conservative_repair_e1`

### 2.3 固定 protocol
完全复用 `P2-L` 的 canonical StageB compare protocol：
- same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
- same pseudo-state handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
- same bounded StageB schedule：`post40_lr03_120`
- same upstream view gate：`gated_rgb0192`
- same replay evaluator

### 2.4 参考锚点
- canonical baseline：`canonical_baseline_post40_lr03_120`
- previous SPGM：`canonical_spgm_keep_post40_lr03_120`

### 2.5 新 repair arms
1. `spgm_repair_a_keep111_eta0_wf025`
   - `cluster_keep=(1.0, 1.0, 1.0)`
   - `support_eta=0.0`
   - `weight_floor=0.25`
2. `spgm_repair_b_keep1109_eta025_wf020`
   - `cluster_keep=(1.0, 1.0, 0.9)`
   - `support_eta=0.25`
   - `weight_floor=0.20`

---

## 3. 结果

### 3.1 主指标对照

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs baseline | ΔPSNR vs prev SPGM |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical baseline | `24.029824` | `0.872286` | `0.081774` | `0` | `+0.087528` |
| previous SPGM | `23.942296` | `0.869559` | `0.083590` | `-0.087528` | `0` |
| repair A | `24.002119` | `0.871324` | `0.082440` | `-0.027705` | `+0.059822` |
| repair B | `23.995951` | `0.871153` | `0.082594` | `-0.033873` | `+0.053655` |

best repair arm：A（PSNR / SSIM / LPIPS 全部最好）

### 3.2 机制对照

四条臂最关键的共同点是：
- rejection 没变：repair A / B 仍然都是 `96/120` iter rejection，拒掉 `225 / 260`
- `grad_keep_ratio_xyz_mean` 也几乎没变，仍在 `0.7290` 左右

真正被修复的是“压制强度”：
- baseline `grad_weight_mean_xyz_mean = 0.7290`
- previous SPGM `= 0.3586`
- repair A `= 0.5497`
- repair B `= 0.4977`

这说明 conservative repair 的作用不是“改 active set”，而是把同一 active set 上过强的软压制往回拉。

### 3.3 loss 侧变化
- baseline：`loss_real_last = 0.12154`，`loss_pseudo_last = 0.02706`
- previous SPGM：`0.11920 / 0.02922`
- repair A：`0.12055 / 0.02777`
- repair B：`0.12033 / 0.02803`

这说明 repair A/B 已经把“real 更低、pseudo 明显更高”的失衡往 baseline 拉回去了，尤其 A 最接近 baseline 的 pseudo loss。

---

## 4. 怎么解读这轮 repair

1. `P2-L` 的核心诊断被证实了：当前 deterministic SPGM 的主要问题确实是 suppress 过强，而不是别的链路没对齐。
2. conservative repair 是有效方向：只改 policy 强度，不动 protocol，其 replay 就能比 previous SPGM 回升约 `+0.054 ~ +0.060 PSNR`。
3. 但它仍没完全回到 baseline：最好 A 臂仍差 `-0.0277 PSNR / -0.00096 SSIM / +0.00067 LPIPS`。
4. 因而现在不太值得继续只做“再轻一点/再重一点”的软压制扫参了；下一步更该改的是“怎么选”，不是“压多重”。

---

## 5. 当前结论

当前可以把结论压成一句话：

> conservative deterministic repair 已经证明 current SPGM 的主要问题是 suppressive policy 过强，而且这个问题可以部分修回来；但仅靠更温和的软压制还不足以追平 canonical baseline。下一步应以 repair A 为新 anchor，进入 selector-first policy 改写。

---

## 6. 下一步建议

1. 把 `spgm_repair_a_keep111_eta0_wf025` 作为新的 SPGM repair anchor。
2. 在它的基础上做 selector-first policy：
   - cluster 内 quantile / top-k keep
   - support ratio 先参与 hard select，再参与 soft weight
   - 目标是先真正改变 active set，再保留适度 soft weight
3. 在 selector-first 结果出来之前，不继续推进：
   - stochastic drop
   - `xyz+opacity`
   - 更长 iter 放大
