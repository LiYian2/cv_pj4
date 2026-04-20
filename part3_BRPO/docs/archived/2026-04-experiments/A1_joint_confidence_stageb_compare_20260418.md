# A1：unified RGB-D joint confidence StageB formal compare（2026-04-18）

> 目标：在不改 SPGM policy、不改 StageA.5 / StageB schedule 的前提下，只把 observation consumption 从 `RGB-first + depth-sidecar` 改成 `joint_confidence_v2 + joint_depth_v2`，回答 unified trusted support 本身是否值得保留。

---

## 1. 一句话结论
A1 在 canonical StageB protocol + repair A dense_keep 下给出了第一条明确正信号：
1. `a1_joint_repair_a_dense_keep` 明确超过 in-run `control_repair_a_rgbfirst_sidecar`
2. A1 同时在 PSNR / LPIPS 上超过 canonical baseline `RGB-only v2 + gated_rgb0192 + post40_lr03_120`
3. 更关键的是：A1 并没有扩大 mask coverage，反而把 render fallback 几乎清空；收益来自 RGB / depth 共享同一 trusted support，而不是“喂更多 depth mass”
4. 因此 Re10k 当前主线从 selector-first 正式转到 A1 confirmation / landing；A2 只在需要扩大 unified support 时再跟进

---

## 2. 运行身份

### 2.1 运行根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_stageb_compare_e1`

### 2.2 代码身份
- repo commit：`33b30ea`

### 2.3 joint signal 根
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_re10k_signal_v2_from_e15`

### 2.4 arms
1. `control_repair_a_rgbfirst_sidecar`
2. `a1_joint_repair_a_dense_keep`

### 2.5 固定 protocol
1. same StageA.5 handoff：`stageA5_v2rgbonly_xyz_gated_rgb0192_80`
2. same sequential pseudo-state handoff：`init_pseudo_camera_states_json = pseudo_camera_states_final.json`
3. same StageB schedule：`post40_lr03_120`
4. same real/pseudo weighting：`lambda_real=1.0 / lambda_pseudo=1.0`
5. same SPGM policy：repair A dense_keep（`keep=(1,1,1), eta=0.0, weight_floor=0.25`）
6. 唯一改变的是 observation consumption：
   - control：`brpo_v2_raw + train_mask + target_depth_for_refine_v2`
   - A1：`joint_confidence_v2 + joint_confidence_v2 + joint_depth_v2`

---

## 3. 主结果

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs control | ΔPSNR vs canonical baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| control rgb-first sidecar | `24.001931` | `0.871322` | `0.082440` | `0` | `-0.027893` |
| A1 joint confidence | `24.031512` | `0.872048` | `0.081744` | `+0.029580` | `+0.001687` |

补充：
- A1 vs control：`+0.029580 PSNR / +0.000726 SSIM / -0.000696 LPIPS`
- A1 vs canonical baseline：`+0.001687 PSNR / -0.000238 SSIM / -0.000029 LPIPS`

---

## 4. 机制层观察

### 4.1 control 复现是稳定的
- in-run control 相对 P2-T old control 仅 `-0.000043 PSNR`，说明本次 compare 没有明显 protocol drift

### 4.2 收益不是来自 coverage 变大
- control `mean_confidence_nonzero_ratio ≈ 0.019582`
- A1 `mean_confidence_nonzero_ratio ≈ 0.019570`
- coverage 基本持平，A1 甚至略窄

### 4.3 A1 把 depth 从 sidecar 变成了同口径 trusted support
- control `mean_target_depth_verified_ratio ≈ 0.041339`，但 `render_fallback_ratio ≈ 0.958661`
- A1 `mean_target_depth_verified_ratio ≈ 0.019570`，`render_fallback_ratio ≈ 0.000012`
- 也就是说，A1 没有靠更多 fallback / dense depth 取胜，而是用更严格但一致的 support 域取胜

### 4.4 SPGM 不是本轮解释主因
- 两臂都保持 repair A dense_keep，`selected_ratio / grad_weight / rejection` 基本一致
- 当前差异主要来自 observation semantics，而不是 SPGM policy 变化

---

## 5. 实现状态
1. 已新增 `pseudo_branch/brpo_v2_signal/joint_confidence.py`
2. 已在 signal builder 写出：
   - `joint_confidence_v2.npy`
   - `joint_confidence_cont_v2.npy`
   - `joint_depth_target_v2.npy`
   - `joint_meta_v2.json`
3. 已在 consumer 接通：
   - `--stageA_rgb_mask_mode joint_confidence_v2`
   - `--stageA_depth_mask_mode joint_confidence_v2`
   - `--stageA_target_depth_mode joint_depth_v2`
4. 已修 generic depth path，避免继续把 `confidence_mask` 错代 `depth_confidence_mask`

---

## 6. 结论
1. A1 已经不只是“理论合理”或“smoke 通过”，而是在 canonical StageB protocol 上给出了明确正结果
2. 当前更大的结构瓶颈确实在 observation semantics；这一步先于 B1/B2/B3 是对的
3. A2 的角色不再是“必须先补 coverage 才有可能有效”，而是“在 A1 已经成立后，若仍要扩大 unified support，再做 geometry-constrained widen”
4. selector-first 保留为 near-parity reference arm，不再作为当前 Re10k 主线

---

## 7. 下一步建议
1. 先把 A1 作为当前 observation 主线做 confirmation / landing
2. 仅当后续需要进一步扩大 unified support 时，再推进 A2
3. 在 Re10k A1 判断稳定前，不急于把这条线平移到 DL3DV


---

## 8. A1+A2 扩展实验（2026-04-18）

### 8.1 三臂 compare 结果

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs control | ΔPSNR vs A1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| control rgb-first sidecar | 24.106 | 0.8734 | 0.08145 | 0.0 | - |
| A1 joint confidence | 24.120 | 0.8736 | 0.08146 | +0.013 | 0.0 |
| A1+A2 geometry expand | 23.834 | 0.8683 | 0.08295 | -0.272 | **-0.286** |

### 8.2 关键发现

1. **A2 扩张带来显著负效果**：A1+A2 相比 A1 下降 -0.286 PSNR
2. Coverage 从 A1 的 1.96% 扩张到 A2 的 6.05%，但引入的低置信度区域损害了整体渲染质量
3. A2 的 geometry-constrained expansion（双侧投影一致性 + overlap + fusion weight）虽然理论上合理，但实际扩张的区域质量不足以支撑训练

### 8.3 结论

1. A1 的收益来自 unified trusted support 的质量，而不是 coverage 的大小
2. 单纯扩大 coverage（即使有几何约束）不能带来质量提升
3. A2 作为 widening 方案需要重新设计：
   - 更严格的置信度筛选（不能仅依赖 geometry）
   - 可能需要引入质量预测/评估机制
   - 或者放弃 widening，直接从 A1 推进到 B1/B2/B3

### 8.4 运行身份

- 运行根：/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_a2_stageb_compare_e1
- 代码：同 A1 compare，额外使用 pseudo_branch/brpo_v2_signal/support_expand.py
- signal_v2 根：/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260418_a1_re10k_signal_v2_from_e15
