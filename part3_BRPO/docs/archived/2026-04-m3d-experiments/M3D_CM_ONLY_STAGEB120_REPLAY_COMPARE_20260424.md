# M3D_CM_ONLY_STAGEB120_REPLAY_COMPARE_20260424.md

> 更新时间：2026-04-24 14:31 (CST)

## 1. purpose
验证 `exact_brpo_upstream_target_v1` 在 consumer 侧若不再使用 `valid_mask` 与 `target_confidence`，而是仅用裸 `C_m` 做 shared RGB/depth supervision，是否会改善 replay。

本次修改没有重建旧 signal 链条；直接复用了已经裁剪为 mainline-only 的 signal roots：
- `/home/bzhang512/my_storage_1T/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_full/sparse_signal`
- `/home/bzhang512/my_storage_1T/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_full/dense3d_q070_signal`

新增 consumer-side loss mode：
- `exact_shared_cm_cm_only_v1`
- 与 `exact_shared_cm_v1` 的唯一区别：不再把 `valid_mask` 与 `target_confidence` 乘进 effective mask；即只用离散 `C_m`。

## 2. important code-grounded note
在当前 exact-upstream 导出结果中，`C_m > 0` 与 `valid_mask > 0` 实际是完全重合的；因此这次 ablation 真正测试到的核心，几乎就是“去掉 `target_confidence`”这件事。

从 live smoke 统计看，切到裸 `C_m` 后，监督总量会放大约：
- sparse: `1.55x`
- q0.90: `1.76x`
- q0.80: `1.71x`
- q0.70: `1.70x`

## 3. protocol
- compare root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_cm_only_compare_stageB120_replay`
- common anchor:
  - `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/refined_gaussians.ply`
  - `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80/pseudo_camera_states_final.json`
- stage budget: `StageB 120 iter`
- arms:
  - `sparse_current` → `sparse_signal + exact_shared_cm_v1`
  - `sparse_cm_only` → `sparse_signal + exact_shared_cm_cm_only_v1`
  - `q070_current` → `dense3d_q070_signal + exact_shared_cm_v1`
  - `q070_cm_only` → `dense3d_q070_signal + exact_shared_cm_cm_only_v1`

## 4. results
### 4.1 replay
- `sparse_current`: PSNR `24.00395`, SSIM `0.87177`, LPIPS `0.08248`
- `sparse_cm_only`: PSNR `24.00123`, SSIM `0.87146`, LPIPS `0.08287`
- `q070_current`: PSNR `23.66591`, SSIM `0.86585`, LPIPS `0.08577`
- `q070_cm_only`: PSNR `23.65668`, SSIM `0.86553`, LPIPS `0.08616`

### 4.2 delta vs current
- `sparse_cm_only - sparse_current`
  - PSNR `-0.00272`
  - SSIM `-0.00030`
  - LPIPS `+0.00039`
- `q070_cm_only - q070_current`
  - PSNR `-0.00923`
  - SSIM `-0.00032`
  - LPIPS `+0.00038`

### 4.3 StageB final losses
- `sparse_current`: total `0.16888`, pseudo `0.04652`, rgb `0.00785`, depth `0.04102`
- `sparse_cm_only`: total `0.17569`, pseudo `0.05341`, rgb `0.00851`, depth `0.04745`
- `q070_current`: total `0.17212`, pseudo `0.05188`, rgb `0.00949`, depth `0.04523`
- `q070_cm_only`: total `0.18286`, pseudo `0.06263`, rgb `0.01217`, depth `0.05410`

## 5. interpretation
结论很直接：当前 consumer 侧改成“只用裸 `C_m`，不乘 `valid_mask` / `target_confidence`”没有带来改善，反而在 sparse 与 q070 上都造成小幅但一致的 replay 退化，同时 RGB/depth/pseudo loss 全部变差。

由于当前 live exact-upstream 导出里 `C_m > 0` 与 `valid_mask > 0` 完全重合，因此这次退化可以近似解释为：
- 主要不是“去掉 valid_mask”的影响
- 而是“去掉 `target_confidence`，把原本被连续几何置信度抑制的监督整体放大”带来的负效应

这与此前 structural forensic 的判断一致：
- dense3d 当前新增 supervision 的大头来自 single-branch 区域
- 这些区域不应简单地按裸 `C_m` 全量放大
- 当前 `target_confidence` 至少在现有 live 语义下起到了必要的抑制作用，而不是多余噪声项

## 6. answer to the ablation question
在当前 fixed route 下，`exact_shared_cm_cm_only_v1` 不是更优替代品；`exact_shared_cm_v1` 仍然更好。

因此下一步不应走“彻底去掉 target_confidence / 只留裸 `C_m`”这条线。若还要继续沿这个方向做方法排查，更值得测的是更局部的合同修正，例如：
1. 只在 `both` 区域弱化/移除 continuous weighting
2. `single` 区域保留 continuous down-weight
3. 或直接对照 paper / author 口径检查 single-branch 区域是否本就不该以当前方式同时进入 shared RGB+depth loss
