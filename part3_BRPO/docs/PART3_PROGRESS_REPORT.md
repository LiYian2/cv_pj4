# Part3 Progress Report

## 1. Pipeline Overview

Part3 pipeline 分为三层：

上游双链：
- Legacy Depth 鏈：depth verify → mask → depth target
- Signal v2 RGB 鏈：fused RGB correspondence → RGB mask → depth supervision

下游 Refine 鏈：
- StageA：pseudo pose + exposure update（不更新 Gaussian）
- StageA5：xyz update（pseudo-side only）
- StageB：joint refine（pseudo + real branch）

辅助机制：
- Abs prior：抑制 pose drift
- Local Gaussian gating：pseudo-side 只更新 support-visible Gaussian

## 2. Legacy Depth Chain

Step 1: internal_eval_cache（source of truth）
- camera_states.json：位姿、内参
- after_opt/render_rgb/：pseudo RGB
- after_opt/render_depth_npy/：pseudo depth
- after_opt/point_cloud.ply：当前地图

Step 2: Select + Difix
- Select：确定 pseudo frame + left/right reference
- Difix：pseudo RGB 经左右修复

Step 3: Verify（几何验证）
- matcher 找 pseudo ↔ ref 匹配点
- backproject → 3D → project 回 pseudo
- 检查 reprojection + depth error
- 输出：seed_support（~1.56%），projected_depth

Step 4: Propagation（mask 扩张）
- seed_support → train_mask（~19%）
- mask 扩张，depth 数值未扩张

Step 5: M3/M5 Depth Target
- M3：projected + fallback blend
- M5：confidence-aware densify
- 问题：verified depth ~1.56%，大部分 fallback

核心问题：depth seed 决定 RGB mask，语义混了

## 3. Signal v2 RGB Chain

Step 1: Fusion v2（BRPO-style）
- 用 depth/geometry 做 target↔reference 权重
- 输出：fused RGB，fusion_weight，overlap_conf

Step 2: RGB Mask Inference v2
- fused RGB ↔ reference RGB correspondence matching
- 不依赖旧的 depth verify
- 输出：raw_rgb_confidence_v2（~1.96%）

Step 3: Depth Supervision v2
- 参考 RGB mask + projected depth
- 独立生成，不依赖 train-mask

核心差异：
- Legacy：depth verify 决定 mask（RGB ~19%，depth ~1.56%）
- v2：RGB correspondence 独立定义（RGB ~1.96%，语义更干净）

## 4. Refine Chain

StageA：只更新 pose + exposure，不更新 Gaussian
StageA5：pseudo-side xyz update
StageB：joint refine（pseudo + real）
- 问题：后段不稳定（20iter PASS → 120iter regression）

Abs Prior：(lambda_abs_t, lambda_abs_r) = (3.0, 0.1)
- 抑制 drift，不显著改善 depth

Local Gating：hard gating based on signal threshold
- legacy 上 threshold 收益低

## 5. Experimental Progress

P0：Abs Prior Calibration
- (3.0, 0.1) 有效区间
- 纯 StageA replay 无信息

P1A：Legacy vs v2-rgb-only vs full-v2
- v2-rgb-only 不是明显坏方向
- full-v2 depth 过窄（~1.96%）

P2D：Threshold Diagnostics
- current threshold 对 legacy 是 no-op

P2E：Legacy Threshold Calibration
- 继续细磨 threshold 收益低

P2F：v2-rgb-only StageA5
- v2 vs legacy：PSNR +0.07
- branch-specific gating 有正增益

P2G：v2 StageB Short Compare
- v2 StageB gated：PSNR +0.062
- v2 vs legacy StageB：PSNR +0.09~0.11

P2H：StageB 120iter Verify
- 120iter 退回到负区间
- gating 工作但不解决后段稳定性

## 6. Current Status

主候选：RGB-only v2 + gated_rgb0192

关键判断：
1. RGB-only v2 优于 legacy（StageA5 + StageB short）
2. StageB 后段不稳定（20iter PASS → 120iter regression）
3. 不该优先做 raw RGB densify

不建议：legacy threshold calibration，raw RGB densify，xyz+opacity/soft gating

## 7. Next Steps

第一优先级：StageB Stabilization
- window localization：定位回落窗口
- post-short-run schedule：lr 降档 + lambda 再平衡

第二优先级：Branch-specific Calibration
- v2-rgb-only 自己的 threshold 口径

暂缓：full v2 depth，raw RGB densify，回到 legacy

## 8. Reproducibility Assets

脚本：
- prepare_stage1_difix_dataset_s3po_internal.py
- brpo_build_mask_from_internal_cache.py
- build_brpo_v2_signal_from_internal_cache.py
- run_pseudo_refinement_v2.py

文档：DESIGN.md, STATUS.md, CHANGELOG.md

基于 2026-04-16 实验和文档整理。
