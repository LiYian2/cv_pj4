# Noah.md - SPGM v1 Progress Tracker

> 2026-04-16 23:27 CST
> 当前用途：压缩后快速恢复 P2-K 的真实状态。这里记录的是“现在可信的结论”，不是乐观版阶段完成口径。

## 1. 当前可信结论
1. SPGM v1 的 wiring 已经接通：`stats.py / score.py / policy.py` 都真实接到 `maybe_apply_pseudo_local_gating()` 里了。
2. 它不是 no-op：StageA.5 / StageB / 120iter 中都能看到 grad 被真实调制。
3. 但当前 `20260416_spgm_v1_*` compare 不是对 canonical `post40_lr03_120 / gated_rgb0192` baseline 的 apples-to-apples 对照。
4. 当前 120iter replay 结果确实为负：baseline `24.136 / 0.87385 / 0.08189`，exp `23.594 / 0.85926 / 0.08912`。
5. 这轮负结果当前更应解释成：`protocol drift + deterministic keep 压制过强`，而不是“SPGM 已完成验证”。

## 2. 当前 run identity
### 代码版本
- commit: `6ad7bfd`

### 关键目录
- StageA.5 compare: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageA5_compare_e1/`
- StageB 40iter compare: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_e1/`
- StageB 120iter + replay: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_spgm_v1_stageB_compare_120iter/`
- forensic audit doc: `docs/P2K_spgm_v1_forensic_audit_20260416.md`

### 当前真实输入
- PLY: `after_opt/point_cloud/point_cloud.ply`
- pseudo_cache: `20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2: `20260414_signal_enhancement_e15_compare/signal_v2`
- pseudo ids: `23 / 57 / 92 / 127 / 162 / 196 / 225 / 260`

## 3. 当前确认的关键问题
### 3.1 实验 protocol 漂移
当前 StageB-120 compare 实际使用：baseline `hard_visible_union_signal`、exp `spgm_keep`、`stageB_post_switch_iter=0`、没有 `init_pseudo_camera_states_json`。所以它不是当前项目主参考臂 `post40_lr03_120 / gated_rgb0192` 的正式对照。

### 3.2 当前 SPGM 更像“强压制”而不是“更优选择”
120iter mean：baseline `grad_keep_ratio_xyz ≈ 0.739`，exp `spgm_active_ratio ≈ 0.739`；baseline `grad_weight_mean_xyz ≈ 0.739`，exp `grad_weight_mean_xyz ≈ 0.292`。也就是说 active set 基本没变，变的是权重被明显压低了。

### 3.3 已确认的小实现坑
- `score.py` 的 density entropy 归一化标尺不对（按 `B` 而不是 `log(B)`）
- `spgm_density_mode` 目前是 dead param
- SPGM accepted-count/history 统计写法对未来 reject-run 不严谨

## 4. 下一步不要做什么
- 不要直接把这轮结果写成“SPGM 已完成验证”
- 不要直接跳 stochastic drop / xyz+opacity / 更长 iter
- 不要把当前 raw-120 baseline 当成 canonical project baseline

## 5. 下一步应该做什么
1. 先读 `docs/P2K_spgm_v1_forensic_audit_20260416.md`
2. 把 compare protocol 拉回 canonical `post40_lr03_120 / gated_rgb0192`
3. 先测 conservative deterministic SPGM：`cluster_keep=(1.0,1.0,1.0)` 或 `(1.0,1.0,0.9)`，更高 `weight_floor`，更弱 `support_eta`
4. 先修 3 个小实现坑，再继续扫参

## 6. 给下一个 agent 的一句话
这轮 P2-K 说明的是“SPGM 路接通了，但 compare 漂了，而且当前 deterministic keep 压制过强”。先做 forensic repair，再谈是否推进 SPGM。
