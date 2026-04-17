# P2-I：RGB-only v2 gated StageB 回落窗口定位（2026-04-16）

## 1. 目的

在 `P2-H` 已确认当前主候选 `RGB-only v2 + gated_rgb0192` 到 `StageB-120iter` 会 regression 之后，这一轮不再只问“会不会回落”，而是进一步定位：
1. 当前 gated 主候选的正向窗口到底能保持到哪里；
2. 回落是从一开始就持续变差，还是在某个预算点之后才出现明显 cliff；
3. 这个 cliff 更像是 gating 失效，还是 `StageB` 后段训练动力学本身失稳。

同时，这一轮也服务于用户最新补的 `docs/SPGM_landing_plan_for_part3_BRPO.md`：在决定后续主线是继续做 `StageB` 后段调度，还是转向 SPGM 落地前，先把当前 `StageB` 的真实窗口看清楚。

## 2. 协议

### 2.1 固定输入
- internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- after_opt baseline ply:
  - `<internal_cache_root>/after_opt/point_cloud/point_cloud.ply`
- pseudo cache:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- signal_v2 root:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- real branch:
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`
- StageA.5 handoff:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80`

### 2.2 共同 StageB 参数
- `signal_pipeline=brpo_v2`
- `stage_mode=stageB`
- `stageA_iters=0`
- `num_pseudo_views=4`
- `num_real_views=2`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `stageA_lambda_depth_seed=1.0`
- `stageA_lambda_depth_dense=0.35`
- `stageA_lambda_depth_fallback=0.0`
- `stageA_lambda_abs_t=3.0`
- `stageA_lambda_abs_r=0.1`
- `stageA_abs_pose_robust=charbonnier`
- `stageA_abs_pose_scale_source=render_depth_trainmask_median`
- `stageA5_trainable_params=xyz`
- `stageA5_lr_xyz=1e-4`
- `seed=0`
- gating 固定为：
  - `pseudo_local_gating=hard_visible_union_signal`
  - `pseudo_local_gating_params=xyz`
  - `pseudo_local_gating_min_verified_ratio=0.01`
  - `pseudo_local_gating_min_rgb_mask_ratio=0.0192`
  - `pseudo_local_gating_max_fallback_ratio=0.995`

### 2.3 窗口预算
- `stageB_iters = {20, 40, 80, 120}`

### 2.4 输出路径
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2i_stageB_window_localization_e1`

## 3. replay 结果

### 3.1 锚点
- after_opt baseline:
  - PSNR = `23.948911476135255`
  - SSIM = `0.8734854695973573`
  - LPIPS = `0.0787797965799217`
- StageA.5 start (`P2-F gated_rgb0192`):
  - PSNR = `23.924801070601852`
  - SSIM = `0.8723195815527881`
  - LPIPS = `0.07993138824348096`

### 3.2 budget sweep

| budget | PSNR | SSIM | LPIPS | ΔPSNR vs after_opt | ΔPSNR vs StageA.5 | rejection iters | grad_keep_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | `24.01060678693983` | `0.8730183219468152` | `0.08036298079899064` | `+0.06169531080457702` | `+0.08580571633797973` | `18` | `0.7363207012414932` |
| 40 | `24.022349823845758` | `0.8725288130618908` | `0.08122208812446506` | `+0.0734383477105034` | `+0.0975487532439061` | `35` | `0.7354410834610462` |
| 80 | `24.019801464787236` | `0.8717923497712171` | `0.0824685257204153` | `+0.07088998865198093` | `+0.09500039418538364` | `66` | `0.7388732429593802` |
| 120 | `23.914431981687194` | `0.8687328830913261` | `0.08483810117123304` | `-0.03447949444806042` | `-0.01036908891465771` | `96` | `0.7290211913486322` |

### 3.3 直接观察

1. 不是只有 `20iter` 才有效：
   - PSNR 在 `20 / 40 / 80` 都高于 `after_opt` baseline，也高于 `StageA.5` 起点；
   - 其中 `40iter` 是本轮 PSNR 最好点。
2. 但 improvement 并不均匀：
   - SSIM 最好点在 `20iter`；
   - LPIPS 也是 `20iter` 最好，之后单调变差；
   - 到 `80iter` 时，虽然 PSNR 仍高，但 SSIM 已低于 `StageA.5` 起点。
3. 真正的 cliff 出现在 `80 -> 120`：
   - PSNR `-0.10537`
   - SSIM `-0.00306`
   - LPIPS `+0.00237`
   - `120iter` 时三项都已劣于 `StageA.5` 起点，且也低于 `after_opt` baseline。

## 4. 机制层分析

### 4.1 gating 仍然持续工作

四个 budget 的被拒 pseudo id 都稳定是 `225 / 260`；说明这一轮并不存在“前面 gate 生效，后面突然彻底 no-op”的问题。

- `20iter`: `18` 次 rejection，`grad_keep_mean≈0.7363`
- `40iter`: `35` 次 rejection，`grad_keep_mean≈0.7354`
- `80iter`: `66` 次 rejection，`grad_keep_mean≈0.7389`
- `120iter`: `96` 次 rejection，`grad_keep_mean≈0.7290`

所以：
- `StageB` 的后段 regression 不是因为 gate 失效；
- 即使在 `120iter`，pseudo-side 更新范围仍在被持续裁剪。

### 4.2 120 的 cliff 更像后段训练动力学失稳

`120iter` 这一臂最关键的信号不是 replay 本身，而是 loss trajectory：
- `80iter` 时：
  - `loss_total≈0.04915`
  - `loss_real≈0.02773`
- `100iter` 时：
  - `loss_total≈0.13386`
  - `loss_real≈0.11178`
- `120iter` 时：
  - `loss_total≈0.14524`
  - `loss_real≈0.11909`

也就是说：
- `20 -> 80` 期间，real loss 是一路往下掉的；
- 到 `80 -> 120`，real loss 明显反弹；
- 这和 replay cliff 是同步出现的。

因此当前更像是：
- `StageB` 在 `80` 左右之前仍处于一个可接受窗口；
- 真正的问题是后段 schedule / objective balance 没稳住，而不是“这条线从一开始就是错的”。

## 5. 当前判断

1. `StageB` 不是完全没价值：当前 gated 主候选的正向窗口至少延伸到了 `40`，在 PSNR 口径下甚至能延伸到 `80`；
2. 但它也不是可以继续无脑拉长 budget 的稳定主线：`120` 已经出现明确 cliff；
3. 由于 gating 持续工作、而 regression 仍然发生，所以后续如果继续只靠 view-level gate，很可能只能缓解，不能根治；
4. 这正好和用户新补的 SPGM 文档形成呼应：当前更缺的不是更多 view filter，而是更强的 per-Gaussian update management。

## 6. 下一步建议

### 6.1 对 StageB 自身

`StageB` 仍值得保留一轮**有边界**的稳定化优化，但不值得再做开放式长跑调参。

最合理的一轮 follow-up 是：
1. 以 `40 / 80` 为锚点；
2. 只改后段调度，不改 signal 语义；
3. 第一批变量优先：
   - `40` 或 `80` 之后的 `xyz lr` 降档；
   - `lambda_real : lambda_pseudo` 再平衡；
   - 必要时把 early-stop 作为 fallback baseline。

### 6.2 对主工程方向

如果要决定“继续优化 StageB，还是转向 SPGM”，当前更合理的结论是：
- `StageB` 还值得做一轮小而硬的 schedule stabilization compare；
- 但主工程重心应开始转向 `SPGM` 第一版落地，而不是继续把大量时间耗在无边界的 StageB 长跑调参上。

更具体地说：
1. 短期：补一轮 bounded `StageB stabilization` compare，验证 cliff 能不能用后段调度压住；
2. 主线：把 `docs/SPGM_landing_plan_for_part3_BRPO.md` 当作活跃落地文档，开始准备 `StageA.5 + xyz-only + deterministic keep` 的 SPGM 第一版实现；
3. 若 bounded stabilization 仍然压不住 `80 -> 120` 的 cliff，则把 `StageB` 降级成 early-stop / short-budget 分支，主线正式转向 SPGM。
