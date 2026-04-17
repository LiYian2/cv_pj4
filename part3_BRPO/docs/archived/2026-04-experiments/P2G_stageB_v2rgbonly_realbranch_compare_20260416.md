# P2-G：signal-aware-8 固定下的 RGB-only v2 StageB real-branch short compare（2026-04-16）

## 1. 目的

在 `P2-F` 已确认 `RGB-only v2` 在 `StageA.5` 上明显优于 legacy 之后，这一轮继续做固定 `signal-aware-8` 的 `StageB` real-branch short compare，回答两个问题：
1. `RGB-only v2` 的优势能不能延续到 joint refine（pseudo + real branch）？
2. branch-specific gating（`min_rgb_mask_ratio=0.0192`）在 `StageB` 里是否仍然有正向价值？

## 2. 协议与输入

### 固定输入
- signal-aware-8 pseudo set：沿用 canonical E1 winner（`[23, 57, 92, 127, 162, 196, 225, 260]`）对应的 pseudo cache，不扩 pseudo 数量
- signal_v2 root:
  - `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2`
- real branch:
  - `train_manifest=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
  - `train_rgb_dir=/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`
- replay internal cache root:
  - `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`

### 共同 StageB 参数
- `signal_pipeline=brpo_v2`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=20`
- `num_pseudo_views=4`
- `num_real_views=2`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `stageA_rgb_mask_mode=brpo_v2_raw`
- `stageA_depth_mask_mode=train_mask`
- `stageA_target_depth_mode=target_depth_for_refine_v2`
- `stageA_depth_loss_mode=source_aware`
- `stageA_lambda_abs_t=3.0`
- `stageA_lambda_abs_r=0.1`
- `StageA5 trainable params = xyz`

### 输出路径
- `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2g_stageB_v2rgbonly_realbranch_compare_e1`

## 3. 对照臂

这次不是从同一个 `StageA.5` 输出起跑做纯 StageB 局部 ablation，而是用两条自然分支分别进入 StageB：

1. `stageB_from_stageA5_v2rgbonly_ungated_20`
   - 起点：`P2-F` 的 `stageA5_v2rgbonly_xyz_ungated_80`
2. `stageB_from_stageA5_v2rgbonly_gated_rgb0192_20`
   - 起点：`P2-F` 的 `stageA5_v2rgbonly_xyz_gated_rgb0192_80`

这样做回答的是：在自然的 `StageA -> StageA.5 -> StageB` 流程下，当前主候选分支能否把优势带进 joint refine。

## 4. replay 结果

part2 after_opt baseline（internal cache 自带）：
- PSNR = `23.94890858685529`
- SSIM = `0.8734854221343994`
- LPIPS = `0.0787798319839769`

### Arm-U：v2 StageB ungated
- replay PSNR = `23.994533609460902`
- replay SSIM = `0.8728713633837524`
- replay LPIPS = `0.0803846628025726`
- delta vs after_opt:
  - PSNR `+0.04562502260561274`
  - SSIM `-0.0006140587506470174`
  - LPIPS `+0.0016048308185957066`

### Arm-G：v2 StageB gated_rgb0192
- replay PSNR = `24.01060622886375`
- replay SSIM = `0.8730179640981887`
- replay LPIPS = `0.08036303558835277`
- delta vs after_opt:
  - PSNR `+0.06169764200846117`
  - SSIM `-0.0004674580362107328`
  - LPIPS `+0.001583203604375874`

### gated vs ungated
- PSNR: gated `+0.01607261940284843`
- SSIM: gated `+0.00014660071443628464`
- LPIPS: gated `-2.1627214219832602e-05`

结论：
- 在 `StageB` real-branch 下，`RGB-only v2` 的 branch-specific gating 不只是“没坏”，而是比 ungated 明显更好；
- 提升幅度已经比 `StageA.5` 上更可见，不再只是 1e-3 量级的轻微抖动。

## 5. 和 legacy StageB 的关系

对比之前 legacy StageB 参考臂：

- `v2 StageB ungated - legacy StageB ungated`:
  - PSNR `+0.09168190426296619`
  - SSIM `+0.0019863784313202126`
  - LPIPS `-0.0013723662330044628`

- `v2 StageB gated - legacy StageB gated`:
  - PSNR `+0.10779952296504192`
  - SSIM `+0.0021335381048697144`
  - LPIPS `-0.001394561360831617`

这说明：
- `RGB-only v2` 的优势并没有在 StageB 里被吃掉；
- 相反，它在带 real branch 的 joint refine 下仍然明显优于 legacy；
- 因而这条支路现在已经可以从“值得保留的 probe”升级成“当前主候选 refine 分支”。

## 6. real branch / gating / pose

### real branch 是否被误伤
- `loss_real_last`：
  - ungated `0.14847081899642944`
  - gated `0.1484547257423401`

几乎完全一致，说明 pseudo-side gating 没有明显伤到 real branch 的 global anchor 功能。

### gating 是否真的在工作
- ungated：`iters_with_rejection = 0 / 20`
- gated：`iters_with_rejection = 18 / 20`
- gated 总拒绝：`22 / 80` sample eval
- 被拒 pseudo：`225`（13 次）、`260`（9 次）
- 原因全是：`rgb_mask_ratio`

并且：
- `grad_keep_ratio_xyz_mean`
  - ungated `1.0`
  - gated `0.7363207012414932`

说明这次 StageB gating 不是 no-op，而是确实在按 `RGB-only v2` 的支路语义过滤 pseudo-side 高斯更新。

### pose 侧是否变坏
最终 pose aggregate：
- ungated mean trans `0.0004997`, mean rotF `0.0025288`
- gated mean trans `0.0004743`, mean rotF `0.0022010`

gated 甚至略好，至少没有出现“为了 map 指标把 pose 稳定性明显换掉”的证据。

## 7. 当前判断

1. `RGB-only v2` 已经通过了 `StageB` real-branch short compare，不再只是 `StageA.5` 局部好看；
2. branch-specific gating（`min_rgb_mask_ratio=0.0192`）在 `StageB` 里也保留了正增益；
3. real branch 没有被明显误伤；
4. 因而当前更合理的工程结论是：
   - 把 `RGB-only v2 + gated_rgb0192` 提升为当前主候选 refine 分支；
   - 后续如果继续验证，优先做更长一点的 `StageB` 或更完整 schedule verify；
   - 暂不把主要精力转去 raw RGB densify。

## 8. 现在是否要做 RGB densify

这轮 `StageB` 结果进一步强化了前一轮判断：**现在还不该直接做 raw RGB densify。**

因为：
- 即使 raw RGB support 图看起来点状，`RGB-only v2 + gated_rgb0192` 依然已经在 `StageA.5` 和 `StageB` 两层都明显优于 legacy；
- 所以当前主问题不是“raw RGB 太稀所以必须先把它铺开”，而是“在这个更干净的支路上，应该如何做更长 schedule / 更完整 joint verify”；
- 若后续真的要扩张，更合理的下一步仍然是：受几何约束的 support/depth expand，而不是直接把 `raw_rgb_confidence_v2` 做 morphology/diffusion 式 densify。

## 9. 下一步建议

当前最合理的下一步是：
1. 以 `RGB-only v2 + gated_rgb0192` 为当前主候选；
2. 做一轮更长一点的 `StageB` verify（或更接近完整 schedule 的中等预算验证），确认这个短跑优势不是偶然；
3. 只有当更长验证显示 coverage 仍是主瓶颈时，再讨论受几何约束的 support/depth expand；
4. 暂不直接做 raw RGB densify。
