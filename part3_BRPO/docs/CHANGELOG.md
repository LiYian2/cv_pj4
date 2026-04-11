# CHANGELOG.md - Part3 Stage1 过程记录

> 本文件记录每次工作的过程、发现和结果。
> 更新规则：以日期为单位记录；允许为修正文档一致性而整理旧条目，但不要把“现状”写进本文件。

---

## 2026-04-12

### Mask problem route：M1 fused-first verification + compatibility layer

根据 `CURRENT_MASK_PROBLEM.md` 与 `SOLVE_MASK_PROBLEM.md` 的判断，开始把主线从“继续扩 Stage A”转到“先解决 upstream mask problem”。

这一轮的重点不是 refine，而是 verification / pack / consumer 之间的 upstream 结构修正。

代码修改：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `docs/SOLVE_MASK_PROBLEM.md`

关键实现：
- `verification_mode=branch_first|fused_first` 双模式落地；
- `fused_first` 模式可直接消费 `fusion/samples/<id>/target_rgb_fused.png`；
- verification 输出新增 `seed_support_left/right/both/single`；
- 同时保留 `support_*` 与 `confidence_mask_brpo_*` 兼容层，不污染旧实验与现有 consumer；
- `pack` 已能在 `verification_mode=fused_first` 下把 `seed_support_*` 带入 sample。

真实检查：
- `fused_first` 已在当前 3-frame prototype 上跑通；
- 但与旧 `branch_first` 相比，coverage 改善并不明显；
- 这验证了：**单纯改 verification 顺序还不够，M2 的 propagation 仍然必要。**

### Mask problem route：M2 train mask propagation 第一轮实现

在 M1 之后，开始做 `seed_support → train_confidence_mask` 的第一轮工程化传播。

新增文件：
- `pseudo_branch/brpo_train_mask.py`

关键实现：
- verify 阶段新增 `train_mask_mode=propagate`；
- 新增：
  - `train_confidence_mask_brpo_{left,right,fused}`
  - `train_support_{left,right,both,single}`
- 当前训练真正消费的 mask 通过 alias 保持兼容：
  - `confidence_mask_brpo_*` → 指向 train mask
  - `support_*` → 指向 seed support
- propagation 仍保留在 verify/pack upstream，不塞回 refine。

中途问题与修复：
- 第一版 M2 因重复落盘（seed/train/alias 全部物理保存）触发 `No space left on device`；
- 随后把 alias 改成 symlink，保留兼容的同时避免重复写大文件；
- 清掉失败的中间 verify 目录后重跑，验证通过。

真实结果：
- 机制成立，coverage 从 `~1%` 量级显著抬升；
- 但默认参数明显过宽：
  - frame 10: `~75.1%`
  - frame 50: `~68.0%`
  - frame 120: `~68.9%`
- 说明 propagation 机制是对的，但默认超参不能直接作为最终 training mask 定义。

### Mask problem route：M2.5 propagation 合理区间小范围研究

为了判断当前 propagation 的“合理区间”，没有继续大规模重跑 verify，而是基于现有 `fused_first + seed_support` 结果做了离线小范围 sweep。

目的：
- 不是寻找唯一最优超参；
- 而是先判断：哪些组合明显太稀、哪些组合明显太稠、哪些组合更像训练可用区间。

代表性结果：
- `radius=1, tau_rel_depth=0.01, tau_rgb_l1=0.03` → `avg_fused_nonzero ≈ 8.2%`
- `radius=1, tau_rel_depth=0.02, tau_rgb_l1=0.03` → `≈ 9.7%`
- `radius=2, tau_rel_depth=0.01, tau_rgb_l1=0.05` → `≈ 19.4%`
- `radius=2, tau_rel_depth=0.02, tau_rgb_l1=0.05` → `≈ 25.1%`
- `radius=3, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 52.3%`
- `radius=5, tau_rel_depth=0.03, tau_rgb_l1=0.08` → `≈ 70.7%`

阶段判断：
- 当前 propagation 的合理研究区间明显更接近 `10% ~ 25%` coverage；
- 因此后续不应继续使用当前过宽默认值，而应优先围绕：
  - `radius ∈ {1,2}`
  - `tau_rel_depth ∈ [0.01, 0.02]`
  - `tau_rgb_l1 ∈ [0.03, 0.05]`
  做进一步收紧。

### Mask problem route：M3 blended `target_depth_for_refine` 落地

在 M2.5 之后，正式进入 M3：把 upstream depth target 从“render-depth fallback 占位符”升级成可审计的 blended target。

新增文件：
- `pseudo_branch/brpo_depth_target.py`

代码修改：
- `pseudo_branch/brpo_reprojection_verify.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

关键实现：
- verification 侧新增 pseudo-view sparse verified depth 输出：
  - `projected_depth_left.npy`
  - `projected_depth_right.npy`
  - `projected_depth_valid_left.npy`
  - `projected_depth_valid_right.npy`
- pack 阶段正式构造：
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`
  - `verified_depth_mask.npy`
- `target_depth_for_refine` 当前定义为：
  - `both` → 双边 verified depth average
  - `left/right` → 单边 verified depth
  - unsupported → `render_depth` fallback

真实 smoke：
- 新开 prototype 目录：
  - `.../internal_prepare/re10k1__internal_afteropt__brpo_proto_v4_stage3/`
- 在 3-frame（10/50/120）上重跑 `verify → pack` 成功；
- sample 层已真实出现：
  - `projected_depth_*`
  - `target_depth_for_refine.npy`
  - `target_depth_for_refine_source_map.npy`
  - `target_depth_for_refine_meta.json`

代表性结果：
- frame 10 verified depth ratio：`~1.63%`
- frame 50 verified depth ratio：`~1.51%`
- frame 120 verified depth ratio：`~1.55%`
- 其余区域为 `render_depth` fallback

阶段判断：
- M3 已完成 upstream depth target 的真实落地；
- 当前 `target_depth_for_refine.npy` 已不再只是 schema 预留位；
- 但当前 verified depth 覆盖偏保守，更像 sparse correction source，而不是大覆盖 supervision 主体。

提交：
- `c7965b4` — `part3: implement Stage M3 blended target_depth_for_refine pipeline`

### Mask problem route：M4 Stage A consumer 显式接入新 target

在 M3 之后，继续做 M4：让 `run_pseudo_refinement_v2.py` 明确消费新的 train mask 与 blended depth target，而不是继续依赖含糊 fallback。

代码修改：
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- Stage A 现在显式支持：
  - `stageA_mask_mode=train_mask|seed_support_only|legacy|auto`
  - `stageA_target_depth_mode=blended_depth|render_depth_only|target_depth_for_refine|target_depth|render_depth|auto`
- history 现在会显式记录：
  - 实际 confidence source / path
  - 实际 depth source / path
  - effective mask/depth mode
  - mask coverage
  - verified depth ratio / render fallback ratio
- 通过 sample 内显式文件读取 `train_confidence_mask_brpo_*` 与 `target_depth_for_refine.npy`，不再让 consumer 靠隐式默认值漂移。

真实 smoke：
- 目录：
  - `.../2026-04-12_m4_stageA_smoke/blended_depth/`
  - `.../2026-04-12_m4_stageA_smoke/render_depth_only/`
- `train_mask + blended_depth`：
  - mean mask coverage `~19.4%`
  - mean verified depth ratio `~1.56%`
  - `loss_depth ≈ 0.00478`（非零）
- `train_mask + render_depth_only`：
  - `loss_depth ≈ 0`
- 额外 debug quick smoke：
  - `seed_support_only + blended_depth` 2 iter 可跑，mean mask coverage `~1.56%`

阶段判断：
- M4 consumer 已真实接通；
- `blended_depth` 与 `render_depth_only` 在 Stage A 中已能被区分；
- 这说明 M3→M4 链路不是“名义接通”，而是 depth signal 已实际进入 loss。

提交：
- `63abe22` — `part3: wire Stage A to explicit M4 mask and depth modes`

### Mask problem route：M4.5 Stage A 长一点对照（blended vs render-only）

在 M4 smoke 之后，继续做一轮更长一点的 Stage A eval，目的不是追最终指标，而是回答：
- depth signal 是否真的被消费；
- 当前 depth signal 在更长一点的 Stage A 中是否会动起来；
- `blended_depth` 与 `render_depth_only` 的差异是否只停留在 smoke 级别。

实验目录：
- `.../2026-04-12_m45_stageA_eval/blended_depth_long/`
- `.../2026-04-12_m45_stageA_eval/render_depth_only_long/`
- `.../2026-04-12_m45_stageA_eval/analysis/summary.json`
- `.../2026-04-12_m45_stageA_eval/analysis/compare.txt`

实验口径：
- `target_side=fused`
- `confidence_mask_source=brpo`
- `stageA_mask_mode=train_mask`
- `num_pseudo_views=3`
- `stageA_iters=300`
- 唯一变量：
  - `stageA_target_depth_mode=blended_depth`
  - vs `stageA_target_depth_mode=render_depth_only`

结果：
- 两组 `loss_rgb` 基本一致：
  - `0.00915 -> 0.00734`
- `blended_depth_long`：
  - `loss_depth ≈ 0.004783`（非零）
  - `loss_total first -> last: 0.00784 -> 0.02269`
- `render_depth_only_long`：
  - `loss_depth ≈ 4.4e-07 ≈ 0`
  - `loss_total first -> last: 0.00641 -> 0.02207`
- 当前 `pose_reg` 会随着迭代持续增长，两组量级接近；
- 当前 `blended_depth` 虽然已非零，但在 300 iter 中基本不下降。

阶段判断：
- `blended_depth` 与 `render_depth_only` 的差异已经真实存在；
- 但当前差异主要体现在“depth loss 是否非零”，而不是“depth 是否明显驱动了更好的优化轨迹”；
- 更准确的结论是：

**depth signal 已接入，但当前 Stage A 对它的利用仍偏弱。**

因此下一步应进入 **M4.6 / depth-flatness diagnosis**，而不是直接跳到 Stage B。

### 文档恢复后的同步补写

由于误删并恢复了 `docs/`，本轮重新按写作规范同步补回：
- `docs/STATUS.md`
- `docs/DESIGN.md`
- `docs/CHANGELOG.md`

补回原则：
- `STATUS.md` 只写当前状态，不追加过程细节；
- `DESIGN.md` 只写当前设计判断与默认实验方案；
- `CHANGELOG.md` 记录 M1 / M2 / M2.5 / M3 / M4 / M4.5 的真实过程与结果；
- 避免把“现状”和“历史过程”混写。

## 2026-04-11

### Internal cache / replay：Phase 1 与 Phase 2 跑通

在 Re10k-1 的 80-frame smoke rerun 上，完成了 internal cache 导出与 replay 机制验证。

新增/修改：
- `third_party/S3PO-GS/utils/internal_eval_utils.py`
- `third_party/S3PO-GS/slam.py`（opt-in internal cache 导出）
- `part3_BRPO/scripts/replay_internal_eval.py`

导出结果结构：
- `internal_eval_cache/manifest.json`
- `internal_eval_cache/camera_states.json`
- `internal_eval_cache/before_opt/`
- `internal_eval_cache/after_opt/`

关键确认：
- `color_refinement()` 只优化 gaussians，不更新 pose；
- 因此当前 cache 设计改为：**camera states 共享一份，before/after 各自保存自己的 PLY 与 render cache**。

same-ply consistency check：
- after_opt：internal `31.4016 / 0.95257 / 0.04713` vs replay `31.3581 / 0.95206 / 0.04761`
- before_opt：internal `23.7945 / 0.88950 / 0.12889` vs replay `23.7256 / 0.88842 / 0.13007`

结论：replay 与官方 internal eval 已足够接近，可以用于后续 baseline/refined 公平比较。

### Internal cache / replay：Phase 3 smoke 协议验证

使用同一份 Re10k-1 smoke internal cache，对比：
- baseline sparse PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_sparse/Re10k-1_part2_s3po/2026-04-04-00-43-29/point_cloud/final/point_cloud.ply`
- current E refined PLY：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/E_joint_realdensify_freezegeom_lambda0p5_tertile/refined.ply`

after_opt replay：
- baseline sparse：`21.3486 / 0.79084 / 0.10690`
- E refined：`19.0380 / 0.69438 / 0.23764`

delta（E - baseline）：
- `PSNR -2.3106`
- `SSIM -0.09646`
- `LPIPS +0.13073`

结论：在 smoke internal protocol 下，current E refined PLY 明显劣于 baseline sparse PLY。
但这仍是 **80-frame smoke**，还不是 full rerun 的最终正式结论。

### BRPO confidence mask：Phase B / Phase C 原型

基于 full internal cache：
`/home/bzhang512/my_storage_500G/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache/`

新增文件：
- `part3_BRPO/pseudo_branch/brpo_reprojection_verify.py`
- `part3_BRPO/pseudo_branch/brpo_confidence_mask.py`
- `part3_BRPO/scripts/brpo_verify_single_branch.py`
- `part3_BRPO/scripts/brpo_build_mask_from_internal_cache.py`

工程决策：
- BRPO 新线与旧 EDP 线隔离，不复用 `epipolar_depth.py` / `build_pseudo_cache.py`；
- verification 采用 **方案 B**：按需用 `ref pose + stage PLY` 现渲染 KF depth；
- 当前阶段不改 refine loss，只先跑 verified support / confidence mask。

Phase B（left 单分支）在 `after_opt` 的 3 个样本帧上跑通：
- 平均 `support_ratio_vs_matches = 0.8687`
- 平均 `mean_reproj_error = 1.1048 px`
- 平均 `mean_rel_depth_error = 0.0196`

Phase C（left + right + mask fusion）在同 3 个样本帧上跑通：
- 平均 `support_ratio_left = 0.01352`
- 平均 `support_ratio_right = 0.01003`
- 平均 `support_ratio_both = 0.00803`
- 平均 `support_ratio_single = 0.00749`
- 平均 `mean_reproj_error_left = 1.1048 px`
- 平均 `mean_reproj_error_right = 1.1354 px`
- 平均 `mean_rel_depth_error_left = 0.0196`
- 平均 `mean_rel_depth_error_right = 0.0241`

说明：
- verified support 没有出现“几乎全空”的坏现象；
- 双边交集 `support_both` 能稳定落到可见结构区域；
- 这说明 `internal cache + on-demand ref depth render + bidirectional reprojection verification` 这条新链已具备进入 mask-only replacement ablation 的条件。

### Internal prepare：Phase 3 正式并入 `select → difix → verify → pack`

在 full internal cache 上，把 internal prepare 正式扩成：

```text
select → difix → verify → pack
```

关键实现：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py` 增加 `verify` stage；
- `scripts/brpo_build_mask_from_internal_cache.py` 支持 `--pseudo-left-root / --pseudo-right-root`；
- verification 可以直接消费 `difix/left_fixed` 与 `difix/right_fixed`；
- `pack` 阶段可自动把 verification 产物接入 `pseudo_cache/samples/<id>/`。

关键 prototype run：
- `re10k1__internal_afteropt__difix_proto_v2`
- `re10k1__internal_afteropt__brpo_proto_v3`

当前 3-frame prototype 使用显式 frame ids：
- `10`
- `50`
- `120`

说明：
- 这轮不是自动 `midpoint / tertile` 选点，而是显式 `frame_ids`；
- `Difix left/right`、`BRPO verification` 与 `pseudo_cache` 已在同一条 internal prepare 流程内打通。

### Phase 4：mask-only ablation（legacy vs brpo）

在同一份 after_opt PLY、同一份 3-frame internal pseudo_cache、同一份 v1 fixed-pose RGB-only refine 配置下，完成：
- `confidence_mask_source=legacy`
- `confidence_mask_source=brpo`

两组 refine 输出：
- legacy：`.../2026-04-11_mask_only_ablation_proto/legacy_left/refined.ply`
- brpo：`.../2026-04-11_mask_only_ablation_proto/brpo_left/refined.ply`

训练内现象：
- `brpo` 的 pseudo loss 下降更快、最终更低；
- 但 replay 结果并未更好。

replay eval（same internal camera states, after_opt）结果：
- baseline internal after_opt：`23.9489 / 0.87349 / 0.07878`
- legacy refine：`22.3862 / 0.84365 / 0.10834`
- brpo refine：`21.9324 / 0.82049 / 0.13134`

结论：
- 当前 BRPO verification 链条**能跑通**；
- 但当前 BRPO mask 在 `v1 fixed-pose RGB-only refine` 下**不优于 legacy mask**；
- 更可能的问题是：当前 mask 语义与当前 refine 消费方式不匹配，而不是 verification 完全失效。

### Phase 5：auditability / provenance 修补

根据 Phase 4 结果，继续检查了 `verify → pack → refine` 的 provenance 是否完整可审计。

发现的问题：
- verification 之前虽然已经支持消费 Difix left/right repaired 图，但 `verification_meta.json` 的 provenance 记录不够完整；
- `prepare_stage1_difix_dataset_s3po_internal.py` 的 sample manifest 对 `.json` 类型 BRPO 产物路径命名不够规范；
- `run_pseudo_refinement.py` 的 history 之前没有写出每个 sample 实际吃到的 target / confidence 来源。

对应补丁：
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/run_pseudo_refinement.py`

补丁后能力：
- verification 会记录 branch 实际输入、override 状态、policy 与 matcher/source 信息；
- pack 会记录更完整的 `source_meta` 与 `verification_*_path`；
- refine history 会记录每个 pseudo sample 的 `target_rgb_path / confidence_path / confidence_source_kind / coverage`。

说明：
- 这一步当前只完成了**代码与编译验证**；
- 若要让产物也带上新 provenance，需要在后续 Phase 6/7 中重跑 verify / pack / refine。

### Phase 6：schema solidification 第一轮实现与真实 smoke

本轮开始正式进入 Phase 6，不再停留在文档规划。

代码修改：
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/run_pseudo_refinement.py`

关键实现：
- `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `fusion` stage，`all` 流程改为 `select → difix → fusion → verify → pack`；
- `fusion` 直接复用 `pseudo_fusion.py`，先实现 **rgb-only fused**，产出 `target_rgb_fused.png`、`confidence_mask_fused.npy/png`、`fusion_meta.json`；
- `pack` 阶段新增 `target_depth_for_refine.npy` 预留位，并将 fused 产物接入 `pseudo_cache/samples/<id>/`；
- `brpo_confidence_mask.py` 从单一 `confidence_mask_brpo.npy` 扩展为：
  - `confidence_mask_brpo_left.npy/png`
  - `confidence_mask_brpo_right.npy/png`
  - `confidence_mask_brpo_fused.npy/png`
  同时保留旧的 `confidence_mask_brpo.npy/png` 作为兼容 alias；
- `run_pseudo_refinement.py` 的 BRPO consumer 改为按 `target_side` 解析 confidence。

真实验证：
- 在现有 prototype `re10k1__internal_afteropt__brpo_proto_v3` 上，实际重跑了：
  `fusion → verify → pack`
- `verify` 目录中已真实产出新的 `left/right/fused` BRPO masks；
- `pack` 初次验证时发现一个真实漏链：verify 已产出新 masks，但 `maybe_link_brpo_artifacts()` 仍按旧白名单链接，导致 sample 里没有带入新字段；
- 随后补上 pack linkfix，并只重跑 `pack`，确认 `pseudo_cache/samples/10/` 中已真实出现：
  - `target_rgb_fused.png`
  - `target_depth_for_refine.npy`
  - `confidence_mask_brpo_left/right/fused.npy`

consumer smoke：
- 直接调用 `load_pseudo_viewpoints(...)`，确认以下三种模式都能正确解析本地 sample 内的新文件：
  - `left + brpo`
  - `right + brpo`
  - `fused + brpo`
- 进一步跑通了一次真实 refine smoke：
  - `target_side=fused`
  - `confidence_mask_source=brpo`
  - `num_iterations=2`
  - 成功进入训练循环并正常落盘。
