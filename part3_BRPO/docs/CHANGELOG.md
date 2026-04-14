# CHANGELOG.md - Part3 Stage1 过程记录

> 本文件记录每次工作的过程、发现和结果。
> 更新规则：以日期为单位记录；允许为修正文档一致性而整理旧条目，但不要把“现状”写进本文件。

---

## 2026-04-15

### Fusion：BRPO-style `target ↔ reference overlap confidence` 第一版落地

代码变更：
- `pseudo_branch/pseudo_fusion.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮实现：
1. `pseudo_fusion.py` 不再以旧的 `branch confidence × render-depth gate × global view scalar` 为主语义；
2. 新版 fusion 会基于：
   - pseudo render depth
   - pseudo / left-ref / right-ref camera states
   - left / right reference rendered depth
   计算每一侧的 `target ↔ reference overlap confidence`；
3. 左右分支权重现在由 `overlap_conf_left/right` 直接归一化得到，再对 DiFix 候选图做 residual fusion；
4. `stage_fusion()` 现已真正喂入几何输入，而不是旧的 RGB-only / uniform-confidence placeholder 调用。

新增导出：
- `fusion_weight_left.npy`
- `fusion_weight_right.npy`
- `overlap_conf_left.npy`
- `overlap_conf_right.npy`
- `overlap_mask_left.npy`
- `overlap_mask_right.npy`
- `projected_depth_left.npy`
- `projected_depth_right.npy`
- `ref_depth_left_render.npy`
- `ref_depth_right_render.npy`
- 对应 `diag/` 可视化图

兼容性处理：
- 若调用方没有提供 pseudo/ref states 或 reference depth，新版 `pseudo_fusion.py` 仍保留一个 legacy fallback，避免旧脚本直接崩掉；
- 但 `prepare_stage1_difix_dataset_s3po_internal.py::stage_fusion()` 主路径已切到新的几何版 fusion。

验证：
- 已通过远端 `py_compile`：
  - `pseudo_branch/pseudo_fusion.py`
  - `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- 已完成 `stage_fusion()` 直连 smoke：
  - 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_stage_fusion_direct_smoke/`
- 已完成一次带真实 DiFix 左右图的 direct fusion smoke：
  - 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_fusion_direct_difix_50/`

当前判断：
- 这一步已经和 `docs/BRPO_fusion_mask_spgm_subset_refine.md` 中关于 fusion 的主判断对齐：
  - fused image 的权重定义已从“left ↔ right agreement 的辅助语义”切回“target ↔ reference overlap confidence”主语义；
- 但当前只完成了 BRPO 口径中的 fusion 部分；
- 最终训练 mask 仍不是 `post-fusion RGB / correspondence` 的独立推理结果；
- 因此接下来的主工作不应回到旧 `seed_support -> train_mask -> target_depth` 路径补丁，而应继续完成：
  1. fused RGB mask v2
  2. depth supervision v2
  3. local Gaussian gating / subset refine

### Signal v2：fused RGB mask + isolated depth supervision 第一版落地

代码变更：
- `pseudo_branch/brpo_v2_signal/__init__.py`
- `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
- `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

本轮实现：
1. 新增 `signal_v2` 独立路径，不覆盖旧 `brpo_confidence_mask.py / brpo_train_mask.py / brpo_depth_target.py` 主线；
2. `rgb_mask_inference.py` 现在会直接在 fused RGB 上，与 left/right reference RGB 做 reciprocal correspondence 匹配；
3. 新版 `raw_rgb_confidence_v2` / `raw_rgb_confidence_cont_v2` 来自 fused RGB ↔ reference RGB correspondence，本轮不依赖旧 depth seed / train-mask；
4. `depth_supervision_v2.py` 会基于：
   - `projected_depth_left/right`
   - `fusion_weight_left/right`
   - `raw_rgb_confidence_v2`
   单独生成：
   - `target_depth_for_refine_v2_brpo.npy`
   - `target_depth_source_map_v2_brpo.npy`
   - `depth_supervision_mask_v2_brpo.npy`
5. 以上新产物全部写到 `signal_v2/` 下，不改写 legacy `pseudo_cache/` 样本中的旧文件名。

新增导出：
- `signal_v2/frame_<id>/raw_rgb_confidence_v2.npy`
- `signal_v2/frame_<id>/raw_rgb_confidence_cont_v2.npy`
- `signal_v2/frame_<id>/rgb_support_{left,right,both,single}_v2.npy`
- `signal_v2/frame_<id>/target_depth_for_refine_v2_brpo.npy`
- `signal_v2/frame_<id>/target_depth_source_map_v2_brpo.npy`
- `signal_v2/frame_<id>/depth_supervision_mask_v2_brpo.npy`
- `signal_v2/frame_<id>/rgb_mask_meta_v2.json`
- `signal_v2/frame_<id>/depth_meta_v2_brpo.json`
- 对应 `diag/` 可视化图与 `summary.json`

验证：
- 已通过远端 `py_compile`：
  - `pseudo_branch/brpo_v2_signal/__init__.py`
  - `pseudo_branch/brpo_v2_signal/rgb_mask_inference.py`
  - `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
  - `scripts/build_brpo_v2_signal_from_internal_cache.py`
- 已完成 1-frame smoke：
  - prepare root：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_stage_fusion_direct_smoke/`
  - 输出目录：`.../signal_v2/`
- frame 50 的 smoke 结果：
  - `raw_rgb_confidence_nonzero_ratio ≈ 0.01865`
  - `support_ratio_both ≈ 0.00904`
  - `depth verified_ratio ≈ 0.01865`
  - `mean_abs_rel_correction_verified ≈ 0.03556`

当前判断：
- 这一步和分析文档的 `mask` / `depth` 语义分工是一致的：
  - mask 主语义已回到 fused RGB / correspondence；
  - depth supervision 已独立成服务 refine 的另一条线；
- 当时这个版本还没有接入 `run_pseudo_refinement_v2.py`；
- 同日后续更新见下一小节：refine consumer 已在 `signal_pipeline=brpo_v2` 路径下完成最小接入并跑通 3-frame smoke。

### Signal v2：3-frame coverage eval + refine consumer smoke 接通

代码变更：
- `scripts/run_pseudo_refinement_v2.py`

本轮接入：
1. 新增 `--signal_pipeline {legacy, brpo_v2}`
2. 新增 `--signal_v2_root`
3. RGB mask 已支持 `brpo_v2_raw / brpo_v2_cont`
4. depth mask 已支持 `brpo_v2_depth`
5. depth target 已支持 `brpo_v2 / target_depth_for_refine_v2_brpo`

3-frame coverage eval：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_v2_coverage_tmp/signal_v2/`
- frame `10 / 50 / 120` 平均：
  - `rgb support left ≈ 1.56%`
  - `rgb support right ≈ 1.20%`
  - `rgb support both ≈ 0.93%`
  - `raw_rgb_confidence_nonzero_ratio ≈ 1.82%`
  - `depth verified_ratio ≈ 1.82%`
  - `both_weighted_ratio ≈ 1.40%`
  - `left_only_ratio ≈ 0.42%`
  - `render_fallback_ratio = 0`
  - `mean_abs_rel_correction_verified ≈ 3.15%`
- 解释：
  1. v2 mask 现在确实是一个更“窄但干净”的 fused-RGB correspondence mask；
  2. depth supervision 已不再靠旧 `train_mask` 扩覆盖，而是严格跟着新的 RGB 可信区域和 projected depth 走。

StageA refine smoke：
- 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260415_signal_v2_refine_smoke`
- 关键结果：
  - `signal_pipeline = brpo_v2`
  - first sample RGB mask mode = `brpo_v2_raw`
  - first sample depth mask mode = `brpo_v2_depth`
  - first sample depth kind = `target_depth_for_refine_v2_brpo`
  - `mean_confidence_nonzero_ratio ≈ 1.82%`
  - `mean_target_depth_verified_ratio ≈ 1.82%`
  - StageA `3 iter` 成功跑完，并产出：
    - `stageA_history.json / refinement_history.json`
    - `pseudo_camera_states_{stageA,final}.json`
    - `refined_gaussians.ply`

当前判断：
- 当前状态已经不是“文件存在但没接上”，而是 `fusion + mask v2 + depth supervision v2 + refine consumer` 四段都已打通；
- 但这还只是 wiring / smoke 成功，不等于已经证明这版窄 coverage 优于 legacy；
- 下一步应该直接做一轮 apples-to-apples short compare：`legacy vs signal_v2`，使用同一组 pseudo、同一组 iter、同一组 refine 超参；
- local Gaussian gating 仍是后续必须补的结构项，不应因为 smoke 通过就后移得过远。

## 2026-04-13

### A+B 实施：split+scaled abs prior + 梯度贡献诊断

代码变更：
- `pseudo_branch/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `pseudo_branch/pseudo_camera_state.py`
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/diagnose_stageA_loss_contrib.py`（新增）

关键实现：
- Stage A abs prior 从 legacy 单标量扩展为 split+scaled 形式：
  - `--stageA_lambda_abs_t`
  - `--stageA_lambda_abs_r`
  - `--stageA_abs_pose_robust {charbonnier,huber,l2}`
  - `--stageA_abs_pose_scale_source {...}`
  - `--stageA_abs_pose_fixed_scale`
- history 新增：
  - `loss_abs_pose_trans/rot`
  - `abs_pose_rho_norm/theta_norm`
  - `scene_scale_used`
  - `grad_contrib`
- 新增在线梯度日志：
  - `--stageA_log_grad_contrib`
  - `--stageA_log_grad_interval`

### A 小网格：6组 × 80 iter

扫描网格：
- `lambda_t ∈ {0.3, 1.0, 3.0}`
- `lambda_r ∈ {0.03, 0.1}`

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_small_grid/`

结果摘要：
- `loss_depth_last` 约在 `0.06837 ~ 0.06854`，组间差异小；
- `abs_pose_rho_norm_last` 从 `~7.93e-4`（弱约束）降到 `~1.83e-4`（强约束）；
- 结论：abs prior 对 drift 抑制明确有效，但对 depth 改善不显著。

### top3 深跑：3组 × 300 iter

输入：
- 基于小网格按“drift优先且兼顾depth”筛出的 top3。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_deep300_top3/`

结果摘要：
- `lt3.0_lr0.1`：drift 最小（`rho≈2.19e-4, theta≈1.17e-4`），depth 不优；
- `lt1.0_lr0.1`：depth 末值最低（`~0.06837`），但 drift 明显更大（`rho≈4.67e-4`）；
- 结论延续：目前没有出现“显著压 drift 且显著改善 depth”的统一最优组。


### A.5 midpoint 执行（xyz / xyz+opacity）

实现与数据：
- 新增 `pseudo_branch/gaussian_param_groups.py`（micro 参数组：`xyz` / `xyz+opacity`）；
- `pseudo_branch/pseudo_refine_scheduler.py` 增加 `StageA5Config` 与 `build_stageA5_optimizers()`；
- `scripts/run_pseudo_refinement_v2.py` 增加 `stageA5` 模式及 CLI：
  - `--stage_mode {stageA,stageA5}`
  - `--stageA5_trainable_params {xyz,xyz_opacity}`
  - `--stageA5_lr_xyz --stageA5_lr_opacity`
- midpoint pseudo cache：`re10k1__internal_afteropt__midpoint_proto_v1`，样本帧 `17/51/86/121/156/190/225/260`。

实验（80 iter，num_pseudo_views=8，fused legacy mask）：
- 输出目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint/`
- 汇总：`summary_stageA5_midpoint_80iter.json`
- 结果（last10 mean）：
  - StageA baseline：`loss_total=0.005028, loss_depth=0.007047`
  - StageA5(xyz)：`0.005191, 0.007429`（劣于 baseline）
  - StageA5(xyz+opacity)：`0.004416, 0.006119`（优于 baseline）
  - StageA5(xyz)+abs(1.0/0.1)：`0.006291, 0.007335`（劣于 baseline）

过程问题：
- `prepare_stage1_difix_dataset_s3po_internal.py --stage verify` 在当前环境触发 MASt3R 依赖报错：`NameError: load_model_as_safetensor`；
- 因此本轮 A.5 使用 fused legacy mask 路径先完成机制对照，BRPO verify 依赖修复单列后续处理。

### 与 2026-04-12（mask 刚修好、depth 问题暴露时）的设计差异

1. 当时主问题：确认 depth 信号是否接通、闭环是否成立；
2. 现在主问题：在闭环已成立条件下，做 drift-depth 可解释权衡；
3. 当时重点在 upstream coverage 与链路真伪；现在重点在 Stage A 参数化与可解释诊断；
4. 不进 Stage B 的理由从“链路未充分可信”转为“tradeoff 未固化”。


### depth-heavy top3：3组 × 300 iter（Step-1）

实验设置：
- 复用 top3 参数组：`(3.0,0.1) / (3.0,0.03) / (1.0,0.1)`
- `stageA_beta_rgb=0.3`（depth-heavy）
- 其余同 A+B 主口径，启用 grad contrib 记录。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_ab_deep300_top3_depthheavy/`

结果：
- best-drift：`lt3.0_lr0.1`（`rho≈5.19e-4, theta≈2.40e-4`）
- best-depth：`lt1.0_lr0.1`（`loss_depth≈0.06814`）
- 结论：tradeoff 在 depth-heavy 下未消失，仍是“drift组”和“depth组”分离。

### Step-2 梯度贡献汇总

汇总文件：
- `.../2026-04-13_ab_deep300_top3_depthheavy/grad_contrib_summary.json`
- `.../2026-04-13_ab_deep300_top3_depthheavy/grad_contrib_summary_cross.json`

核心结论：
- `rot` 与 `trans` 的主导更新项主要来自 `rgb` 与 `depth_total`（二者交替主导）；
- `abs_pose_trans/rot` 在均值上显著小于数据项，当前更像稳定边界，不是主驱动项。

### Step-3 口径固化

固化两套 Stage A 临时默认：
1. drift-prioritized：`beta_rgb=0.7, lambda_abs_t=3.0, lambda_abs_r=0.1`
2. depth-prioritized：`beta_rgb=0.3, lambda_abs_t=1.0, lambda_abs_r=0.1`

说明：这两套用于可解释对照，不代表已得到统一最优。


### 下游价值验证（StageA 三口径）

实验设置：
- 组A：`noabs_beta0.7_300`
- 组B：`drift_prio_beta0.7`（`lt3.0_lr0.1`）
- 组C：`depth_prio_beta0.3`（`lt1.0_lr0.1`）
- 在 internal cache `after_opt` 的 270 帧 non-KF replay 上比较。

输出目录：
- `/home/bzhang512/CV_Project/output/part3_stage1_internal/re10k-1/full/2026-04-13_value_eval/`
- 汇总：`value_eval_report.json`

结果：
- noabs: `PSNR 23.948825 / SSIM 0.8734876 / LPIPS 0.0787778`
- drift-prio: `23.948666 / 0.8734792 / 0.0787801`
- depth-prio: `23.948893 / 0.8734833 / 0.0787813`
- 相对 baseline (`23.948909 / 0.8734854 / 0.0787798`) 的差异在 `1e-4~1e-6` 量级。

结论：
- 当前 Stage A 口径虽然改变了 drift 行为，但未转化为可见的下游 replay 提升；
- 主要原因是当前仅更新 3 个 pseudo frame pose，对 270 帧总体指标影响过小。

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


### Mask problem route：M5-0 depth signal diagnosis

在 M4.5 之后，没有直接继续推进 Stage B，而是先做一轮更结构化的 diagnosis，确认当前 depth signal 到底卡在哪里。

新增文件：
- `scripts/analyze_m5_depth_signal.py`

实验目录：
- `.../2026-04-12_m50_m51_eval/analysis/m50_depth_signal.json`

关键结果：
- `train_mask coverage ≈ 19.41%`
- `verified depth coverage ≈ 1.56%`
- `verified_within_train_mask_ratio ≈ 8.05%`
- `render_fallback_within_train_mask_ratio ≈ 91.95%`

解释：
- 当前 depth loss 不是没接上；
- 但在训练真正消费的 mask 区域里，绝大多数像素仍然只是 fallback depth；
- 这说明旧的 M3 target 在 train-mask 内部的“新几何信息占比”确实太低。

阶段判断：
- M5-0 证明旧问题真实存在；
- 下一步值得先尝试 densify correction field，而不是继续堆 M4.5 式对照。

### Mask problem route：M5-1 densify depth correction field

根据 M5 规划，开始新增一层受控的 local depth correction densify，而不是去放宽 verify 阈值。

新增文件：
- `pseudo_branch/brpo_depth_densify.py`
- `scripts/materialize_m5_depth_targets.py`

代码修改：
- `pseudo_branch/brpo_depth_target.py`

关键实现：
- 在 verified seed 上构造 `log-depth correction`，而不是传播 absolute depth；
- 先实现首版 `patch-wise constant correction`；
- 输出：
  - `target_depth_for_refine_v2.npy`
  - `target_depth_dense_source_map.npy`
  - `depth_correction_seed.npy`
  - `depth_correction_dense.npy`
  - `depth_seed_valid_mask.npy`
  - `depth_dense_valid_mask.npy`
  - `depth_densify_meta.json`

默认参数第一次运行结果：
- 只把 densified 区域提升到 `~2.08%`
- 总 non-fallback 只有 `~3.64%`
- 结论：太保守，不足以支撑下游对照

随后做了小范围 sweep，目标不是追求极大 coverage，而是找一个“明显增强但仍受控”的配置。

当前选中的一组参数：
- `patch_size = 11`
- `stride = 5`
- `min_seed_count = 4`
- `max_seed_delta_std = 0.08`

在这组参数下：
- `seed_ratio ≈ 1.56%`
- `densified_ratio ≈ 14.21%`
- `total_nonfallback_ratio ≈ 15.77%`
- `render_fallback_ratio ≈ 84.23%`

更关键的是，在 train-mask 内部：
- `nonfallback_within_train_mask_ratio ≈ 81.22%`
- `seed_within_train_mask_ratio ≈ 8.05%`
- `dense_within_train_mask_ratio ≈ 73.17%`

阶段判断：
- M5-1 upstream 是可行的；
- 这一步已经把真正有新几何信息的区域从 seed 级抬到了 train-mask 的大部分区域；
- 因此后续值得测试 source-aware depth loss，而不是继续纠结 upstream coverage 是否还太低。

提交：
- `8a9c224` — `part3: add M5 depth signal analysis and densify targets`

### Mask problem route：M5-2 source-aware depth loss 接入 Stage A

在 M5-1 之后，继续把 Stage A 接到新的 densified target，并把 depth loss 按 source 拆开。

代码修改：
- `pseudo_branch/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- `run_pseudo_refinement_v2.py` 新增：
  - `stageA_target_depth_mode=blended_depth_m5`
  - `stageA_depth_loss_mode=legacy|source_aware`
- `pseudo_loss_v2.py` 新增：
  - `loss_depth_seed`
  - `loss_depth_dense`
  - `loss_depth_fallback`
- 当前 source-aware 默认权重：
  - `lambda_depth_seed = 1.0`
  - `lambda_depth_dense = 0.35`
  - `lambda_depth_fallback = 0.0`

实验目录：
- `.../2026-04-12_m52_stageA_loss_eval/`

对照结果：
1. `M5 + legacy depth loss`
   - `loss_depth ≈ 0.02770`
   - 300 iter 中基本完全不动
2. `M5 + source-aware depth loss`
   - `loss_depth ≈ 0.06867`
   - 其中：
     - `loss_depth_seed ≈ 0.05796`
     - `loss_depth_dense ≈ 0.03062`
     - `loss_depth_fallback ≈ 0`
   - 300 iter 中同样基本完全不动

阶段判断：
- source-aware depth loss 已经真实接通；
- fallback 也已基本从 depth loss 主体中剔除；
- 但 depth 仍然不动，说明问题已经不再只是 coverage 或 fallback 稀释。

提交：
- `d700194` — `part3: add M5 source-aware stageA depth loss`

### Stage A diagnosis：确认 `theta/rho` 在 renderer 中 forward 丢弃、backward 仍回梯度

在 M5-2 之后，继续追查为什么 loss 仍完全不动，并审核 refine 过程本身是否存在更底层的问题。

新增文件：
- `scripts/diagnose_stageA_gradients.py`

实验目录：
- `.../2026-04-12_m53_stageA_diagnosis/m53_gradients.json`

代码审计先确认：
- `cam_rot_delta / cam_trans_delta` 确实传进了 `gaussian_renderer`
- optimizer 也确实在更新它们

但真正关键的 diagnosis 结果是：

1. **forward sensitivity probe 极其异常**
   - 人工把 `cam_rot_delta / cam_trans_delta` 设到 `0.01 / 0.01`、`0.05 / 0.05`、`0.1 / 0.1`、`0.2 / 0.2`
   - 重新 render RGB / depth
   - 与 base render 比较，结果始终：
     - `rgb_mean_abs_change = 0.0`
     - `depth_mean_abs_change = 0.0`

2. **backward 却返回了非零 pose 梯度**
   - `mean_grad_rgb_rot ≈ 0.265`
   - `mean_grad_rgb_trans ≈ 0.0786`
   - `mean_grad_depth_legacy_rot ≈ 0.504`
   - `mean_grad_depth_legacy_trans ≈ 0.197`
   - `mean_grad_depth_src_rot ≈ 1.732`
   - `mean_grad_depth_src_trans ≈ 0.550`

3. **pose regularization 在初始化为 0 时梯度也是 0**
   - 因此最开始没有任何“拉回 pose”的约束力

阶段判断：
- 当前最可疑的问题已经被代码链路审计确认：

**Stage A 的 `theta/rho` 在 Python 层被传入，但在 C++/CUDA forward 中被完全丢弃；与此同时 backward 仍然单独构造并返回 `dL_dtau`。**

这也解释了为什么：
- loss 基本不下降；
- pose delta 却会持续增大；
- 当前 Stage A 的 pose refine 结果不应被直接信任。

进一步的代码证据是：
- `diff_gaussian_rasterization/__init__.py::_RasterizeGaussians.forward()` 的 `args` 不包含 `theta/rho`；
- `rasterize_points.h/.cu::RasterizeGaussiansCUDA` 的 forward 函数签名也没有 `theta/rho`；
- 但 `_RasterizeGaussians.backward()` 会从 `_C.rasterize_gaussians_backward(...)` 收到 `grad_tau`，再拆成 `grad_theta / grad_rho`。

因此当前明确不建议直接进入 Stage B。

补充修正判断：
- 经过进一步代码链路审计，这里更准确的结论不是“CUDA 单纯写坏了一个本该 forward 直接吃 `theta/rho` 的接口”；
- S3PO 原始代码里，`cam_rot_delta / cam_trans_delta` 本来就是一步一清的增量 residual，`slam_frontend.py` / `slam_backend.py` 会在每次 `optimizer.step()` 后立刻调用 `update_pose(viewpoint)`，把 `tau` 折回 `R/T` 并清零；
- 当前 Part3 Stage A 的真正问题是：**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**


### Stage A 修复：补回 S3PO residual pose 闭环 + 最小验证

在继续追根因之后，进一步梳理了 S3PO 原始代码对 `cam_rot_delta / cam_trans_delta` 的使用方式。

关键发现：
- 原始 S3PO 不是把 `theta/rho` 当作持续直接作用于 forward render 的状态量；
- 它把它们当作**一步一清的 pose residual**；
- `slam_frontend.py` / `slam_backend.py` 在每次 `optimizer.step()` 后都会立刻调用 `update_pose(viewpoint)`；
- `pose_utils.py::update_pose()` 会把 `tau=[rho, theta]` 折回 `R/T` 并清零。

所以当前 Part3 Stage A 的真正问题不是“CUDA 单纯坏了”，而是：

**我们沿用了 residual 参数和 backward 梯度，但没有沿用 `update_pose()` 这一步，因此把 S3PO 的增量式 pose 优化闭环用断了。**

代码修改：
- `pseudo_branch/pseudo_camera_state.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `apply_pose_residual_()`：
  - 用当前 `tau=[cam_trans_delta, cam_rot_delta]` 计算新的 `w2c`；
  - 折回 `vp.R / vp.T`；
  - 刷新 `world_view_transform / full_proj_transform / camera_center`；
  - 清零 residual。
- `run_pseudo_refinement_v2.py` 现在在每次 `optimizer.step()` 后，默认会对本轮参与优化的 pseudo views 调 `apply_pose_residual_()`；
- 同时新增开关：
  - `--stageA_apply_pose_update`（默认开启）
  - `--stageA_no_apply_pose_update`（用于保留旧坏链路做对照）

最小验证 1：手动 residual -> fold 到 R/T -> render
- 修复后，render 会立刻随 pose 变化：
  - `(0.01, 0.01)`：`rgb_mean ≈ 0.0679`, `depth_mean ≈ 0.1648`
  - `(0.05, 0.05)`：`rgb_mean ≈ 0.1604`, `depth_mean ≈ 0.4364`
  - `(0.10, 0.10)`：`rgb_mean ≈ 0.2412`, `depth_mean ≈ 0.8877`

最小验证 2：M5 source-aware 80 iter smoke 对照
- `no_apply_pose_update`（旧坏链路）：
  - `loss_depth: 0.06867 -> 0.06867`（完全平）
  - `pose_updates_total = 0`
  - 最终 residual 堆积在 `rot_norm ~ 0.27~0.41`、`trans_norm ~ 0.13`
- `apply_pose_update`（修复后闭环）：
  - `loss_total: 0.02701 -> 0.02570`
  - `loss_depth: 0.06867 -> 0.06826`
  - `loss_depth_seed: 0.05796 -> 0.05760`
  - `loss_depth_dense: 0.03062 -> 0.03046`
  - `pose_updates_total = 240`
  - 最终 residual 全部回到 `0.0`

阶段判断：
- 这说明当前 Stage A 已经不再是之前那种“forward 不动、backward 乱推”的假优化；
- 但也要老实说：修复闭环之后，depth 只是开始轻微下降，还没有出现非常强的下降趋势；
- 所以下一步不应直接进入 Stage B，而应先做修复后的中等规模验证，再决定是否需要继续加强 loss / 训练长度 / 更深层 renderer 改动。


### Stage A 修复后 300-iter 参数/规模验证

在补回 `apply_pose_residual_()` 之后，没有直接进入下一阶段，而是先做一轮修复后的 300-iter 参数/规模验证，确认当前问题更像训练规模、权重还是结构。

实验目录：
- `.../2026-04-12_m55_pose_fix_scale_eval/default_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/depth_heavy_300`
- `.../2026-04-12_m55_pose_fix_scale_eval/no_exposure_300`
- 汇总：`.../analysis/m55_summary.json`

实验口径：
1. `default_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.01`
2. `depth_heavy_300`
   - `beta_rgb=0.3`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=1.0`
   - `lr_exp=0.01`
3. `no_exposure_300`
   - `beta_rgb=0.7`
   - `lambda_depth_seed=1.0`
   - `lambda_depth_dense=0.35`
   - `lr_exp=0.0`

结果：
- `default_300`
  - `loss_total: 0.02701 -> 0.02577`
  - `loss_depth: 0.06867 -> 0.06816`
  - `loss_depth_seed: 0.05796 -> 0.05750`
  - `loss_depth_dense: 0.03062 -> 0.03047`
- `depth_heavy_300`
  - `loss_total: 0.06475 -> 0.06364`
  - `loss_depth: 0.08857 -> 0.08747`
  - `loss_depth_seed: 0.05796 -> 0.05731`
  - `loss_depth_dense: 0.03062 -> 0.03016`
- `no_exposure_300`
  - `loss_total: 0.02701 -> 0.02696`
  - `loss_rgb: 0.00915 -> 0.00928`
  - `loss_depth: 0.06867 -> 0.06822`

附加分析：真实 pose 累计变化（init -> final）
- `default_300`
  - mean `trans_delta_l2 ≈ 0.00138`
  - mean `rot_delta_rad ≈ 0.00498`
- `depth_heavy_300`
  - mean `trans_delta_l2 ≈ 0.00390`
  - mean `rot_delta_rad ≈ 0.00453`
- `no_exposure_300`
  - mean `trans_delta_l2 ≈ 0.00141`
  - mean `rot_delta_rad ≈ 0.00463`

阶段判断：
- 修复后，depth 已经不是完全平线；
- 但三组设置都只表现出“弱下降”，没有出现质变；
- `depth_heavy_300` 的确更强一点，但提升仍有限；
- 冻结 exposure 会让 RGB 更差，但不会显著改善 depth；
- 这说明问题已经从“闭环断了”进一步收敛成：

**当前更像是权重/尺度与结构设计问题，而不再是链路完全断开。**

新的结构发现：
- 因为现在每步都会 `apply_pose_residual_()` 并清零 residual，当前 `pose_reg_loss = ||cam_rot_delta|| + ||cam_trans_delta||` 会在训练中几乎始终为 0；
- 这意味着 `stageA_lambda_pose` 当前已经无法约束“累计位姿偏移”；
- 因此下一步需要重点判断：是否要补一个基于 `R/T` 相对初值的 **absolute pose prior**，而不是继续只对 residual 本身做正则。

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


### Mask problem route：M5-4 absolute pose prior 接入与 smoke 对照

目标：把 `absolute_pose_prior.md` 从设计文档落地为可执行代码，并验证它在当前 Stage A 闭环下是否有效。

代码修改：
- `third_party/S3PO-GS/utils/pose_utils.py`
- `pseudo_branch/pseudo_camera_state.py`
- `pseudo_branch/pseudo_loss_v2.py`
- `pseudo_branch/pseudo_refine_scheduler.py`
- `scripts/run_pseudo_refinement_v2.py`

关键实现：
- 新增 `SO3_log / SE3_log`；
- `make_viewpoint_trainable()` 存 `R0/T0`；
- 新增 `absolute_pose_prior_loss`（基于当前 `w2c` 相对初始 `w2c0` 的 `SE3_log`）；
- Stage A 增加 `--stageA_lambda_abs_pose`，并写入 history 与导出状态；
- `export_view_state` 新增 `pose_w2c_initial / abs_pose_rho / abs_pose_theta / abs_pose_norm`。

最小验证：
- `py_compile` 全通过；
- CLI `--help` 可见 `--stageA_lambda_abs_pose`；
- `SE3_exp -> SE3_log` roundtrip 误差约 `3.95e-09`。

60-iter smoke（同一 pseudo_cache，seed=0）：
- `default_noabs` vs `default_abs(0.1)`：几乎重合；
- `depth_heavy_noabs` vs `depth_heavy_abs(0.1)`：几乎重合；
- `default_abs(100)`：
  - `abs_pose_norm mean 0.001535 -> 0.000338`（drift 明显收紧）
  - 但 `loss_depth` 从 `0.068364` 变为 `0.068543`（略变差）。

阶段判断：
- absolute prior“接入成功”≠“参数可用”；
- 当前结果更像权重尺度问题：`0.1` 太弱，`100` 太强；
- 下一步应做 300-iter 权重扫描与尺度化 prior，而不是直接拿 `100` 当默认值。

### 与 2026-04-11（mask problem 初始阶段）的设计差异

相较昨天，当前设计变化是实质性的：

1. 昨天主矛盾是 **upstream coverage 不足**（先做 verify/seed/train-mask/M3/M5）；
2. 现在 upstream 已可用，主矛盾转为 **Stage A 结构标定**（闭环已修，开始做 abs prior）；
3. 昨天主要在“把 depth 信号接进来”，现在是“如何在 drift 约束与 depth 对齐之间找平衡”；
4. 因此当前不进入 Stage B 的理由也变化了：
   - 不是“链路没接通”；
   - 而是“已接通但参数区间未定”。

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


### A.5 300iter + replay 对照（baseline vs xyz+opacity）

运行目录：
- `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageA5_midpoint_300iter_withply/`

关键结果：
- StageA baseline replay(270): `PSNR 23.948911 / SSIM 0.873485 / LPIPS 0.0787798`
- StageA5(xyz+opacity) replay(270): `PSNR 23.970262 / SSIM 0.873834 / LPIPS 0.0787174`
- delta(xyz+opacity - baseline): `PSNR +0.021351 / SSIM +0.000348 / LPIPS -6.23e-05`

结论：
- 与此前“3帧改动后 replay 几乎不变”不同，midpoint 8 帧 + A.5(xyz+opacity) 已出现可复现的下游正向信号；
- `xyz-only` 仍非优先路径。


### StageB conservative Phase0/1/2 执行（2026-04-13）

- 新增计划文档：`docs/STAGEB_CONSERVATIVE_ENTRY_PLAN.md`
- 新增执行报告：`docs/STAGEB_PHASE0_2_EXEC_REPORT_20260413.md`
- `run_pseudo_refinement_v2.py` 新增 `stageB` 模式（conservative joint）：
  - real anchor 稀疏视角分支（`--train_manifest`, `--num_real_views`, `--lambda_real`）
  - pseudo RGBD masked 分支（`--lambda_pseudo`）
  - joint 优化 pseudo camera 参数 + micro gaussian 参数（`xyz/xyz+opacity`）
- 本次 gate run（120iter）相对 A.5 baseline replay：
  - `PSNR +0.29383`
  - `SSIM +0.004369`
  - `LPIPS -0.0002153`
- gate 判定：`PASS`


### StageB Phase3 long run（300iter）

- run: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity`
- replay: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260413_stageB_phase3_longrun/stageB300_conservative_xyz_opacity/replay/stageB300_conservative_xyz_opacity/replay_eval.json`
- delta(vs A.5): PSNR -0.179275, SSIM -0.004948, LPIPS +0.002430
- gate: FAIL


### Strict midpoint8 + M5 + source-aware full pipeline execution（2026-04-13 夜）

新建完整执行与分析报告：
- `docs/MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md`

本轮严格协议：
- midpoint 8 帧：`17/51/86/121/156/190/225/260`
- 先对 midpoint pseudo cache 补 `target_depth_for_refine_v2.npy`（M5）
- 然后运行：
  - `StageA(source_aware)`
  - `StageA.5(source_aware + xyz_opacity)`
  - `StageB120(conservative source-aware)`
- 每阶段都做 270-frame replay

结果（vs part2 after_opt baseline）：
- StageA：几乎完全不变（噪声级）
- StageA.5：`PSNR -0.11066 / SSIM -0.00794 / LPIPS +0.00878`
- StageB120：`PSNR -0.78787 / SSIM -0.02304 / LPIPS +0.01903`

关键分析结论：
1. `final_delta_summary` 会误导地显示 pose residual 为零；但真实 `pose_w2c` 仍有小幅变化；
2. StageA replay 不变的真正原因不是“pose 绝对没动”，而是 **StageA 不更新 Gaussian，导出的 PLY 与输入 hash 完全一致**；
3. 当前 `StageA -> A.5 -> B` 不是严格顺序相机 handoff，A.5 / B 都不会读取前一阶段的 `pseudo_camera_states_final.json`；
4. 当前 midpoint8 M5 的有效 non-fallback 几何监督偏弱（train-mask 内约 `18.65%`），不足以支撑强 pose 提升；
5. 现在更需要修 pipeline 结构，而不是继续把问题当成纯参数扫描。


### P0 repair: stage handoff + StageA reporting fix + repaired sequential rerun（2026-04-13 夜）

代码修复：
- `pseudo_branch/pseudo_camera_state.py`
  - 新增：
    - `load_exported_view_states(...)`
    - `apply_loaded_view_state_(...)`
    - `summarize_true_pose_deltas(...)`
- `scripts/run_pseudo_refinement_v2.py`
  - 新增 CLI：
    - `--init_pseudo_camera_states_json`
    - `--init_pseudo_reference_mode {keep,reset_to_loaded}`
  - 新增 handoff summary 记录
  - 新增 true pose delta summary / aggregate
  - 原 `final_delta_summary` 改为更准确的 `residual_delta_summary_post_foldback`

验证：
- `repair_seq_check/` short smoke 已确认：
  - `StageA final -> StageA.5 init` 完全对齐
  - `StageA.5 final -> StageB init` 完全对齐
- `repair_seq_rerun/` 已完成：
  - `StageA80`
  - `StageA.5(80, from StageA)`
  - `StageB120(from StageA.5)`

结果变化：
- handoff 修复后，A.5 / StageB 相对修复前 strict run 有小幅改善；
- 但整体仍然未超过 baseline，说明 handoff bug 只是其中一个结构问题，不是唯一瓶颈。

过程复盘沉淀：
- 以后 staged pipeline 设计必须先验证：
  1. handoff continuity
  2. evaluation-artifact continuity
  3. true-state summary correctness
  4. short smoke before long run

### P1 bottleneck review: signal / method mismatch for midpoint8 M5（2026-04-13 夜）

代码与数据链路复查：
- 复查 `brpo_train_mask.py` / `brpo_depth_densify.py` / `brpo_depth_target.py` / `pseudo_loss_v2.py` / `pseudo_refine_scheduler.py` / `gaussian_param_groups.py` / `run_pseudo_refinement_v2.py`
- 复查 midpoint8 M5 pseudo cache 与 `repair_seq_rerun/` 实验产物

新增确认的关键事实：
- pseudo RGB 与 pseudo depth 都只在 `train_confidence_mask_brpo_fused.npy` 非零区域生效；
- 在 `lambda_depth_fallback=0` 下，fallback depth 区域不提供 depth 梯度；
- midpoint8 M5 的平均有效几何支持仅约 `3.49%`，平均 fallback 约 `96.51%`；
- support 区域内 `target_depth_for_refine_v2` 相对 `render_depth` 的平均修正仅约 `1.53%`；
- A.5 / StageB 却直接打开全局 Gaussian `xyz/opacity`，导致 supervision scope 与 optimization scope 明显不匹配。

新增量化结论：
- StageA80 的平均 pose 平移改变量仅约 scene scale 的 `0.118%`；
- A.5 相对 baseline PLY：`74.1%` Gaussians 的 xyz 改动 `>1e-3`；
- StageB120 相对 baseline PLY：`86.2%` Gaussians 的 xyz 改动 `>1e-3`。

工程判断更新：
- 当前主瓶颈更接近 `signal weak + underconstrained global refinement`；
- 后续优先事项应从“继续扫同一组 refine 超参”转向：
  1. pseudo 选点 / signal 质量复查；
  2. A.5 / B 的可训练 Gaussian 作用域收缩；
  3. 将 support/correction 量化检查加入长跑前固定 gate。

### 2026-04-14：新增下一阶段执行计划（signal semantics + stable refinement）

新增文档：
- `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
- 同步更新：`docs/hermes.md`

本轮不是代码实现，而是基于更新后的 `DEBUG_CONVERSATION.md`、BRPO 差异分析和真实代码链路，形成下一阶段的工程方案。核心收敛：
1. 先做 `continuous confidence + agreement-aware support`；
2. 把 RGB/raw confidence 与 depth/train-mask 语义分离；
3. densify 改成 confidence-aware；
4. StageB 改成两段式 curriculum，并在 pseudo 分支补 local Gaussian gating；
5. full SPGM 暂不作为第一优先级，而是后续 stabilizer 候选。
### 2026-04-14：S1.1 完成（continuous confidence + agreement-aware support）

代码更新：
- `pseudo_branch/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮落地内容：
1. 在离散 `1.0 / 0.5 / 0.0` 之外，新增 raw continuous confidence 输出；
2. both-support 新增基于双侧 projected-depth 的 agreement-aware 连续权重；
3. verification 产物新增：
   - `raw_confidence_mask_brpo_{left,right,fused}.*`
   - `raw_confidence_mask_brpo_cont_{left,right,fused}.*`
   - `confidence_mask_brpo_agreement.*`
4. sample pack / link 阶段已可带上这些新 artifact。
### 2026-04-14：S1.2 完成（RGB/depth mask semantics split）

代码更新：
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/pseudo_loss_v2.py`

本轮落地内容：
1. refine CLI 新增：`--stageA_rgb_mask_mode`、`--stageA_depth_mask_mode`、`--stageA_confidence_variant`；
2. `load_stageA_pseudo_views(...)` 现已分别解析 `conf_rgb` 与 `conf_depth`；
3. `build_stageA_loss*` 现已支持 RGB / depth 分别使用不同 mask；
4. history / per-view stats 已新增 RGB / depth mask 与 coverage 记录。
### 2026-04-14：S1.3 完成 + 2-frame smoke/ablation

代码更新：
- `pseudo_branch/brpo_depth_densify.py`
- `pseudo_branch/brpo_depth_target.py`
- `scripts/materialize_m5_depth_targets.py`

本轮落地内容：
1. densify 新增 confidence-aware patch acceptance；
2. 支持 both-rich patch 放宽 seed-count、single-dominant patch 收紧 delta-std；
3. M5 materialize 新增 `--use-continuous-confidence`、`--min-patch-confidence`、`--both-seed-count-relax`、`--single-std-tighten`。

2-frame smoke：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_semantics_smoke`
- 已确认 verify/pack 可真实产出 `raw_confidence_mask_brpo_cont_*` 与 `confidence_mask_brpo_agreement.*`，并成功进入 pseudo_cache sample。

2-frame compare：
- 目录：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_semantics_compare`
- baseline densify：`mean_dense_valid_ratio≈0.02505`
- conf-aware densify：`mean_dense_valid_ratio≈0.00129`
- 说明首轮 confidence-aware 阈值过于保守，densify coverage 被压得过小。

2-frame StageA-10iter ablation：
- baseline(shared train-mask/discrete) 与 conf-aware(split raw/train + continuous) 均已跑通；
- conf-aware 组的 RGB/depth 作用域显著收缩，pose drift 更小，但 loss 更高；
- 当前结论是：语义分离与 confidence-aware wiring 已打通，但 S1.3 需要先回调阈值，再进入更正式的 8-frame 短跑对照。
### 2026-04-14 晚：8-frame retuned compare

执行目录：
- `.../20260414_signal_semantics_mid8`
- `.../20260414_signal_semantics_mid8_compare`

本轮完成：
1. 8-frame verify/pack 联调；
2. retuned confidence-aware densify（`min_patch_confidence=0.12, both_relax=2, single_std_tighten=0.9`）；
3. 8-frame StageA-20iter baseline vs conf-aware 小对照。

关键量化：
- baseline densify：`mean_dense_valid_ratio≈0.02269`
- retuned conf-aware：`mean_dense_valid_ratio≈0.01634`
- baseline StageA20：`loss_total≈0.03673`, `pose_mean_trans≈0.00116`
- conf-aware StageA20：`loss_total≈0.04392`, `pose_mean_trans≈0.00128`

结论：
- 相比首轮过严版本，retuned conf-aware densify 已回到可用 coverage；
- 但当前 8-frame 短跑下，conf-aware 组还没有稳定优于 baseline；
- 因此下一步仍应先做 E1 小网格回调，而不是直接进入 StageB 两段式 curriculum。


### 2026-04-14 夜：E1 support-aware pseudo selection 第一轮完成

代码新增：
- `scripts/select_signal_aware_pseudos.py`

运行产物：
- `.../20260414_signal_enhancement_e1/reports/signal_aware_selection_report.json`
- `.../20260414_signal_enhancement_e1/manifests/signal_aware_selection_manifest.json`
- `docs/SIGNAL_AWARE_SELECTION_E1_REPORT_20260414.md`

本轮完成内容：
1. 对 8 个 KF gap 枚举 `{1/3, 1/2, 2/3}` 候选；
2. 基于 internal cache 的 render_rgb/render_depth 做 lightweight verify；
3. 用 `both_ratio + verified_ratio + correction_magnitude + balance` 做候选打分；
4. 生成一组 `signal-aware-8` 选中的 pseudo ids。

关键结果：
- midpoint8：`[17, 51, 86, 121, 156, 190, 225, 260]`
- signal-aware-8：`[23, 57, 92, 127, 162, 196, 225, 260]`
- `6/8` 个 gap 发生变化，前 6 个 gap 全部偏向 `2/3`
- aggregate 上，`verified_ratio / continuous confidence / balance / score` 均高于 midpoint8；`support_ratio_both` 略低。

结论更新：
- `midpoint` 不是当前 case 的稳定最优 pseudo 位置；
- E1 方向成立，但当前仍属于 lightweight render-based verify；
- 在进入 E2 之前，应先用 `signal-aware-8` 补一轮 apples-to-apples verify/pack + 8-frame StageA short compare。

### 2026-04-14 深夜：E1.5 完成（support-aware pseudo selection 正式短对照）

新增产物：
- `.../20260414_signal_enhancement_e15_compare/`
- `.../20260414_signal_enhancement_e15_compare/e15_compare_summary.json`
- `docs/SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414.md`

本轮完成内容：
1. 用 `signal-aware-8 = [23, 57, 92, 127, 162, 196, 225, 260]` 重跑正式 `fusion -> verify -> pack`；
2. 复用 midpoint8_compare 的同一组 baseline / conf-aware M5 densify 参数；
3. 复用同一组 StageA-20iter baseline / conf-aware 参数，做 apples-to-apples 短对照。

关键结论：
- raw signal：`verified_ratio` 与 `continuous confidence` 有小幅提升，`support_ratio_both` 略降；
- baseline densify：`mean_dense_valid_ratio 0.02269 -> 0.02961`
- conf-aware densify：`mean_dense_valid_ratio 0.01634 -> 0.01831`
- baseline StageA20：`loss_total 0.03673 -> 0.03659`
- conf-aware StageA20：`loss_total 0.04392 -> 0.04340`

结论更新：
- support-aware pseudo selection 已通过正式短对照，不再只是 lightweight proposal；
- 它的收益主要先体现在 `verified signal -> dense coverage` 的传导；
- E1 可以收口，下一步可进入 E2（dual-pseudo allocation）。


### 2026-04-14 深夜：E2 完成（dual-pseudo allocation 正式对照）

代码更新：
- `scripts/select_signal_aware_pseudos.py`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`

本轮落地内容：
1. `select_signal_aware_pseudos.py` 新增 `--topk-per-gap` 与 `--allocation-policy`，支持每 gap 输出多 pseudo；
2. selection manifest 现会写入 `gap_index / allocation_rank / allocation_policy / candidate_fraction / selection_score / selection_label`；
3. `prepare_stage1_difix_dataset_s3po_internal.py` 新增 `--selection-manifest`，可直接消费 selection manifest；
4. pseudo-cache schema 升级到 `pseudo-cache-internal-v1.5`，sample/meta/manifest 均保留 allocation 元信息；
5. 已完成 E2 正式链路：`select -> fusion -> verify -> pack -> baseline/conf-aware densify -> StageA-20`。

运行产物：
- `.../20260414_signal_enhancement_e2_compare/`
- `.../20260414_signal_enhancement_e2_compare/e2_compare_summary.json`
- `docs/SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md`

关键结论：
- E2 的 top2 实际稳定落成 `1/2 + 2/3`，而不是固定 `1/3 + 2/3`；
- raw signal 相比 midpoint8 仍有轻微正向，但幅度有限；
- densify 层相对 midpoint8 为正，但弱于 E1 `signal-aware-8`；
- short StageA-20 baseline / conf-aware 两条线都明显差于 E1；
- 当前 default winner 仍应保持 E1 `signal-aware-8`；若进入 E3，应优先基于 E1 winner 做 multi-anchor verify，而不是直接沿用 E2 16-pseudo set。


### 2026-04-14 更深夜：补充 pipeline judgement + 归档部分已完成计划文档

新增文档：
- `docs/MASK_DEPTH_CONFIDENCE_PIPELINE_JUDGEMENT_20260414.md`

本轮新增结论：
1. `mask / depth / confidence` pipeline 不是伪方向，但其作为主增益来源的边际收益已接近见顶；
2. E1 已吃到这条线上最有价值的上游收益；
3. E2 说明“多 pseudo”在当前 consumer 机制下会稀释 winner，不应再默认沿这一路线继续扩张；
4. 若继续推进，这条线更适合作为 E3 final probe，而不是继续高强度主线投入。

文档归档：
- `docs/M5_STAGEA-A.5-B_RUNBOOK.md -> docs/archived/2026-04-superseded-plans/`
- `docs/part3_BRPO_M5_depth_densify_and_loss_plan.md -> docs/archived/2026-04-superseded-plans/`
- `docs/part3_stageA_pre_stageB_engineering_plan.md -> docs/archived/2026-04-superseded-plans/`

原因：以上文档对应的执行阶段/计划已完成或已被当前 `SIGNAL_ENHANCEMENT.md` 与最新 STATUS/DESIGN 判断取代。
