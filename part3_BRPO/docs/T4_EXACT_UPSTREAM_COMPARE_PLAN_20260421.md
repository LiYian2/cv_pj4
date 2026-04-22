# T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md

> 角色：T~ exact-upstream compare 执行文档
> 范围：定义 Phase T4 的完整执行顺序，分成 `T4.0 smoke` 与 `T4.1 formal compare`
> 目的：在不混入新 G~/R~ 改动的前提下，先确认 exact-upstream path 真正被端到端消费，再做固定 control 的 formal replay compare
> 上位文档：`docs/T_direct_brpo_alignment_engineering_plan.md`
> 相关 authority：`docs/current/STATUS.md`、`docs/current/DESIGN.md`、`docs/current/CHANGELOG.md`

---

## 0. 结论先说

这轮 T4 不能直接从 Noah 当前的 smoke 结果跳到 full compare，原因有二：
1. 当前 smoke 虽然已经证明 exact backend / exact-upstream builder / exact loss contract 都有代码接线，但还缺一轮真正的 end-to-end consumer smoke；
2. 当前 exact backend smoke 使用 `branch_first`，但 `summary_meta.json` 里 `pseudo_left_root` / `pseudo_right_root` 仍是 `null`，因此还没有真正证明“branch-native verifier input”这一步已经被实际跑通。

因此这轮 T4 必须按下面顺序执行：
1. `T4.0a`：branch-native exact backend smoke
2. `T4.0b`：exact-upstream signal smoke
3. `T4.0c`：exact-upstream consumer smoke
4. `T4.1a`：full exact backend build
5. `T4.1b`：full exact-upstream signal build
6. `T4.1c`：4-arm formal compare
7. `T4.1d`：统一 replay summary + verdict

如果 `T4.0` 任一步失败，不进入 `T4.1`。

---

## 1. 固定实验底座（整轮 T4 不允许漂移）

### 1.1 固定环境
- Project root：`/home/bzhang512/CV_Project/part3_BRPO`
- Python：`/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- PYTHONPATH：`/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`

### 1.2 固定数据输入
- Internal cache root：`/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- Prepare root：`/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare`
- Pseudo cache：`/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline`
- PLY：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply`
- Init pseudo camera states：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json`
- Train manifest：`/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json`
- Train rgb dir：`/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb`

### 1.3 固定 pseudo frame 集
来自 `20260414_signal_enhancement_e15_compare/manifests/pseudo_selection_manifest.json`：
- frame ids = `[23, 57, 92, 127, 162, 196, 225, 260]`

### 1.4 固定 branch-native verifier inputs
- left branch root：`/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/difix/left_fixed`
- right branch root：`/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/difix/right_fixed`

### 1.5 固定训练 protocol
这轮 compare 必须沿用 `20260420_a1_exact_brpo_target_compare_e1` 的 canonical StageB protocol，不允许顺手换 schedule：
- `joint_topology_mode=brpo_joint_v1`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=120`
- `stageB_post_switch_iter=40`
- `stageB_post_lr_scale_xyz=0.3`
- `stageB_post_lr_scale_opacity=1.0`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `num_real_views=2`
- `num_pseudo_views=4`
- `seed=0`
- `init_pseudo_reference_mode=keep`

### 1.6 固定 G~/R~ control
这轮 compare 不允许混新的 G~/R~ 改动：
- G~ 固定为 clean summary-only no-action control：
  - `pseudo_local_gating=spgm_keep`
  - `pseudo_local_gating_params=xyz`
  - `pseudo_local_gating_spgm_manager_mode=summary_only`
  - `pseudo_local_gating_spgm_policy_mode=dense_keep`
  - `pseudo_local_gating_spgm_ranking_mode=v1`
  - `pseudo_local_gating_spgm_density_mode=opacity_support`
  - `pseudo_local_gating_spgm_cluster_keep_near=1.0`
  - `pseudo_local_gating_spgm_cluster_keep_mid=1.0`
  - `pseudo_local_gating_spgm_cluster_keep_far=1.0`
  - `pseudo_local_gating_spgm_support_eta=0.0`
  - `pseudo_local_gating_spgm_weight_floor=0.25`
- R~ 固定为 `brpo_joint_v1`

---

## 2. T4.0：Smoke（3 步，任一步失败都不进入 formal compare）

---

### T4.0a：branch-native exact backend smoke

#### 目标
验证 exact backend 真正吃的是 branch-native 输入，而不是名义上的 `branch_first` + 默认 fused/render path。

#### 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_backend_branch_native_smoke`

#### 运行范围
- 只跑 frame `23`

#### 执行命令模板
```bash
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python \
  /home/bzhang512/CV_Project/part3_BRPO/scripts/brpo_build_mask_from_internal_cache.py \
  --internal-cache-root /data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache \
  --stage-tag after_opt \
  --frame-ids 23 \
  --verification-mode branch_first \
  --verifier-backend-semantics exact_branch_native_v1 \
  --pseudo-left-root /data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/difix/left_fixed \
  --pseudo-right-root /data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/difix/right_fixed \
  --output-root <T4.0a_output_root>
```

#### 必查 artifact
- `<T4.0a_output_root>/summary_meta.json`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/exact_backend_meta.json`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/support_left_exact.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/support_right_exact.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/projected_depth_left_exact.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/projected_depth_right_exact.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/provenance_left.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/provenance_right.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/hit_count_left.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/hit_count_right.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/occlusion_reason_left.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/occlusion_reason_right.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/confidence_left_exact.npy`
- `<T4.0a_output_root>/exact_backend_v1/frame_0023/confidence_right_exact.npy`

#### 通过标准
必须同时满足：
1. `summary_meta.json` 里 `verifier_backend_semantics=exact_branch_native_v1`
2. `summary_meta.json` 里 `pseudo_left_root` / `pseudo_right_root` 不为 `null`
3. `exact_backend_meta.json` 里 `target_proxy_semantics=branch_native_exact`
4. exact bundle arrays 全部存在
5. `left_stats` / `right_stats` 非空，support/projected-depth 统计有效

#### 失败即停的信号
1. `pseudo_left_root` / `pseudo_right_root` 仍为 `null`
2. exact bundle 缺文件
3. `verification_mode=branch_first` 但实际路径还是 fused/render 默认路径

---

### T4.0b：exact-upstream signal smoke

#### 目标
验证 `build_brpo_v2_signal_from_internal_cache.py` 真的消费了 T4.0a 的 exact backend bundle，并生成 `exact_brpo_upstream_target_v1` signal。

#### 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_upstream_signal_smoke`

#### 运行范围
- 只跑 frame `23`

#### 执行命令模板
```bash
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python \
  /home/bzhang512/CV_Project/part3_BRPO/scripts/build_brpo_v2_signal_from_internal_cache.py \
  --internal-cache-root /data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache \
  --prepare-root /data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare \
  --stage-tag after_opt \
  --frame-ids 23 \
  --use-exact-backend \
  --exact-backend-root <T4.0a_output_root>/exact_backend_v1 \
  --output-root <T4.0b_output_root>
```

#### 必查 artifact
- `<T4.0b_output_root>/summary_meta.json`
- `<T4.0b_output_root>/frame_0023/exact_brpo_upstream_target_observation_meta_v1.json`
- `<T4.0b_output_root>/frame_0023/pseudo_depth_target_exact_brpo_upstream_target_v1.npy`
- `<T4.0b_output_root>/frame_0023/pseudo_confidence_exact_brpo_upstream_target_v1.npy`
- `<T4.0b_output_root>/frame_0023/pseudo_source_map_exact_brpo_upstream_target_v1.npy`
- `<T4.0b_output_root>/frame_0023/pseudo_valid_mask_exact_brpo_upstream_target_v1.npy`
- `<T4.0b_output_root>/frame_0023/pseudo_target_confidence_exact_brpo_upstream_target_v1.npy`

#### 通过标准
必须同时满足：
1. `summary_meta.json` 里 `exact_upstream_backend_enabled=true`
2. `summary_meta.json` 里 `exact_backend_root` 指向 T4.0a 的 exact backend root
3. observation meta 里：
   - `target_field_semantics=exact_upstream_v1`
   - `target_loss_contract=exact_shared_cm_v1`
   - `no_render_fallback=true`
4. 新 exact-upstream arrays 全存在
5. 对照 `exact_brpo_full_target_v1`，`depth/source/confidence` 不是纯 metadata 改名，而是有实质差异

#### 失败即停的信号
1. `exact_backend_root` 没接到 T4.0a
2. exact-upstream meta 缺 `exact_shared_cm_v1`
3. signal 仍走 old `render_depth` fallback
4. array 只是拷贝 exact_full_target 旧结果

---

### T4.0c：exact-upstream consumer smoke

#### 目标
验证训练 consumer 真的在吃 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1`，不是 builder exact、训练仍回到 legacy。

#### 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_upstream_consumer_smoke`

#### 运行范围
- 只跑 1 iter StageB smoke
- 使用完整 pseudo frame 集（8 个 sample），但不做 formal compare

#### 执行命令模板
```bash
export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO
/home/bzhang512/miniconda3/envs/s3po-gs/bin/python \
  /home/bzhang512/CV_Project/part3_BRPO/scripts/run_pseudo_refinement_v2.py \
  --ply_path /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply \
  --pseudo_cache /home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline \
  --output_dir <T4.0c_output_root> \
  --signal_pipeline brpo_v2 \
  --signal_v2_root <T4.0b_output_root> \
  --pseudo_observation_mode exact_brpo_upstream_target_v1 \
  --stageA_depth_loss_mode exact_shared_cm_v1 \
  --joint_topology_mode brpo_joint_v1 \
  --stage_mode stageB \
  --stageA_iters 0 \
  --stageB_iters 1 \
  --stageB_post_switch_iter 40 \
  --stageB_post_lr_scale_xyz 0.3 \
  --stageB_post_lr_scale_opacity 1.0 \
  --lambda_real 1.0 \
  --lambda_pseudo 1.0 \
  --num_real_views 2 \
  --num_pseudo_views 4 \
  --train_manifest /home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json \
  --train_rgb_dir /home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb \
  --init_pseudo_camera_states_json /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json \
  --init_pseudo_reference_mode keep \
  --pseudo_local_gating spgm_keep \
  --pseudo_local_gating_params xyz \
  --pseudo_local_gating_spgm_manager_mode summary_only \
  --pseudo_local_gating_spgm_policy_mode dense_keep \
  --pseudo_local_gating_spgm_ranking_mode v1 \
  --pseudo_local_gating_spgm_density_mode opacity_support \
  --pseudo_local_gating_spgm_cluster_keep_near 1.0 \
  --pseudo_local_gating_spgm_cluster_keep_mid 1.0 \
  --pseudo_local_gating_spgm_cluster_keep_far 1.0 \
  --pseudo_local_gating_spgm_support_eta 0.0 \
  --pseudo_local_gating_spgm_weight_floor 0.25 \
  --seed 0
```

#### 必查 artifact
- `<T4.0c_output_root>/stageA_history.json`
- `<T4.0c_output_root>/stageB_history.json`
- `<T4.0c_output_root>/refinement_history.json`

#### 通过标准
必须同时满足：
1. `stageA_history.json` / `stageB_history.json` 正常产出
2. `effective_source_summary` 里：
   - `signal_v2_root=<T4.0b_output_root>`
   - `pseudo_observation_mode_requested=exact_brpo_upstream_target_v1`
   - `stageA_depth_loss_mode=exact_shared_cm_v1`
3. `pseudo_sample_meta[*]` 里：
   - `pseudo_observation_mode_effective=exact_brpo_upstream_target_v1`
   - `pseudo_observation_meta_path` 指向 `exact_brpo_upstream_target_observation_meta_v1.json`
4. 不出现回退到 `exact_brpo_full_target_v1 + legacy` 的迹象
5. `mean_target_depth_render_fallback_ratio=0` 或等价 no-fallback 证据成立

#### 失败即停的信号
1. consumer 端报错
2. effective mode 不是 `exact_brpo_upstream_target_v1`
3. `stageA_depth_loss_mode` 实际没变成 `exact_shared_cm_v1`
4. pseudo meta 仍指向旧 exact_full_target artifact

---

## 3. T4.1：Formal compare（4 步）

`T4.0` 全过后才进入 `T4.1`。

---

### T4.1a：full exact backend build

#### 目标
对全部 8 个 pseudo frames 导出真正 branch-native 的 exact backend bundle。

#### 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_backend_branch_native_full`

#### 执行口径
与 `T4.0a` 相同，但 frame ids 换成全部 8 个：
- `[23, 57, 92, 127, 162, 196, 225, 260]`

#### 通过标准
1. `summary_meta.json` 中 frame ids 正确
2. `exact_backend_v1/frame_*` 对 8 个 frame 全存在
3. 所有 frame 的 meta 都是 `exact_branch_native_v1`
4. `pseudo_left_root` / `pseudo_right_root` 仍正确写入，不是 smoke-only 偶然成功

---

### T4.1b：full exact-upstream signal build

#### 目标
基于 `T4.1a` 的 full exact backend bundle，构建完整 `exact_brpo_upstream_target_v1` signal root，作为 compare 的唯一新 signal 输入。

#### 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_upstream_signal_full`

#### 执行口径
与 `T4.0b` 相同，但 frame ids 换成全部 8 个。

#### 通过标准
1. 8 个 frame 的 exact-upstream artifact 全存在
2. summary meta 正确记录 `exact_backend_root`
3. 8 个 frame 的 `exact_brpo_upstream_target_observation_meta_v1.json` 都成立
4. 无 frame 回退为 legacy / render fallback contract

---

### T4.1c：4-arm formal compare

#### 目标
用固定 topology + fixed G control 比较：
- historical strong baseline
- current semantics-clean control
- exact consumer / upstream-proxy control
- new exact-upstream target arm

#### compare 输出根
建议：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/<RUN_DATE>_t4_exact_upstream_compare_e1`

#### 4 个 arms
1. `oldA1_newT1_summary_only`
   - 角色：历史外参 / 最终旧强基线
2. `exactBrpoCm_oldTarget_v1_newT1_summary_only`
   - 角色：当前最干净的 semantics-clean control
3. `exactBrpoFullTarget_v1_newT1_summary_only`
   - 角色：consumer exact / upstream proxy control
4. `exactBrpoUpstreamTarget_v1_newT1_summary_only`
   - 角色：这轮 T4 的目标臂

#### 为什么这轮只保留 4 臂
- `old A1`：回答“新 exact-upstream 最终能不能追上历史强基线”
- `exact_brpo_cm_old_target_v1`：回答“新 exact-upstream 相对当前 semantics-clean control 的位置”
- `exact_brpo_full_target_v1`：回答“上游 exact 化相对 consumer-exact 的净增益”
- `exact_brpo_upstream_target_v1`：当前目标臂

`exact_brpo_cm_hybrid_target_v1` / `exact_brpo_cm_stable_target_v1` 本轮不是 mandatory arm；只有在结果模糊时才补第二轮 ablation。

#### 共同参数（除 arm-specific mode 外全部固定）
```text
--ply_path /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/refined_gaussians.ply
--pseudo_cache /home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/pseudo_cache_baseline
--signal_pipeline brpo_v2
--joint_topology_mode brpo_joint_v1
--stage_mode stageB
--stageA_iters 0
--stageB_iters 120
--stageB_post_switch_iter 40
--stageB_post_lr_scale_xyz 0.3
--stageB_post_lr_scale_opacity 1.0
--lambda_real 1.0
--lambda_pseudo 1.0
--num_real_views 2
--num_pseudo_views 4
--train_manifest /home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/split_manifest.json
--train_rgb_dir /home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part2_s3po/sparse/rgb
--init_pseudo_camera_states_json /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/stageA5_v2rgbonly_xyz_gated_rgb0192_80/pseudo_camera_states_final.json
--init_pseudo_reference_mode keep
--pseudo_local_gating spgm_keep
--pseudo_local_gating_params xyz
--pseudo_local_gating_spgm_manager_mode summary_only
--pseudo_local_gating_spgm_policy_mode dense_keep
--pseudo_local_gating_spgm_ranking_mode v1
--pseudo_local_gating_spgm_density_mode opacity_support
--pseudo_local_gating_spgm_cluster_keep_near 1.0
--pseudo_local_gating_spgm_cluster_keep_mid 1.0
--pseudo_local_gating_spgm_cluster_keep_far 1.0
--pseudo_local_gating_spgm_support_eta 0.0
--pseudo_local_gating_spgm_weight_floor 0.25
--seed 0
```

#### arm-specific 配置

##### Arm 1：oldA1_newT1_summary_only
```text
--signal_v2_root /data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260414_signal_enhancement_e15_compare/signal_v2
--pseudo_observation_mode off
--stageA_rgb_mask_mode joint_confidence_v2
--stageA_depth_mask_mode joint_confidence_v2
--stageA_target_depth_mode joint_depth_v2
```

##### Arm 2：exactBrpoCm_oldTarget_v1_newT1_summary_only
```text
--signal_v2_root /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_full_brpo_target_signal_full
--pseudo_observation_mode exact_brpo_cm_old_target_v1
--stageA_depth_loss_mode source_aware
```

##### Arm 3：exactBrpoFullTarget_v1_newT1_summary_only
```text
--signal_v2_root /data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_full_brpo_target_signal_full
--pseudo_observation_mode exact_brpo_full_target_v1
--stageA_depth_loss_mode legacy
```

##### Arm 4：exactBrpoUpstreamTarget_v1_newT1_summary_only
```text
--signal_v2_root <T4.1b_output_root>
--pseudo_observation_mode exact_brpo_upstream_target_v1
--stageA_depth_loss_mode exact_shared_cm_v1
```

#### 必查 artifact
每个 arm 都必须有：
- `stageA_history.json`
- `stageB_history.json`
- `refined_gaussians.ply`
- `replay_eval/replay_eval_meta.json`

#### 失败即停的信号
1. 新 arm 用错 `signal_v2_root`
2. 新 arm 的 `stageA_depth_loss_mode` 没有变成 `exact_shared_cm_v1`
3. formal compare 中混入新的 G~/R~ 设置
4. replay eval 缺臂

---

### T4.1d：统一 replay summary + verdict

#### 主比较顺序
1. `exact_upstream` vs `exact_full`
2. `exact_upstream` vs `exact_cm_old_target`
3. `exact_upstream` vs `old A1`

#### 成功 / 部分成功 / 失败判定

##### A. 成功
同时满足：
1. `exactBrpoUpstreamTarget_v1_newT1_summary_only` 明显优于 `exactBrpoFullTarget_v1_newT1_summary_only`
2. 它相对 `exactBrpoCm_oldTarget_v1_newT1_summary_only` 的 gap 明显缩小
3. 对 `old A1` 的负差距明显缩小，或接近/超过 `old A1`

结论口径：
- T~ upstream 第一轮落地成功
- 下一步不是继续补 consumer，而是做确认 rerun + doc sync

##### B. 部分成功 / 无明显增益
满足：
- `exact_upstream` 与 `exact_full` 基本持平，或只改善噪声级
- 仍停留在 old A1 下方的弱负区

结论口径：
- 说明“upstream 接线是对的，但当前 upstream semantics 还不够强”
- 下一步进入 `T5 backend forensics`，而不是回到 consumer-side 小修补

##### C. 失败
满足任一项：
1. `exact_upstream < exact_full`
2. 明显劣于 `exact_cm_old_target`
3. 结果回撤到比 current exact_full 还弱的水平

结论口径：
- 不继续开更多 formal compare
- 先做 backend forensics：
  - support 覆盖为何变稀
  - exact confidence 是否过严
  - projected-depth valid ratio 为什么下降
  - provenance/hit_count/occlusion_reason 为什么还没有效转化成 target 质量

---

## 4. T4 做完后的下一步原则

### 如果 T4 赢了
下一步：
1. 把 `exact_brpo_upstream_target_v1` 升为新的 T control candidate
2. 做确认 rerun / replay 复核
3. 同步 `STATUS.md` / `DESIGN.md` / `CHANGELOG.md`

### 如果 T4 不赢但接近
下一步进入 `T5`：
- 不再补 consumer-side 小权重
- 转去让 `provenance / hit_count / occlusion_reason / depth_variance` 真正参与 target field builder / confidence policy

### 如果 T4 明显更差
下一步：
- 先做 backend 诊断，不做第二轮 formal compare
- 优先定位 exact backend support / valid 区域为何丢失

---

## 5. 一句话执行口径

这轮 T4 不是“直接拿 Noah 当前 smoke 接着跑 compare”，而是先补齐真正的 `branch-native exact backend -> exact-upstream signal -> exact consumer` 三步 smoke，确认 exact-upstream path 端到端成立后，再在固定 `new T1 + clean summary-only G~ control` 下做 4-arm formal replay compare，核心只回答一件事：`exact_brpo_upstream_target_v1` 相对 `exact_brpo_full_target_v1` 是否带来了真正的上游语义增益。