> 目标: 覆盖旧规划，改成 grounded 的 A/B 方案，把 online mapping 当前的 pseudo depth target 从“ref depth 回投影”切到“pseudo GT RGB + 左右 KF GT RGB 经过 MASt3R 得到的 direct pseudo depth”，并明确区分“只换 target depth 数值”和“放开 depth-only supervision contract”两步。

## 0. 先纠正旧规划里的关键前提

旧文档里最需要覆盖掉的一点是：
- 不能把 `pts3d_2_in_1.z` 直接视为“继承 ref render depth 的 metric scale”。
- `pts3d_1` / `pts3d_2_in_1` 都是 MASt3R pair 自己预测出来的几何表达；`pts3d_2_in_1` 只是“第二张图的点以第一张图坐标系表达”，不是当前 Gaussian render depth 的真实 metric 延伸。

因此，正确的 scale 处理不是：
- `pts3d_2_in_1.z` 天然就是 metric，然后拿它去校 `pts3d_1.z`

而是：
- 用当前 exact backend 已经算好的 `projected_depth_left/right_exact` + `support_left/right_exact` 作为 metric anchor
- 在这些 exact anchor 区域上，对 `pts3d_1.z` 做 robust scale alignment

也就是说，新路线不是“完全脱离当前 exact backend”，而是：
- 用 MASt3R direct pseudo depth 提供新的 depth 形状
- 用现有 exact projected depth 提供局部 metric 锚点

## 1. 当前 live depth pipeline 的真实情况

当前 online mapping 里的 pseudo depth target 不是 MASt3R direct depth，而是：
1. 在 pseudo / left KF / right KF 位姿下对当前 Gaussian map render depth
2. 用 matcher 只提供 pseudo↔ref pixel 对应
3. 在每个 ref 分支上：`ref_depth -> backproject -> world -> project to pseudo -> reproj_z`
4. 得到 `projected_depth_left_exact` / `projected_depth_right_exact`
5. 再用 exact support + confidence 组合成最终 `pseudo_depth_target_exact_brpo_upstream_target_v1`

所以当前 target depth 的本质是：
- “当前 map 对自己做双侧 keyframe 重投影得到的 depth target”
而不是：
- “pseudo GT RGB 直接经过 MASt3R 估的 dense pseudo depth”

## 2. 新方案的总体判断

这个改动应该拆成两个阶段：

### Phase A: 只换 target depth 的数值来源
目标：
- 只回答一个干净问题：当前没效果，到底是不是因为 `target_depth` 本身来自 self-projection，所以几乎不给 map 注入新几何

做法：
- 保留当前 exact verifier / exact support / exact C_m / valid_mask / target_confidence / exact_shared_cm_v1 不动
- 只把 `target_depth` 从“projected depth composition”替换成“MASt3R direct pseudo depth，经 exact projected depth anchor 后的 composition”

这是一个“值替换、contract 不替换”的最小实验。

### Phase B: 如果 A 没明显起色，再放开 depth-only contract
目标：
- 检查是不是 `exact_shared_cm_v1` 把 depth supervision 依然锁得太死，导致新的 dense depth 虽然算出来了，但大部分根本没进入 loss

做法：
- 保留 RGB 仍然使用严格 exact C_m
- depth 改用自己的 `depth_valid_mask × depth_confidence`（可再与 verify_union 相交）
- 新增 split contract，例如 `exact_split_directdepth_v1`

这一步属于 supervision contract 变更，不应和 Phase A 混在一起上线。

## 3. Phase A 的精确定义

### 3.1 输入图像
Phase A 里 depth 生成明确使用：
- pseudo frame: `pseudo_state["image_path"]` 对应的原始 dataset GT RGB
- left/right ref: 左右 keyframe 的原始 GT RGB

注意：
- verifier matching 仍然可以继续使用当前 fused pseudo RGB（例如 Difix 后的 `pseudo_fused_rgb.png`）
- 但 direct depth 必须使用 pseudo GT RGB，而不是 fused / restored pseudo RGB

因此，runtime backend 中必须显式保存：
- `pseudo_gt_rgb_path`
不能在后面被 `pseudo_state["image_path"] = pseudo_fused_rgb.png` 覆盖掉之后丢失。

### 3.2 每个分支的 direct depth 生成
对每个分支分别做一遍：
- `(pseudo_gt_rgb, left_ref_rgb)` -> MASt3R pair forward
- `(pseudo_gt_rgb, right_ref_rgb)` -> MASt3R pair forward

从每次 pair forward 中提取：
- `pts3d_1[..., 2]` 作为 pseudo-side raw direct depth
- `conf1` 作为 pseudo-side confidence 诊断信息

这一步的目标是得到：
- `direct_depth_left_raw`
- `direct_depth_right_raw`

### 3.3 metric scale anchoring
对每个分支分别用 exact projected depth 做 anchor：
- `anchor_mask_left = support_left_exact & projected_depth_left_exact > 0 & direct_depth_left_raw > 0`
- `scale_left = median(projected_depth_left_exact / direct_depth_left_raw)` on anchor mask
- `direct_depth_left_metric = scale_left * direct_depth_left_raw`

右侧同理。

原则：
- 不使用 `pts3d_2_in_1.z` 充当 metric ground
- 不直接和 dataset mono_depth 做主锚定
- 第一版统一使用当前 exact projected depth 做局部 robust metric anchor

### 3.4 target depth composition
Phase A 里仍然沿用当前 exact-upstream 的 target composition 语义：
- both-available: 按 `confidence_left_exact/right_exact` 做加权
- left-only: 直接用 `direct_depth_left_metric`
- right-only: 直接用 `direct_depth_right_metric`
- no render fallback

但这里“被组合的 depth 数值”不再是：
- `projected_depth_left/right_exact`
而改为：
- `direct_depth_left/right_metric`

### 3.5 Phase A 中保持不变的东西
以下都不改：
1. exact verifier 的 `support_left/right_exact`
2. exact C_m (`both->1.0, xor->0.5, none->0.0`)
3. `valid_mask`
4. `target_confidence`
5. `exact_shared_cm_v1` depth loss contract
6. pseudo RGB target / fused RGB / Difix 路线

这样可以保证 Phase A 真正只是在测：
- “把 target depth 的数值来源从 self-projected 换成 MASt3R direct depth anchored 到 exact metric 后，是否会改变 mapping 结果”

## 4. 为什么要先做 A，不直接做 B

因为当前系统里，影响 online mapping 没效果的原因可能有两类：

1. depth target 本身没提供新几何
- 因为现在 target depth 主要来自当前 map 对自己的回投影
- 这会让 pseudo depth loss 更像 self-consistency，而不是外部新约束

2. 新 depth 即使更好，也没真正进 loss
- 因为 `exact_shared_cm_v1` 的 effective mask 是：
  - `C_m × valid_mask × target_confidence`
- 这会把 supervision 继续锁在 exact verifier 的支持域里

如果 A 和 B 同时做，就没法区分到底是哪一层在限制效果。

## 5. 预期实现落点

### 5.1 `pseudo_branch/integration/runtime_exact_backend.py`
新增/修改：
1. `RuntimeExactBackendConfig.depth_generation_mode`
   - 默认: `projected`
   - 新模式: `mast3r_direct_exact_anchor_v1`
2. 保存 `pseudo_gt_rgb_path`
3. 新增 direct depth helper：
   - 调用 shared MASt3R pair forward
   - 提取 `pts3d_1[..., 2]`
   - 用 exact projected depth + support 做 scale anchor
4. 导出 debug artifact：
   - `direct_depth_left_mast3r_raw.npy`
   - `direct_depth_right_mast3r_raw.npy`
   - `direct_depth_left_mast3r_exact_anchor.npy`
   - `direct_depth_right_mast3r_exact_anchor.npy`
   - `direct_depth_mast3r_exact_anchor_meta.json`
5. `RuntimeExactBackendBundle` 额外携带：
   - `direct_depth_left`
   - `direct_depth_right`
   - `direct_depth_meta`

### 5.2 `pseudo_branch/target/depth_supervision_v2.py`
扩展 `build_exact_upstream_depth_target()`：
- 保留原 projected-depth 语义
- 允许传入 `target_depth_left_override` / `target_depth_right_override`
- 当 override 存在时，仍沿用 exact support/confidence 的 target composition 逻辑，但 depth 数值改用 override
- summary 中额外写清：
  - `target_depth_override_applied`
  - `depth_input_semantics`
  - `target_field_semantics`

### 5.3 `pseudo_branch/observation/pseudo_observation_brpo_style.py`
扩展 `build_exact_brpo_upstream_target_observation()`：
- 把 Phase A 的 override depth 透传到 `build_exact_upstream_depth_target()`
- summary/policy 中明确写清当前不是 projected depth，而是：
  - `mast3r_direct_depth_anchored_by_exact_projected`

### 5.4 `pseudo_branch/integration/runtime_signal_builder.py`
根据 `exact_bundle.exact_meta["depth_generation"]["mode"]`：
- `projected` -> 保持现状
- `mast3r_direct_exact_anchor_v1` -> 调 exact-upstream observation builder，但传入 direct depth override

### 5.5 `third_party/S3PO-GS/utils/slam_backend.py`
把 `depth_generation_mode` 从 config 透传进 `RuntimeExactBackendConfig`。

## 6. 验证标准

### 6.1 模块连接验证
需要证明的不只是“代码能跑”，而是：
- runtime exact backend 真的生成了 direct depth artifacts
- signal builder 真的使用了 override depth
- runtime pseudo record 中 `target_depth_runtime.npy` 已经不等于老的 projected-depth target
- meta / summary 明确记录 `depth_generation_mode = mast3r_direct_exact_anchor_v1`

### 6.2 直接数值验证
至少检查一帧：
- `target_depth_runtime.npy` 与旧 `projected_depth_left/right_exact` 的 compose 不再完全相同
- 在有效 source_map 区域内，新的 target depth 与 direct anchored depth compose 一致
- summary 中 `target_depth_override_applied = true`

### 6.3 实验验证
第一组正式实验应复用昨天 D6 设置，只新增一个变量：
- `depth_generation_mode: mast3r_direct_exact_anchor_v1`

保持：
- midpoint_only
- 1 pseudo/gap
- exact_shared_cm_v1
- update_real_pose=false
- use_gauss_newton=false
- Difix on
- 其他 trigger / iter / force keyframe 设置与 D6 对齐

## 7. Phase B 预告（本次不实现）

如果 Phase A 结果仍几乎不变，下一步不是继续调 scale，而是进入 Phase B：
- 新增 `exact_split_directdepth_v1`
- RGB 仍然严格使用 exact C_m
- depth 改用独立 depth-valid / depth-confidence contract
- 从而允许 dense direct depth 真正扩大 depth supervision 作用域

这一步必须作为第二个变量单独 compare，不能与 Phase A 一起首轮混改。

## 8. 当前推荐执行顺序

1. 覆盖旧文档为本版本
2. 实现 Phase A
3. 做 function-level smoke + artifact-level verification
4. 按 D6 protocol 跑一组新实验
5. 检查：
   - target depth 是否真换掉
   - loss / history / meta 是否证明新路径已被消费
6. 再决定是否进入 Phase B

## 9. 本文档的最终立场

- 旧规划的“大方向”对：要把 pseudo depth 从 self-projection 转向 pseudo GT RGB 的 MASt3R direct depth
- 旧规划最关键的“scale 理解”不对，必须改成“用 exact projected depth 做 metric anchor”
- 工程上应该先做 A（值替换），再做 B（contract 替换）
- A 成功与否，将直接告诉我们：问题更多在 target depth 数值来源，还是在 exact_shared_cm_v1 的 supervision contract