# S3PO Online Mapping：当前 Pipeline、Phase 3 Compare 计划与 Phase 4 规划（2026-05-04）

## 0. 目的

这份文档做三件事：
1. 简要记录当前已经修改后的 online mapping pipeline；
2. 记录接下来 Phase 3 representative compare 的实验规划；
3. 记录后续 Phase 4 的扩展规划。

更完整的背景与落地记录可参考：
- `docs/S3PO_PIPELINE_MAPPING_INTEGRATION_PLAN_20260503.md`
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`

---

## 1. 当前修改后的 pipeline（简要）

### 1.1 主线已经从什么，转成了什么

当前 pseudo supervision 的工程主线已经不再是：
- `after_opt` 之后再单独跑 `run_pseudo_refinement_v2.py`，或
- `slam.py::_maybe_run_brpo_pseudo_continuation()` 这种 post-hoc continuation。

当前主线已经转成：
- 在 S3PO backend 的 keyframe event 内，直接激活 pseudo slot；
- 直接基于 live runtime state 构建 exact backend bundle / exact-upstream signal / runtime pseudo record；
- 然后在同一个 backend keyframe path 里，附加执行一个 conservative pseudo-aware mapping block。

也就是说，pseudo supervision 现在已经从“after_opt 之后的 detached consumer”转成“backend mapping 内部的一个 online block”。

### 1.2 当前 live code path

当前 online mapping 主线的关键代码路径是：
- `pseudo_branch/integration/runtime_slot_selector.py`
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/integration/runtime_signal_builder.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py`
- `pseudo_branch/integration/runtime_debug_export.py`
- `third_party/S3PO-GS/utils/slam_frontend.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`

其中语义分工如下：

1. `slam_frontend.py`
   - 缓存 runtime camera states；
   - 在 keyframe message 里把这些 runtime states 传给 backend。

2. `slam_backend.py`
   - 先跑原生 real-only `map(self.current_window, ...)`；
   - 然后在 keyframe path 内调用 `_maybe_prepare_brpo_runtime_slots(...)`；
   - 为 newly-closed gap 选择 pseudo slot；
   - 构建 `exact_backend_v1/`、`signal_v2/`、`runtime_pseudo_record/`；
   - 如果打开 pseudo gradient，再调用 `_run_brpo_runtime_pseudo_mapping(...)`。

3. `slam_backend_brpo.py`
   - 已不再只是 continuation runner；
   - 现在保留 `run_brpo_pseudo_continuation(...)` 作为 control/reference；
   - 同时新增 `run_brpo_pseudo_mapping(...)`，供 online mapping 直接消费 runtime pseudo records；
   - 当前 online mapping 采用 conservative 设置：`split_pseudo_authority=true`，`pseudo_scene_mask_mode=both_only`。

### 1.3 当前 runtime dataflow

当前 runtime dataflow 可以简化成：

`real keyframe event`
→ `select runtime pseudo slot`
→ `build runtime exact backend bundle`
→ `build runtime exact-upstream signal`
→ `build runtime pseudo record`
→ `run conservative pseudo-aware mapping block`
→ `write event_summary / pseudo_mapping_summary / brpo_pseudo_history`

### 1.4 当前已完成的验证边界

当前已经真实完成并验证过的是：

1. Phase 1 parity：runtime rebuild 的 exact-upstream 核心数组与历史 reference bitwise 一致。
2. Phase 2 trigger/no-op：`current_window=[0,34]` 时能正确激活 midpoint pseudo `frame_id=17`，且 `pseudo_map_iters=0` 时 scene 不变。
3. Phase 3 conservative smoke：同一 gap 上，`pseudo_map_iters=2`、`pseudo_scene_mask_mode=both_only` 的 online pseudo-aware mapping 已真实执行成功，并写出：
   - `event_summary.json`
   - `pseudo_mapping_summary.json`
   - `pseudo_mapping/brpo_pseudo_history.json`

当前 smoke 还确认了一个重要点：
- 在 `update_real_pose=false` 的 conservative online block 下，real pose 不再被误更新；
- pseudo 已经真实进入 backend mapping loop，而不是只停留在 shell/build artifact 层。

### 1.5 当前仍然故意保守的地方

当前 live landing 仍然是保守版，不是 full policy：
- `placement_mode=midpoint_only`
- `max_pseudo_per_gap=1`
- `num_pseudo_views_per_step=1`
- `pseudo_scene_mask_mode=both_only`
- pseudo block 是 keyframe event 内的额外 mapping block，不改 tracking
- densify/prune 仍沿用原 S3PO backend 机制，不让 pseudo 直接改 densify/prune 统计
- 当前 smoke 使用的是 small-iter conservative setting，不代表 full online compare 已完成

---

## 2. 当前对 pipeline 的判断

当前更准确的判断不是“phase4 可以直接开始”，而是：
- Phase 1/2 shell/build integration 已完成；
- Phase 3 的 first conservative landing 已完成；
- 但还缺一轮更有代表性的 Phase 3 compare，来确认这条 online mapping 主线不是只在最小 smoke 下成立。

所以接下来最自然的顺序是：
- 先完成 Phase 3 representative compare；
- 再决定是否推进到 Phase 4 的更强 policy 扩展。

---

## 3. Phase 3 representative compare 计划

### 3.1 目标

目标不是立刻追求“超过某条 standalone winner”，而是先回答三个更基础的问题：

1. 当前 online integration shell 在真实 representative branch 上是否稳定；
2. active pseudo-aware mapping 是否比 no-op control 更容易造成 scene collapse；
3. 如果退化，问题更像 slot/builder 错误，还是 online dynamics 问题。

### 3.2 第一轮 compare 的控制原则

第一轮 compare 保持少变量，避免把 phase3 验证和 phase4 扩展混在一起：
- 沿用当前 live online mapping 主线；
- 不切换到新 policy family；
- tracking 不改；
- densify/prune 机制不改；
- pseudo 仍采用 `both_only` conservative scene authority；
- 先不做 multi-gap / top-k / stronger pseudo authority；
- matcher 先沿用当前已打通的 conservative runtime path，不把“matching policy compare”与“online dynamics compare”混成一轮。

### 3.3 compare arms

第一轮至少做三臂：

1. `baseline_s3po`
   - 完全不打开 online pseudo mapping；
   - 作为原始 backend mapping 参照。

2. `phase3_noop_control`
   - `brpo_online_mapping.enabled=true`；
   - 但 `pseudo_map_iters=0` 或 `lambda_pseudo=0`；
   - 作用是验证 integration shell 本身是否破坏 baseline。

3. `phase3_active_online_pseudo`
   - 打开当前 conservative online pseudo-aware mapping；
   - 例如：`midpoint_only + one pseudo per gap + both_only + small/medium pseudo_map_iters`；
   - 作用是测 active pseudo block 的真实在线影响。

如果第一轮 active arm 表现稳定，再考虑补一条更轻的 ablation，例如：
- `phase3_active_pose_only_authority`
- 或 `phase3_active_lower_pseudo_iters`

但这不作为第一轮必须项。

### 3.4 代表性 branch 选择原则

branch 的选择要满足两点：
- 不是最小 smoke；
- 也不是一上来就 full multi-gap 大跑。

因此第一轮更适合：
- 选一个代表性的 Re10k-1 online branch；
- keyframe 数量和 run 长度明显大于 smoke；
- 但仍保留清晰的 debug surface，确保 event-level summary 可审计。

### 3.5 输出与诊断面

这轮 compare 不只看最终 replay/render 指标，还要同时保留三层输出：

1. 最终评估层
   - baseline S3PO evaluation
   - before/after pseudo mapping 的渲染指标

2. event-level debug 层
   - `event_summary.json`
   - `pseudo_mapping_summary.json`
   - 每个激活 slot 的 exact / signal / record sidecar

3. history 层
   - `brpo_pseudo_history.json`
   - 重点看：
     - `loss_real`
     - `loss_pseudo`
     - `loss_pseudo_pose`
     - `loss_pseudo_scene`
     - sampled pseudo ids
     - pseudo effective mask stats

这样如果结果变差，可以先区分：
- shell/no-op 就坏了；
- slot/signal 本身异常；
- 还是 pseudo-aware mapping dynamics 本身不稳定。

### 3.6 第一轮通过标准

第一轮通过标准设得务实一些：

1. `phase3_noop_control` 应与 `baseline_s3po` 接近；
2. `phase3_active_online_pseudo` 至少不能出现 continuation 那种明显 scene collapse；
3. history/debug 必须能明确解释 active arm 的行为；
4. 如果 active arm 仍退化，也要能定位是 shell 问题、slot/builder 问题，还是 online dynamics 问题。

这轮先不把“必须优于 standalone winner”作为 gate。

---

## 4. 后续 Phase 4 规划

Phase 4 不是“让当前 conservative phase3 再多跑几次”，而是把 policy 从 conservative single-gap landing 扩成正式 compare 主线。

### 4.1 Phase 4 目标

把当前：
- midpoint only
- each gap <= 1 pseudo
- both_only conservative authority
- extra pseudo block after real map

扩成：
- 更完整的 pseudo selection / allocation policy；
- 更完整的 multi-gap / multi-pseudo online integration；
- 并评估 pseudo 对 backend mapping dynamics 的更深层作用范围。

### 4.2 计划中的 Phase 4 扩展轴

1. slot policy 扩展
   - 支持每个 gap 多个 pseudo 候选；
   - 支持 top-k 或 allocation policy；
   - 不再只限 midpoint only。

2. pseudo scheduling 扩展
   - 决定 pseudo 是：
     - 在 real-only map 之后追加 block；
     - 还是在 real map 之前；
     - 还是直接 unified 到同一个 mapping block。

3. authority 扩展
   - 从当前 `both_only` conservative mode 出发；
   - 评估是否扩到更完整的 valid support；
   - 但必须保留清晰 compare identity，不能静默改语义。

4. pseudo 数量扩展
   - 从 single-gap / single-pseudo，扩到 multi-gap / small multi-pseudo；
   - 观察是否出现系统性 scene collapse。

5. backend statistics 扩展
   - 评估 pseudo 是否需要影响 densify/prune 的统计；
   - 当前 phase3 不做这件事，phase4 才讨论。

### 4.3 Phase 4 的前提

进入 phase4 之前，最好先满足：
- phase3 representative compare 已完成；
- no-op shell 对 baseline 不构成明显破坏；
- active single-gap online pseudo 至少是可解释、可复现、无明显灾难性 collapse 的。

如果这些前提还没过，直接做 phase4 只会把问题维度继续放大。

---

## 5. 当前建议的执行顺序

建议的实际顺序是：

1. 先完成一轮 Phase 3 representative compare；
2. 用结果判断当前 online pseudo-aware mapping 是“已经可扩展”，还是“还要先做 conservative dynamics 修正”；
3. 如果通过，再进入 Phase 4：multi-gap / multi-pseudo / richer policy；
4. 如果不过，优先修正 phase3 online dynamics，而不是跳去做更强 policy。

---

## 6. 当前一句话结论

当前状态不是“已经该直接开始 phase4”，而是：
- phase3 的 first landing 已经完成；
- pseudo 已真实进入 backend mapping loop；
- 下一步应先做一轮 representative phase3 compare，把这条主线跑扎实；
- 然后再推进 phase4 的策略扩展。
