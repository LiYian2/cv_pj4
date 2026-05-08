# S3PO Pipeline Mapping Integration Plan (2026-05-03)

## 0. Goal

把 Part3 的 pseudo optimization 主线从“standalone / after_opt 后处理”正式转成“融入 S3PO backend mapping pipeline 的在线/准在线 pseudo-aware mapping”。

这里的决定不是继续给 `run_pseudo_refinement_v2.py` 或 `after_opt continuation` 做安全的小修补，而是做一个更有用、也更符合当前诊断结论的结构性转向：

- 保留 Part3 已经打磨出来的 exact M~/T~ supervision 语义；
- 放弃把 standalone consumer shell 当成未来主引擎；
- 也不把问题继续缩成“只调 optimization loss”；
- 而是把 pseudo 作为 backend mapping 的正式输入，进入 S3PO 的 map state transition。

一句话：后续主线不是“再做一次更稳的 after_opt continuation”，而是“让 pseudo 在 mapping 过程中进入 backend”。

---

## 1. 当前 live code flow：我们实际上在做什么

下面结论来自已检查的 live 代码与文档：

- `docs/PIPELINE.md`
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/design/REFINE_DESIGN.md`
- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/refine/pseudo_loss_v2.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- `pseudo_branch/refine/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`
- `third_party/S3PO-GS/slam.py`
- `third_party/S3PO-GS/utils/slam_backend.py`
- `third_party/S3PO-GS/utils/slam_utils.py`
- `third_party/S3PO-GS/utils/internal_eval_utils.py`

### 1.1 当前 standalone winner 的真实数据流

当前权威文档已经把 standalone winner 固化为：

`part2 S3PO full rerun`
→ `internal_eval_cache`
→ `prepare_stage1_difix_dataset_s3po_internal.py`
→ `build_brpo_v2_signal_from_internal_cache.py` / exact backend / signal_v2
→ `run_pseudo_refinement_v2.py`
→ replay eval

也就是说，当前 Part3 胜出的不是一个“在线插入 pseudo 的 S3PO 变体”，而是一条明显的两阶段离线路线：

1. Part2/S3PO 先完成自己的 run，并通过 `third_party/S3PO-GS/slam.py` + `utils/internal_eval_utils.py` 导出 `internal_eval_cache`；
2. Part3 再从这个 cache 离线构造 `pseudo_cache`、`signal_v2`、`exact_backend_v1`；
3. 最后由 `scripts/run_pseudo_refinement_v2.py` 在一个单独的 consumer shell 里消费它们。

### 1.2 当前 after_opt continuation 的真实位置

`third_party/S3PO-GS/slam.py` 当前的 BRPO integration 入口是：

- 正常 S3PO run 完成
- `after_opt` eval/export 完成
- `save_gaussians(..., "final_after_opt")`
- 然后才进入 `_maybe_run_brpo_pseudo_continuation()`
- 最后再做 `after_pseudo_opt` eval/export

所以当前 continuation 的工程位置是“post-after_opt continuation”，不是 mapping 内插。

### 1.3 S3PO backend 的真实语义：不是纯 local，但也不只是一个 loss loop

`third_party/S3PO-GS/utils/slam_backend.py::map()` 的真实行为是：

- 始终以 `current_window` 为核心视图集；
- 每轮对 `current_window` 中所有 keyframe render + loss；
- 再从窗口外随机抽 2 个 viewpoint 加入 mapping loss；
- 然后统一 backward；
- 再执行 densify / prune / opacity reset / Gaussian optimizer step / keyframe optimizer step / pose update。

因此 S3PO 的 backend 不是“只在一个纯局部子图上做 BA”，但它也绝不是“只多加一个 optimization loss”这么简单。它是一个以 current_window 为中心、但作用到全局 Gaussian scene 的 mapping process。

这也是这次转向的关键：后续集成目标应该是 mapping integration，而不是 optimization-only integration。

---

## 2. 为什么现在必须转向 mapping integration

过去这段时间已经足够说明几件事：

1. 单纯修 standalone consumer shell，不足以让方法语义接近你要的 BRPO-in-S3PO。
2. 单纯修 after_opt continuation，也不足以解决 scene degradation；它依然发生在“图已经建完之后再额外开一轮 pseudo-aware continuation”的时机上。
3. 现在最有价值的改变，不是继续偷偷做安全的小调整（再调权重、再缩 window、再做一个温和 compare），而是直接改变 pseudo 进入系统的层级：从 post-hoc consumer / continuation，变成 backend mapping input。

所以本计划明确选择：

- 不再把 `run_pseudo_refinement_v2.py` 当未来主引擎；
- 不再把 `slam.py::_maybe_run_brpo_pseudo_continuation()` 当最终 landing 目标；
- 主线转向：把 Part3 pseudo supervision 融入 S3PO backend mapping。

---

## 3. 目标架构：我们要做成什么样

## 3.1 目标不是“重写一套新 SLAM”

第一阶段不改 frontend tracking，不改 keyframe 判定，不改 color refinement 主逻辑。

我们要做的是：

- 保持 S3PO 现有 frontend / backend 主框架；
- 在 backend keyframe mapping 路径中加入 pseudo slot activation 与 pseudo-aware mapping；
- 让 pseudo 的生成、打包、消费发生在 backend mapping flow 中，而不再依赖离线 `internal_eval_cache -> prepare -> signal_v2 -> standalone refine` 这一长链。

## 3.2 目标结构（v1）

每当 backend 收到新 keyframe 并建立/更新 `current_window` 时：

1. 根据当前 keyframe 序列与策略，决定哪些 pseudo slot 现在“可激活”；
2. 用当前 live scene state + 相邻 GT/keyframe 信息，为这些 pseudo slot 直接构建 Part3 exact supervision bundle；
3. 把这些 pseudo bundle 转成 backend 可直接消费的 pseudo viewpoints / target fields；
4. 在同一个 backend mapping 过程中，执行 pseudo-aware mapping iterations；
5. scene / pose / densify / prune 的结果继续流入后续 keyframe 与后续 mapping，而不是等整条 run 结束后再补一次 continuation。

一句话：pseudo 应当进入 backend mapping state machine，而不是只进入一个事后的 optimize pass。

---

## 4. 现有代码应该如何重组

## 4.1 总原则

当前很多能力已经有了，但位置不对：

- exact M~/T~ 语义已经在 Part3 里打磨出来；
- backend pseudo loss / pseudo record loader / continuation shell 也已经有一部分可复用；
- 问题不是“没有组件”，而是组件还被组织在离线 producer + standalone consumer / post-after_opt continuation 这两种壳里。

所以工程主线不是重写全部，而是“提纯可复用核心 + 改壳”。

## 4.2 必须提纯成 library 的部分

### A. 从离线脚本里提纯 runtime builder

当前这些脚本仍然是 CLI/offline 入口：

- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

后续必须把其中真正有用的核心逻辑抽成可复用函数，而不是继续让 S3PO backend 通过“导出到磁盘再 shell 调脚本”来调用它们。

建议新增一个新的 live integration 层，例如：

- `pseudo_branch/integration/runtime_slot_selector.py`
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/integration/runtime_signal_builder.py`
- `pseudo_branch/integration/runtime_pseudo_bundle.py`
- `pseudo_branch/integration/runtime_debug_export.py`

它们分别负责：

1. 选择当前应该激活哪些 pseudo slot；
2. 用 live scene / camera state 直接构建 exact backend bundle；
3. 构建 exact-upstream signal（包括 `C_m` / `valid_mask` / `target_confidence` / target depth / source map）；
4. 组装成 backend pseudo records；
5. 可选写盘，便于和旧 `signal_v2` / `exact_backend_v1` 做 parity/debug。

### B. 从 standalone refine 里保留 supervision contract，不保留执行壳

当前 `scripts/run_pseudo_refinement_v2.py` 的价值主要在 supervision contract，不在执行壳。

需要保留/复用的是：

- `exact_brpo_upstream_target_v1` 的 bundle 语义；
- `build_stageA_loss_exact_shared_cm()` / `build_stageA_loss_paper_brpo_split()` 这类 loss contract；
- pseudo pose / exposure / abs prior 等约束的定义方式；
- diagnostics/history 结构。

不应继续保留为主壳的是：

- standalone StageA/StageB orchestration；
- 以 `stageA_history.json` 反向加载 pseudo samples 的消费方式；
- “先 prepare 完全部 pseudo_cache，再交给单独 refine 脚本”的执行结构。

### C. backend continuation 代码要拆成“共享 engine”和“具体入口”

当前 `third_party/S3PO-GS/utils/slam_backend_brpo.py` 里混有两类逻辑：

1. 真正通用的 backend pseudo loss / sampling / optimization 机制；
2. 明显只适合 `stageA_history.json` / `after_opt continuation` 的离线入口逻辑。

应该拆分为：

- shared pseudo-aware backend engine
- offline continuation entry（保留作 control）
- online/backend-mapping entry（新主线）

这样后续可以继续保留 continuation 作 regression control，但主线切到 online mapping。

---

## 5. 具体代码改动规划

## 5.1 Part3 repo 改动

### 5.1.1 新增 runtime integration package

建议在 `pseudo_branch/` 下新增一个明确的 integration 层，例如：

- `pseudo_branch/integration/__init__.py`
- `pseudo_branch/integration/runtime_slot_selector.py`
- `pseudo_branch/integration/runtime_exact_backend.py`
- `pseudo_branch/integration/runtime_signal_builder.py`
- `pseudo_branch/integration/runtime_pseudo_builder.py`
- `pseudo_branch/integration/runtime_debug_export.py`

职责建议如下：

1. `runtime_slot_selector.py`
   - 输入：当前 `kf_indices`、`current_window`、可用非 KF frame、策略配置
   - 输出：当前 iteration/keyframe 事件要激活的 pseudo slot 列表
   - 第一版只做最简单、最可控的策略：newly-closed gap / midpoint-only / top1-per-gap

2. `runtime_exact_backend.py`
   - 复用 `brpo_build_mask_from_internal_cache.py` 里真正的 matching + exact backend 逻辑
   - 输入不再是 `internal_eval_cache` 路径，而是 live camera states / live render / live refs
   - 输出：`support_left/right_exact`、`projected_depth_left/right_exact`、`confidence_left/right_exact`、`provenance` 等

3. `runtime_signal_builder.py`
   - 复用 `build_brpo_v2_signal_from_internal_cache.py` 里的 exact-upstream target/signal 逻辑
   - 直接生成当前 pseudo frame 对应的 exact signal bundle
   - 输出字段必须和现有 exact-upstream consumer 对齐

4. `runtime_pseudo_builder.py`
   - 把 runtime signal 转成 `BackendPseudoViewRecord`
   - 不再依赖 `stageA_history.json`
   - 可以重用 `backend_pseudo_view_loader.py` 的 record schema，但输入源要改成 in-memory bundle

5. `runtime_debug_export.py`
   - 第一阶段一定要有
   - 作用：把 runtime 生成的 pseudo bundle 按旧 `signal_v2/frame_xxxx` / `exact_backend_v1/frame_xxxx` 风格写到 debug root，便于做 parity 和定位错误

### 5.1.2 调整现有 CLI 脚本角色

以下脚本继续保留，但角色改变：

- `scripts/prepare_stage1_difix_dataset_s3po_internal.py`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `scripts/run_pseudo_refinement_v2.py`

它们后续应成为：

- offline control
- regression reference
- debug/probe tool

而不是线上主路径。

## 5.2 third_party/S3PO-GS 改动

### 5.2.1 `utils/slam_backend.py`

这是主 landing 点。

当前 `keyframe` 消息分支会：

- 更新 `self.current_window`
- `add_next_kf(...)`
- 建立 keyframe optimizer
- 调 `self.map(self.current_window, ...)`
- prune
- push_to_frontend

新主线需要在这里加入 pseudo-aware mapping 调度。推荐方式：

1. 保留现有 real-only `map()` 作为基线实现；
2. 新增一个新的 pseudo-aware mapping 入口，例如：
   - `map_with_brpo_pseudo(...)`
   - 或 `run_brpo_mapping_step(...)`
3. 在 keyframe backend 分支按 config 决定调用：
   - 只跑 real-only map
   - 或先 real map 再 pseudo-aware map
   - 或把 pseudo 直接并入一个统一 mapping loop

第一版建议选择中间路线：

- 仍以 S3PO backend map shell 为核心；
- 但 pseudo-aware 部分作为 keyframe event 内的额外 mapping block；
- 暂时不改 frontend，不改 tracking。

### 5.2.2 `utils/slam_backend_brpo.py`

把它从“continuation-specific runner”提升为“shared pseudo-aware backend engine”。

建议拆成两层：

- shared engine：
  - pseudo sampling
  - pseudo loss compute
  - split authority / scene mask mode
  - shared history/diagnostics
- entry adapters：
  - `run_brpo_pseudo_continuation(...)` 保留
  - 新增 `run_brpo_pseudo_mapping(...)` 或 `map_with_runtime_pseudo_records(...)`

### 5.2.3 `slam.py`

`slam.py` 的职责不应再是主 integration 壳，只做：

- 读取 config；
- 创建 backend/fronted；
- 在 final eval/export 处按需要导出 debug/summary。

当前 `_maybe_run_brpo_pseudo_continuation()` 不删除，但降级为：

- offline control path
- compare baseline path

新增一套 config block 用于 online mapping integration，例如：

```yaml
Results:
  brpo_online_mapping:
    enabled: true
    trigger: keyframe
    placement_mode: midpoint_only
    max_pseudo_per_gap: 1
    pseudo_map_iters: 20
    num_pseudo_views_per_step: 1
    lambda_real: 1.0
    lambda_pseudo: 1.0
    lambda_depth: 1.0
    beta_rgb: 0.7
    lambda_pose: 0.01
    lambda_exp: 0.001
    lambda_abs_t: 3.0
    lambda_abs_r: 0.1
    pseudo_scene_mask_mode: both_only
    debug_export_root: ...
```

### 5.2.4 `utils/internal_eval_utils.py`

在线 mapping 主线不能再依赖 `internal_eval_cache` 才能运行；

但 `internal_eval_cache` 仍然保留两个作用：

1. offline reference / parity compare
2. 最终 before_opt / after_opt / after_pseudo_opt 评估导出

也就是说，`internal_eval_cache` 从“主依赖”降级为“debug 与评估边车”。

## 5.3 Part2 / orchestration 层改动

当前 Part2 的 S3PO rerun 主线把 `internal_eval_cache` 作为下游 Part3 的生产源。

转向 online mapping 后，Part2/Part3 边界需要改成：

- Part2 不再只是“先跑完，再吐离线 cache 给 Part3”；
- 而是 S3PO run 本身在 backend mapping 阶段直接调用 Part3 integration library；
- 最终只在需要 compare/debug 时导出 `after_opt` / `after_pseudo_opt` snapshot。

这意味着：

- “Part2 产出供 Part3 离线消费”将降级为 reference flow；
- “Part3 optimization 融入 S3PO backend mapping”成为新 live flow。

---

## 6. 建议的实施阶段

## Phase 0: 冻结当前 control，不再继续无意义 continuation 探索

目标：先停掉当前无效主线扩展。

动作：

- 保留当前 `after_opt continuation` 代码与代表性 compare 作为 control；
- 不再继续基于它做更多 matcher sweep / 小权重 sweep / 安全微调；
- 文档口径明确：continuation 只是过渡 control，不是最终 landing 目标。

验收：

- control 路径可重复跑；
- 新主线不依赖继续修它。

## Phase 1: 提纯 runtime builders（最关键的基础设施）

目标：把离线 producer 变成 live library。

动作：

1. 从 `prepare_stage1_difix_dataset_s3po_internal.py` 提纯 pseudo slot record / ref pairing / input assembly；
2. 从 `brpo_build_mask_from_internal_cache.py` 提纯 exact backend runtime builder；
3. 从 `build_brpo_v2_signal_from_internal_cache.py` 提纯 exact signal runtime builder；
4. 建立 runtime pseudo bundle → `BackendPseudoViewRecord` 的 in-memory path；
5. 加 debug export，使 runtime 输出可以按旧目录格式落盘。

验收：

- 给定一个固定 frame_id，在离线 prepare root 和 runtime builder 上，能得到可对齐的 exact backend / signal 结果；
- support mask / valid mask / target_confidence / target depth 的关键数组在容忍误差内一致；
- 不需要 `stageA_history.json` 就能构造 backend pseudo record。

## Phase 2: backend keyframe event 中接入 pseudo slot activation

目标：pseudo 不再只在 run 结束后出现，而是在 keyframe backend path 中被触发。

动作：

1. 在 `utils/slam_backend.py` 的 keyframe 分支中，加入 pseudo slot selector；
2. 第一版只允许最简单触发策略：
   - newly-closed gap
   - midpoint-only
   - 每个 gap 最多 1 个 pseudo
3. 每次 keyframe event 都把当前激活的 pseudo bundle 写到 debug export root；
4. 暂时可以先只构建 pseudo records，不开 pseudo gradient，先验证调度和数据正确性。

验收：

- backend 日志能明确打印：当前 keyframe、新增 gap、激活 pseudo slot、关联 left/right refs；
- debug root 中能看到 per-slot exact bundle 与 signal bundle；
- 无 pseudo gradient 的情况下，run 结果与 baseline S3PO 基本一致。

## Phase 3: pseudo-aware backend mapping loop

目标：真正让 pseudo 进入 mapping，而不是只生成 bundle。

动作：

1. 新增 `map_with_brpo_pseudo(...)` 或等价入口；
2. 复用 shared pseudo-aware backend engine；
3. 第一版只开放最保守的 pseudo authority：
   - pseudo pose/exposure 可动；
   - pseudo scene authority 默认 `both_only`；
   - densify/prune 仍沿用 S3PO backend 机制；
   - tracking 不变。
4. 将 pseudo-aware mapping block 直接放在 keyframe backend path 中执行。

验收：

- pseudo-aware mapping 可以在真实 run 中稳定执行；
- history 中能同时看到 real mapping 与 pseudo mapping 的 loss stats；
- replay/original-pose 评估不再只是 continuation 那种“run 结束再补一轮”。

## Phase 4: 从 conservative online integration 扩到 full policy

目标：让 online integration 从 smoke/control 变成正式 compare 主线。

动作：

1. 支持每个 gap 多个 pseudo 候选 / top-k 策略；
2. 支持更完整的 pseudo selection policy（midpoint / allocation / manifest-driven）；
3. 决定 pseudo 是插在 real-only map 前、后，还是 unified 在同一 mapping block；
4. 评估是否需要让 pseudo 影响 densify/prune 的统计，而不只是 loss。

验收：

- 新主线可以在 Re10k-1 上完成完整 run；
- 与 control（baseline S3PO / after_opt continuation / standalone winner）有清晰 compare；
- 结果可复现。

---

## 7. 测试与验收梯度

后续执行必须按这个顺序，不要跳。

### Test 1: pure-function parity test

目的：证明 runtime builder 没改坏 Part3 语义。

做法：

- 选 1 个已有 prepare root + 1 个 frame；
- 离线 `signal_v2/frame_xxxx` / `exact_backend_v1/frame_xxxx` 作为 reference；
- 用 runtime builder 重新构建；
- 逐项比较：
  - support_left/right
  - both/single mask
  - projected depth
  - target depth
  - valid mask
  - target confidence

通过标准：

- 离散 support / valid 类数组应完全一致或接近 bitwise 一致；
- 连续 target/depth/conf 数值应在合理误差内一致；
- 若不一致，必须先修 parity，再做下一步。

### Test 2: backend trigger smoke

目的：证明 pseudo slot 能在 live backend keyframe path 被正确激活。

做法：

- 小数据 / 少 keyframe run；
- pseudo integration 打开，但 pseudo loss 关闭；
- 只检查 selector / runtime bundle / debug export。

通过标准：

- 日志与 debug root 都能看到正确 pseudo slot；
- baseline 指标基本不变。

### Test 3: no-op parity / shell safety test

目的：证明新集成壳本身不会悄悄改坏 S3PO。

做法：

- `enabled=true`，但 `lambda_pseudo=0` 或 `pseudo_map_iters=0`；
- 与 baseline S3PO compare。

通过标准：

- 指标应接近 baseline；
- 若明显劣化，说明 integration shell 本身有问题。

### Test 4: single-gap online pseudo smoke

目的：第一次真正验证 online pseudo-aware mapping。

做法：

- 只对 1 个 gap 激活 1 个 pseudo；
- `both_only` scene authority；
- 小 iteration；
- 保留 debug export。

通过标准：

- 不崩；
- history 可解释；
- scene 不出现 continuation 那种立即性灾难性下降。

### Test 5: representative full online branch

目的：代替之前 continuation 单臂试验。

做法：

- 选代表性 Re10k-1 branch；
- 跑完整 online pseudo-aware mapping；
- 与 baseline S3PO、after_opt continuation、standalone winner 三方比较。

通过标准：

- 至少不能明显破坏 baseline；
- 如果仍退化，要能从 history/debug 判断是 slot builder 问题还是 online dynamics 问题。

### Test 6: multi-gap / multi-pseudo run

目的：验证它不是只在最小 smoke 下成立。

做法：

- 开多个 gap；
- 允许少量 pseudo 并行；
- 保持 conservative scene authority。

通过标准：

- 不出现系统性 scene collapse；
- debug/history 仍可审计。

---

## 8. 关键验收标准

本项目转向成功，不是看“代码能跑”，而要满足下面几条。

### 8.1 结构验收

- 主路径不再依赖先导出 `internal_eval_cache` 再 shell 跑 Part3 脚本；
- pseudo supervision 在 backend mapping 中被构建与消费；
- `run_pseudo_refinement_v2.py` 明确退为 reference/control。

### 8.2 语义验收

- 复用的是 Part3 exact M~/T~ 语义，而不是退回 legacy `pseudo_refinement()`；
- pseudo 不是只在 after_opt 之后出现；
- pseudo slot 与相邻 keyframe 的关系在 mapping 中被显式保留。

### 8.3 动力学验收

- new shell 本身不能在 `lambda_pseudo=0` 时就破坏 baseline；
- online pseudo-aware mapping 的劣化/收益必须能被 history/debug 定位；
- 不接受“只是又能跑一个 continuation compare”。

### 8.4 结果验收

- 至少先达到“不显著破坏 baseline S3PO”；
- 然后再追求优于 current continuation；
- 最终再看能否与 standalone winner 做有意义对比。

---

## 9. 明确不做什么

这次转向过程中，以下事情不应再作为主线：

1. 不再继续把 `run_pseudo_refinement_v2.py` 打磨成最终引擎。
2. 不再继续把 `after_opt continuation` 当最终形态，只保留为 control。
3. 不先改 frontend tracking。
4. 不先改 color refinement。
5. 不先做更多 dense/sparse / q-threshold / 小权重 sweep。
6. 不做“为了安全先小修一个更稳的 continuation”这种偷偷保守策略。

后续真正有用的改变，是把 pseudo 放进 backend mapping pipeline。

---

## 10. 推荐的落地顺序（执行版）

后续真正执行时，建议严格按以下顺序推进：

1. 建 runtime integration package，并把离线脚本逻辑提纯成 pure functions；
2. 做 single-frame parity，验证 runtime builder 与旧 signal/exact artifacts 一致；
3. 把 pseudo slot activation 接进 `slam_backend.py` keyframe path，但先不开 pseudo gradient；
4. 做 no-op parity，证明 integration shell 不破坏 baseline；
5. 接 pseudo-aware mapping block，先 conservative `both_only`；
6. 跑 representative online branch；
7. 再决定是否扩到 multi-gap / stronger pseudo authority。

如果后面需要正式执行，这份文档就是实施顺序与验收标准。
