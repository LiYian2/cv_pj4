# REFINE_SLAM_STYLE_EXEC_PLAN_20260430

更新时间：2026-04-30 20:18 (Asia/Shanghai)

## 目标

本文件不是“如何继续安全地补现有 consumer shell”，而是一个明确的工程落地规划：把 Part3 当前的 refine 主线改造成更接近 S3PO 的 SLAM-style joint pose-map optimization，同时保留我们已经查明并积累下来的 exact M~/T~ supervision 语义。

## 当前执行状态（2026-04-30 晚）

按压缩后的三阶段执行口径：

1. **Phase 1 已完成**：`pseudo_branch/refine/backend_pseudo_bundle.py`、`backend_pseudo_view_loader.py`、`backend_pseudo_loss.py` 已落地，remote py_compile 与 direct smoke 都已通过。
2. **Phase 2 核心已完成**：`third_party/S3PO-GS/utils/slam_backend_brpo.py` 与 `third_party/S3PO-GS/slam.py` hook 已落地，`SLAM._maybe_run_brpo_pseudo_continuation()` 的 actual hook smoke 已通过。产物根：`/data/bzhang512/tmp/s3po_brpo_hook_manual_smoke/`。
3. **当前仍未完成的部分**：还没有做 canonical full compare；目前验证边界是 hook-level after_opt smoke，而不是整条 frontend rerun compare。
4. **为支撑 smoke/debug 新增的配套改动**：`third_party/S3PO-GS/utils/slam_backend.py` 已支持 `S3PO_COLOR_REFINEMENT_ITERS` override，用于把默认 26k color refinement 缩到 debug 可承受范围。

---

## 总体判断

当前应保留的是 supervision 语义，不应保留的是 refine 执行器。

更具体地说：

- 保留：current exact M~ / exact upstream T~ / signal bundle 语义
- 替换：`run_pseudo_refinement_v2.py` 这套离线 consumer + micro Gaussian optimizer 壳
- 新主线：S3PO-style post-`after_opt` backend continuation

这里的第一目标不是“先做最小改动确认不炸”，而是把优化范式从错误壳切换到正确壳。第一阶段可以控制边界，但边界控制不等于继续沿用 current shell。

## 1. 目标架构（一句话）

新架构应是：

`S3PO normal run -> after_opt live scene -> BRPO pseudo backend continuation -> after_pseudo_opt eval/export`

也就是说，不再用“导出 PLY -> standalone consumer 重新读 PLY -> pseudo StageA/B”作为主 refine 路线，而是在 S3PO backend 里继续做一个新的 pseudo-aware mapping stage。

## 2. 新架构必须满足的结构条件

### 2.1 real keyframe window 必须回到优化中心

新的 joint stage 必须以 S3PO backend 当前 real keyframe window 为主体，而不是像 current StageB 那样只从 manifest 里抽两个固定 real RGB anchor。

这意味着 real branch 必须满足：

- real keyframe pose 可优化
- real RGB-D mapping loss 继续存在
- real views 来自 live backend window，而不是离线兼容 loader

### 2.2 Gaussian 优化必须回到 backend optimizer

不能继续用 `build_micro_gaussian_param_groups()` 给 `_xyz` / `_opacity` 单独起一套 optimizer。新的主线应直接复用：

- `gaussians.training_setup(opt_params)`
- `gaussians.optimizer`
- `gaussians.update_learning_rate(...)`
- `densify_and_prune(...)`
- `reset_opacity_nonvisible(...)`

需要讨论的是 pseudo stage 是否一开始就允许 densify/prune 全开；但无论开关如何，执行器必须先回到 backend optimizer 路线，而不是保留 micro optimizer 壳。

### 2.3 pseudo views 必须作为 backend 附加约束视图，而不是主训练对象

新的 pseudo stage 中，pseudo views 应是附加约束，不应再成为 optimization 主体。优化中心必须是：

- real window keyframes
- full Gaussian map
- pseudo support 作为补充监督

这会直接改变当前 pseudo branch 主导、real branch 过弱的结构问题。

### 2.4 要有真正独立的 final appearance refinement

current shell 把 StageB 当成 refine 终点是不够的。新主线应恢复两段后期优化：

- joint pose-map stage
- pose 基本收敛后的 appearance refinement stage

这一步可以先复用 S3PO 的 `color_refinement()` 框架，再逐步引入 BRPO 的 confidence-weighted image refine 语义。

## 3. 第一落地边界：不是 online tracking 改写，而是 after_opt backend continuation

为了避免把 tracking / keyframe insertion / online map 全部搅在一起，第一步建议把重构边界设在：

- S3PO 正常跑完
- `after_opt` scene 已经稳定
- 然后进入新的 BRPO pseudo backend continuation

这不是“最小改动思维”，而是“先替换 refine 执行器，再决定是否把 producer 进一步在线化”的工程分层。它已经是有意义的大改，因为 joint optimizer 壳已经换掉了。

## 4. 具体执行计划

### Phase 0：冻结 current shell 为 reference，不再继续扩它

目标：把 current shell 明确降级为对照线，而不是继续加主功能。

动作：

1. 保留 `run_pseudo_refinement_v2.py` 作为 reference / replay baseline。
2. 明确后续新功能不再优先落在这个脚本里。
3. 把 current shell 的职责收缩为：
   - 旧对照
   - 快速 consumer-only probe
   - 数据比对工具

验收：

- 文档口径上不再把 `run_pseudo_refinement_v2.py` 称为未来主 refine 引擎。

### Phase 1：抽离 pseudo bundle loader 与 exact loss contract

目标：把 supervision 语义从 standalone 脚本中抽出来，变成 backend 可复用模块。

建议新模块：

- `pseudo_branch/refine/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`

建议内容：

1. 定义 `PseudoBundleSample` / `PseudoBundleBatch` 数据结构，显式承载：
   - target RGB
   - target depth
   - confidence mask (`C_m`)
   - valid mask
   - target confidence
   - source map
   - pseudo viewpoint init pose / exposure state
2. 把 current exact shared-C_m loss 封成 backend 可直接调用的函数，不再绑死在 StageA/StageB consumer 代码里。
3. 明确区分：
   - real mapping loss
   - pseudo exact RGB-D loss
   - pose / exposure prior
   - optional appearance refine loss

验收：

- 不启动 standalone refine，也能在单个 pseudo sample 上独立计算 exact pseudo loss。
- loader 能直接从当前 `signal_v2` / `exact_backend_v1` 产物构造 pseudo sample 对象。

### Phase 2：实现 backend continuation runner

目标：不再“读 PLY 重新开局”，而是从 S3PO `after_opt` 的 live scene 继续优化。

建议实现位置二选一：

方案 A（更干净，推荐）：
- 新建 `third_party/S3PO-GS/utils/slam_backend_brpo.py`
- 由它复用 `slam_backend.py` 的 mapping 机制，并加上 pseudo branch

方案 B（改动更集中）：
- 直接在 `third_party/S3PO-GS/utils/slam_backend.py` 增加 `map_with_pseudo(...)` / `brpo_pseudo_continuation(...)`

建议先走方案 A，原因是：

- 不污染现有纯 S3PO backend
- 方便并排比较 old mapping 与 new pseudo mapping
- 失败时更容易回退

这个 runner 需要做的事：

1. 接收 live `gaussians`、real `viewpoints`、current window。
2. 接收 pseudo bundle root 或已加载 pseudo bundle。
3. 构建 pseudo viewpoints，并把它们纳入同一 backend optimization loop。
4. 复用 `gaussians.optimizer` 与 real keyframe pose optimizer，而不是再建 micro Gaussian optimizer。
5. 每轮同时计算：
   - real mapping loss（RGB-D）
   - pseudo exact loss（RGB-D, masked）
   - pose/exposure priors
6. 统一 backward / step。

验收：

- 新 runner 在不调用 `run_pseudo_refinement_v2.py` 的前提下，能完成一个 `after_opt -> after_pseudo_opt` smoke。
- 输出可以直接保存新的 gaussians 与 camera states。

### Phase 3：恢复真正的三段式优化日程

目标：把当前“StageA + StageB 脚本式拼接”换成 backend 内的显式调度。

建议阶段：

1. `PseudoPoseWarmup`
   - 只动 pseudo pose / exposure
   - real map 基本冻结或仅 very light maintenance
2. `JointPseudoMapping`
   - real keyframe poses + pseudo poses + Gaussian map 一起优化
   - 这是主 stage
3. `AppearanceRefine`
   - pose 基本冻结
   - 以 image quality 为主，做 final refine

建议新模块：

- `pseudo_branch/refine/backend_scheduler.py`
- `pseudo_branch/refine/backend_stage_configs.py`

验收：

- 日志与 history 能明确分开记录三段式阶段。
- `after_pseudo_opt` 不是 current StageB 的换皮，而是真正的 backend schedule。

### Phase 4：把入口挂到 `slam.py` after_opt 之后

目标：让新 refine 成为 S3PO 完整 pipeline 的标准 continuation stage。

建议入口改动：

- `third_party/S3PO-GS/slam.py`

建议流程：

1. 正常完成 before_opt / after_opt / color_refinement
2. 检查是否启用 BRPO pseudo continuation
3. 若启用：
   - 加载 pseudo bundle
   - 调 backend continuation runner
   - 保存 `after_pseudo_opt`
   - 再跑一轮 evaluation/export

建议新增产物：

- `after_pseudo_opt/point_cloud/point_cloud.ply`
- `after_pseudo_opt/camera_states.json`
- `after_pseudo_opt/replay_eval/...`
- `after_pseudo_opt/brpo_pseudo_history.json`

验收：

- 一次 S3PO run 可以自然生成 `after_opt` 与 `after_pseudo_opt` 两个后期状态。
- 新路径不需要再调用 standalone refine 脚本作为主执行器。

### Phase 5：validation ladder（按壳替换而不是按小技巧扫参）

目标：验证我们换掉的是正确的壳，而不是又堆一层复杂性。

建议验证顺序：

1. one-scene / one-pseudo-sample backend smoke
2. 8 pseudo views backend smoke
3. 保持 current exact sparse pseudo bundle，比较：
   - old standalone consumer shell
   - new backend continuation shell
4. 在 new backend shell 下再比较：
   - sparse 2D
   - dense3d q070 / q090
5. 只在 new backend shell 初步成立后，才继续考虑更进一步的 online integration

关键判据：

- 不再出现 current shell 那种“real RGB 更好但 replay 更坏”的明显弱锚症状
- stronger pseudo route 的 early-positive / late-collapse 应显著减轻
- dense3d 是否真正转正，要在 new backend shell 下重新评估，而不是沿用 old shell 结论

## 5. 代码落点建议

### 应新增

- `pseudo_branch/refine/backend_pseudo_bundle.py`
- `pseudo_branch/refine/backend_pseudo_view_loader.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`
- `pseudo_branch/refine/backend_scheduler.py`
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`
- 可选：`scripts/run_brpo_backend_continuation.py`（仅作为 smoke / debug wrapper，不作为最终主执行器）

### 应改造

- `third_party/S3PO-GS/slam.py`
- 必要时 `third_party/S3PO-GS/utils/slam_backend.py` 的公共 helper 抽取
- `pseudo_branch/refine/__init__.py` 以导出新 backend 子模块

### 应降级为 reference

- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/gaussian_management/gaussian_param_groups.py` 中的 micro optimizer 路线

这里的“降级”不是马上删除，而是停止把它们当未来主线。

## 6. 第一轮实现时不要再做的事

为了防止工程又滑回 current shell 旋钮战，第一轮实现中不要把以下动作当主目标：

- 继续在 `run_pseudo_refinement_v2.py` 上堆新的 StageB 变体
- 先扫一轮 `xyz / xyz_opacity / lr / real:pseudo ratio`
- 先把 `C_m` 改 continuous 再看能不能救
- 先把 dense3d q 再扫一遍

这些都可以存在，但都必须退回到新 backend 壳打通之后再讨论。

## 7. 第一轮工程验收标准

这轮大改不应以“有没有少量提升”验收，而应以“主壳是否真的换掉”验收。最低验收标准应是：

1. 新路径不再以 standalone PLY consumer 作为主 refine 执行器。
2. 新路径直接在 S3PO backend continuation 中优化。
3. real keyframe poses 在 joint stage 中重新进入优化回路。
4. Gaussian 优化走 backend optimizer 而不是 micro xyz-only optimizer。
5. pseudo exact supervision 作为 backend 附加约束视图接入。
6. 产物里有 `after_pseudo_opt`，而不只是另一个 standalone `refined_gaussians.ply`。

只要这六条还没成立，就不算真正进入了 SLAM-style joint pose-map optimization。

## 8. 最终口径

后续真正有意义的工作不是“继续修 current refine”，而是“保留 exact supervision，替换 refine 引擎”。

因此本规划的核心不是再给 current shell 打补丁，而是把 Part3 的后期优化主线改造成：

- real-window-centered
- backend-optimizer-based
- pseudo-aware
- full pose-map joint
- with final appearance refinement

这才是本项目下一步值得投入的大方向。
