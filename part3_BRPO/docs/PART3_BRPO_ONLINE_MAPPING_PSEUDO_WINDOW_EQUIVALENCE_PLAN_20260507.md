# Part3 BRPO Online Mapping：Pseudo 作为 Loss Window 等价成员的落地规划

**日期**: 2026-05-07
**状态**: 待实施
**优先级**: 最高
**目标**: 把当前的 pseudo side-branch online mapping，改成“pseudo 不参与新 keyframe / fusion，但在优化 loss 中与 real keyframe 等价、只是多一个 mask”的 joint-primary 拓扑。

---

## 0. 结论摘要

当前 live online mapping 与作者口述语义之间，最关键的结构差异不是参数，而是 **优化拓扑**：

1. 当前 keyframe event 先跑标准 `self.map(...)`，再额外跑一次 pseudo mapping side branch。
2. 标准 `map()` 只消费 real keyframes + 2 个随机 real views，pseudo 不在主 mapping window 里。
3. 当前 BRPO joint engine 虽然已经能在一个 `total_loss` 里累加 real + pseudo loss，但它只是主 mapping 之后的补跑块，不是 primary mapping executor。
4. 当前多数实验设置 `update_real_pose=false`，因此 pseudo loss 不能直接进入 real pose 的优化闭环。
5. 当前常用 `split_pseudo_authority=true` + `pseudo_scene_mask_mode=all_valid`，在 live 代码里会把同一条 pseudo 记录以两份 loss 重复消费，这并不等价于“一个带 mask 的 window member”。

因此，本规划的核心不是“再调 lambda / GN / pseudo 个数”，而是把 pseudo 从 **后挂监督支路** 提升成 **joint mapping optimization batch 中的等价成员**。

---

## 1. 当前 live 行为：代码级证据

### 1.1 Keyframe event 的执行顺序仍然是 real mapping → pseudo side branch

**文件**: `third_party/S3PO-GS/utils/slam_backend.py`

`keyframe` 分支的 live 路径（约 981-1058 行）当前是：

```python
self.map(self.current_window, iters=iter_per_kf, up_pose=True)
self.map(self.current_window, prune=True)
prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)
```

这说明 pseudo 不是 primary mapping loop 的成员，而是标准 S3PO mapping 之后额外再跑的 BRPO block。

### 1.2 标准 `map()` 完全不包含 pseudo

**文件**: `third_party/S3PO-GS/utils/slam_backend.py`

`map()`（约 512-680 行）当前只做：

1. `current_window` 中 real keyframes 的 render + loss
2. `random_viewpoint_stack` 中随机 2 个 real 非窗口视角的 loss
3. Gaussian densify/prune/opacity reset
4. real keyframe optimizer step + `update_pose(viewpoint)`

当前 `map()` 不读：
- `brpo_runtime_pseudo_records`
- `prepare_payload`
- 任何 pseudo mask / pseudo viewpoint

### 1.3 现有 BRPO engine 其实已经具备 joint loop 雏形

**文件**: `third_party/S3PO-GS/utils/slam_backend_brpo.py`

`_run_joint_pseudo_engine()`（约 481-607 行）已经会：

1. 遍历 `current_window` real views 计算 `real_loss_sum`
2. 可选采样 `extra_real_candidates`
3. 遍历 `sampled_pseudo` 计算 pseudo loss
4. 组装统一 `total_loss`
5. 对 gaussians / real pose / pseudo pose / exposure 做 step

也就是说，当前不是“完全没有 joint executor”，而是 **joint executor 存在，但被放在主 mapping 之后作为 side branch 调用**。

### 1.4 当前 pseudo 并没有作为“单条 masked member”被消费

**文件**: `third_party/S3PO-GS/utils/slam_backend_brpo.py`

当前常见设置：

```yaml
split_pseudo_authority: true
pseudo_scene_mask_mode: all_valid
```

但 `_scene_record_for_mode()`（约 401-420 行）在 `all_valid` 下直接 `return record`。
因此每个 pseudo sample 会被分成：

1. `pseudo_pose_loss`
2. `pseudo_scene_loss`

而且两份 loss 对应的 record 本质是同一份 mask/target。

这与“pseudo 就是一个带 mask 的 keyframe member”不一致；当前更像“把同一个 pseudo 记录拆成两份 authority 逻辑反复使用”。

### 1.5 当前若直接把 BRPO engine 替换成主 mapping，还会丢掉标准 map 的一部分语义

这个点必须提前记录，否则会在落地时悄悄退化：

1. `map()` 默认包含 2 个随机 real 非窗口视角。
2. `BRPOMappingConfig` 当前 **没有** `extra_real_views` 字段。
3. `_run_joint_pseudo_engine()` 虽然写了 `extra_real_views = int(getattr(cfg, "extra_real_views", 0))`，但 live config 解析里没有这一项，且现有 yaml 也没有设置它。

因此如果直接用 BRPO engine 取代 `map()` 而不补 `extra_real_views`，会导致 primary mapping 与现有 baseline 的 real-view supervision contract 不一致。

---

## 2. 为什么必须改：这不是小调参，而是 pipeline correctness

论文作者新增澄清是：

1. pseudo 不参与提供新的 keyframe / fusion；
2. 但在 loss refine 上，pseudo 与 keyframe 是等价成员，只是 pseudo 的 loss 带 mask；
3. 如果 keyframe pose optimization 能影响别的 keyframe 或非 keyframe pose，那么 pseudo 也应通过同一套 refine 体系产生影响。

当前 live 代码不满足第 2 条和第 3 条：

- pseudo 没进主 mapping loop
- pseudo 只是后挂 side branch
- 多数配置还冻结 real pose
- pseudo 常被拆成两份 authority 重复计算

因此现在即使看到“pseudo mapping 在跑”“Gaussian 在动”，也不能说明 pipeline 已经对齐作者所说的 BRPO online mapping。

---

## 3. 目标语义与边界条件

本改动要实现的不是“把 pseudo 注册成真实 keyframe”，而是下面这组更精确的语义：

### 3.1 必须满足

1. pseudo **不进入** `self.current_window` 的持久 keyframe 集合。
2. pseudo **不进入** `self.viewpoints` 作为 tracking / fusion / archived camera 的主容器。
3. pseudo **不触发** `add_next_kf()`、`add_new_keyframe()` 或任何新 Gaussian 初始化。
4. 但在每次 keyframe mapping event 中，优化 batch 应包含：
   - real current-window views
   - optional extra real views
   - pseudo runtime records
5. 每个 pseudo sample 在 joint optimization 中应当表现为：
   - 一条普通 supervision member
   - 只是 RGB / depth loss 带自己的 mask
6. pseudo 与 real 的 loss 必须在同一个 backward / 同一轮 optimizer step 中生效。

### 3.2 第一版明确不做

1. 不把 pseudo 变成真正的 SLAM keyframe。
2. 不让 pseudo 参与 `add_next_kf()` 的新 Gaussian 注入。
3. 第一版不要求 pseudo 直接参与 densify/prune 统计；这可以先保持 real-only visibility policy，避免额外结构噪声。

---

## 4. 建议的落地策略：保留 legacy side-branch，对外新增 joint-primary 模式

不建议直接覆盖现有 side-branch 路径。更稳妥的方式是：

1. 保留 `topology_mode=side_branch` 作为当前 control / 回归基线；
2. 新增 `topology_mode=joint_primary` 作为论文对齐主线；
3. 所有正式 compare 都显式标注 topology mode，避免把两种拓扑混成一个 family。

这样做有两个好处：

- 不会把历史 D/E 结果语义污染掉；
- joint-primary 出现问题时，能快速回到 side-branch 做定位。

---

## 5. 具体改动方案（按文件拆解）

### 5.1 `third_party/S3PO-GS/utils/slam_backend.py`

#### 5.1.1 扩展 `Results.brpo_online_mapping` 配置解析

**位置**: `_resolve_brpo_online_mapping_cfg()`

建议新增字段：

```yaml
Results:
  brpo_online_mapping:
    topology_mode: joint_primary        # side_branch | joint_primary
    extra_real_views: 2                # 与原 map() 对齐
    pseudo_window_equivalence: true    # 进入“单条 masked member”语义
    propagate_pseudo_delta_to_neighbors: false
    update_real_exposure: true         # 与 update_real_pose 解耦
    prune_policy: real_only            # first landing
```

**理由**：

- `topology_mode` 用来明确区分老路径和新路径。
- `extra_real_views` 必须显式补回，不然主 mapping 与旧 `map()` 的 real supervision 不等价。
- `update_real_exposure` 不能继续与 `update_real_pose` 绑死，否则 joint-primary + `update_real_pose=false` 会错误丢失 real exposure optimization。
- `propagate_pseudo_delta_to_neighbors=false` 是为了避免 heuristic pose propagation 与“等价 window member”语义混在一起。

#### 5.1.2 改写 keyframe event 的调度顺序

**当前顺序**：

```python
self.map(self.current_window, iters=iter_per_kf, up_pose=True)
self.map(self.current_window, prune=True)
prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
self._run_brpo_runtime_pseudo_mapping(cur_frame_idx, prepare_payload)
```

**目标顺序**：

- `side_branch` 模式：保持原样
- `joint_primary` 模式：

```python
prepare_payload = self._maybe_prepare_brpo_runtime_slots(cur_frame_idx)
self._run_brpo_runtime_joint_primary_mapping(cur_frame_idx, prepare_payload, iter_per_kf)
self._run_real_only_prune_visibility_pass(cur_frame_idx)   # first landing 可保留 real-only prune
```

注意点：

1. `joint_primary` 模式下不能先跑一次 `self.map(...)`，否则 real 会先在无 pseudo 条件下被优化，再额外补一遍 pseudo，仍然是 side-branch 语义。
2. `prune=True` 现有实现依赖 `map()` 自己计算 visibility；因此最好抽出一个新的 real-only prune/visibility pass，而不是继续复用 `map(prune=True)` 的整段逻辑。

#### 5.1.3 新增 primary mapping 调用 helper

建议新增：

- `_run_brpo_runtime_joint_primary_mapping(...)`
- `_run_real_only_prune_visibility_pass(...)`

前者负责：
- 从 `prepare_payload` 中解析 runtime pseudo records
- 构造 `BRPOMappingConfig`
- 调 `run_brpo_pseudo_mapping(...)`
- 写 joint-primary 专属 event summary / history

后者负责：
- 基于 real keyframes 计算 visibility
- 执行 prune / opacity-reset / densify policy
- 保证第一版结构风险最低

### 5.2 `third_party/S3PO-GS/utils/slam_backend_brpo.py`

#### 5.2.1 扩展 `BRPOMappingConfig`

建议新增字段：

```python
topology_mode: str = "side_branch"
extra_real_views: int = 2
pseudo_window_equivalence: bool = False
update_real_exposure: bool = True
propagate_pseudo_delta_to_neighbors: bool = False
prune_policy: str = "real_only"
```

其中：

- `extra_real_views` 用来补齐原始 `map()` 的 random-view contract。
- `pseudo_window_equivalence` 用来切换“每个 pseudo 只作为一条 masked member loss”语义。
- `update_real_exposure` 把 exposure 与 pose 解耦。
- `propagate_pseudo_delta_to_neighbors` 默认应在新主线里关掉。

#### 5.2.2 让 real exposure 与 real pose 解耦

当前 `_build_joint_pose_optimizers()` 的调用方式是：

```python
include_real_pose=bool(cfg.update_real_pose)
include_real_exposure=bool(cfg.update_real_pose)
```

这会导致：

- 只要 `update_real_pose=false`
- real exposure 也一起关掉

这不等价于旧 `map()`（旧 `map()` 始终为 current_window real views 建 exposure optimizer）。

**修复要求**：

```python
include_real_pose=bool(cfg.update_real_pose)
include_real_exposure=bool(cfg.update_real_exposure)
```

并在 config 侧显式控制。

#### 5.2.3 新模式下禁止 split-authority 重复消费同一 pseudo

当前 live `split_pseudo_authority=true + all_valid` 会把同一 record 算成两次 pseudo loss。

对于 `pseudo_window_equivalence=true` 的新模式，应改成：

- 每个 sampled pseudo record 只调用一次 `compute_backend_pseudo_exact_loss(...)`
- mask 就来自这条 record 自己的 `confidence_mask` / `valid_mask` / `target_confidence`
- 不再额外派生 `scene_record`

即：

```python
for record in sampled_pseudo:
    pseudo_member_loss, pseudo_stats = compute_backend_pseudo_exact_loss(...)
    pseudo_member_loss_sum += pseudo_member_loss
    total_loss += lambda_pseudo * pseudo_member_loss
```

而不是当前：
- `pseudo_pose_loss`
- `pseudo_scene_loss`
- 双路拆分

#### 5.2.4 关闭 pseudo→neighbor 的 heuristic 传播

`_propagate_pseudo_pose_to_neighbors_()` 是一种“把 pseudo pose delta 人工摊给左右 KF”的 patch。

这不等价于“在同一 joint loop 中通过共享 loss/optimizer 影响 real pose”。

因此新主线建议：

- `side_branch` / legacy compare 可以保留此开关
- `joint_primary + pseudo_window_equivalence` 下默认关闭

换句话说，第一版要先测“纯 joint-window member 语义”的效果，避免把 heuristic propagation 和主改动混在一起。

#### 5.2.5 增加更强的 history 审计字段

当前 `brpo_pseudo_history.json` 记录了 loss，但还不足以证明“pseudo 已成为等价 member”。

建议新增字段：

- `topology_mode`
- `pseudo_window_equivalence`
- `real_window_ids`
- `sampled_extra_real_indices`
- `sampled_pseudo_ids`
- `num_real_window_members`
- `num_extra_real_members`
- `num_pseudo_members`
- `num_real_pose_optimized`
- `num_real_exposure_optimized`
- `num_pseudo_pose_optimized`
- `num_pseudo_exposure_optimized`
- `neighbor_pose_propagation_applied`

这样后续审计时，不需要再靠口头推断当前这一步到底是谁进了 joint loss。

### 5.3 `part3_BRPO/configs/*.yaml`

joint-primary 落地后，应新建一组独立 config family，而不是静默覆盖 E2/E3：

例如：

- `e4_jointprimary_paper_split_rgbonly.yaml`
- `e5_jointprimary_paper_split_rgbonly_realpose.yaml`

推荐第一版默认：

```yaml
Results:
  brpo_online_mapping:
    topology_mode: joint_primary
    pseudo_window_equivalence: true
    extra_real_views: 2
    split_pseudo_authority: false
    pseudo_scene_mask_mode: none
    propagate_pseudo_delta_to_neighbors: false
    update_real_exposure: true
```

如果走 RGB-only：

```yaml
use_depth: false
lambda_depth: 0.0
```

### 5.4 不建议做的错误改法

1. **不要**把 pseudo 直接塞进 `self.current_window` / `self.viewpoints`。
   - 这会污染 keyframe / tracking / fusion 语义。
2. **不要**保留 `split_pseudo_authority=true + all_valid` 然后声称 pseudo 已经等价于 window member。
   - 这还是双路 authority，不是单条 member supervision。
3. **不要**只开 `update_real_pose=true` 就认为目标达成。
   - 若拓扑仍是 side-branch，本质问题没变。

---

## 6. 第一版建议的实施顺序

### Phase A：最小结构切换

目标：先把 side-branch 改成 joint-primary，但保持 prune policy 保守。

1. 新增 `topology_mode` / `extra_real_views` / `update_real_exposure`
2. keyframe event 改为 conditional dispatch
3. `joint_primary` 模式直接调用 BRPO engine 做主 mapping
4. 仍然只用 real-only visibility 做 prune / densify / opacity reset

### Phase B：pseudo 等价 member 语义收紧

1. `pseudo_window_equivalence=true`
2. 强制 `split_pseudo_authority=false`
3. 禁用 neighbor propagation heuristic
4. 增加 history 审计字段

### Phase C：实验 compare

建议实验顺序：

1. `E2`：side_branch + rgb-only + real pose frozen（现有 control）
2. `E3`：side_branch + rgb-only + update_real_pose=true（过渡 control）
3. `E4`：joint_primary + rgb-only + real pose frozen
4. `E5`：joint_primary + rgb-only + update_real_pose=true

其中真正回答“pseudo 是否进入主 loss window”的是：
- `E2 -> E4`

回答“pseudo 进入主 loss window 后，是否能进一步推动 real pose”的是：
- `E4 -> E5`

---

## 7. 审计方案：如何证明 pseudo 已变成 online mapping loss window 的等价成员

这一部分必须做得比过去更严格，否则很容易又落入“日志里写了 pseudo mapping 在跑，所以应该对了”的错觉。

### 7.1 静态代码审计

必查点：

1. `joint_primary` 模式下，`keyframe` 分支不能先调用 `self.map(..., up_pose=True)`。
2. `BRPOMappingConfig` 必须真的包含 `extra_real_views` / `update_real_exposure` / `pseudo_window_equivalence`。
3. `_run_joint_pseudo_engine()` 在 `pseudo_window_equivalence=true` 时，每个 pseudo 记录只能有一次 `compute_backend_pseudo_exact_loss(...)` 调用路径。
4. `neighbor propagation` 在 joint-primary 主线上必须可见地关闭。

### 7.2 单事件 smoke 审计

目标：只验证拓扑是否正确，不看最终 PSNR。

建议条件：

- 单个 gap
- 1 个 pseudo slot
- `pseudo_map_iters=2`
- `num_pseudo_views_per_step=1`

需要看到的证据：

1. event summary 中有：
   - `topology_mode = joint_primary`
   - `num_real_window_members`
   - `num_extra_real_members = 2`（或你显式配置的值）
   - `num_pseudo_members = 1`
2. `brpo_pseudo_history.json` 中每步只记录一份 pseudo member loss，而不是 `pose + scene` 两份重复 loss。
3. `current_window` 仍然只包含 real KF ids，例如 `[67, 33, 0]`，而不是把 pseudo frame id 塞进去。

### 7.3 梯度 / 参数更新审计

如果 `update_real_pose=true`，必须确认这不是 heuristic propagation 造成的假象。

需要记录并检查：

1. 当前步 real pose optimizer 的 param group 数量 > 0
2. `neighbor_pose_propagation_applied = false`
3. real `cam_rot_delta/cam_trans_delta` 的变化来自 optimizer step + `update_pose(viewpoint)`
4. 不是 `_propagate_pseudo_pose_to_neighbors_()` 的写入

最好新增 log / history 记录：

- `real_pose_delta_norm_before_step`
- `real_pose_delta_norm_after_step`
- `pseudo_pose_delta_norm_before_fold`

### 7.4 产物流审计

目标：确认 pseudo 没污染 keyframe/fusion 语义。

需要核对：

1. `internal_eval_cache/manifest.json` 中 `kf_indices` 仍然只是真实 keyframes。
2. `camera_states.json` 不会多出 pseudo frame 的持久 camera state。
3. `self.viewpoints` 导出的 after/before camera 数仍对应真实数据帧，不含临时 pseudo record。

### 7.5 正式实验解释标准

joint-primary 上线后，只有同时满足以下条件，才可以说“pseudo 已作为 online mapping loss window 等价成员落地”：

1. 静态代码证据成立
2. 单事件 smoke 里 pseudo member 是单条 masked loss，不再拆两份 authority
3. `current_window` / `kf_indices` 仍是 real-only
4. 若 `update_real_pose=true`，real pose 更新来自同一 joint loop，而不是 heuristic 邻居传播

少任何一条，都只能说“更接近了”，不能说“已经对齐作者语义”。

---

## 8. 预期影响与风险

### 8.1 预期收益

1. pseudo supervision 从“后挂补充项”变成“主 mapping 优化成员”。
2. 若 pseudo 确实有价值，它对 real pose / shared gaussians 的影响路径会更直接。
3. 后续评估“为什么没 PSNR 提升”时，终于能排除“拓扑压根不对”这个主因。

### 8.2 风险

1. joint-primary 可能改变原始 real mapping 稳定性，因此必须保留 side-branch control。
2. 如果不补 `extra_real_views`，会静默改变 baseline real supervision contract。
3. 若 prune/densify 直接让 pseudo 参与，第一版可能引入额外不稳定性；因此建议先 real-only prune policy。

---

## 9. 一句话版实施原则

**不要把 pseudo 变成“新 keyframe”；要把 pseudo 变成“主 mapping joint loss 里的一条 masked member”。**

这是这次改动最重要的语义边界。
