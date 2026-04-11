# BRPO-style 两阶段 Refine 工程落地方案

> 目标：在你当前 `part3_BRPO/` 的 standalone refine 基础上，落地一版更接近 BRPO 的 **two-stage joint optimization**。
> 
> 本文默认：
> 1. `internal cache` 会先补齐；
> 2. `BRPO-style confidence mask` 会作为前置模块先落地；
> 3. 本文优先考虑“先做可跑通的 standalone v2”，后续再决定是否并入 S3PO backend。

---

## 1. 当前实现 vs BRPO：差异先说清楚

### 1.1 你当前 refine 主体是什么

当前主入口是：
- `scripts/run_pseudo_refinement.py`

当前实际做的是：
- 读一张已有 PLY
- 读 pseudo cache
- 固定 pseudo 相机 viewpoint
- 对 Gaussian 原地优化
- 支持 real branch / pseudo branch / densify / prune / pseudo param freeze

这是一个 **standalone map refinement**。

### 1.2 旧版 bak 做过什么

`run_pseudo_refinement.py.bak_before_fix` 最早做过：
- RGB + depth 混合 pseudo loss
- 但没有 real-anchor 稳定器
- 没有 pseudo pose delta
- 没有 exposure stabilization
- 有效 depth support 稀时非常容易失效

### 1.3 当前实现和 BRPO 的根本差异

当前实现的问题不只是“没用 depth”。更本质的是：
- pseudo 相机是固定的
- 一旦 pseudo supervision 和地图不一致，只能把误差解释进 Gaussian appearance / opacity
- 这更像 fixed-pose appearance tuning

BRPO 的关键不同是：
1. **先做 pose deltas + exposure stabilization**
2. 然后 joint optimize：
   - Gaussian 参数
   - camera poses
3. RGB / depth loss 都是 confidence-mask-weighted
4. 不是只训 appearance

所以这轮真正要补的是：
**让 pseudo supervision 有机会先校正 pose，再去改地图。**

---

## 2. 本文目标边界

这一版只做下面这条最小可行链：

```text
existing PLY + internal pseudo cache + BRPO-style confidence mask
    ↓
Stage A: pose delta + exposure stabilization
    ↓
Stage B: Gaussian + pseudo pose joint refinement
    ↓
refined ply + updated pseudo camera states + history
```

**本阶段不做：**
- 不改 S3PO frontend tracking
- 不把 pseudo frame 塞回 keyframe queue
- 不先做 full backend integration
- 不先做 scene perception Gaussian management

目标是先让这条优化链在 standalone 下成立。

---

## 3. 推荐的总体工程形态

我建议你把这一版定义为：

**`run_pseudo_refinement_v2.py`**

而不是继续在当前 `run_pseudo_refinement.py` 上无限叠逻辑。

原因：
- 当前 v1 已经承担太多兼容逻辑
- v2 会多出 pseudo camera optimization / exposure / depth target / stage schedule
- 分文件更容易对照实验和回退

### 推荐新增文件

```text
part3_BRPO/scripts/run_pseudo_refinement_v2.py
part3_BRPO/pseudo_branch/pseudo_camera_state.py
part3_BRPO/pseudo_branch/pseudo_loss_v2.py
part3_BRPO/pseudo_branch/pseudo_refine_scheduler.py
```

---

## 4. 输入 / 输出 / 中间产物

### 4.1 输入

v2 的输入建议固定为：
- `--ply_path`
- `--pseudo_cache`
- `--train_manifest`
- `--train_rgb_dir`
- `--output_dir`

pseudo sample 级别最少需要：
- `camera.json`
- `refs.json`
- `render_rgb.png`
- `render_depth.npy`
- `target_rgb_fused.png` 或 left/right target
- `confidence_mask_brpo.npy`
- `target_depth_for_refine.npy`（本阶段建议新增）

### 4.2 输出

建议输出目录改成：

```text
output_dir/
├── refined.ply
├── pseudo_camera_states_init.json
├── pseudo_camera_states_stageA.json
├── pseudo_camera_states_final.json
├── refinement_history.json
├── stageA_history.json
├── stageB_history.json
└── diag/
```

### 4.3 中间产物

建议显式保存：
- Stage A 后的 pseudo pose deltas
- Stage A 后的 exposure 参数
- Stage B 后的最终 pseudo camera states
- 每个 sample 的有效 mask ratio
- 每个 sample 的 RGB / depth loss

这样后面才能判断：
- 是 pose 没动起来
- 还是 pose 动了但 map 没收益

---

## 5. target_depth：这一版建议怎么处理

本阶段先不要强行回到旧 EDP-only 路线。

### 5.1 建议的第一版来源

第一版建议：
- 高置信 verified 区域：可用 reprojection 得到的 depth
- 中置信区域：退回 `render_depth.npy`
- 低置信区域：不监督

也就是新增一张：
- `target_depth_for_refine.npy`

它不是纯 EDP，也不是纯 render depth，而是：
**mask-aware blended depth target**。

### 5.2 为什么这么做

这样比“全图 render depth”强，因为有外部几何校验；
又比“全图 EDP”稳，因为 support 稀时不会整张图失效。

---

## 6. 两阶段优化：推荐定义

## Stage A：Pseudo pose delta + exposure stabilization

### 6.1 目标

先不急着改 Gaussian 几何，让 pseudo 相机先和当前地图对齐一点。

### 6.2 优化变量

对每个 pseudo sample，引入：
- `cam_rot_delta`
- `cam_trans_delta`
- `exposure_a`
- `exposure_b`

### 6.3 固定项

Stage A 建议：
- Gaussian 几何固定
- Gaussian appearance 固定或只允许极小更新
- densify / prune 关闭

### 6.4 loss

Stage A 只用 pseudo branch：

```text
L_stageA = β_A * L_rgb_masked + (1 - β_A) * L_depth_masked + λ_pose * L_pose_reg + λ_exp * L_exp_reg
```

其中：
- `L_rgb_masked`：confidence-mask-weighted RGB loss
- `L_depth_masked`：confidence-mask-weighted depth loss
- `L_pose_reg`：约束 delta 不要发散
- `L_exp_reg`：约束 exposure 不要飘太远

### 6.5 建议的正则

```text
L_pose_reg = ||cam_rot_delta||_2 + w_t * ||cam_trans_delta||_2
L_exp_reg  = |exposure_a| + |exposure_b|
```

### 6.6 Stage A 输出

输出：
- 更新后的 pseudo 相机状态
- 每个 sample 的 pose delta / exposure

只要 Stage A 结束后 pseudo render 更贴近 target，就算成功。

---

## 7. Stage B：Gaussian + pseudo pose joint refinement

### 7.1 目标

在 Stage A 已有较好 pseudo pose 初值的前提下，开始 joint refine：
- Gaussian 参数
- pseudo camera poses
- 可选 exposure

### 7.2 优化变量

Gaussian 侧：
- `xyz`
- `scaling`
- `rotation`
- `f_dc`
- `f_rest`
- `opacity`

Pseudo camera 侧：
- `cam_rot_delta`
- `cam_trans_delta`
- 可选 `exposure_a / exposure_b`

### 7.3 总 loss

建议写成：

```text
L_total = λ_real * L_real
        + λ_pseudo * ( β * L_rgb_masked + (1-β) * L_depth_masked )
        + λ_s * L_scale_reg
        + λ_pose * L_pose_reg
        + λ_exp * L_exp_reg
```

其中：
- `L_real`：当前已有 real sparse train anchor loss
- `L_rgb_masked`：用 `confidence_mask_brpo`
- `L_depth_masked`：只在 mask 支持区域监督
- `L_scale_reg`：沿用当前 isotropic / scale regularization

### 7.4 这一版和当前 v1 的核心区别

不是“加了个 depth loss”这么简单，而是：
1. pseudo pose 不再固定
2. depth 终于进入 joint objective
3. real anchor 继续稳定整体 geometry
4. pseudo supervision 不再只能把误差塞进 appearance

---

## 8. 参数组与优化器建议

### 8.1 Gaussian 参数组

沿用当前分组：
- `xyz`
- `f_dc`
- `f_rest`
- `opacity`
- `scaling`
- `rotation`

### 8.2 新增 pseudo camera 参数组

建议每个 pseudo sample 对应：
- `rot_delta: nn.Parameter(3)` 或四元数增量
- `trans_delta: nn.Parameter(3)`
- `exposure_a: nn.Parameter(1)`
- `exposure_b: nn.Parameter(1)`

### 8.3 实现建议

不要把这些参数散在脚本里，建议抽成：
- `pseudo_camera_state.py`

每个 sample 一个对象，负责：
- 从 `camera.json` 初始化 base pose
- 应用 delta 得到当前 pose_c2w
- 导出当前状态

---

## 9. run_pseudo_refinement_v2 的推荐流程

```text
load gaussians
load real views
load pseudo samples
load pseudo camera state objects
    ↓
Stage A loop
    - render pseudo with current pose
    - compute masked RGBD loss
    - optimize pseudo pose delta + exposure
    - save stageA states
    ↓
Stage B loop
    - sample real views -> L_real
    - sample pseudo views -> masked RGBD loss
    - optimize gaussian params + pseudo pose delta (+ exposure)
    - optional densify/prune with stricter rule
    ↓
save refined.ply + pseudo camera states + history
```

---

## 10. densify / prune：这一版怎么处理

这一版不要延续“默认开 densify 再看”的思路。

### 10.1 建议

Stage A：
- `disable_densify = True`
- `disable_prune = True`

Stage B：
- 第一版建议仍然 **先关 densify**
- 或者只允许 `densify_stats_source = real`

### 10.2 理由

因为这轮先要回答的是：
- pseudo pose + RGBD joint objective 是否成立

如果这时还开 densify，很容易把结论搞混：
- 到底是 supervision 有效
- 还是高斯数爆了导致 appearance 变平滑

所以第一版 v2 应该尽量减少结构性变量。

---

## 11. standalone 还是并入 S3PO backend

### 11.1 我对 BRPO 的判断

从论文写法和实验行为看，BRPO 更像：
- 一套自定义 3DGS optimization pipeline
- 不是简单“重新跑一遍 S3PO”
- 也不是只读已有 PLY 的极轻量 post-refine

### 11.2 对你当前工程的建议

这轮建议先做：
- **standalone v2**

不要一开始就并入 S3PO backend。原因：
- 你现在最缺的是机制验证，不是系统集成
- 先在 standalone 下证明：
  - pseudo pose delta 能优化
  - confidence-weighted RGBD loss 有意义
- 之后再决定要不要并入 backend

### 11.3 什么时候再考虑 backend integration

当下面三件事都成立后再考虑：
1. Stage A 确实能稳定减小 pseudo residual
2. Stage B 比 v1 更稳
3. internal replay 能证明 refined PLY 真正受益

在这之前，standalone 更合适。

---

## 12. 建议新增 / 修改的文件

### 12.1 新增

```text
part3_BRPO/scripts/run_pseudo_refinement_v2.py
part3_BRPO/pseudo_branch/pseudo_camera_state.py
part3_BRPO/pseudo_branch/pseudo_loss_v2.py
part3_BRPO/pseudo_branch/pseudo_refine_scheduler.py
```

#### `pseudo_camera_state.py`
职责：
- base pose + delta 管理
- current pose 导出
- state serialization

#### `pseudo_loss_v2.py`
职责：
- masked RGB loss
- masked depth loss
- pose / exposure regularization
- total loss 组装

#### `pseudo_refine_scheduler.py`
职责：
- Stage A / Stage B 切换
- iteration schedule
- 哪些 param group 在哪个 stage 开启

### 12.2 修改

#### `build_pseudo_cache.py`
新增：
- 生成 `target_depth_for_refine.npy`
- 供 v2 使用

#### `run_pseudo_refinement.py`
不建议继续大改。
只保留 v1 baseline 角色。

---

## 13. 第一版建议的超参策略

### Stage A

```text
iters_stageA        = 200 ~ 500
β_A                 = 0.7
λ_pose              = 0.01
λ_exp               = 0.001
optimize_gaussian   = False
```

### Stage B

```text
iters_stageB        = 1000 ~ 2000
λ_real              = 1.0
λ_pseudo            = 0.5
β                   = 0.7 ~ 0.9
λ_s                 = 沿用当前 scale reg
λ_pose              = 0.005
λ_exp               = 0.0005
densify             = off (first try)
```

这里只给方向，不强行定死。

---

## 14. 建议的最小实验顺序

不要一上来做很多 ablation，先做最小四步。

### Exp-1

`Stage A only`
- 固定 Gaussian
- 只优化 pseudo pose + exposure
- 看 pseudo RGB residual 是否下降

### Exp-2

`Stage A + Stage B, no depth`
- 先验证两阶段机制本身
- 只用 BRPO mask 下的 RGB loss

### Exp-3

`Stage A + Stage B, with depth`
- 再加入 `target_depth_for_refine`

### Exp-4

和 v1 对照
- v1 RGB-only
- v2 RGB-only
- v2 RGBD

只有这四步稳定后，再考虑 densify / opacity-only / geometry freeze 等细调。

---

## 15. 第一版成功标准

这一版先不要拿“最终论文分数”做目标，先看机制是否成立。

成功标准按顺序是：
1. Stage A 能稳定优化 pseudo pose
2. Stage B 不比 v1 更容易崩
3. 用 internal replay 评估时，v2 比 v1 更稳定或更优
4. depth 加入后没有再次出现旧版“support 稀导致直接崩”的情况

只要达到这四条中的前三条，这一版就值得继续推进。

---

## 16. 本文和 confidence mask 文档的关系

两份文档的分工是：

- `BRPO_confidence_mask_engineering_plan.md`
  - 解决“哪些 pseudo 像素可信”

- `BRPO_two_stage_refine_engineering_plan.md`
  - 解决“可信 pseudo supervision 如何进入 pose + Gaussian joint optimization”

推荐顺序：

```text
internal cache
→ confidence mask
→ two-stage refine
```

不要颠倒。
