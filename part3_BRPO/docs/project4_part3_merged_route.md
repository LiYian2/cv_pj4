# Project 4 Part 3：合并后的可执行路线（给 Agent）

> 适用范围：**不包含**“先跑出第一版 S3PO 结果（PLY / poses）”以及“Difix 对 pseudo RGB 的修复”本身。  
> 本文从这些结果**已经有了**开始，描述后续该怎么做。  
> 核心约束：**pseudo view 只参与 mapping，不参与 tracking。**

---

## 0. 先定死的系统边界

当前可行的第一版路线，不是“把 pseudo view 加进去再重跑一遍完整 S3PO SLAM”，而是：

1. 先完成一次 **real-only S3PO**，拿到：
   - 当前 Gaussian scene（内存中的 `GaussianModel` 或等价导出结果）
   - 真实 keyframe poses
   - 可用的邻接参考帧关系

2. 在 **外部 pseudo 模块** 中，对每个 pseudo sample 做：
   - pseudo pose 选取
   - pseudo render（RGB / depth）
   - Difix 修复（这部分默认你已有）
   - 模块 3：几何置信 / `target_depth` / 诊断图 / 整图分数
   - 可选：RAP offline audit

3. 然后只在 **S3PO backend** 里加一个新的 **pseudo-assisted refinement / mapping-only** 阶段，把
   `{pseudo_camera, target_rgb, target_depth, confidence_mask}`
   当作额外监督加入高斯优化。

**不做的事：**
- 不把 pseudo view 当真实 keyframe 回灌到 frontend tracking
- 不让 pseudo 相机进入 `cam_rot_delta / cam_trans_delta` 的窗口位姿优化
- 第一版不重写 S3PO 的 tracking 主流程

这条边界是根据 repo 结构定的：当前 S3PO 的 frontend 负责真实帧 tracking + keyframe selection，backend 负责高斯插入和 window mapping；如果把 pseudo 伪装成普通 keyframe，会直接卷入 pose optimizer 和 keyframe window 逻辑，改动范围会显著扩大。

---

## 1. 总 pipeline（合并后路线）

## Stage 1：最小可行版本（先做）

### 1.1 输入前提
默认你已经有：
- 第一版 **real-only S3PO** 结果
- 真实 keyframe poses
- 当前 Gaussian scene
- Difix 修复后的 pseudo RGB（或至少有可用的 pseudo RGB）
- pseudo pose

### 1.2 Stage 1 主流程

```text
real-only S3PO scene + real keyframe poses
    -> pseudo pose scheduler
    -> pseudo render: I_gs, D_gs
    -> Difix repaired pseudo RGB (已有)
    -> Epipolar Depth Priors (EDP) 生成 target_depth
    -> build C_geom = visibility * depth_consistency * pose_decay
    -> optional flow veto
    -> save pseudo sample cache:
       {pseudo_camera, target_rgb, target_depth, confidence_mask, diag}
    -> S3PO backend-only pseudo refinement
    -> masked pseudo RGBD loss + real-view anchor loss
    -> updated Gaussian scene
```

### 1.3 Stage 1 的关键思想
Stage 1 的重点不是把 pseudo view 深度嵌入 S3PO 前端，而是先把 **pseudo supervision** 这条链跑通：

- `target_rgb`：来自 pseudo RGB（通常是 Difix 修复后的结果）
- `target_depth`：**不要直接等于当前 render depth**，优先使用 **EDP / epipolar depth** 或其 blend 版本
- `confidence_mask`：来自模块 3 的 `C_geom`
- 最后只在 backend 做 **mapping-only refinement**

---

## 2. 模块 3（pseudo branch 的核心）

模块 3 的位置在：

```text
pseudo render / Difix 修复 之后
S3PO backend mapping 之前
```

它不在 tracking 里，不在 keyframe selection 里，也不在 S3PO frontend 主循环里。

### 2.1 模块 3 输入

每个 pseudo sample 的输入应固定为：

- `pseudo_camera`
- 当前 Gaussian scene
- pseudo 视角下 render 的：
  - `I_t_gs`
  - `D_t_gs`
- 左右真实参考帧：
  - `I_L`, `I_R`
  - `cam_L`, `cam_R`
- 可选：
  - flow estimator（如 RAFT）
  - Difix 修复后的 pseudo RGB

### 2.2 模块 3 输出

模块 3 最终输出四类东西：

1. `confidence_mask`
   - 逐像素（第一版）或逐 patch（增强版）置信图
   - 第一版核心是 `C_geom`

2. `target_depth`
   - pseudo 相机视角下，后续 depth loss 要对齐的目标深度
   - 第一版应尽量来自 **EDP**
   - 不是简单复用 `D_t_gs`

3. `diag`
   - 诊断图，便于 debug / ablation
   - 建议至少保存：
     - `epipolar_distance`
     - `left_score_map`
     - `right_score_map`
     - `visibility_map`

4. `view_score`
   - 整图分数
   - 给整图拒收 / RAP offline audit 用

---

## 3. Stage 1 的 EDP 落地（先放在 pseudo branch，不先改 S3PO 前端）

EDP 在这里的角色不是“替代整个 `C_geom`”，而是：

- 生成 `target_depth`
- 作为 `C_geom` 中 `depth_consistency` 的核心来源

### 3.1 计算逻辑

对 pseudo sample 和某个参考帧 `j`：

1. 用 flow 网络从 pseudo 视角到参考帧预测匹配点
2. 对预测点做极线约束修正（投到 epipolar line）
3. 根据修正后的匹配点和相机参数显式算深度，得到 `D_epi^(j)`
4. 用 point-to-epiline distance 做 flow 可靠性检查
5. 左右参考帧各自产生一张 `D_epi`
6. 用更稳的策略（非简单平均）得到最终 `target_depth`

### 3.2 为什么 Stage 1 要先把 EDP 放在 pseudo branch
因为这样不会先破坏 S3PO 原有 real-keyframe 初始化链路。  
这一步主要服务于 **pseudo depth supervision**，先验证：

- EDP 生成的 `target_depth` 是否稳定
- 加入 pseudo depth loss 后是否真的提升结果
- 是否会把 geometry 拖坏

这比一上来深改 frontend 更可控。

---

## 4. Stage 1 的 `C_geom`

第一版建议直接定成：

```math
C_geom^(j)(p) = O_vis^(j)(p) * s_depth^(j)(p) * s_pose^(j)
```

然后左右合成：

```math
C_geom(p) = max(C_geom^(L)(p), C_geom^(R)(p))
```

### 4.1 `O_vis`
可见性 / 遮挡合法性：
- pseudo depth 反投影到 3D
- 投到参考帧
- 要求：
  - 在图像内
  - 深度有效
  - 不与参考视角的深度严重冲突

### 4.2 `s_depth`
第一版最好用 EDP 版本，而不是简单 render-depth consistency：

- `D_epi` 来自 flow + epipolar geometry
- `s_depth` 衡量当前 pseudo render depth 与 `D_epi` 的一致程度
- 同时结合 epipolar distance 做过滤

### 4.3 `s_pose`
参考帧离 pseudo pose 太远时整体降权。  
这一项是一个 **view-level scalar**，不是逐像素图。

### 4.4 第一版是否加 matcher
**不作为 Stage 1 必须项。**

第一版先做：
- `C_geom`
- optional flow veto

而不是一开始再叠 `C_match`。  
先把 geometry-first 这条链路跑通。

---

## 5. flow 和 flow veto（Stage 1 只做弱使用）

### 5.1 flow 是什么
这里的 flow 指 pseudo view 和真实参考帧之间的 **optical flow**，本质是二维位移场。

### 5.2 flow veto 是什么
在这条路线里，flow 不是主几何依据，而是 **否决项**：

- 如果某个区域几何投影看起来“应该一致”
- 但 flow 结果严重偏离极线约束，或者 warp 后残差很大
- 就把该区域降权 / 拒收

所以 Stage 1 的 flow 角色是：
- 帮助得到 `D_epi`
- 帮助做 veto
- **不是**替代几何主干

---

## 6. Stage 1 的 RAP：只做 offline audit，不做 online virtual-KF

### 6.1 这一点必须定死
第一版 **不要**把 pseudo view 当成 virtual keyframe 在线塞回 S3PO SLAM 主循环。

RAP 在 Stage 1 的角色是：
- 对已经生成好的 pseudo sample 做 **离线 3D 审核**
- 输出整图 accept / reject（或 coarse score）
- **不是**进入 frontend / keyframe insertion / pose optimization

### 6.2 RAP 输入

RAP 需要两个点集：

#### (1) pseudo point cloud
由当前 pseudo sample 的高置信区域反投影得到：

- 使用 `target_depth`
- 乘上 `confidence_mask`
- 只保留高置信区域

#### (2) local anchor cloud
不是全局整张 scene 的点云，而是从当前 Gaussian scene 中裁出的局部点集。

第一版建议：
- 用左右真实参考帧可见高斯的并集
- 或 pseudo frustum 内的高斯中心点

### 6.3 为什么不是全局点云
因为 RAP 在这里是“这张 pseudo view 是否与当前局部几何兼容”的审核器。  
如果直接拿全局 scene，会引入太多无关区域。

### 6.4 RAP 在 Stage 1 的输出
第一版只做：
- `accept / reject`
- 或 coarse `rap_score`

不要先做 point-wise gate，不要先嵌入 point replacement。

---

## 7. Stage 1：S3PO repo 内部怎么改

第一版坚持最小侵入：

### 7.1 repo 外新增模块（新写）
建议新建一个独立目录，例如：

```text
part3_brpo_lite/
    pseudo_scheduler.py
    pseudo_renderer.py
    epipolar_depth.py
    geometry_confidence.py
    rap_audit.py
    pseudo_cache.py
```

这些模块都属于 **external pseudo branch**，尽量不要先写进 S3PO 主 repo。

### 7.2 repo 内改动范围（仅三处）

#### (1) `slam.py`
新增一个 **pseudo_refinement stage**，放在：
- `frontend.run()` 结束之后
- `color_refinement` 之前或并列

作用：
- 触发 backend-only pseudo refinement

#### (2) `utils/slam_backend.py`
新增：
- 新 message handler，例如 `"pseudo_refinement"`
- 新函数：`pseudo_refinement(pseudo_samples)`

它只做：
- real-view anchor loss
- pseudo-view masked loss
- Gaussian optimization

不应把 pseudo 相机并入 keyframe pose optimizer

#### (3) `utils/slam_utils.py`
新增：
- `get_loss_mapping_pseudo(...)`

作用：
- 计算 masked pseudo RGB loss
- 计算 masked pseudo depth loss

### 7.3 Stage 1 不建议改的地方
- `utils/slam_frontend.py`
- tracking 主流程
- keyframe selection 逻辑
- `"keyframe"` 消息语义
- backend 的 `"keyframe"` 分支

理由很简单：这些地方都默认 keyframe 是真实帧，把 pseudo 塞进去会显著放大系统改动范围。

---

## 8. Stage 1 的新 loss

第一版 backend 里的总体思路是：

```text
loss_total = loss_real_anchor + lambda_pseudo * loss_pseudo
```

其中：

### 8.1 real-view anchor loss
继续用现有 `get_loss_mapping(...)`  
作用：
- 防止 pseudo supervision 把 geometry 拖偏
- 给 refinement 一个稳定锚点

### 8.2 pseudo-view loss
新加：

```math
L_pseudo
=
lambda_rgb * L_rgb_masked
+
lambda_d * L_depth_masked
```

其中：

```math
L_rgb_masked
=
|| C \odot (I_hat - I_target) ||_1 / ||C||_1
```

```math
L_depth_masked
=
|| C \odot (D_hat - D_target) ||_1 / ||C||_1
```

这里：
- `I_target` = pseudo RGB（通常是 Difix 修复结果）
- `D_target` = `target_depth`
- `C` = `confidence_mask`

### 8.3 这一步的关键含义
不是给 pseudo 相机继续做 pose 优化，  
而是把 pseudo sample 当成 **mapping-only supervision**。

---

## 9. Stage 2（增强版）

Stage 2 不是推翻 Stage 1，而是在 Stage 1 跑通后再增强。

### 9.1 Stage 2-A：模块 3 自身增强
在 Stage 1 的 external pseudo branch 上继续加：

- patch confidence
- 左右参考帧更稳的 score fusion
- 更强的 flow veto
- RAP offline audit（从 accept/reject 扩展到更细的视图级过滤）

### 9.2 Stage 2-B：把 EDP 深度推进 S3PO 自身 real-keyframe 初始化
这是增强版的重要方向。

也就是去改 S3PO 原本的 real-keyframe depth initialization / replacement 逻辑，让 real keyframe 本身的 `initial_depth` 更稳。

更合理的落点不是抽象上的 `mapping.py / point_manager.py`，而是 repo 里真实存在的链路：

- `FrontEnd.add_new_keyframe(...)`
- 其中的 `process_depth(...)`
- 与之配合的 backend `"keyframe"` 后续插图 / map 逻辑

这一步的目标是：

```text
real keyframe arrives
    -> flow-based matching with adjacent KF
    -> epipolar correction
    -> D_epi
    -> replace / blend original initial_depth
    -> stronger base geometry
```

### 9.3 Stage 2 为什么不先做
因为它会直接触碰 S3PO 自己的 real keyframe pipeline，改动范围比 Stage 1 大。  
更适合作为第二阶段增强，而不是第一阶段主线。

---

## 10. 明确不做 / 先不做

为了避免 agent 后续跑偏，这些点先明确写死：

1. **第一版不把 pseudo view 当 virtual keyframe 在线插入 S3PO。**
2. **第一版不重跑完整 frontend tracking。**
3. **第一版不把 pseudo 相机加入 keyframe pose optimizer。**
4. **第一版不把 RAP 做成 online point-wise gate。**
5. **第一版不先深改 S3PO 的 real-keyframe 初始化链路。**

---

## 11. 最终执行顺序（建议）

### Stage 1
1. 跑完 real-only S3PO（你已完成）
2. 准备 pseudo pose / pseudo RGB（你已有前半部分）
3. 外部模块实现 `target_depth`（EDP）
4. 外部模块实现 `C_geom`
5. 保存 pseudo cache
6. 改 `slam.py`
7. 改 `utils/slam_backend.py`
8. 改 `utils/slam_utils.py`
9. 跑 backend-only pseudo refinement
10. 看结果 / 做 ablation

### Stage 2
11. 加 patch confidence
12. 加 RAP offline audit
13. 把 EDP 推进 `FrontEnd.add_new_keyframe()` / `process_depth(...)`
14. 再做更完整的增强实验

---

## 12. 这一版路线的最终结论

合并后的最稳路线是：

- **Stage 1：external pseudo branch + backend-only pseudo refinement**
  - pseudo view 只做 mapping，不做 tracking
  - EDP 先用于 pseudo `target_depth`
  - 新增 masked pseudo RGBD loss
  - repo 内只做最小范围改动

- **Stage 2：在 Stage 1 稳定后，再增强**
  - patch confidence
  - RAP offline audit
  - EDP 深入 S3PO 自身 real-keyframe depth initialization

这条路线的核心优点是：
- 可行性高
- 改动边界清晰
- 易于做 ablation
- 与当前 S3PO repo 的前后端分工兼容
- 与 BRPO 的“confidence-guided optimization”逻辑一致

---
