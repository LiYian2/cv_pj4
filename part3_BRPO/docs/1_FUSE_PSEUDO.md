# FUSE_PSEUDO.md — 方案1：left/right修复结果融合

> 目标：构造几何一致性更强、扩散幻觉更少的 fused pseudo target，解决位姿更准但图像质量变差的问题。
> 
> 创建时间：2026-04-09

---

## 0. Agent执行须知（先看）

这份文档可以作为实现规划，但**不要无约束地一次性大改完整条链路**。执行时请按 **v1 收敛方案** 落地，先做最小可验证版本，再做扩展实验。核心要求如下。

### 0.1 这次实现的目标边界

v1 的目标是：**在不改变当前训练范式的前提下，引入 fused pseudo target，并验证它是否优于单边 target。**

具体约束：
- **不改当前 refine 的主损失形式**：仍然是 confidence-weighted pseudo RGB loss；
- **不在 v1 里重定义 `target_depth.npy` 的语义**；当前 depth 仍保持现有写法，避免把“RGB fusion”和“depth supervision redesign”混在一起；
- **v1 重点只新增**：
  - `target_rgb_fused.png`
  - `confidence_mask_fused.npy`
  - `confidence_mask_fused.png`
  - `fusion_meta.json`
- `run_pseudo_refinement.py` 只需要新增 `fused` 读入模式，不要顺手重写训练框架。

### 0.2 先收窄实现，再做扩展

请严格按下面顺序推进，不要跳步：

1. **Phase A：cache侧最小实现**
   - 保留当前 `left/right` 产物；
   - 新增 fused RGB / fused confidence；
   - 保持 `target_depth.npy` 与旧版兼容。

2. **Phase B：refine接入**
   - 将 `--target_side` 扩成 `left|right|fused`；
   - `fused` 模式下读取 `target_rgb_fused.png + confidence_mask_fused.npy`；
   - `left/right` 旧逻辑不得破坏。

3. **Phase C：先跑一组最小实验**
   - 先只跑 **E-fused-rgb-only**（`use_depth_agreement=False`）；
   - 不要一上来就把 `A_depth` 当默认主线。

4. **Phase D：确认最小实验跑通后，再补扩展实验**
   - 再做 **E-fused-with-depth**（`use_depth_agreement=True`）作为第二组对照。

### 0.3 关于接口：不要重复造轮子

当前真实代码里：
- `epipolar_depth.py` 已经有 `compute_edp_depth()`；
- 也已经有 `compute_edp_depth_bidirectional()`。

因此 v1 **不要盲目新造一套完全平行的接口**，例如先入为主地重做一个和现有逻辑重复的 `compute_edp_depth_pair()`。更推荐两种做法之一：
- **做法 A（推荐）**：在 `build_pseudo_cache.py` 里继续显式保留 left/right 单边输出，然后在该文件或新建的 `pseudo_fusion.py` 中完成 fusion；
- **做法 B**：扩展现有 `compute_edp_depth_bidirectional()` 的返回内容，使其能提供 v1 需要的 left/right 中间量。

原则：**复用现有接口和数据流，少引入平行抽象。**

### 0.4 关于 `target_depth`：v1 不要动它的定义

这是本次最重要的执行约束之一。

当前 refine 并**没有**直接使用 depth loss；真正进入优化的是 pseudo RGB 和 confidence mask。因此这次实验的变量应该尽量收窄在：
- fused RGB 是否更稳；
- fused confidence 是否更保守有效。

所以：
- **v1 不要把 `target_depth.npy` 也改成 fused-depth 新定义**；
- `target_depth.npy` 可以继续保留当前 winner-take-all / 现有 EDP 结果；
- 等 fused RGB 这条线验证有效后，再单独讨论 depth 语义是否值得重构。

### 0.5 关于 `A_depth`：先做 ablation，不要默认启用

文档中的 `A_depth` 在原理上成立，但从工程和数据分布上看，**一上来就默认开启风险较大**：
- EDP 本身可能偏 sparse；
- 再乘一个 `A_depth` 可能把 support 压得过狠；
- 最后得到的 supervision 可能“更干净但太少”，导致训练信号不足。

因此执行顺序必须是：
- **先做 `rgb-only fusion`**；
- 再把 `depth agreement` 作为第二组 ablation。

### 0.6 失败策略与兼容性

实现时请明确以下规则：
- `left/right` 模式保持完全向后兼容；
- `fused` 模式只在 fused 文件齐备时启用；
- 若指定 `--target_side fused` 但缺少 `target_rgb_fused.png` 或 `confidence_mask_fused.npy`，**应 hard fail 并报清楚错误**，不要静默 fallback 到旧文件；
- 旧 cache 不应因为没有 fused 文件而无法继续使用 `left/right`。

### 0.7 修改代码前的检查与备份建议

执行前先做这几步：

1. **先检查是否有相关脚本正在运行**，避免改到正在被使用的文件：
   - `pseudo_branch/build_pseudo_cache.py`
   - `pseudo_branch/epipolar_depth.py`
   - `pseudo_branch/diag_writer.py`
   - `scripts/run_pseudo_refinement.py`
2. 先记录 `git status`。
3. 这次更推荐：
   - **小改动：直接走 git 提交，不额外手工备份**；
   - **若出现结构性重写**（尤其是 `build_pseudo_cache.py` / `run_pseudo_refinement.py` 大段改动），可额外保留一个临时备份分支或 patch。

原则：**优先 git 版本化，不要散落一堆手工副本；但在大改主入口前，至少保证能一键回退。**

### 0.8 实验执行要求

v1 落地后，实验必须按下面顺序跑：

1. **Smoke check**
   - 用一个已跑通过的 scene（优先 `DL3DV-2 sparse`）
   - 先验证 fused cache 能完整产出：文件数、manifest、sample 读入都正常
   - 再验证 `run_pseudo_refinement.py --target_side fused` 能正常启动并写 history

2. **正式实验 1：E-fused-rgb-only**
   - 基于当前 E config：
     - `lambda_pseudo=0.5`
     - `freeze_geometry_for_pseudo=True`
     - `pseudo_trainable_params=appearance`
     - `densify_stats_source=real`
     - `tertile pseudo`
   - `use_depth_agreement=False`

3. **正式实验 2：E-fused-with-depth`（可选第二步）**
   - 与实验1保持同口径，只多开 `use_depth_agreement=True`

4. **评估要求**
   - 必须继续使用当前主线口径：`GT pose external eval`
   - 必须与现有 E（单边 tertile）做同口径对照
   - 不要把 infer-pose 结果混进主结论

### 0.9 文档更新要求

实现和实验完成后，必须同步更新：
- `docs/STATUS.md`：只写现状，不写过程
- `docs/DESIGN.md`：补 fused target 的设计选择、v1 边界与默认开关
- `docs/CHANGELOG.md`：记录改了哪些文件、跑了哪些实验、结果如何

如果最终只实现了 v1，而没有做 depth-agreement 版本，也要明确写清：
- 哪些内容已落地
- 哪些内容被刻意延后
- 为什么延后

### 0.10 验收标准

交付前至少确认以下几点：
- 新 cache 中 fused 文件齐备，样本数与原 sample 数一致；
- `left/right` 旧逻辑未被破坏；
- `fused` 模式 refine 能正常跑通；
- 至少有一组同口径正式实验结果；
- 文档已更新，且过程与现状分离。

---

## 1. 问题背景

当前每个pseudo sample已有两张修复结果：target_rgb_left.png（以左参考帧为条件）和 target_rgb_right.png（以右参考帧为条件）。训练时只二选一使用，这意味着已付出双向修复代价但只消费单边信息，且单边diffusion的hallucination一旦落在高置信区域就会被直接写回高斯外观参数。

方案1的核心目标不是做更漂亮的平均图，而是通过左右一致性+几何置信+视角权重构造一个更保守、更稳定的fused supervision。

---

## 2. 数学框架

### 2.1 记号

对每个pseudo sample定义：原始render图 I_r，左右修复图 I_L, I_R，左右EDP深度与置信度 D_L, C_L, D_R, C_R，render深度 D_r。

### 2.2 Branch Score

对单边branch定义逐像素score：

S_L(p) = C_L(p) * G_L(p) * V_L

其中：

- C_L(p)：EDP几何置信度
- G_L(p) = exp(-|D_r(p) - D_L(p)| / (tau_d * D_r(p)))：render深度与branch深度的一致性gate
- V_L = alpha1 * mean(C_L) + alpha2 * valid_ratio_L：视角权重，第一版采用简易形式

右侧同理。

### 2.3 Agreement Term

左右一致性项：

A_rgb(p) = exp(-||I_L(p) - I_R(p)||_1 / tau_rgb)

可选深度一致性项（通过参数开关控制）：

A_depth(p) = exp(-|D_L(p) - D_R(p)| / (tau_ld * D_r(p)))

最终agreement取乘积：A(p) = A_rgb(p) * A_depth(p) 或仅 A(p) = A_rgb(p)。

### 2.4 融合权重

最终权重由branch score与agreement共同决定：

W_L(p) = A(p) * S_L(p), W_R(p) = A(p) * S_R(p)

归一化后：

W_tilde_L(p) = W_L(p) / (W_L(p) + W_R(p) + eps)

### 2.5 残差融合

不直接融合RGB，而是融合残差：

R_L(p) = I_L(p) - I_r(p), R_R(p) = I_R(p) - I_r(p)

I_F(p) = I_r(p) + W_tilde_L(p) * R_L(p) + W_tilde_R(p) * R_R(p)

这样若左右都只在局部修某些区域，未修区域自然保留原render；若两边偏离方向相反，不会把颜色平均到发灰。

### 2.6 Fused Confidence

C_F(p) = min(1, (W_L(p) + W_R(p)) / tau_c) * A(p)

同时表达至少有一边branch在此有较高几何可信度和左右两边在此外观基本一致。若某区域只有一边自信但另一边严重不同，A(p)会压低该区域权重。

---

## 3. 落地规划

### Phase 1：双边EDP输出

修改 epipolar_depth.py，新增 compute_edp_depth_pair() 接口，返回 depth_left, conf_left, stats_left, depth_right, conf_right, stats_right, depth_fused, conf_fused。保持 compute_edp_depth() 向后兼容。

### Phase 2：融合核心逻辑

新增 pseudo_branch/pseudo_fusion.py，核心函数：

- compute_branch_score(C, G, V) — 单边branch score
- compute_agreement_map(I_L, I_R, D_L, D_R, D_r, use_depth_agreement, tau_rgb, tau_ld) — 左右一致性
- fuse_residual_targets(I_r, I_L, I_R, W_tilde_L, W_tilde_R) — 残差融合
- build_fused_confidence(W_L, W_R, A, tau_c) — 融合置信度
- run_fusion_for_sample(...) — 入口函数

输出：target_rgb_fused.png, confidence_mask_fused.npy, confidence_mask_fused.png, fusion_meta.json。

### Phase 3：Cache构建集成

修改 build_pseudo_cache.py：

1. 调用 compute_edp_depth_pair() 获取双边输出
2. 写出 depth_left.npy, conf_left.npy, depth_right.npy, conf_right.npy
3. 调用 pseudo_fusion.py 生成融合结果
4. 写出融合产物及 fusion_meta.json
5. 更新 manifest.json，新增字段

### Phase 4：诊断可视化

修改 diag_writer.py，新增：rgb_disagreement.png, weight_left.png, weight_right.png, support_left.png, support_right.png, support_fused.png, confidence_fused.png。

### Phase 5：Refine接入

修改 run_pseudo_refinement.py：将 --use_left / --use_right 改为 --target_side left|right|fused。fused 模式读取 target_rgb_fused.png + confidence_mask_fused.npy，若后者不存在则fallback到旧 confidence_mask.npy。history记录新增 target_side, pseudo_manifest_version, num_fused_samples_used。

### Phase 6：验证实验

在E配置（lambda_pseudo=0.5, freeze-geom, real-densify, tertile）基础上做两组对照：

- E-fused-rgb-only：use_depth_agreement=False
- E-fused-with-depth：use_depth_agreement=True

与E（单边tertile）对比PSNR/SSIM/LPIPS。

---

## 4. 超参数与开关

**超参数**（写入 fusion_meta.json）：

| 参数 | 作用 | 建议初值 |
|------|------|----------|
| tau_rgb | RGB agreement阈值 | 10.0 |
| tau_d | 深度一致性gate相对阈值 | 0.1 |
| tau_ld | 深度agreement相对阈值 | 0.05 |
| tau_c | Fused confidence归一化阈值 | 1.0 |
| alpha1, alpha2 | 视角权重系数 | 0.7, 0.3 |

**开关参数**：

- use_depth_agreement：是否使用 A_depth，默认 True

---

## 5. 文件改动清单

| 文件 | 类型 | 说明 |
|------|------|------|
| pseudo_branch/epipolar_depth.py | 修改 | 新增双边输出接口 |
| pseudo_branch/pseudo_fusion.py | 新增 | 融合核心逻辑 |
| pseudo_branch/build_pseudo_cache.py | 修改 | 集成融合，写新字段 |
| pseudo_branch/diag_writer.py | 修改 | 新增诊断图 |
| scripts/run_pseudo_refinement.py | 修改 | 接入fused target |

---

## 6. 数据结构扩展

每个sample目录新增：

- target_rgb_fused.png
- confidence_mask_fused.npy
- confidence_mask_fused.png
- depth_left.npy, depth_right.npy
- conf_left.npy, conf_right.npy
- fusion_meta.json

manifest.json 新增字段：target_rgb_fused_path, confidence_mask_fused_path, depth_left_path, depth_right_path, conf_left_path, conf_right_path, fusion_meta_path，以及sample级指标 mean_conf_fused, support_ratio_fused, mean_rgb_agreement。

---

## 7. 实施顺序

严格按Phase 1到6执行，每Phase完成后验证输出正确性再进入下一Phase。Phase 1-4为数据准备，Phase 5为训练接入，Phase 6为实验验证。
