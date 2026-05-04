# BRPO Paper-Realign Engineering Plan (2026-05-03)

## 0. 目标

这份规划的目标不是继续把 `exact_brpo_upstream_target_v1 + exact_shared_cm_v1` 打磨得更“强”，而是**把当前 DL3DV 路线拉回更接近 BRPO 论文的 producer / consumer 语义**，减少 RGB 与 depth 的过度耦合，避免 branch-native exact verifier 在上游过早杀死大量 supervision。

本轮的设计目标是：

1. **RGB-side `C_m` 回到 paper-style 语义**：在 fused pseudo frame 上与左右 GT ref 建 correspondence，再离散成三档 `C_m ∈ {1.0, 0.5, 0.0}`。
2. **target-depth producer 独立出来**：继续做几何投影与轻量过滤，但**不再反过来支配 RGB-side `C_m`**。
3. **consumer 解耦**：RGB 只用 `C_m`；depth 再考虑额外 confidence / valid gating，但这些 depth-side 权重**不回流到 RGB**。
4. **保留轻量过滤，不回到 current hard exact verifier**：允许 reciprocal / bounds / positive-depth / 轻量 consistency，但不再用 branch-native exact backend 的重几何合同去共同定义 `C_m` 和 target depth。

---

## 1. 当前问题归纳

### 1.1 当前 exact-heavy 路线的关键耦合

当前 `exact_brpo_upstream_target_v1` 路线里：

- branch RGB matching 使用 `difix/left_fixed`、`difix/right_fixed`
- verifier 的 3D 几何检查使用原始 `after_opt/render_depth_npy/*.npy`
- verified support 同时控制：
  - `C_m`
  - `target_depth`
  - `valid_mask`
  - `target_confidence`
- consumer `exact_shared_cm_v1` 再把它们压成一个共享合同：
  - `effective_mask = C_m * valid_mask * target_confidence`

于是 RGB 和 depth 在 producer / consumer 两侧都被绑得过紧。

### 1.2 当前 DL3DV 上暴露出来的结构性风险

1. `difix` branch RGB 与原始 pseudo depth 真实存在 source mismatch。
2. matcher 有候选，但 exact verifier 大量杀死候选，尤其是 `depth_mismatch`。
3. active support 虽然少，但 loss 对 active mask 是归一化的；因此“少量但偏置很强”的 supervision 依然可能产生很强的错误更新。
4. 当前问题更像是**合同过重 + producer/consumer 混绑**，而不是单一阈值设置错误。

---

## 2. 论文对齐目标（本轮拟采用的语义）

参考 `docs/paper/BRPO_METHOD_extracted_20260424.md`：

- `C_m` 来自 fused pseudo frame `I_fix` 与左右 real refs 的 correspondence sets `M_k / M_{k+1}`。
- `L_rgb` 与 `L_d` 都以 `C_m` 为基本 mask。
- 论文没有 current exact-upstream 里 `valid_mask * target_confidence` 这种共享 consumer 合同。

因此，本轮 paper-realign 的核心口径是：

- **RGB producer：paper-style fused correspondence mask**
- **Depth producer：geometry target builder, but depth-only semantics**
- **Consumer：RGB uses `C_m` only; depth uses `C_m` + optional depth-only confidence**

---

## 3. 新的目标数据流

建议把当前单条 exact-heavy 链拆成两条 producer：

### 3.1 RGB / `C_m` producer（独立）

输入：
- fused pseudo RGB (`I_fix`)
- left GT ref RGB
- right GT ref RGB
- matcher config（`sparse_desc_2d` / `dense_pts3d_3d`）

处理：
1. fused pseudo ↔ left GT 做 matching，得到 fused-image-domain 对应集合 `M_k`
2. fused pseudo ↔ right GT 做 matching，得到 fused-image-domain 对应集合 `M_{k+1}`
3. 对 fused pseudo 的像素网格 rasterize：
   - `p ∈ M_k ∩ M_{k+1}` → `C_m = 1.0`
   - `p ∈ M_k ⊕ M_{k+1}` → `C_m = 0.5`
   - else → `C_m = 0.0`

轻量过滤允许保留：
- matcher reciprocal consistency
- matcher 自身 confidence / quantile candidate 筛选
- bounds / duplicate suppression
- 可选很轻的 connected-component / isolated-point cleanup

**禁止回流的重过滤**：
- 不使用 branch-native exact backend verifier
- 不使用 pseudo-depth ↔ ref-depth exact consistency 去决定 `C_m`
- 不用 `valid_mask` / `target_confidence` 去再改写 `C_m`

### 3.2 target-depth producer（独立）

输入：
- fused pseudo camera state / pose
- left / right ref states
- 几何投影所需深度来源
- 可用时的左右 projected depth

处理原则：
1. target depth 继续由左右参考信息投影到 pseudo/fused 视角生成。
2. 允许使用**轻量几何有效性**：
   - in-bounds
   - positive depth
   - visibility / occlusion basic check
   - 轻量 relative-depth consistency（若保留，作为 depth-only producer 内部过滤）
3. depth producer 的输出只服务 depth：
   - `target_depth`
   - `depth_valid_mask`
   - `depth_confidence`（可选）
   - `depth_source_map`
4. **depth producer 不再回头修改 RGB-side `C_m`**。

重要约束：
- 不对 diffusion RGB 强行“改 depth”；depth 仍保持 geometry-source 生成。
- 但也不再让“branch RGB + old pseudo depth”的组合支配 RGB-side confidence mask。

---

## 4. producer 侧具体工程改动

### 4.1 新增一个 paper-style RGB mask builder mode

建议新增新模式，而不是直接覆盖 current exact 路线：

- 新 observation/mask mode 名称建议：
  - `paper_brpo_cm_v1`
  - 或 `brpo_paper_cm_v1`

建议落点：
- `pseudo_branch/mask/rgb_mask_inference.py`
- `pseudo_branch/mask/brpo_confidence_mask.py`
- `scripts/brpo_build_mask_from_internal_cache.py`

实现要点：
- 新增 fused-image-domain builder：直接读 fused pseudo RGB，而不是 `left_fixed/right_fixed`
- 调用现有 matcher factory：
  - `build_pair_matcher(matcher_mode=..., model_name=..., device=..., dense3d_conf_quantile=...)`
- 分别得到 left/right 对应集合后，直接生成 paper-style 三档 `C_m`
- 保存新的 meta：
  - `matcher_mode`
  - `matcher_meta_left/right`
  - `cm_both_ratio`
  - `cm_single_ratio`
  - `cm_nonzero_ratio`
  - `paper_style=True`
  - `uses_exact_backend=False`

### 4.2 新增一个 paper-style target-depth builder mode

建议新增目标模式，而不是覆盖 `exact_brpo_upstream_target_v1`：

- 新 target/observation mode 名称建议：
  - `paper_brpo_target_v1`
  - 或 `brpo_paper_target_v1`

建议落点：
- `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- `scripts/build_brpo_v2_signal_from_internal_cache.py`

实现要点：
- target-depth 生成基于 fused pseudo domain
- left/right 各自产生 projected depth 与 valid bit
- both/single 组合规则显式写盘
- 如果保留 confidence，则只写成 depth-side field：
  - `pseudo_target_confidence_paper_brpo_target_v1.npy`
- 不再输出 current exact-upstream 那种“RGB/depth 共用合同”的推荐语义

### 4.3 轻量过滤建议

depth producer 内允许保留的过滤：
- `projected in bounds`
- `depth > 0`
- basic visibility
- optional `tau_rel_depth_light`

不建议保留的 heavy 逻辑：
- branch-native verifier
- exact ref-depth-render contract 去定义 RGB mask
- `no_render_fallback=true` 这一类 exact-heavy 语义作为默认主线

---

## 5. consumer 侧具体工程改动

### 5.1 新增 split loss mode，而不是复用 `exact_shared_cm_v1`

建议新增新的 stageA depth loss mode：

- `paper_brpo_split_v1`
- 可选第二个更保守版本：`paper_brpo_split_depthconf_v1`

建议落点：
- `pseudo_branch/refine/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`（若 backend continuation 也要复用）

### 5.2 `paper_brpo_split_v1` 语义

RGB：
- `l_rgb = masked_rgb_loss(render_rgb, target_rgb, C_m)`

Depth：
- `l_depth = masked_depth_loss(render_depth, target_depth, C_m)`

说明：
- `masked_depth_loss()` 内部本来就会要求 `target_depth > 1e-4`
- 所以 depth 端即使外部只传 `C_m`，也不会在无效 target 上生效
- 这已经比 current shared exact contract 轻很多

### 5.3 `paper_brpo_split_depthconf_v1` 语义

RGB：
- `l_rgb = masked_rgb_loss(render_rgb, target_rgb, C_m)`

Depth：
- `depth_mask = C_m * soft_depth_conf`
- `l_depth = masked_depth_loss(render_depth, target_depth, depth_mask)`

说明：
- `soft_depth_conf` 只作用于 depth
- RGB 不再乘 `valid_mask` / `target_confidence`
- 这是本轮最推荐的“paper-aligned but not naive”版本

### 5.4 需要明确禁止的 consumer 行为

在新的 paper split mode 下：
- 不再使用 `effective_mask = C_m * valid_mask * target_confidence` 这种 shared contract
- 不再让 depth-side `valid_mask` 回头裁 RGB supervision
- 不再把 depth-side `target_confidence` 当作 RGB loss 权重

---

## 6. exposure correction 与 scale regularization

### 6.1 exposure correction

**结论：保留，而且当前代码里已经有。**

当前已有实现：
- `pseudo_branch/refine/pseudo_loss_v2.py::apply_exposure()`
- `pseudo_branch/refine/pseudo_loss_v2.py::exposure_reg_loss()`
- `run_pseudo_refinement_v2.py --stageA_lambda_exp`
- backend continuation 也已经支持 `lambda_exp`

因此本轮建议：
- **不要新发明 exposure 机制**
- 直接保留现有 `exposure_a / exposure_b + lambda_exp`
- 但在新 split loss mode 下，继续把 exposure correction 只用于 RGB 渲染项
- depth loss 不需要 exposure 参与

建议默认：
- 第一轮保持当前 `lambda_exp` 量级不变
- 如果 paper-realign 后 RGB-only / RGB-dominant loss 变强，再单独看是否需要微调 `lambda_exp`

### 6.2 scale regularization

**结论：可以规划，但不建议在第一轮 paper-realign patch 里就默认打开。**

原因：
1. 当前主问题首先是 supervision contract 与 producer/consumer 耦合，先不要把 Gaussian 几何正则也混进第一轮因果诊断。
2. 当前 mainline `pseudo_loss_v2.py` 没有明确的 paper-style `L_s` Gaussian scale regularizer；历史 legacy 脚本里存在过 ad-hoc scaling regularization 痕迹，但它不是 current mainline合同的一部分。
3. 如果第一轮同时改 producer、consumer、再加新的 scale reg，会很难判断收益来自哪里。

建议策略：
- **Phase 1：不加新的 scale reg，只保留现有 pose/exposure regularization**
- **Phase 2：如果 paper-realign 后 replay 仍存在 Gaussian over-expansion / unstable opacity / scale drift，再加 optional `lambda_scale_reg` ablation**

若 Phase 2 需要落地，建议：
- 在 `pseudo_loss_v2.py` / `backend_pseudo_loss.py` 增加可选 Gaussian regularization hook
- 第一版只做简单、可控的 scale smoothness / magnitude regularization
- CLI 增加：`--stageA_lambda_scale_reg`
- 默认 `0.0`，只做显式 ablation

---

## 7. 具体代码修改地图

### 7.1 需要修改 / 新增的核心模块

#### Producer: RGB / `C_m`
- `scripts/brpo_build_mask_from_internal_cache.py`
- `pseudo_branch/mask/rgb_mask_inference.py`
- `pseudo_branch/mask/brpo_confidence_mask.py`
- 如有必要：`pseudo_branch/common/flow_matcher.py` / `mast3r_matchers.py`（仅复用，不建议大改）

#### Producer: target depth
- `scripts/build_brpo_v2_signal_from_internal_cache.py`
- `pseudo_branch/target/depth_supervision_v2.py`
- `pseudo_branch/observation/pseudo_observation_brpo_style.py`
- 如有需要：`pseudo_branch/observation/brpo_reprojection_verify.py`（只抽轻量 depth-only 逻辑，不复用 current hard contract）

#### Consumer
- `pseudo_branch/refine/pseudo_loss_v2.py`
- `scripts/run_pseudo_refinement_v2.py`
- `pseudo_branch/refine/backend_pseudo_loss.py`

### 7.2 建议新增的模式名称

Observation / producer：
- `paper_brpo_cm_v1`
- `paper_brpo_target_v1`
- 或 combined bundle 名：`paper_brpo_observation_v1`

Loss / consumer：
- `paper_brpo_split_v1`
- `paper_brpo_split_depthconf_v1`

命名原则：
- 明确与 `exact_brpo_upstream_target_v1` 区分
- 明确这是“paper realign”实验线，而不是继续改写 current exact mainline

---

## 8. 实验落地顺序（建议）

### Phase A：先做 producer/consumer 解耦，不碰大训练

1. **A1: paper-style `C_m` producer smoke**
   - fused ↔ left/right GT matching
   - 产出三档 `C_m`
   - 检查 `cm_both_ratio / cm_single_ratio / cm_nonzero_ratio`

2. **A2: paper-style target-depth producer smoke**
   - 独立产出 `target_depth / depth_valid / depth_confidence / source_map`
   - 确认它不再改写 RGB `C_m`

3. **A3: consumer smoke**
   - `paper_brpo_split_v1`
   - `paper_brpo_split_depthconf_v1`
   - 1 iter / 5 iter short smoke 即可

### Phase B：最小 compare（不要一上来 full replay）

建议先做 4 臂：

1. `current_exact_control`
   - `exact_brpo_upstream_target_v1 + exact_shared_cm_v1`

2. `paper_cm_only_target_old`
   - paper-style `C_m`
   - old / stable target contract
   - split consumer

3. `paper_cm_paper_target_split`
   - paper-style `C_m`
   - paper-style target depth
   - `paper_brpo_split_v1`

4. `paper_cm_paper_target_split_depthconf`
   - paper-style `C_m`
   - paper-style target depth
   - `paper_brpo_split_depthconf_v1`

比较重点：
- support coverage
- both-vs-single ratio
- depth target fill ratio
- depth confidence mean
- StageB20 / StageB40 replay trend
- pseudo RGB/depth loss 曲线

### Phase C：如果 producer/consumer 解耦后明显变稳，再讨论额外项

按顺序考虑：
1. `tau_rel_depth_light` 是否需要扫
2. `lambda_exp` 是否需要小调
3. `lambda_scale_reg` 是否要作为 Phase 2 可选项加入

---

## 9. 风险与注意事项

1. **不要把 current `cm_only` / `rgb_only` ablation 直接当作否定本计划的证据。**
   - 之前 ablation 改的是 consumer，但 producer 仍是 exact-heavy 路线。
   - 本计划改的是 producer + consumer 的合同整体，不是同一个实验问题。

2. **不要把 paper-realign 与 exact-upstream mainline 混成同一模式。**
   - 应作为新实验线单独命名、单独写盘、单独 compare。

3. **第一轮不要同时上 scale regularization。**
   - 否则很难分清改善来自 supervision contract 还是 Gaussian 正则。

4. **target-depth producer 可以轻量，但不要彻底无过滤。**
   - 否则 depth 会过脏。
   - 关键不是“完全去掉 verify”，而是“不要再用重 verify 共同定义 RGB 与 depth”。

---

## 10. 一句话执行建议

**推荐执行。**

最值得先落地的不是继续扫 `tau_rel_depth`，而是：

- **producer 端**：把 `C_m` 拉回 fused pseudo ↔ left/right GT 的 paper-style matching mask；target depth 独立为 depth-only producer；只保留轻量过滤。
- **consumer 端**：新增 split loss mode，让 RGB 只用 `C_m`，depth 再单独使用 `C_m` + optional depth confidence。
- **regularization 端**：保留已有 exposure correction；scale regularization 先作为第二阶段可选 ablation，不默认上主线。

这条线能最干净地回答当前真正的问题：**到底是 current exact-heavy 合同本身过重，还是 depth target 本身就不可靠，还是两者都有。**
