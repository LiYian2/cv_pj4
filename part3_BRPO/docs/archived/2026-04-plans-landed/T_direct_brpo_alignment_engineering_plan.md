# T_direct_brpo_alignment_engineering_plan.md

> 上位文档：`docs/archived/2026-04-plans-landed/BRPO_A1_B3_reexploration_master_plan.md`、`docs/archived/2026-04-reference/part3_BRPO_A1_B3_vs_BRPO_detailed_analysis.md`
> 相关现状：`docs/design/TARGET_DESIGN.md`、`docs/current/STATUS.md`、`docs/current/DESIGN.md`
> 触发原因：用户明确要求 **T~ 也按 direct BRPO 路线规划**，不再把 `stable/fallback consumer-side 小修补` 当成主推进路线。
> 执行顺序：**本计划在 G~ direct BRPO 主线完成首轮落地并拿到 compare 结论后再执行。**
> 文档角色：定义一条新的 **direct BRPO T~ path**，把当前系统从“exact `C_m` + target-side proxy”推进到“upstream verifier / proxy / projected-depth backend 也按 BRPO 语义重建”。

---

## 0. 结论先说

当前 T~ **还没有对齐 BRPO**。

已经对齐到位的是：
- M~ / `C_m` 语义基本已经对齐；
- `exact_brpo_full_target_v1` 已把 consumer 端 target composition 按 strict BRPO-style 方式写出来；
- compare 也已经证明：问题不再主要卡在 consumer-side blending。

但真正决定 T~ 语义的四层还没有完全对齐：
1. **proxy frame semantics**：当前上游仍依赖 `target_rgb_fused.png` 这类 residual-fused pseudo-frame proxy；
2. **verifier backend semantics**：当前 support / projected depth 由 `FlowMatcher + stage PLY render depth + threshold verify` 产生，本质仍是工程 proxy backend；
3. **projected-depth field semantics**：`exact_brpo_full_target_v1` 只是重组现有 `projected_depth_left/right`，没有重建 upstream target field；
4. **loss/consumer contract semantics**：exact target path 的产物虽然更干净，但仍沿用了 current backend 的 field，且 metadata 里仍明确 `recommended_stageA_depth_loss_mode='legacy'`。

所以这轮 T~ 不是继续抠：
- `stable target` 权重，或
- `fallback/render_depth` 的局部参数，或
- `source-aware consumer` 的小修补，

而是直接改成：

> **branch-native verifier input → exact support/projection backend → exact target field → exact T~ consumer/loss contract**

旧 `old / hybrid / stable / current exact_full_target_v1` 路线保留，但只作为 compare control，不再作为目标态。

---

## 1. 当前代码事实（已核对 live code）

### 1.1 `pseudo_branch/pseudo_fusion.py`
当前上游 pseudo-frame proxy 的核心事实：
- `normalize_branch_weights(...)` 只是把 `overlap_conf_left/right` 归一化成 `w_left / w_right`；
- `fuse_residual_targets(...)` 直接做
  - `I_render + W_L * (I_L - I_render) + W_R * (I_R - I_render)`；
- `run_fusion_for_sample(...)` 在 geometry-ready 时，仍把 `target_rgb_fused.png`、confidence masks、fusion weights 作为主要导出物。

这说明当前 `I_t^{fix}` / pseudo target RGB 上游仍是一个 **residual-fused proxy**，不是 direct BRPO target backend 本身。

### 1.2 `scripts/brpo_build_mask_from_internal_cache.py` + `pseudo_branch/brpo_reprojection_verify.py`
当前 verifier / projected-depth backend 的核心事实：
- `brpo_build_mask_from_internal_cache.py` 用 `FlowMatcher` 建立 pseudo/ref 对应；
- `verify_single_branch(...)` 通过：
  1. 从 ref view 的 stage PLY render depth 取点；
  2. 回投到世界坐标；
  3. 再投到 pseudo view；
  4. 用 `tau_reproj_px` 和 `tau_rel_depth` 做 hard threshold；
- `projected_depth_map` 只是把通过验证的 `reproj_z` 选最优写入像素；
- 最终导出的是 `support_left/right`、`projected_depth_left/right`、`projected_depth_valid_left/right` 等工程中间场。

这条链路说明：当前 exact T~ 所消费的 field 仍是 **proxy verifier backend 产物**，而不是一个“已经直接对齐 BRPO target semantics”的 upstream backend。

### 1.3 `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`
旧 target contract 的核心事实：
- `build_depth_supervision_v2(...)` 仍以 `render_depth` 为默认 fallback；
- 只有在 `raw_rgb_confidence >= min_rgb_conf_for_depth` 且 projected depth 有效时，才用 `projected_depth_left/right` 填充 target；
- `source_map` 仍保留 `SOURCE_RENDER_FALLBACK`。

这说明 old T~ 仍是：
- render fallback 主骨架
- projected-depth 局部替换
- rgb gate 驱动的工程化 target contract

它不是 direct BRPO T~ 目标态，但它应继续作为工程 control 保留。

### 1.4 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`
`build_exact_brpo_full_target_observation(...)` 的关键事实：
- 直接基于 `support_left/right` 构造 strict `C_m`；
- 再基于 `projected_depth_left/right + fusion_weight_left/right` 组成 `depth_target`；
- 在 both / left-only / right-only 区域保留 source provenance；
- summary policy 里明确写的是：
  - `confidence_rule = strict BRPO-style C_m ...`
  - `depth_target_rule = strict BRPO target-side proxy ...`
  - `strict_brpo_scope = cm_and_target`
  - `recommended_stageA_depth_loss_mode = legacy`

这说明当前 exact-full-target 已经把 **consumer 端 target composition** 写干净了，但它仍然只是：
- strict BRPO-style consumer contract
- 建立在 current upstream proxy backend 上的 target-side proxy

并没有把 upstream verifier / projection backend 本身重做。

### 1.5 compare 事实与当前判断
已有 compare/array 检查已经给出两个重要结论：
- `exact_brpo_full_target_v1` 相比 `exact_brpo_cm_hybrid_target_v1` 只提升约 `+0.00014 PSNR`；
- 相比 old control 仍约 `-0.013 PSNR`；
- builder 级 target/source contract 几乎相同，frame 级 target array 差异只有极小量级。

所以当前结论应固定为：
- M~ 已基本对齐；
- T~ consumer 已做过 exact 化尝试；
- 剩余瓶颈主要在 **upstream Layer-B proxy/backend**，而不是继续只抠 consumer-side 权重。

---

## 2. 这轮 direct BRPO T~ 的目标态

这轮要对齐的不是“看起来更像 exact target”，而是以下四个明确语义轴。

### 2.1 Proxy 输入语义对齐
目标：把 verifier/backend 的主输入从 residual-fused RGB proxy，提升到 branch-native、provenance-aware 的 target backend 输入。

工程定义：
- `target_rgb_fused.png` 保留为 debug / compare artifact；
- exact T~ backend 不再把 residual-fused RGB 当 authoritative truth；
- verifier 主输入应显式区分：
  - left branch pseudo
  - right branch pseudo
  - optional fused debug view
- exact path 必须保留 branch provenance，不能在最上游先把来源抹平。

### 2.2 Verifier backend 语义对齐
目标：让 T~ 的 support / validity / projected depth 来自一条显式命名的 exact upstream backend，而不是继续复用 current thresholded proxy chain。

工程定义：
- 新增 exact backend mode，显式区分于 current `proxy_flow_v3` 风格实现；
- per-branch support、multi-hit policy、occlusion/depth-valid diagnostics 都要一起输出；
- `branch_first` 应成为 exact path 主默认，`fused_first` 仅保留为 control/diagnostic。

### 2.3 Target field 语义对齐
目标：让 depth target field 真正由 verified branch projection 定义，而不是由 old render-fallback contract 再包一层 exact 名字。

工程定义：
- exact T~ target field 只在 verified union 内定义；
- both / left-only / right-only 的 provenance 必须保留到最终 target/source_map；
- exact path 不再用 `render_depth` 作为默认 target 骨架；
- unsupported 区域应该是 invalid / zeroed exact target，而不是悄悄 fallback 成 old field。

### 2.4 Consumer / loss contract 对齐
目标：让 exact T~ 的 consumer 和 loss 明确消费 exact upstream field，而不是 metadata 写 exact、训练仍走 legacy expectation。

工程定义：
- exact T~ path 需要单独的 depth-loss contract 标识；
- M~ 与 T~ 应共享同一套 exact support family；
- `legacy` depth-loss mode 保留为 control，不再作为 direct BRPO T~ 的默认推荐项。

---

## 3. 工程策略：不是继续补 old/hybrid/stable，而是新增一条平行 exact-upstream path

### 3.1 旧路全部保留，但只作为 control
保留原因：
- old A1 / old T~ 仍是 replay control；
- `exact_brpo_cm_old_target_v1` 是当前最干净的 M~ semantics control；
- `exact_brpo_cm_hybrid_target_v1`、`exact_brpo_cm_stable_target_v1`、`exact_brpo_full_target_v1` 已经构成必要的 ablation ladder。

因此这轮不覆盖旧 mode，而是新增一条显式命名的新 exact-upstream T~ path。

### 3.2 direct T~ path 必须显式命名，不能偷改现有 exact_full_target_v1 的含义
建议新增显式语义轴，例如：
- `verifier_backend_semantics = proxy_flow_v3 | exact_branch_native_v1`
- `target_field_semantics = old_render_fallback_v2 | hybrid_verified_v1 | stable_blend_v1 | exact_upstream_v1`
- `target_proxy_semantics = residual_fused_debug | branch_native_exact`
- `target_loss_contract = legacy_depth_v2 | exact_shared_cm_v1`

这样 compare 才知道自己在比较：
- consumer 端 exact，还是
- upstream backend 端 exact。

### 3.3 这轮 direct T~ 的改动范围只限 T~ backend / consumer，不混入 G~/R~
执行时必须固定：
- G~ 使用当时已确定的 direct/landing control；
- R~ 继续固定主线；
- 不把 topology 或 G~ 改动混入 T~ compare。

否则就会再次失去 T~ 语义因果归因。

---

## 4. 代码改动规划（按文件）

## 4.1 `pseudo_branch/pseudo_fusion.py`

### 当前问题
它现在同时承担：
- 生成可视化 / debug 的 fused pseudo RGB；
- 近似定义 verifier/backend 实际读取的 pseudo target proxy。

这会让 exact T~ 上游在第一步就被 residual-fusion 语义锁死。

### 改动方向
1. 把 `target_rgb_fused.png` 明确降级为 debug/control artifact；
2. 新增 exact path 导出包，显式保存：
   - branch-native pseudo RGB inputs
   - branch-level confidence / overlap provenance
   - optional fused debug image
3. 不让 `normalize_branch_weights(...)` 的简单归一化逻辑继续充当 exact target truth 的定义。

### 目标结果
exact T~ 上游输入不再是“先 fuse 再 verify”，而是“保留 branch provenance，再进入 exact verifier backend”。

## 4.2 `pseudo_branch/brpo_reprojection_verify.py`

### 当前问题
`verify_single_branch(...)` 已能导出 support / projected depth，但它还是 current proxy backend：
- hard threshold reprojection + relative depth agreement；
- 单像素只保留一个 best reproj_z；
- diagnostics 不足以表达 exact-upstream validity semantics。

### 改动方向
1. 保留 current `verify_single_branch(...)` 作为 control backend；
2. 新增 exact backend builder，例如：
   - branch-native support provenance
   - multi-hit resolve policy
   - occlusion / invalid-depth / out-of-bounds reason maps
   - projected-depth confidence / density summary
3. exact path 的 projected-depth 输出要服务 target field builder，而不是只服务当前 proxy compare。

### 目标结果
support / projected depth 不再只是“过阈值就写一个 z”，而是一套可被 exact T~ field 直接消费的 upstream bundle。

## 4.3 `scripts/brpo_build_mask_from_internal_cache.py`

### 当前问题
脚本现在能导出很多中间产物，但 mode 组织仍围绕 current proxy compare：
- `verification_mode = branch_first | fused_first`
- summary/meta 主要描述 current verification policy
- 没有把 exact-upstream backend 单独封成一个命名稳定的 artifact contract

### 改动方向
1. 新增 exact backend 导出模式，例如 `exact_branch_native_v1`；
2. 把下面这些 exact artifact 作为一套稳定 contract 导出：
   - support_left/right exact
   - projected_depth_left/right exact
   - projected_valid_left/right exact
   - provenance / diagnostics / reason maps
3. 在 `summary_meta.json` 中显式写入：
   - verifier backend semantics
   - target proxy semantics
   - exact vs control 标识

### 目标结果
`build_brpo_v2_signal_from_internal_cache.py` 后续可以明确消费 exact-upstream bundle，而不是继续猜这些 field 属于哪一代 proxy 实现。

## 4.4 `pseudo_branch/brpo_v2_signal/depth_supervision_v2.py`

### 当前问题
它代表的是 old T~ contract：
- render-depth fallback 是默认骨架；
- rgb confidence 是 activation gate；
- exact 路线若继续借它兜底，最终只会回到 old contract。

### 改动方向
1. 保留 `build_depth_supervision_v2(...)` 作为 old control；
2. 新增 exact-upstream target field builder，要求：
   - 输入来自 exact verifier bundle；
   - verified union 外不再默认灌 `render_depth`；
   - both/left/right source_map 与 valid_mask 一起导出；
   - field/loss 所需 mask 与 target 同源。

### 目标结果
exact T~ field 和 old render-fallback field 完全分离，避免 exact 名字下偷偷走 old target 骨架。

## 4.5 `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`

### 当前问题
`build_exact_brpo_full_target_observation(...)` 已经是最干净的 consumer-side exact builder，但它仍然复用 current upstream maps。

### 改动方向
1. 保留 `exact_brpo_full_target_v1` 作为“consumer exact / upstream proxy” control；
2. 新增真正的 exact-upstream 版本，例如 `exact_brpo_upstream_target_v1`；
3. 新版本只读取 exact backend 导出的 support / projected-depth / provenance fields；
4. summary policy 中明确区分：
   - `strict_brpo_scope = cm_and_target`
   - `upstream_backend = exact_branch_native_v1`
   - `target_loss_contract = exact_shared_cm_v1`

### 目标结果
compare 时能清楚知道：现在提升的是 upstream T~ 语义，而不是只换了 consumer 包装。

## 4.6 `scripts/build_brpo_v2_signal_from_internal_cache.py`

### 当前问题
它已经能编排 `old / hybrid / stable / exact_full_target_v1` 等 bundle，但还缺一个“exact-upstream T~”的正式生产入口。

### 改动方向
1. 把 exact-upstream T~ 作为单独产物模式接入脚本；
2. 统一输出 metadata：
   - `pseudo_observation_mode`
   - `verifier_backend_semantics`
   - `target_field_semantics`
   - `target_loss_contract`
3. 保证 compare 脚本和 replay 配置能直接引用该模式，而不需要手动拼 field。

### 目标结果
exact-upstream T~ 成为正式 builder mode，而不是一次性实验 patch。

## 4.7 `pseudo_branch/pseudo_loss_v2.py` + `scripts/run_pseudo_refinement_v2.py`

### 当前问题
当前 exact-full-target metadata 已经提示：consumer 端 exact target 仍建议走 `legacy` depth-loss mode。

### 改动方向
1. 新增 exact T~ 专用 depth-loss contract；
2. exact path 下，loss 直接消费 exact valid/support field；
3. legacy depth-loss mode 保留为 control，不再当 exact T~ 默认。

### 目标结果
训练时真正消费的是 exact T~ field，而不是“builder exact、训练 legacy”。

---

## 5. 分阶段执行顺序（明确在 G~ 完成后启动）

## Phase T0：锁定执行前提与 control
启动条件：
- G~ direct BRPO 第一轮落地已完成；
- G~ compare 已给出明确控制臂；
- T~ compare 期间不再同时改 G~/R~。

本阶段动作：
1. 固定 compare control：
   - `old M~ + old T~`
   - `exact M~ + old T~`
   - `exact M~ + exact_brpo_full_target_v1`
2. 固定 exact-upstream T~ 的 mode 命名和 metadata 口径。

## Phase T1：exact verifier/backend bundle 落地
目标：先把 upstream exact bundle 做出来。

本阶段重点：
- `pseudo_fusion.py`
- `brpo_reprojection_verify.py`
- `brpo_build_mask_from_internal_cache.py`

验收标准：
- exact bundle 已能独立导出；
- 它与 current proxy bundle 在 artifact/meta 上可区分；
- provenance / projected-depth / validity diagnostics 完整。

## Phase T2：exact target field builder 落地
目标：基于 exact backend bundle 生产真正的 exact T~ field。

本阶段重点：
- `depth_supervision_v2.py`
- `pseudo_observation_brpo_style.py`
- `build_brpo_v2_signal_from_internal_cache.py`

验收标准：
- exact-upstream target array 与 current hybrid/current exact_full_target_v1 有实质差异；
- `source_map` / `valid_mask` / `depth_target` 同源；
- 不再暗含 old render fallback。

## Phase T3：exact loss contract 接线
目标：让训练真实消费 exact T~ field。

本阶段重点：
- `pseudo_loss_v2.py`
- `run_pseudo_refinement_v2.py`

验收标准：
- exact T~ replay 配置不再依赖 legacy depth-loss recommendation；
- exact valid/support field 已贯穿到 loss 端。

## Phase T4：formal compare 与决策
目标：判断 direct BRPO T~ 是否真正落地，而不是只做了一次 upstream 名义替换。

最小 compare 梯度建议：
1. `exact M~ + old T~`
2. `exact M~ + exact_brpo_cm_hybrid_target_v1`
3. `exact M~ + exact_brpo_full_target_v1`
4. `exact M~ + exact_brpo_upstream_target_v1`

优先观察两类信号：
- 语义信号：artifact/array/source_map 是否真的变了；
- replay 信号：是否摆脱 current `-0.013` 弱负格局。

---

## 6. 成功标准与失败判据

### 6.1 成功标准
至少要同时满足：
1. exact-upstream T~ 的 artifact contract 与 current proxy contract 明确不同；
2. target/source array 不再只是浮点噪声级差异；
3. loss 端真实消费 exact field，而不是退回 legacy；
4. compare 至少证明“upstream exact 化”不再是 current exact_full_target_v1 那种几乎零语义增量。

### 6.2 失败判据
出现以下任一项，就不应把它写成 T~ 已 direct-BRPO 对齐：
1. 只是替换 metadata/命名，但 upstream field 没变；
2. exact-upstream target 最终仍默认灌回 render_depth fallback；
3. training 仍实际走 legacy depth-loss contract；
4. compare 里又同时混入 G~/R~ 改动，导致归因失效。

---

## 7. 一句话 handoff

T~ 现在的问题不是 `C_m` 不够 exact，也不是 consumer-side target composition 还差最后一层小权重；真正没对齐的是 upstream Layer-B：`proxy frame / verifier backend / projected-depth field / loss contract`。因此，G~ 完成后，T~ 的 direct BRPO 主线应是：先重做 exact-upstream backend，再让 consumer 和 loss 真正吃这套 field，而不是继续在 `stable/fallback` consumer patch 上兜圈子。
