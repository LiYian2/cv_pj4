# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 口径统一：M~（Mask）、T~（Target）、G~（Gaussian Management）、R~（Topology）
> 4. 实验细节引用归档文档

---

## 2026-04-20

### 文档重组：口径统一为 M~/T~/G~/R~
- 创建 TARGET_DESIGN.md（T~）、GAUSSIAN_MANAGEMENT_DESIGN.md（G~）、新 REFINE_DESIGN.md（R~）
- 更新 MASK_DESIGN.md（M~），分离 T~ 内容
- 更新 DESIGN.md，四大模块口径统一
- 更新 CHANGELOG.md，历史记录口径统一
- 结论：M~ 与 T~ 最初耦合实现但语义独立，现在各自有独立设计文档。

### M~-T~-E1：exact BRPO target-side compare
- 在 pseudo_observation_brpo_style.py 新增 exact_brpo_full_target_v1：strict M~ + exact T~ + shared-M~ depth loss
- Compare 结果：exact M~ + exact T~ = 24.174488，exact M~ + old T~ = 24.187495，old M~ + old T~ = 24.187737
- 结论：exact M~ 已对齐，exact T~ 在当前 proxy backend 下仍弱 -0.013 PSNR，瓶颈在上游 Layer B。

### M~-S1：strict BRPO M~ checklist + exact/hybrid split
- 新增 A1_STRICT_BRPO_ALIGNMENT_CHECKLIST.md，明确 M~ 与 T~ 分离
- 接入三条 exact M~ ablation：exact_brpo_cm_old_target_v1、exact_brpo_cm_hybrid_target_v1、exact_brpo_cm_stable_target_v1
- Compare 结果：exact M~ + old T~ ≈ old M~ + old T~（差 < 1e-5），说明 M~ 已对齐
- 结论：剩余 gap 在 T~ contract，不在 M~。

### M~-D1：direct BRPO builder + fixed R~ compare
- 新增 build_brpo_direct_observation，产出 hybrid M~ + hybrid T~ bundle
- Compare 结果：hybrid M~ + hybrid T~ = 24.175408，优于 new M~ + new T~ 但弱于 old 约 -0.012
- 结论：hybrid 分支（historical brpo_direct_v1）不是 strict BRPO，应明确标记为 hybrid。

### G~-O1-C1：delayed opacity participation compare
- 在 score.py 拆出 participation_score，在 manager.py 新增 deterministic_opacity_participation
- Compare 结果：opacity vs summary = -0.000573 PSNR，仍弱负
- 结论：O0/O1 wiring 成立，但 delayed opacity 当前不应推进 O2a/b，应先做 C0 诊断。

### M~-V2-H1：fixed R~ 下 conf-only / depth-only hybrid compare
- 补做 conf-only（M~ brpo_style_v2 + T~ old）与 depth-only（M~ old + T~ brpo_style_v2）
- Compare 结果：三臂差异 ~1e-4，不能归因成单侧问题
- 结论：剩余 gap 需 M~ + T~ 联合改动。

### M~-R1：BRPO-style v1 builder + fixed R~ compare
- 新增 pseudo_observation_brpo_style.py，产出 hybrid M~ + hybrid T~ bundle
- Compare 结果：hybrid M~ + hybrid T~ = 24.175377，优于 new M~ + new T~ 但弱于 old
- 结论：shared M~ 方向对，但 verifier/target builder 不够强。

### M~-V1：verifier proxy + fixed R~ compare（negative）
- 新增 pseudo_observation_verifier.py，产出 verify M~
- Compare 结果：verify M~ + new T~ = 24.067703，显著差于 old 和 new
- 结论：verify proxy 已完成排错职责，证明保守 decoupling 不够强。

## 2026-04-19

### G~-R1：deterministic participation controller + compare
- 把 G~ 从 xyz_lr_scale 推进到 pre-render boolean participation control
- Compare 结果：boolean vs summary = -0.0032 PSNR，weak-negative
- 结论：G~ 方法对象已切对，但 keep 配置当前 no-go。

---

## 口径说明

- M~ = Mask（confidence）
- T~ = Target（depth target）
- G~ = Gaussian Management（per-Gaussian gating）
- R~ = Topology（joint loop）

历史记录中的 A1 → M~ + T~，B3 → G~，T1 → R~。