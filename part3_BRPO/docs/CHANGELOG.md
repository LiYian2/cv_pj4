# CHANGELOG.md - Part3 Stage1 过程记录

> **书写规范**：
> 1. 增量更新，倒序排列（最新在最上）
> 2. 每条提炼为 3-5 行：做了什么 → 发现什么 → 结论是什么
> 3. 不重复设计判断（DESIGN.md 负责），不重复状态细节（STATUS.md 负责）
> 4. 实验细节引用归档文档：`[参见 archived/2026-04-experiments/P2X_*.md]`

---

## 2026-04-17

### DL3DV：Phase G baseline + SPGM bring-up
- 基于 canonical DL3DV chain（internal cache / signal-aware selection / canonical prepare root / signal_v2）完成首轮 StageA → StageA.5 → bounded StageB baseline → SPGM repair A，并统一 replay
- 发现：当前 DL3DV pseudo_cache 不含 `target_depth_for_refine_v2`，所以首轮 refine 采用 `signal_pipeline=brpo_v2 + brpo_v2_raw`，depth target 回退到 canonical `target_depth_for_refine`；修正后链路可稳定跑通
- 结论：DL3DV case bring-up 已完成，`canonical_baseline_post40_lr03_120` 首轮略优于 `spgm_repair_a_keep111_eta0_wf025`，selector-first 先不启动 [参见 `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260417_dl3dv_phaseg_baseline_spgm_e1/summary.json`]

### P2-S：support_blend far-keep follow-up compare
- 固定 ranking=`support_blend`，只扫 far keep ratio (`0.90/0.85/0.80`)
- 发现：`far=0.90` 已逼近 repair A parity（`+0.000069 PSNR`）
- 结论：下一步围绕 `0.90` 做窄范围确认，不回强 selector [参见 archived/2026-04-experiments/P2S_*.md]

### P2-R：score/ranking repair compare
- 拆分 `ranking_score` 与 `weighting_score`，新增 `support_blend` ranking
- 发现：support-aware ranking 有小幅回升，但仍低于 repair A
- 结论：ranking 方向有效，但不完全解决问题 [参见 archived/2026-04-experiments/P2R_*.md]

### P2-O：selector-first formal compare
- 以 repair A 为 control，比较 selector S1/S2
- 发现：selector 越强 replay 越差（误删有用更新）
- 结论：当前 ranking 下 selector 会伤 replay，需先修 ranking [参见 archived/2026-04-experiments/P2O_*.md]

### P2-N：selector-first plumbing smoke
- 实现 `selector_quantile` policy，selector 真实生效
- 发现：`selected_ratio < active_ratio`，Gaussian 子集被缩小
- 结论：plumbing 完成，下一步正式 compare [参见 archived/2026-04-experiments/P2N_*.md]

### P2-M：conservative deterministic repair compare
- 更温和 suppress（A/B 两臂），grad weight回升
- 发现：repair A 优于原始 SPGM，但仍低于 canonical baseline
- 结论：suppress 过强是问题，但还需 selector-first [参见 archived/2026-04-experiments/P2M_*.md]

### P2-L：canonical StageB formal compare
- 对齐 protocol 后，SPGM 仍低于 baseline
- 发现：两臂 rejection 一致，grad weight 下降是主变化
- 结论：protocol drift 不是主解释，需进入 repair [参见 archived/2026-04-experiments/P2L_*.md]

### P2-K：forensic audit + code hygiene
- 审查 SPGM v1 实现，发现 entropy 归一化问题
- 修复：entropy 按 `log(B)` 归一化，density_mode 真实接通
- 结论：wiring 正确，但原始 compare protocol 与 baseline 漂离 [参见 archived/2026-04-experiments/P2K_*.md]

---

## 2026-04-16

### P2-J：bounded StageB schedule compare
- 比较 `post40_lr03_120` / `post80_lr03_120` / `post80_lr03_real05_120`
- 发现：`post40_lr03_120` 最佳，能把 120iter cliff 拉回
- 结论：StageB 已有 bounded baseline，不再深调 schedule [参见 archived/2026-04-experiments/P2J_*.md]

### P2-I：StageB 回落窗口定位
- 固定 gated winner，扫 `20/40/80/120` iter
- 发现：PSNR 最佳在 40iter，80→120 出现 cliff
- 结论：StageB 不是完全没价值，但需 bounded schedule [参见 archived/2026-04-experiments/P2I_*.md]

### P2-H：RGB-only v2 StageB-120iter verify
- gated 仍真实工作（96/120 rejection），但两臂都低于 baseline
- 发现：gating 不能阻止 120iter regression
- 结论：需先修 StageB 后段 schedule [参见 archived/2026-04-experiments/P2H_*.md]

### P2-G：StageB real-branch short compare
- `RGB-only v2 + gated_rgb0192` 在 StageB 优于 legacy
- 发现：real branch 未被误伤，gated 小幅正向
- 结论：主候选延续到 joint refine [参见 archived/2026-04-experiments/P2G_*.md]

### P2-F：RGB-only v2 StageA.5 gated compare
- `RGB-only v2` 明显优于 legacy，branch-specific gating 有小幅正增益
- 发现：raw RGB 稀疏不影响 StageA.5 表现
- 结论：gating 主线转到 RGB-only v2 [参见 archived/2026-04-experiments/P2F_*.md]

### P2-E：legacy threshold calibration
- `vr=0.02/0.03` 进入真实 reject 区，但 replay 改善极弱
- 发现：threshold softness 不是 legacy 主瓶颈
- 结论：转向 RGB-only v2 做 branch-specific calibration [参见 archived/2026-04-experiments/P2E_*.md]

### P2-D：signal gate 阈值诊断
- 默认阈值 `0.01/0.01/0.995` 对 legacy 过松，0 rejection 是结构必然
- 发现：legacy 实际 range `verified 1.89%~5.17% / rgb 14.68%~22.05%`
- 结论：需先标定 legacy threshold [参见 archived/2026-04-experiments/P2D_*.md]

### SPGM v1 Phase 0-6 完成
- Phase 0-3：plumbing/stats/score/policy 落地
- Phase 4-6：StageA.5/StageB 接入 + 120iter compare
- 发现：SPGM 真实调制 grad，但 120iter replay 低于 baseline
- 结论：实现稳定，但需 repair / 超参调优 [参见 CHANGELOG 条目 P2-K~P2-S]

---

## 2026-04-15

### P2-C：StageB real-branch short compare
- gated 不会误伤 real branch，但 replay/loss 与 ungated 重合
- 发现：问题从"误伤风险"转为"gate 未产生 rejection"
- 结论：需解释为什么 current threshold 下 0 rejection [参见 archived/2026-04-experiments/P2C_*.md]

### P2-B：StageA.5 gated vs ungated
- gated 只有极轻微 replay 优势，没有明显改善
- 发现：`iters_with_rejection=0/80`，signal gate 未生效
- 结论：转向 StageB 验证 real branch 误伤风险 [参见 archived/2026-04-experiments/P2B_*.md]

### P2：local Gaussian gating 第一版实现
- 新增 `pseudo_local_gating_*` CLI，StageA.5/StageB split backward
- 发现：hard gating 骨架已落地，smoke 通过
- 结论：进入 8-frame compare 验证收益 [参见 STATUS §6.2]

### P1A：signal_v2 StageA-only compare
- `legacy / v2-rgb-only / v2-full` 三臂对照
- 发现：`v2-rgb-only` 可保留，`v2-full` 过窄（verified≈1.96%）
- 结论：v2 分支保留 rgb-only，不直接接管 full depth [参见 STATUS §6.2]

### P0：abs prior 标定 + StageA-only 性质确认
- 固定背景：`lambda_abs_t=3.0, lambda_abs_r=0.1`
- 发现：StageA 不更新 Gaussian，replay-on-PLY 只是 sanity check
- 结论：replay compare 必须移到 StageA.5/StageB [参见 STATUS §6.2]

### Fusion v1 + Signal v2 第一版落地
- fusion 改为 `target↔reference overlap confidence`
- 新增 `signal_v2/` 路径，fused RGB mask + depth supervision v2
- 发现：wiring 通，但需 compare 证明优于 legacy [参见 STATUS §6.2]

---

## 2026-04-14

### E2：dual-pseudo allocation
- `top2 per gap` 实际落成 `1/2+2/3`
- 发现：E2 伪帧池扩大但短跑 loss 回落，次优样本稀释 supervision
- 结论：default winner 保持 E1 `signal-aware-8` [参见 archived/2026-04-plans-landed/SIGNAL_ENHANCEMENT.md]

### E1.5：support-aware selection 正式短对照
- `signal-aware-8` 正式 verify/pack + StageA-20iter 对照
- 发现：verified_ratio / continuous_confidence 有提升，densify coverage 上升
- 结论：E1 收口，收益先传导到 densify [参见 archived/2026-04-plans-landed/SIGNAL_ENHANCEMENT.md]

### E1：support-aware pseudo selection 第一轮
- `signal-aware-8` 选出 `[23,57,92,127,162,196,225,260]`
- 发现：`6/8` gap 改选，偏 2/3 位置
- 结论：midpoint 不是稳定最优，E1 方向成立 [参见 archived/2026-04-plans-landed/SIGNAL_ENHANCEMENT.md]

### S1.3：confidence-aware densify + 8-frame compare
- retuned 阈值回调到可用区间
- 发现：conf-aware 组仍高于 baseline loss
- 结论：wiring 通但参数需继续回调 [参见 archived/2026-04-plans-landed/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md]

### S1.2：RGB/depth mask semantics split
- consumer 显式分离 RGB/raw confidence 与 depth/train-mask
- 发现：loss 侧支持不同 mask
- 结论：语义分工 wiring 完成 [参见 STATUS §3.2]

### S1.1：continuous confidence + agreement-aware support
- 新增 raw continuous confidence 与 agreement-aware support
- 发现：verify/pack 产物已更新
- 结论：upstream semantics 完成 [参见 STATUS §3.2]

---

## 2026-04-13

### P1 bottleneck review：signal/scope mismatch
- midpoint8 M5 有效 depth 仅约 3.49%
- 发现：弱局 supervision 作用在全局 Gaussian，underconstrained
- 结论：主线从参数扫描转为结构修正 [参见 STATUS §6.3]

### P0 repair：stage handoff + reporting fix
- 修复 handoff：StageA/A.5/B 相机状态顺序传递
- 发现：修复后 A.5/B 略好，但仍负
- 结论：handoff bug 是真问题，但不是唯一瓶颈 [参见 STATUS §12]

### StageB Phase3：300iter long run
- 120iter PASS，300iter FAIL（相对 A.5 baseline）
- 发现：StageB 短程有效，长程退化
- 结论：主问题是后段稳定性 [参见 STATUS §9]

### A.5 midpoint 80iter + replay
- `xyz+opacity` 优于 baseline（`+0.0213 PSNR`）
- 发现：8 帧改动后有正向 replay 信号
- 结论：A.5(xyz+opacity) 可作为微调候选 [参见 STATUS §7]

### A+B：split abs prior + grad contrib 诊断
- split+scaled abs prior 落地
- 发现：rot/trans 主驱动来自 RGB/depth，abs prior 是稳定边界
- 结论：StageA 参数化完成 [参见 STATUS §6.2]

---

## 2026-04-12

### M5-2：source-aware depth loss 接入
- depth loss 按 source 拆分，fallback weight=0
- 发现：loss_depth 接通但仍不下降
- 结论：depth signal 弱但已真实进入 loss [参见 STATUS §6.2]

### M5-1：densify depth correction field
- densified coverage 从 ~1.56% 提到 ~14.21%
- 发现：train-mask 内 non-fallback 提到 ~81.2%
- 结论：upstream densify 可行 [参见 STATUS §6.2]

### M5-0：depth signal diagnosis
- verified depth 仅占 train-mask ~8%
- 发现：fallback 占主导
- 结论：需 densify correction field [参见 STATUS §6.2]

### M4.5：StageA 长对照（blended vs render-only）
- `blended_depth` 非零，`render_depth_only` ≈0
- 发现：depth 已接入，但 StageA 利用偏弱
- 结论：depth signal 弱有效 [参见 STATUS §6.2]

### M4：StageA consumer 显式接入新 target
- 显式 mask/depth mode，不再依赖隐式 fallback
- 发现：consumer wiring 完成
- 结论：upstream→consumer 链路打通 [参见 STATUS §6.2]

### M3：blended target_depth_for_refine 落地
- pseudo-view sparse verified depth + blended target
- 发现：verified depth coverage ~1.5%
- 结论：depth target 不再是占位符 [参见 STATUS §6.2]

### M2.5：propagation 合理区间研究
- 研究区间收敛到 10%~25% coverage
- 发现：radius=1/2，tau_rel_depth=0.01~0.02 更合理
- 结论：默认参数过宽，需收紧 [参见 STATUS §6.2]

### M2：train mask propagation 第一版
- seed_support → train_mask propagation 落地
- 发现：默认参数 coverage ~70%，过宽
- 结论：机制成立，参数需收紧 [参见 STATUS §6.2]

### M1：fused-first verification + compatibility layer
- `fused_first` mode 落地
- 发现：单纯改 verification order 改善有限
- 结论：需 M2 propagation [参见 STATUS §6.2]

### Phase 6：schema solidification
- fusion/verify/pack 真实产出新 schema
- 发现：consumer smoke 通过
- 结论：canonical schema 第一轮打通 [参见 STATUS §6.1]

### Phase 5：auditability/provenance 修补
- verification_meta、pack source_meta 增强
- 发现：provenance 链可审计
- 结论：链路完整性修复 [参见 STATUS §6.1]

### Phase 4：mask-only ablation（legacy vs brpo）
- 同 pseudo_cache、同 refine 配置，比较 mask 来源
- 发现：brpo mask 在 v1 fixed-pose RGB-only refine 下不优于 legacy
- 结论：mask 语义与 refine 消费方式不匹配 [参见 STATUS §6.1]

---

## 2026-04-11

### Phase 3：internal prepare → verify → pack
- BRPO verification 接入 internal prepare 流程
- 发现：3-frame prototype 通过
- 结论：internal route 链路打通 [参见 STATUS §6.1]

### Phase 2：same-ply replay consistency
- internal after_opt replay 与官方 eval 对齐
- 发现：replay 机制可用于 baseline/refined 比较
- 结论：replay pipeline 可信 [参见 STATUS §6.1]

### Phase 1：internal cache 导出
- before_opt/after_opt PLY + render cache 导出
- 发现：camera states 共享，PLY 各自保存
- 结论：internal cache 结构确立 [参见 STATUS §2.1]

### BRPO Phase B/C：verification 原型
- left/right 单分支 + 双分支 fusion
- 发现：verified support ratio ~0.8%~1.3%，非全空
- 结论：verification 机制可行 [参见 STATUS §6.1]