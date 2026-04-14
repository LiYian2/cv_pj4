# hermes.md

> 用途：给后续 Hermes 自己看的快速回忆文档。对话断了、上下文丢了、或者用户直接让我“先回忆一下”时，先看这份，再按这里给出的文档路径继续展开。
> 维护原则：每做完一个阶段性任务后轻量覆盖更新；不追求完整日志，只保留“现在最该知道的状态、看哪里、下一步方向”。

## 1. 现在先怎么用这份文档

如果用户让我回忆最近做了什么，按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/STATUS.md`
3. 再看 `docs/DESIGN.md`
4. 再看 `docs/CHANGELOG.md`
5. 如需恢复更细的分析链路，再看专项报告或 `docs/DEBUG_CONVERSATION.md`

如果用户问“你记不记得昨天/上次做的审查”，默认不要先去翻聊天记录，先看本文件列出的“最近关键结论”和“推荐回顾文档”。

## 2. 当前项目位置（最新人工摘要）

当前主线是 Part3 BRPO 在 re10k case 上的 midpoint8 + M5 + StageA/A.5/StageB 结构复查与修正。

当前已经确认的状态：
1. 之前的 stage handoff bug 是真问题，而且已经修掉：现在 `StageA final -> StageA.5 init -> StageB init` 可以严格连续。
2. 但 handoff 修复后 pipeline 仍未转正，所以主瓶颈不只是 handoff。
3. 当前更深层的主矛盾已经收敛为：`signal weak + supervision scope / optimization scope mismatch`。
4. 换句话说：midpoint8 M5 提供的真实有效几何监督太弱，而 A.5 / StageB 又在更新全局 Gaussian `xyz/opacity`，所以容易出现 pseudo-side loss 降、replay 退化。

## 3. 最近最重要的结论

### 3.1 关于昨晚那轮 P1 bottleneck review

那次调研的核心结论不是“再调一轮超参”，而是：
1. `train_mask` 平均约 18.7%，但真正 non-fallback depth 只有约 3.49%，fallback 约 96.5%。
2. support 区域内 `target_depth_for_refine_v2` 相对 `render_depth` 的平均修正仅约 1.53%；折算整图，平均相对修正量约 0.053%。
3. 所以当前 pseudo signal 不是完全没有，而是“有效几何部分太稀、太弱”。
4. 同时 A.5 / StageB 打开的是全局 Gaussian `xyz/opacity`，这让弱而局部的 supervision 去驱动大范围全局扰动。
5. 因此当前最贴切的定性是：underconstrained global refinement。

### 3.2 当前方法层面的优先级判断

不要默认继续在固定 midpoint8 + 全局 xyz/opacity 上扫同一组 refine 超参。
当前优先级应转向：
1. pseudo 选点 / signal 质量复查
2. 收缩 A.5 / B 的可训练 Gaussian 作用域
3. 把 `support ratio + correction magnitude + replay` 固化成长跑前 gate

## 4. 想快速回顾时，看哪些文档

### 必看（从高到低）
1. `docs/hermes.md`
   - 用途：超短回忆入口；先找方向和文档导航。
2. `docs/STATUS.md`
   - 用途：看“当前状态、已完成、进行中、下一步”。
3. `docs/DESIGN.md`
   - 用途：看“为什么这么判断、设计矛盾在哪里”。
4. `docs/CHANGELOG.md`
   - 用途：看“最近具体做了哪些修改和结论沉淀”。

### 和昨晚 bottleneck review 直接相关的专项文档
1. `docs/MIDPOINT8_M5_P1_BOTTLENECK_REVIEW_20260413.md`
   - 最直接的昨晚调研落盘版；要恢复完整结论优先看它。
2. `docs/MIDPOINT8_M5_STAGEA_A5_B_EXEC_REPORT_20260413.md`
   - 看严格 midpoint8 + M5 + StageA/A.5/B full pipeline 的执行与发现。
3. `docs/MIDPOINT8_M5_PIPELINE_REPAIR_PLAN_20260413.md`
   - 看之前为什么先盯 handoff / pipeline continuity。
4. `docs/DEBUG_CONVERSATION.md`
   - 这是用户手工整理的聊天摘录；适合补语气、上下文和当时的完整推理展开。
5. `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md`
   - 当前下一阶段主执行计划；重点看 continuous confidence、RGB/depth mask 语义分离、StageB 两段式与 local Gaussian gating。

## 5. 当前推荐口径（回答用户时尽量先给这个）

如果用户问“现在问题到底是什么”，优先回答：
- 不是主要卡在 handoff bug 了；
- 现在主矛盾是 midpoint8 M5 的有效几何信号偏弱；
- 而 A.5 / StageB 当前的全局 Gaussian refine 作用域过大；
- 所以属于 `weak local supervision -> global refinement` 的结构失配。

如果用户问“下一步该做什么”，优先回答：
1. continuous confidence + agreement-aware support
2. RGB/raw confidence 与 depth/train-mask 语义分离
3. confidence-aware densify
4. StageB 两段式 curriculum + local Gaussian gating
5. 固定 signal gate，再决定是否进长跑或更深 joint stage

## 6. 下次回来先检查什么

1. 本文件是否需要根据最近任务轻量更新
2. `STATUS.md` / `DESIGN.md` / `CHANGELOG.md` 是否和当前口径一致
3. 最近一次专项报告或执行计划是否已经写出并在这里挂上路径
4. superseded 的计划/执行类 md 可移入 `docs/archived/` 对应子目录
5. 如果用户问的是“昨晚/上次做了什么”，先从本文件定位，再去对应报告，不要直接凭模糊会话记忆回答

## 7. 给下一个 Hermes 的一句话

先看 `docs/hermes.md`，再去 `MIDPOINT8_M5_P1_BOTTLENECK_REVIEW_20260413.md` 和 `STATUS/DESIGN/CHANGELOG`；当前最重要的判断是：问题已从 handoff bug 收敛到 `signal weak + global refine scope mismatch`，不要把精力默认浪费在继续扫同一套 midpoint8 + 全局 xyz/opacity 超参上。
补充：2026-04-14 已完成 S1.1 第一版实现。若要回顾本次代码落地点，优先看 `docs/CHANGELOG.md` 最新条目和 `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md` 中 S1.1 标记。
补充：2026-04-14 已完成 S1.2 第一版实现；consumer 侧现在能区分 RGB/raw confidence 与 depth/train-mask。下一步优先看 `confidence-aware densify`，然后再做小规模 StageA/A.5 semantics ablation。
当前最新落点（2026-04-14）：S1.1/S1.2/S1.3 都已做完第一版实现，并完成了 2-frame smoke + 2-frame StageA-10iter 小对照。下一次回来先看：1) `docs/STATUS.md` 最新两节 2) `docs/CHANGELOG.md` 最新 S1.3 条目 3) `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments/20260414_signal_semantics_compare/`。最重要的 handoff 判断：代码链路已通，但 S1.3 首轮阈值过严，下一步先回调 densify 阈值，再做 8-frame 短跑对照，不要直接进长跑。
最新 handoff（2026-04-14 晚）：已经完成 8-frame verify/pack、retuned conf-aware densify、以及 8-frame StageA-20iter baseline vs conf-aware 小对照。现在不要直接跳到 StageB curriculum。下一次回来先看：1) `docs/STATUS.md` 第18节 2) `docs/DESIGN.md` 第15节 3) `.../20260414_signal_semantics_mid8_compare/`。当前最该做的是 E1 小网格回调（继续调 conf-aware densify/语义组合），目标是先让 8-frame 短跑结果至少接近或不差于 baseline，再决定是否进入 plan 第4步。
当前下一阶段主计划已切换为 `docs/SIGNAL_ENHANCEMENT.md`。如果下次回来要继续，不要先推进 StageB curriculum，先按 `SIGNAL_ENHANCEMENT.md` 做 E1/E2/E3：support-aware pseudo selection -> dual-pseudo allocation -> multi-anchor verify。做完 signal enhancement，再回到 `docs/SIGNAL_SEMANTICS_AND_STABLE_REFINEMENT_PLAN_20260414.md` 继续后续 consumer-side 计划。

最新 handoff（2026-04-14 深夜）：E1 `support-aware pseudo selection` 第一轮已完成。新脚本：`scripts/select_signal_aware_pseudos.py`；结果目录：`.../20260414_signal_enhancement_e1/`；专项报告：`docs/SIGNAL_AWARE_SELECTION_E1_REPORT_20260414.md`。本轮选中的 `signal-aware-8` 为 `[23, 57, 92, 127, 162, 196, 225, 260]`，相对 midpoint8 的 `[17, 51, 86, 121, 156, 190, 225, 260]`，有 `6/8` 个 gap 改选，且前 6 个 gap 全部偏向 `2/3`。当前最重要的新判断：`midpoint` 不是这个 case 的稳定最优点，E1 方向成立；但这轮仍是 lightweight render-based verify，不是 full fused-first apples-to-apples rerun。所以下次回来不要直接进 E2 或 StageB，先用 `signal-aware-8` 补一次正式 verify/pack + 8-frame StageA short compare，再决定是否冻结 E1 结论并进入 E2。

最新 handoff（2026-04-14 更晚）：E1.5 已完成。正式 compare 目录：`.../20260414_signal_enhancement_e15_compare/`；总结报告：`docs/SIGNAL_AWARE_SELECTION_E15_COMPARE_20260414.md`。结论已从“E1 方向成立”升级为“E1 通过正式短对照”：raw signal 上 `verified_ratio` 和 `continuous confidence` 小幅提升，densify 上 baseline `dense_valid_ratio 0.02269 -> 0.02961`、conf-aware `0.01634 -> 0.01831`，StageA-20 baseline/confaware 的 `loss_total` 也都略优于 midpoint8。当前推荐动作：不要再回头争论 midpoint 是否足够；E1 可以收口，下一步进入 E2（dual-pseudo allocation）。

## 8. 云服务器环境信息（固定配置）

### SSH alias
- `Group8DDY`（大小写敏感，必须精确匹配）

### Python 环境
s3po部分：
- Conda env: `s3po-gs`
- 完整路径: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate s3po-gs`

difix部分：
- Conda env: `reggs`
- 完整路径: `/home/bzhang512/miniconda3/envs/reggs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate reggs`


### PYTHONPATH（每次执行必须显式设置）
- 必须包含: `/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`
或reggs的环境
### 项目路径
- Part3 BRPO root: `/home/bzhang512/CV_Project/part3_BRPO`
- 实验输出: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```

### 注意事项
- SSH alias 必须用 `Group8DDY`（不是 `group8ddy`）
- 远端执行必须显式设置 PYTHONPATH


最新 handoff（2026-04-14 更深夜）：E2 已完成正式对照。目录：`.../20260414_signal_enhancement_e2_compare/`；总结报告：`docs/SIGNAL_DUAL_PSEUDO_E2_COMPARE_20260414.md`。本轮已把 multi-pseudo 的 manifest/schema/pseudo_cache 支持打通：`select_signal_aware_pseudos.py` 新增 `topk-per-gap`，`prepare_stage1_difix_dataset_s3po_internal.py` 可直接消费 selection manifest，pseudo-cache schema 升级到 `pseudo-cache-internal-v1.5`。当前 case 中，E2 的 top2 实际稳定落成 `1/2 + 2/3`，最终 16 帧为 `[23, 17, 57, 51, 92, 86, 127, 121, 162, 156, 196, 190, 225, 231, 260, 266]`。结果判断：E2 相比 midpoint8 仍有轻微正向，但没有超过 E1 的 `signal-aware-8`；densify 层 `baseline 0.02621 < 0.02961`、`conf-aware 0.01737 < 0.01831`，StageA-20 baseline/confaware 的 loss 也都明显差于 E1。所以当前 default winner 仍应保持 E1 `signal-aware-8`。如果下一步进入 E3（multi-anchor verify），优先基于 E1 winner 做，不要默认沿用 E2 16-pseudo set。
