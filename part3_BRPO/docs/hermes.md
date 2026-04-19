# hermes.md

> 用途：Part3 BRPO 压缩/重启后的第一入口。先看这份，再按这里列的顺序继续。
> 维护原则：只保留当前真实状态、当前执行顺序、关键文档入口和固定环境信息；不重复写完整 changelog。

## 1. 现在先怎么用这份文档

如果用户让我“先回忆一下现在做到哪了”，按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/A1_joint_confidence_stageb_compare_20260418.md`
3. 再看 `docs/P2T_selector_confirmation_precision_compare_20260418.md`
4. 再看 `docs/BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md`
5. 再看新写的五个工程落地方案：
   - `docs/A1_unified_rgbd_joint_confidence_engineering_plan.md`
   - `docs/A2_geometry_constrained_support_depth_expand_engineering_plan.md`
   - `docs/B1_spgm_manager_shell_engineering_plan.md`
   - `docs/B2_scene_aware_density_proxy_and_state_score_engineering_plan.md`
   - `docs/B3_deterministic_state_management_engineering_plan.md`
6. 再看 `docs/archived/2026-04-experiments/P2S_supportblend_farkeep_followup_compare_20260417.md`
7. 再看 `docs/archived/2026-04-experiments/P2R_spgm_score_ranking_repair_compare_20260417.md`
8. 再看 `docs/archived/2026-04-experiments/P2M_spgm_conservative_repair_compare_20260417.md`
9. 再看 `docs/STATUS.md`
10. 再看 `docs/DESIGN.md`
11. 再看 `docs/CHANGELOG.md`

## 2. 当前项目位置（本次更新后的真实状态）

当前主线结论已进一步收窄：
1. canonical StageB baseline 仍是 `RGB-only v2 + gated_rgb0192 + post40_lr03_120`
2. Re10k 上 repair A 仍是当前最稳 SPGM anchor
3. `P2-T` 已完成 confirmation / precision sweep：selector-first 仍处在 practical-parity 区，但没有在 `0.88 / 0.90-repeat / 0.92 / 0.95` 里形成稳定正胜
4. 当前真正值得保留的 selector 微窗口只剩 `far≈0.90 ~ 0.92`，身份是 reference arm，不是主线
5. A1 已完成 Re10k canonical StageB formal compare：`joint_confidence_v2 + joint_depth_v2` 明确超过 rgb-first sidecar control，但 A2 compare 已给出负信号：A1+A2 相比 A1 下降 -0.286 PSNR，说明 geometry-constrained expansion 的当前实现引入了有害的低置信度区域
6. **A2 已完成 compare，结果是负信号**：A1+A2 相比 A1 下降 -0.286 PSNR，说明 geometry-constrained expansion 引入的低置信度区域损害了整体渲染质量
7. **主线已推进到 B1/B2/B3**：A2 widening 策略失败后，不再继续 A 线 widening；当前主线是 B1 SPGM manager shell → B2 state score → B3 deterministic state management
8. 因此当前执行顺序是：B1 → B2 → B3；A2 作为 widening 方案暂缓，后续若需推进需重新设计更严格的置信度筛选

更重要的是：
- 新的 `BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md` 理论方向仍然是对的
- A1 的第一轮结果说明：当前更大的结构瓶颈确实在 observation semantics，不是先做更重的 SPGM state management
- A1 的收益来自 unified trusted support，而不是更大 coverage；后续若要扩大 coverage，应从 A1 出发做 A2，而不是回到 depth-sidecar

## 3. 当前固定判断

### 3.1 selector-first 当前应如何理解
1. selector-first 不再是“明显负结果”；它已经真实进入 practical-parity 区
2. 但 `P2-T` 已说明：它目前仍是“贴近 repair A”，不是“稳定越过 repair A”
3. 因此它现在的身份应是：
   - 一个可保留的参考候选臂
   - 不是当前主 anchor
4. 如果以后回到 selector-first，只看 `0.90 ~ 0.92` 的极窄窗口；不再继续扫更强 keep，也不再做更宽 far-keep sweep

### 3.2 新文档两大方向的当前优先级
在新文档提出的 A/B 两条线里，当前顺序固定为：
1. A1：unified RGB-D joint confidence
2. A2：geometry-constrained support/depth expand
3. B1：SPGM manager shell
4. B2：scene-aware density proxy + state score
5. B3：deterministic state management

### 3.3 为什么先 A 后 B
因为当前更大的理论差距仍然在 observation：
- live code 现在已经有 RGB builder + depth builder + joint confidence sidecar + consumer 新 mode
- A1 第一轮 compare 也已经证明：只改 unified trusted support，就足以带来明确正信号
- 因此在 A1 没站稳前，先把 SPGM 做成 manager 仍然是在管理一个尚未完全确认的新 observation 语义系统

## 4. 当前必须按顺序完成的任务

### Task 1：B1 SPGM manager shell（当前主线）
A1 第一轮 StageB formal compare 已经给出 clear positive；现在先把它作为 observation 主线确认并沉淀。
- 文档：`docs/A1_joint_confidence_stageb_compare_20260418.md`
- 工程文档：`docs/A1_unified_rgbd_joint_confidence_engineering_plan.md`
- 当前目标：确认 A1 是否直接升级为新的 canonical observation，或是否只需小修就可稳定保留

### Task 2：B2 scene-aware density proxy + state score
A2 当前 widening 策略已验证失败。若后续需要扩大 unified support，需重新设计更严格的置信度筛选；扩张 unified support，不回到 raw RGB densify 或 depth-sidecar fallback。
- 文档：`docs/A2_geometry_constrained_support_depth_expand_engineering_plan.md`

### Task 3：B3 deterministic state management
只在 B1/B2 已站稳后，才做 deterministic state action；先 lr scaling，再 very mild opacity decay。
- 文档：`docs/B1_spgm_manager_shell_engineering_plan.md`

### Task 4（暂缓）：A2 重新设计
补 scene-aware density proxy 与 `state_score`，明确分离 `ranking_score` 和 `state_score`。
- 文档：`docs/B2_scene_aware_density_proxy_and_state_score_engineering_plan.md`


状态：暂缓，主线推进 B 线。
- 文档：`docs/B3_deterministic_state_management_engineering_plan.md`

## 5. 每一步对应文档
1. `docs/A1_joint_confidence_stageb_compare_20260418.md`
2. `docs/P2T_selector_confirmation_precision_compare_20260418.md`
3. `docs/A1_unified_rgbd_joint_confidence_engineering_plan.md`
4. `docs/A2_geometry_constrained_support_depth_expand_engineering_plan.md`
5. `docs/B1_spgm_manager_shell_engineering_plan.md`
6. `docs/B2_scene_aware_density_proxy_and_state_score_engineering_plan.md`
7. `docs/B3_deterministic_state_management_engineering_plan.md`

如果需要真正动手实现，先读对应文档，不要只从总规划文档直接跳代码。

## 6. 现在明确不做的事
1. 不直接上 stochastic / Bernoulli opacity masking
2. 不把 A.5 直接并入 StageB
3. 不做 raw RGB densify
4. 不继续无边界扫 StageB schedule
5. 不继续做新的 far-keep precision sweep，除非 A1/A2 做完后有新理由回来看 `0.90 ~ 0.92`
6. 不在 repair A 尚未被 selector-first 稳定替代前，把 Re10k 的 selector-first follow-up 生硬平移到 DL3DV

## 7. 给下一个 Hermes 的一句话
现在不要把当前状态误读成“selector-first 还明显负”或“下一步还该继续磨 0.90 sweep”。更准确的状态是：`P2-T` 已经把 selector-first 的信息压缩完成——它真实处于 practical-parity 区，值得保留的只剩 `0.90 ~ 0.92` 微窗口，但它没有稳定越过 repair A。随后 A1 已在 canonical StageB formal compare 里给出 clear positive：`joint_confidence_v2 + joint_depth_v2` 明确超过 rgb-first sidecar control，但 A2 compare 已给出负信号：A1+A2 相比 A1 下降 -0.286 PSNR，说明 geometry-constrained expansion 的当前实现引入了有害的低置信度区域。因此当前主线是 A1 confirmation / landing；A2 需要重新设计（更严格的置信度筛选），B1/B2/B3 继续排在后面。

## 8. 云服务器环境信息（固定配置）

### SSH alias
- `Group8DDY`（大小写敏感，必须精确匹配）

### Python 环境
s3po 部分：
- Conda env: `s3po-gs`
- 完整路径: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate s3po-gs`

difix 部分：
- Conda env: `reggs`
- 完整路径: `/home/bzhang512/miniconda3/envs/reggs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate reggs`

### PYTHONPATH（远端执行必须显式设置）
- 必须包含: `/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`

### 项目路径
- Part3 BRPO root: `/home/bzhang512/CV_Project/part3_BRPO`
- 实验输出默认优先: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments`
- 旧输出: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```

### 注意事项
- SSH alias 必须用 `Group8DDY`（不是 `group8ddy`）
- 远端执行必须显式设置 `PYTHONPATH`
- `/` 盘很紧；新实验默认写 `/data2`，文档更新保持轻量
