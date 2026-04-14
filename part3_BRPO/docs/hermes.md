# hermes.md

> 用途：给后续 Hermes 自己看的快速回忆文档。对话断了、上下文丢了、或者用户直接让我“先回忆一下”时，先看这份，再按这里给出的文档路径继续展开。
> 维护原则：轻量覆盖更新，不做第二份 changelog，只保留当前最该知道的状态、文档入口和下一步重点。

## 1. 现在先怎么用这份文档

如果用户让我回忆最近做了什么，默认按这个顺序：
1. 先看本文件 `docs/hermes.md`
2. 再看 `docs/STATUS.md`
3. 再看 `docs/DESIGN.md`
4. 再看 `docs/CHANGELOG.md`
5. 如果当前问题是“fusion / mask / depth / refine 下一步怎么改”，优先看这两份新方案：
   - `docs/BRPO_fusion_rgb_mask_depth_v2_engineering_plan_20260415.md`
   - `docs/BRPO_local_gaussian_gating_engineering_plan_20260415.md`

如果用户问“你记不记得上次对 BRPO 差异的判断”，不要先翻聊天记录，先看：
1. `docs/BRPO_fusion_mask_spgm_subset_refine.md`
2. 再看这次落地方案文档

## 2. 当前项目位置（最新人工摘要）

当前主线已经从“继续在旧 mask/depth 链路上调参”切到“两条明确的新改造线”：
1. BRPO-style fusion + RGB mask / depth supervision v2
2. local Gaussian gating / 子集 refine

现在的关键判断不是“handeoff bug 有没有修掉”。那个问题已经修掉了，但修掉之后 pipeline 仍未转正。当前更深层的主矛盾已经收敛为：
- `signal weak`
- `supervision scope / optimization scope mismatch`

更具体地说：
1. 当前有效几何监督仍偏稀、偏弱；
2. 当前 fusion 和 mask 的主语义仍不够接近 BRPO；
3. 当前 A.5 / StageB 仍让弱局部监督去推动过大的全局 Gaussian 更新范围。

## 3. 当前最重要的结论

### 3.1 关于 signal 侧

当前 `mask / depth / confidence` pipeline 不是完全错误，但它作为主增益来源已经很接近见顶。
E1 `signal-aware-8` 是当前 default winner；E2 已说明“增加 pseudo 数量”不能替代“提高 winner 质量”。

因此现在真正值得继续推进的，不是再在旧 `seed_support -> train_mask -> target_depth_for_refine/v2` 路径上叠更多小补丁，而是：
- 把 fusion 改回 target↔reference overlap confidence 主语义；
- 把 RGB mask 和 depth supervision 彻底解耦；
- 建一条与旧链路隔离的 v2 signal path。

### 3.2 关于 map-side refine

当前最值得优先做的 map-side 改动不是 full SPGM，而是 local Gaussian gating / 子集 refine。
原因很直接：当前真正的结构问题是“弱局部监督在推动全局 Gaussian `xyz/opacity`”。

所以后续默认判断应是：
- 先做 local gating，让 pseudo branch 只更新可见且过 gate 的 Gaussian 子集；
- 再看 replay 是否止跌；
- 再决定要不要继续往更重的 SPGM / management 方向走。

## 4. 当前必看的专项文档

### 第一优先级：这次新写的两份工程落地方案
1. `docs/BRPO_fusion_rgb_mask_depth_v2_engineering_plan_20260415.md`
   - 用途：指导如何重写 fusion，并建立与旧 mask/depth 路径隔离的 v2 signal pipeline。
   - 核心口径：`fusion 用 depth/geometry；mask 用 fused RGB correspondence；depth supervision 单独生成。`
2. `docs/BRPO_local_gaussian_gating_engineering_plan_20260415.md`
   - 用途：指导如何在 `run_pseudo_refinement_v2.py` 中落地 local Gaussian gating / subset refine。
   - 核心口径：`pseudo branch 只更新 local visible subset，real branch 保留全局纠偏。`

### 第二优先级：理论判断与现状回顾
1. `docs/BRPO_fusion_mask_spgm_subset_refine.md`
   - 用途：看为什么当前应该优先做 fusion/mask 语义重排 + local gating，而不是直接上 full SPGM。
2. `docs/STATUS.md`
   - 用途：看当前状态、最近阶段推进结果。
3. `docs/DESIGN.md`
   - 用途：看设计判断为何从“signal semantics”进一步收敛到“新版 fusion/mask + local gating”。
4. `docs/CHANGELOG.md`
   - 用途：看具体实现与实验过程。
5. `docs/SIGNAL_ENHANCEMENT.md`
   - 用途：看 E1/E2 为什么收敛到 `signal-aware-8` 是当前 winner。

## 5. 当前回答用户时的推荐口径

如果用户问“现在最大的问题是什么”，优先回答：
- 不是 handoff bug 了；
- 当前主问题是 `signal weak + optimization scope too global`；
- 其中 signal 侧最该改的是 fusion/mask/depth 的语义重排；
- map-side 最该改的是 local Gaussian gating，而不是继续全局 refine。

如果用户问“下一步该做什么”，优先回答：
1. 写清楚并执行 BRPO-style fusion + RGB mask / depth supervision v2 的工程方案
2. 写清楚并执行 local Gaussian gating / 子集 refine 的工程方案
3. 用 E1 `signal-aware-8` 做短跑验证，不要默认退回 midpoint8
4. 没证明新链路有效之前，不要继续在旧链路里高强度加复杂度

## 6. 下次回来先检查什么

1. 这两份新方案文档是否已经开始落地到代码：
   - `BRPO_fusion_rgb_mask_depth_v2_engineering_plan_20260415.md`
   - `BRPO_local_gaussian_gating_engineering_plan_20260415.md`
2. `run_pseudo_refinement_v2.py` 是否已经新增：
   - `signal_pipeline = brpo_v2`
   - `pseudo_local_gating_*` CLI
3. 是否已经新建隔离目录：
   - `pseudo_branch/brpo_v2_signal/`
   - `pseudo_branch/local_gating/`
4. 是否已经在新的 pseudo cache / signal outputs 中把新旧 artifact 名字彻底隔开
5. 是否已经用 E1 winner 做了至少一轮 2-frame 或 8-frame smoke / short compare

## 7. 给下一个 Hermes 的一句话

现在不要再把注意力主要放在旧 `train_mask / target_depth_for_refine/v2` 路径里调阈值。当前最高优先级已经切到两份新工程方案：
- `docs/BRPO_fusion_rgb_mask_depth_v2_engineering_plan_20260415.md`
- `docs/BRPO_local_gaussian_gating_engineering_plan_20260415.md`

前者负责把 signal 语义重排到更接近 BRPO；后者负责把 pseudo supervision 对 Gaussian 的作用范围收回来。默认从这两份文档开始恢复上下文，而不是再从旧阶段计划里找下一步。

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
- 实验输出: `/home/bzhang512/my_storage_500G/CV_Project/output/part3_BRPO/experiments`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```

### 注意事项
- SSH alias 必须用 `Group8DDY`（不是 `group8ddy`）
- 远端执行必须显式设置 `PYTHONPATH`
