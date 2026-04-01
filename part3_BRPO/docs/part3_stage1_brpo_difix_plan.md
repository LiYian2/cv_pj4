# Part3 Stage1 深度分析与执行规划（BRPO-inspired + Difix）

版本状态：v1（已完成项目说明、BRPO论文、Difix3D代码、Part2/Part2_S3PO结构核对）
更新时间：2026-04-01

---

## 0. 阅读与核对范围（已完成）

### 0.1 已阅读资料

1. 课程项目说明 PDF：/home/bzhang512/CV_Project/Project_4__Generative_Sparse_View_3D_Reconstruction.pdf
2. BRPO 论文 PDF：/home/bzhang512/CV_Project/part3_BRPO/docs/pseudo_view.pdf
3. 现有 Part3 分析：/home/bzhang512/CV_Project/part3_BRPO/docs/part3_confidence_pipeline_notes.md
4. Difix3D 仓库与核心源码：
	- /home/bzhang512/CV_Project/third_party/Difix3D/README.md
	- /home/bzhang512/CV_Project/third_party/Difix3D/src/inference_difix.py
	- /home/bzhang512/CV_Project/third_party/Difix3D/src/model.py
	- /home/bzhang512/CV_Project/third_party/Difix3D/src/mv_unet.py
	- /home/bzhang512/CV_Project/third_party/Difix3D/src/dataset.py
5. Part2/Part2_S3PO 规划与目录约定：
	- /home/bzhang512/CV_Project/part2/docs/PART2_DATA_PIPELINE_UNIFIED_SPEC.md
	- /home/bzhang512/CV_Project/part2_s3po/docs/2026-03-30_s3po_pipeline_plan.md
	- /home/bzhang512/CV_Project/part2/configs/reggs_re10k1_re10k-ckpt_sr50_nv9_sm2_comparison_check.yaml
	- /home/bzhang512/CV_Project/part2_s3po/configs/s3po_re10k1_sparse_full.yaml

### 0.2 关键事实锚点（用于后文推导）

1. Part3 明确要求：pseudo-view generation + hybrid optimization + confidence fusion + consistency-aware optimization（可用 optical flow/reprojection error 下调不可靠区域权重）。
	- 项目文档抽取文本锚点：Project_4 提取文本 grep 结果行 38-41。
2. 你的现有笔记主线与此一致，并且已给出“RegGS -> pseudo -> confidence -> RegGS”的最小闭环。
	- 参考：part3_confidence_pipeline_notes.md:80-95, 522-605, 768-779。
3. Difix 官方仓库直接支持 reference-conditioned 推理（difix_ref），且是单步 diffusion 方案。
	- 参考：Difix3D README.md:43-57, 120-131。
4. Difix 本地推理脚本在 ref_image 非空时自动启用 mv_unet 分支。
	- 参考：inference_difix.py:15, 31-36, 46-53。
5. Difix mv_unet 源码里有固定 num_views=2 的显式假设，天然对应“当前伪视图 + 一张参考图”的双视图结构。
	- 参考：mv_unet.py:75-77。
6. Part2 数据规范强调：保留 full sequence，稀疏划分在运行时由框架参数决定；大量不变资产推荐 symlink。
	- 参考：PART2_DATA_PIPELINE_UNIFIED_SPEC.md:19-21, 48-56, 129-138。

---

## 1. 你的 Stage1 方案可行性分析

## 1.1 总结结论

结论：可行，且是高性价比的一阶段方案。

前提：

1. 严格防止 test 泄漏（pseudo 使用帧不得进入最终评测集）。
2. pseudo supervision 必须做逐像素或至少 patch+像素级置信降权。
3. 先走不训练新网络的闭环，再谈 BRPO 全量复现组件。

## 1.2 与课程要求的对齐度

你的方案与课程 Part3 对齐度高，原因如下：

1. pseudo-view generation：你选择 BRPO 风格 + Difix，完全契合课程给出的参考方向。
2. hybrid optimization：你明确将 pseudo views 回灌 RegGS/S3PO 训练。
3. confidence/consistency：你明确要自己实现逐像素置信机制，并可以引入 flow/reprojection。
4. 结果对比：可自然构建 Sparse only vs Sparse+pseudo 的实验矩阵。

## 1.3 与 BRPO 原论文的“可复现子集”关系

BRPO 原文包含：deblur UNet + 双参考 diffusion 候选 + overlap fusion + confidence mask + scene perception gaussian management。

你的 stage1 去掉 deblur UNet 是合理的工程裁剪：

1. 论文 supplementary 显示其 UNet 是单独训练得到（非现成即插即用），复现成本高。
2. 当前目标是可验证闭环，不是论文逐模块 full reproduction。
3. 仅保留“双参考候选 + 融合 + 置信掩码 + masked优化”，仍能覆盖课程核心加分点。

## 1.4 Difix 适配性分析

### 正向结论

1. Difix_ref 可直接用于 image + ref_image 条件修复（README 已给最小调用范式）。
2. 推理接口天然支持目录批处理，便于对 pseudo 批量执行。
3. model.py 的 sample 函数在 ref_image 存在时按双视图堆叠输入，可直接对应左右参考分别跑两次。

### 工程注意点

1. mv_unet 的 num_views 固定为 2：这意味着 stage1 不应试图一次塞入多张参考图，需要“左参考一次、右参考一次”两路推理后再融合。
2. 官方 src/dataset.py 当前存在明显变量名错误（img_t/output_t/ref_t 在转换前未定义），所以不建议在 stage1 做 Difix再训练，优先使用预训练推理。
3. 默认分辨率参数是 576x1024（inference_difix.py），但代码会在输出时回 resize 到原图尺寸，仍建议你在 pipeline 内统一 pseudo 渲染分辨率，避免多次重采样。

## 1.5 RegGS 与 S3PO 作为回灌目标的可行性对比

### RegGS（建议一阶段主线）

优点：

1. 你的 part2 已有 RegGS 参数体系和输出约定，最容易复用。
2. 更符合“离线组织训练集后再跑 infer/refine/metric”的 pipeline 思维。
3. 与 confidence map 的离线构造天然匹配。

风险：

1. 需要明确 pseudo 帧如何以可读格式加入 RegGS 输入结构（建议通过 part3 专用 adapter 生成标准化 scene 目录）。

### S3PO（建议二阶段副线）

优点：

1. outdoor 表现潜力高。

风险：

1. 是单入口 SLAM 流程，不是 RegGS 式拆分管线。
2. 你现有文档已记录 S3PO upstream 有运行稳定性风险点（参数/同步逻辑等），一阶段引入会增加不确定性。
3. pseudo 回灌更适合作 mapping-only 场景，不适合直接污染 tracking。

结论：Stage1 先 RegGS-only 回灌，S3PO 放到 Stage1.5/2 验证更稳。

---

## 2. Stage1 执行方案（可直接落地）

## 2.1 目标定义

在不改动核心第三方模型代码的前提下，完成以下闭环：

1. 用 Part2 稀疏输入得到 baseline 重建。
2. 在 pseudo poses 渲染 pseudo RGB/depth。
3. 用 Difix_ref 生成左右参考候选并融合。
4. 生成逐像素 confidence map。
5. 组装增强训练集并重跑 RegGS。
6. 在 untouched final test 上比较指标。

## 2.2 一阶段建议的数据与实验优先级

建议顺序：

1. Re10k-1（室内、调试成本低，先跑通全链路）。
2. DL3DV-2（户外过渡）。
3. 405841（最复杂，放到后面）。

建议 pseudo 数量：

1. 每个 train gap 先 1 个 midpoint（最稳）。
2. 若收益明确，再尝试每 gap 2 个（quarter points）。

## 2.3 目录与路径规划（重点）

说明：以下是建议结构，当前仅规划不创建。

### A. 代码与配置（放在 part3_BRPO 内）

1. /home/bzhang512/CV_Project/part3_BRPO/configs/
2. /home/bzhang512/CV_Project/part3_BRPO/scripts/
3. /home/bzhang512/CV_Project/part3_BRPO/notebooks/
4. /home/bzhang512/CV_Project/part3_BRPO/checks/
5. /home/bzhang512/CV_Project/part3_BRPO/reports/

### B. 数据组织（放在 dataset 下，遵循 Part2 风格）

每个场景建议新增：

1. /home/bzhang512/CV_Project/dataset/<SCENE>/part3_brpo_stage1/

内部建议：

1. manifests/
	- split_manifest.json（train/pseudo/final_test ids）
	- pseudo_pose_manifest.json（pseudo id 对应 pose/intrinsics/ref ids）
2. rendered/
	- raw_rgb/
	- raw_depth/
3. difix/
	- left_fixed/
	- right_fixed/
	- fused/
4. confidence/
	- cmaps/
	- qc_tables/
5. augmented_train/
	- images/
	- cameras.json
	- intrinsics.json
	- pseudo_meta.json

规则：

1. 不变的大图像资产用 symlink。
2. 变换后的派生文件（fused/confidence/manifests）写新文件。

### C. 输出组织（与现有 output/results/plots 口径一致）

1. 原始运行输出：
	- /home/bzhang512/CV_Project/output/part3_brpo/<SCENE>/<EXP_NAME>/
2. 指标汇总：
	- /home/bzhang512/CV_Project/results/part3/final/
	- /home/bzhang512/CV_Project/results/part3/qc/
3. 可视化：
	- /home/bzhang512/CV_Project/plots/part3/main/
	- /home/bzhang512/CV_Project/plots/part3/appendix/

## 2.4 关键执行文件规划（尤其置信度验证）

说明：以下为推荐脚本清单，当前仅规划。

1. 01_build_split_manifest.py
	- 输入：Part2 scene 全序列 + 稀疏 train 规则
	- 输出：train/pseudo/final_test 划分清单
	- 核心检查：pseudo 与 final_test 互斥

2. 02_render_pseudo_from_baseline.py
	- 输入：baseline 模型输出 + pseudo pose 清单
	- 输出：raw_rgb/raw_depth
	- 核心检查：每个 pseudo id 都有可用渲染结果

3. 03_run_difix_bidirectional.py
	- 输入：raw_rgb + left_ref/right_ref
	- 输出：left_fixed/right_fixed
	- 调用策略：同一 pseudo 跑两次 Difix_ref（左参考一次，右参考一次）

4. 04_overlap_fusion.py
	- 输入：left_fixed/right_fixed + depth/pose
	- 输出：fused pseudo view + overlap 权重图
	- 公式遵循 BRPO overlap 思路（可见性、深度一致、位姿衰减）

5. 05_confidence_map_stage1.py（最关键）
	- 输入：fused、raw_depth、相机参数、左右参考
	- 输出：逐像素 confidence map C_final
	- 建议结构：
	  - C_geom：重投影可见性 + 深度一致性 + pose consistency
	  - C_patch：matcher 支持率（左/右双向）
	  - C_flow（可选）：flow forward-backward 不一致 veto
	  - C_final = smooth( clamp( C_geom * C_patch * C_flow ) )
	- 产物：
	  - 置信图 png/npy
	  - patch 统计表 csv
	  - 每帧质量评分 json（mean/min/valid_ratio/support_ratio）

6. 06_filter_and_pack_augmented_trainset.py
	- 输入：fused + C_final + gate 阈值
	- 输出：augmented_train 目录
	- 建议 gate：
	  - 全图均值置信低于阈值则剔除该 pseudo frame
	  - 局部低置信区域仅降权不删除

7. 07_run_reggs_part3_stage1.py
	- 输入：augmented_train + reggs config template
	- 输出：part3 baseline/retrain 对比结果

8. 08_eval_compare_part3.py
	- 输入：sparse-only 与 sparse+pseudo 的评测结果
	- 输出：
	  - 主表：PSNR/SSIM/LPIPS
	  - 附表：pseudo 帧数量、通过率、平均置信度

## 2.5 置信度模块的最低可用实现细节（MVP）

### 几何主干（必须有）

1. 对每个 pseudo 像素用深度反投影到 3D。
2. 投影到左右参考帧，判断是否在视野内与可见。
3. 计算深度一致得分，结合相对位姿平移衰减。
4. 得到 C_geom。

### 特征补强（建议有）

1. 用 LoFTR 或 DISK+LightGlue 做 pseudo-fused 与左右参考匹配。
2. 以 16x16 或 32x32 patch 聚合支持率。
3. 得到 C_patch 并上采样。

### 光流 veto（可选）

1. 只在明显不一致区域做 down-weight。
2. 不作为主干监督来源。

### 输出到训练的接口形式

1. 每张 pseudo 存一张 C_final（0-1 float map）。
2. 训练时用于 masked RGB/depth loss。
3. 支持 view-level score（例如 mean(C_final)）做整图再降权。

---

## 3. 可执行里程碑与实验矩阵

## 3.1 里程碑（建议）

M1：Re10k 单场景闭环跑通

1. baseline -> render -> difix -> confidence -> retrain -> compare 全链路通。
2. 先不上 flow，仅 geometry+patch。

M2：Re10k ablation

1. w/o confidence
2. geometry only
3. geometry+patch
4. geometry+patch+flow(optional)

M3：迁移到 DL3DV

1. 保持同样脚本框架，仅替换配置和 matcher默认项。

M4：迁移到 405841

1. 重点看大基线下 pseudo 通过率和几何失配风险。

## 3.2 最小实验矩阵（建议）

每个场景至少做：

1. Sparse-only baseline
2. Sparse + Difix fused（无 confidence）
3. Sparse + Difix fused + C_geom
4. Sparse + Difix fused + C_geom*C_patch

如果时间够，再补：

5. Sparse + Difix fused + C_geom*C_patch*C_flow

---

## 4. 批判性思考与自检

## 4.1 该方案最容易失败的点

1. 数据泄漏：pseudo 帧与 final test 混用导致指标虚高。
2. 几何链路断裂：某些场景深度/pose 不稳定，C_geom 大面积失效。
3. Difix 幻觉放大：无置信约束直接回灌会污染几何。
4. 置信过严：过度降权导致 pseudo 基本不起作用。
5. S3PO 早接入：单入口 SLAM 流程引入额外系统不稳定，拖慢 stage1 主目标。

## 4.2 自检问题清单（每次实验前后必须回答）

1. 本次 pseudo ids 是否全部从 final test 中排除？
2. 每张 pseudo 是否都有 left/right 参考与有效 pose/intrinsics？
3. C_final 的统计是否合理（均值分布、低置信占比）？
4. retrain 后提升是否稳定出现在 untouched final test，而不是仅训练相关帧？
5. 指标变化与可视化是否一致（避免只看单一指标）？

## 4.3 方案边界与不确定性（诚实项）

1. 目前尚未读取你未来补齐的“part2渲染图真实产物”，因此本规划先按接口契约设计。
2. BRPO 论文中的 confidence mask 具体网络细节在公开文本里有限，本规划采用可解释的工程替代实现。
3. Difix 训练链路当前源码有明显 loader 变量错误，stage1 默认不走再训练路线，仅用预训练推理。

---

## 5. 一句话结论

你的 part3 一阶段路线是正确且可落地的：

先以 RegGS 为主线，用 Difix_ref 做双参考候选修复，再以“几何主导 + patch验证 + 像素平滑”的置信图进行 masked 回灌优化；这条线最符合课程要求、工程风险最低、且最容易形成可解释的报告结果。

---

## 6. 本文关键引用锚点（便于后续复核）

1. 课程 Part3 要求：Project_4 提取文本 grep 行 38-41。
2. 现有 Part3 主线笔记：part3_confidence_pipeline_notes.md:80-95, 403-431, 522-605, 648-705, 708-739。
3. Difix_ref quickstart：third_party/Difix3D/README.md:43-57。
4. Difix 推理入口参数与 ref 分支：third_party/Difix3D/src/inference_difix.py:14-24, 31-36, 46-53。
5. model.py 中 ref_image 双视图输入：third_party/Difix3D/src/model.py:253-259。
6. mv_unet 固定双视图假设：third_party/Difix3D/src/mv_unet.py:75-77。
7. Part2 split/目录契约：part2/docs/PART2_DATA_PIPELINE_UNIFIED_SPEC.md:19-21, 48-56, 129-138, 199-205。
8. RegGS config 输入输出路径样式：part2/configs/reggs_re10k1_re10k-ckpt_sr50_nv9_sm2_comparison_check.yaml:24-27。
9. S3PO 单入口与风险点：part2_s3po/docs/2026-03-30_s3po_pipeline_plan.md:12, 65-78。

---

## 7. 增量补充（v1.1，仅新增可执行细节）

说明：本节是对现有 v1 规划的“执行层补丁”，不改动既有结论。

## 7.1 脚本输入/输出契约（建议固定）

为避免后续脚本互相猜字段，建议先冻结最小契约。

| 脚本 | 最小输入 | 最小输出 | 失败即停条件 |
|---|---|---|---|
| 01_build_split_manifest.py | scene 全帧 cameras/intrinsics + train 规则 | split_manifest.json | train/pseudo/final_test 交集非空 |
| 02_render_pseudo_from_baseline.py | baseline 场景输出 + pseudo_pose_manifest.json | rendered/raw_rgb, rendered/raw_depth | 任一 pseudo 缺 RGB 或 depth |
| 03_run_difix_bidirectional.py | raw_rgb + left/right refs | difix/left_fixed, difix/right_fixed | 左右任一路输出缺失 |
| 04_overlap_fusion.py | left/right fixed + depth + pose | difix/fused + overlap_weights | 融合后尺寸与输入不一致 |
| 05_confidence_map_stage1.py | fused + refs + depth + pose | confidence/cmaps + qc_tables | C_final 有 NaN 或全 0 |
| 06_filter_and_pack_augmented_trainset.py | fused + C_final + gate 配置 | augmented_train/ 标准输入结构 | 通过 gate 的 pseudo 数量为 0 |
| 07_run_reggs_part3_stage1.py | augmented_train + reggs 配置模板 | output/part3_brpo/... | 训练中断或无 eval 文件 |
| 08_eval_compare_part3.py | baseline/retrain 指标文件 | results/part3/final 主表 + qc | final_test 帧数不一致 |

## 7.2 split_manifest.json 建议字段（固定 schema）

建议最小字段：

1. scene_name
2. frame_count
3. train_ids
4. pseudo_ids
5. final_test_ids
6. rules（记录生成规则，如 midpoint_per_gap）
7. hash（可选，来源 cameras.json 的哈希，用于版本一致性检查）

约束：

1. train_ids, pseudo_ids, final_test_ids 两两互斥。
2. 三者并集必须等于全帧集合。
3. pseudo_ids 一旦写入 manifest，不可在同实验中重分配到 final_test。

## 7.3 confidence 模块默认公式与阈值（先给可跑默认）

### 7.3.1 逐项得分

默认建议：

$$
C_{geom}=V\cdot \exp\left(-\frac{|\Delta z|}{\tau_z}\right)\cdot \exp\left(-\frac{\|t_{rel}\|}{\tau_t}\right)
$$

$$
C_{patch}=\frac{1}{2}(s_{left}+s_{right})
$$

其中：

1. $V\in\{0,1\}$ 为可见性。
2. $s_{left}, s_{right}\in[0,1]$ 为 patch 匹配支持率。

### 7.3.2 融合与平滑

先用乘法主干，便于直接抑制坏区域：

$$
C_{raw}=C_{geom}\cdot C_{patch}\cdot C_{flow}
$$

$$
C_{final}=\text{clip}(\text{gaussian\_blur}(C_{raw}),0,1)
$$

默认：若未启用 flow，则 $C_{flow}=1$。

### 7.3.3 一阶段默认阈值（可直接开跑）

1. Re10k-1：
	- tau_z = 0.08
	- tau_t = 0.60
	- patch_size = 16
	- pseudo_keep_mean_conf >= 0.30
2. DL3DV-2 / 405841（先更宽松，防止全部被拒）：
	- tau_z = 0.12
	- tau_t = 0.90
	- patch_size = 16 或 32
	- pseudo_keep_mean_conf >= 0.25

## 7.4 防泄漏与一致性校验（建议脚本内强制）

所有运行前自动执行以下 hard checks：

1. split 泄漏检查：
	- train ∩ pseudo = 空
	- train ∩ final_test = 空
	- pseudo ∩ final_test = 空
2. 指标一致性检查：
	- baseline 与 retrain 的 final_test_ids 必须完全相同。
3. 文件一致性检查：
	- augmented_train 中 pseudo 的 image_name 必须在 pseudo_meta.json 有映射。
4. 尺寸一致性检查：
	- fused RGB、confidence map、render depth 三者分辨率一致。

任一检查失败直接退出，不允许继续训练。

## 7.5 M1 通过标准（定义完成态）

Re10k-1 单场景 M1 认为“通过”需同时满足：

1. 全链路脚本可顺序执行完成（01 到 08）。
2. 至少有 1 张 pseudo frame 通过 gate 并实际参与 retrain。
3. retrain 指标文件可读且 final_test 帧数与 baseline 一致。
4. 至少 1 项主指标方向正确：
	- PSNR 上升，或
	- LPIPS 下降，或
	- SSIM 上升。
5. 质检表存在并可追溯每张 pseudo 的 mean_conf/support_ratio。

## 7.6 文档维护规则（后续增量）

后续只追加，不回写已确认章节，建议采用：

1. v1.2：给出可直接执行的参数模板，并记录首轮真实运行参数与失败样例。
2. v1.3：记录 ablation 结果与阈值修正。
3. v1.4：记录迁移到 DL3DV/405841 的差异设置。

这样能保持“结论层”稳定，“执行层”迭代可追踪。

---

## 8. v1.2 执行参数模板（可直接套用）

本节目标：把 01-08 脚本的参数和文件契约落成“可抄模板”。

## 8.1 统一命名模板

建议统一 run key：

1. dataset_key：re10k_1 / dl3dv_2 / 405841
2. scene_key：main 或 scene_A/B/C
3. exp_key：brpo_difix_stage1
4. setting_key：m1_midpoint_geom_patch
5. run_id：YYYYMMDD_HHMMSS

拼接示例：

```text
{dataset_key}__{scene_key}__{exp_key}__{setting_key}__{run_id}
```

这样可直接映射 output/results/plots 同名目录，减少后续汇总时的 join 成本。

## 8.2 顶层配置文件建议（单场景一份）

建议文件：

```text
part3_BRPO/configs/stage1/{dataset_key}_{scene_key}_m1.yaml
```

模板：

```yaml
project:
	dataset_key: re10k_1
	scene_key: main
	exp_key: brpo_difix_stage1
	setting_key: m1_midpoint_geom_patch
	run_id: 20260401_000000

paths:
	cv_root: /home/bzhang512/CV_Project
	scene_input_root: /home/bzhang512/CV_Project/dataset/Re10k-1/part2/reggs_re10k1_fullseq_256
	part3_scene_root: /home/bzhang512/CV_Project/dataset/Re10k-1/part3_brpo_stage1/main
	output_root: /home/bzhang512/CV_Project/output/part3_brpo/re10k_1/main
	results_root: /home/bzhang512/CV_Project/results/part3

split:
	mode: midpoint_per_gap
	enforce_disjoint: true
	min_train_views: 9

render:
	source_model: reggs_baseline
	save_depth: true
	image_ext: png

difix:
	model_name: difix_ref
	steps: 1
	cfg_scale: 3.0
	seed: 42
	batch_size: 4

fusion:
	method: overlap_weighted
	depth_tau: 0.08
	pose_tau: 0.60

confidence:
	use_patch: true
	patch_size: 16
	use_flow_veto: false
	blur_kernel: 5
	clamp_min: 0.0
	clamp_max: 1.0

gate:
	pseudo_keep_mean_conf: 0.30
	min_valid_ratio: 0.15

retrain:
	target_model: reggs
	freeze_pseudo_pose: true
	pseudo_rgb_weight: 1.0
	pseudo_depth_weight: 0.5

eval:
	strict_final_test_only: true
	metrics: [psnr, ssim, lpips]

qc:
	fail_on_nan_confidence: true
	fail_on_split_leakage: true
	fail_on_missing_artifacts: true
```

## 8.3 01-08 脚本参数模板（按顺序）

### 8.3.1 01_build_split_manifest.py

输入参数建议：

1. --config
2. --scene-input-root
3. --out-manifest

输出文件模板：

```json
{
	"scene_name": "re10k_1_main",
	"frame_count": 196,
	"train_ids": [0, 24, 49, 73, 98, 122, 147, 171, 195],
	"pseudo_ids": [12, 36, 61, 85, 110, 134, 159, 183],
	"final_test_ids": [1, 2, 3],
	"rules": {
		"mode": "midpoint_per_gap",
		"notes": "one midpoint per adjacent train pair"
	}
}
```

### 8.3.2 02_render_pseudo_from_baseline.py

输入参数建议：

1. --config
2. --split-manifest
3. --pseudo-pose-manifest
4. --baseline-output-root

输出目录：

```text
rendered/raw_rgb/{pseudo_id}.png
rendered/raw_depth/{pseudo_id}.npy
rendered/render_qc.json
```

### 8.3.3 03_run_difix_bidirectional.py

输入参数建议：

1. --config
2. --pseudo-pose-manifest
3. --raw-rgb-dir
4. --left-ref-dir
5. --right-ref-dir

输出目录：

```text
difix/left_fixed/{pseudo_id}.png
difix/right_fixed/{pseudo_id}.png
difix/difix_qc.json
```

### 8.3.4 04_overlap_fusion.py

输入参数建议：

1. --config
2. --left-fixed-dir
3. --right-fixed-dir
4. --raw-depth-dir
5. --pseudo-pose-manifest

输出目录：

```text
difix/fused/{pseudo_id}.png
difix/overlap_weights/{pseudo_id}.npy
difix/fusion_qc.csv
```

### 8.3.5 05_confidence_map_stage1.py

输入参数建议：

1. --config
2. --fused-dir
3. --raw-depth-dir
4. --pseudo-pose-manifest
5. --matcher-backend

输出目录：

```text
confidence/cmaps/{pseudo_id}.npy
confidence/cmaps_png/{pseudo_id}.png
confidence/qc_tables/per_frame_confidence.csv
confidence/qc_tables/patch_support.csv
```

### 8.3.6 06_filter_and_pack_augmented_trainset.py

输入参数建议：

1. --config
2. --split-manifest
3. --fused-dir
4. --confidence-dir

输出目录：

```text
augmented_train/images/
augmented_train/cameras.json
augmented_train/intrinsics.json
augmented_train/pseudo_meta.json
augmented_train/filter_log.csv
```

### 8.3.7 07_run_reggs_part3_stage1.py

输入参数建议：

1. --config
2. --augmented-train-root
3. --reggs-config-template
4. --out-run-root

输出目录：

```text
output/part3_brpo/{dataset_key}/{scene_key}/{run_key}/
```

最少要求产物：

1. config 实际快照
2. eval_test.json
3. 训练日志摘要

### 8.3.8 08_eval_compare_part3.py

输入参数建议：

1. --baseline-run-root
2. --retrain-run-root
3. --split-manifest
4. --out-results-root

输出目录：

```text
results/part3/final/part3_stage1_main_table.csv
results/part3/final/part3_stage1_per_scene.json
results/part3/qc/part3_stage1_eval_qc.csv
```

## 8.4 首轮推荐默认值（M1-Re10k）

用于第一轮“先跑通”默认：

1. split: midpoint_per_gap
2. difix: steps=1, cfg_scale=3.0, seed=42
3. confidence: patch_size=16, use_flow_veto=false
4. gate: pseudo_keep_mean_conf >= 0.30
5. retrain: freeze_pseudo_pose=true

成功后再做两项最小 ablation：

1. 去掉 C_patch（只保留 C_geom）
2. 开启 flow veto（观察是否过度降权）

## 8.5 失败样例记录模板（为 v1.3 做准备）

建议首轮开始就记录以下表头：

```csv
dataset_key,scene_key,run_key,pseudo_id,fail_stage,fail_reason,mean_conf,valid_ratio,matcher_support,action
```

其中 action 仅允许三类：

1. drop_frame
2. relax_threshold
3. rerun_with_fix

这样 v1.3 可直接基于失败统计做阈值收敛。
