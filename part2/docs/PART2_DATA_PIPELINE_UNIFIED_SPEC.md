# Part2 Data Pipeline Unified Spec

更新日期：2026-03-30

## 1. 文档目标与适用范围

本文档合并以下三类内容，作为 Part2 数据处理与运行抽帧协议的统一规范：

1. Pipeline 帧使用审计（运行时真实行为）。
2. 各数据源输入格式分析（Re10k-1 / DL3DV-2 / 405841）。
3. Adapter 设计与导出契约（目标格式、命名、目录职责、验收规则）。

目标是提供一份可直接执行、可直接追溯源码、可直接支撑实验汇报的单一文档，避免多文档之间重复、漂移或口径不一致。

## 2. 先给结论（执行层）

Part2 的核心原则有三条：

1. 数据准备阶段保留全序列，不在导出目录手动稀疏化。
2. 真实 train/test 划分由 RegGS infer/refine/metric 源码在运行时计算，notebook 里的 split preview 仅用于核对。
3. 评测口径需显式标注：当前本地默认是 sample_rate 子采样 test，不是“所有非训练帧”。

这三条是后续所有结果解释和实验对比的前置条件。

## 3. 责任边界：谁负责什么

### 3.1 Notebook 层（我们维护）

Notebook 主要负责“准备与编排”：

1. 场景适配与全序列导出（例如 01a）。
2. 生成和写入 run 配置（sample_rate、n_views、new_submap_every 等）。
3. 可视化 split preview，提前核对 train/test ids。
4. 调用 run_infer.py、run_refine.py、run_metric.py。

关键点：notebook 不改写 RegGS 内部 split 协议，只负责传参与审查。

### 3.2 RegGS 源码层（third_party/RegGS）

RegGS 在 infer/refine/metric 三阶段分别计算 split，并据此实际执行训练与评测。换言之，真正生效的帧使用策略以源码为准。

在当前 Re10KDataset 实现下，split 的帧基数来自数据集中 pose 序列长度（本质对应导出场景中的总帧数），而不是 notebook 里人为写的 preview 数组。

## 4. 运行时抽帧协议（单一真值）

infer/refine/metric 的 split 公式一致：

1. test_ids = frame_ids[int(sample_rate/2) :: sample_rate]
2. remain_ids = frame_ids \ test_ids
3. train_ids = 在 remain_ids 上等间距取 n_views 个索引

这意味着：

1. sample_rate 与 n_views 是两条独立控制链路。
2. test 先由 sample_rate 决定，train 再由 n_views 在 remain 上决定。
3. train 与 test 互斥，但 test 不是所有非 train 帧。

### 4.1 306 帧示例（DL3DV，sample_rate=30，n_views=10）

给定 frame_ids = 0..305：

1. test_ids 从 15 开始步长 30：
   15, 45, 75, 105, 135, 165, 195, 225, 255, 285
2. remain_ids 为其余帧。
3. train_ids 在 remain 上等距取 10 张，典型结果：
   0, 33, 67, 101, 136, 169, 203, 237, 271, 305

因此会看到两组都“有规律”的帧号：一组由 sample_rate 直接控制（test），另一组由 n_views 间接控制（train）。

## 5. 与论文评测口径的关系

论文常见口径是：训练帧等距采样，测试集为所有非训练帧。

本地当前默认口径是：测试集为 sample_rate 子采样帧。

结论：两者定义不完全一致，论文数字与本地数字不能直接横向对比；报告中必须显式标注评测协议。

## 6. 本地实证与代码一致性

现有输出结果已与上述协议相互印证：

1. DL3DV 8/30 的 eval_test.json 中 test 帧序列符合 15 起点、步长 30 的子采样规律。
2. Re10k 8/30 的 eval_test.json 同样符合相同规则。

此外，本地 patch 主要集中在输出兼容和 PLY 回退读取等位置，未改动 split 公式关键行；因此当前 split 行为仍可视为 RegGS 原生逻辑。

## 7. 数据源输入矩阵（场景级）

### 7.1 Re10k-1

输入现状：

1. 已有平铺 images 序列。
2. 已有 cameras.json，包含 cam_quat、cam_trans 与归一化内参字段。
3. 文件命名可排序。

适配结论：

Re10k-1 最接近目标格式，属于低风险 adapter，可作为首个稳定基线。

### 7.2 DL3DV-2

输入现状：

1. 图像在 rgb/frame_*.png。
2. intrinsics.json 与 cameras.json 已存在。
3. 姿态字段结构与 RegGS 目标格式接近。

适配结论：

主要工作是命名规范化与一致性导出（例如统一零填充数字命名），属于中低风险 adapter。

### 7.3 405841

输入现状：

1. 图像在 FRONT/rgb。
2. 内参在 calib/*.txt，姿态在 gt/*.txt（4x4 矩阵）。
3. 数据形态偏原始，非直接 JSON 结构。

适配结论：

需要解析、坐标/约定检查与格式转换，是三者中复杂度最高的 adapter，应在 Re10k-1 和 DL3DV-2 稳定后再推进。

## 8. 统一 Adapter Contract（必须满足）

### 8.1 主规则

1. 导出目录保留 full frame sequence。
2. 不在导出阶段手动稀疏化。
3. 稀疏视角实验交由 RegGS 运行参数控制。

### 8.2 派生数据规则

1. 大体积且不变的资产优先 symlink（图像、复用资产）。
2. 发生变换的资产写新文件（resize 后图片、重写后的 intrinsics/cameras、配置与报告）。

即：unchanged large assets -> symlink；transformed assets -> write new files。

### 8.3 目标导出格式

每个导出场景应满足：

- images/0000.png ...
- intrinsics.json（归一化内参）
- cameras.json（逐帧位姿）

### 8.4 必要输出能力

每个 adapter 都要能稳定提供：

1. 有序帧列表。
2. 导出分辨率对应内参。
3. 每帧 pose。
4. 稳定 scene name。

### 8.5 必须明确的解析项

每个上游数据集都必须在 adapter 中明确定义：

1. 图像来源路径与排序规则。
2. GT 内参与 GT 位姿来源。
3. 图像是可 symlink 还是必须重写。
4. 分辨率变化后 normalized intrinsics 计算方法。
5. 输出 scene 名称生成规则。

### 8.6 内参规则

intrinsics.json 采用归一化字段：

- fx/W
- fy/H
- cx/W
- cy/H

若分辨率变化，必须同步更新内参，禁止沿用旧值。

### 8.7 位姿规则

cameras.json 每帧至少包含：

- cam_quat
- cam_trans

导出约定必须与 RegGS 在 datasets.py 的读取约定一致。

### 8.8 命名规则

图像命名需可排序且无歧义，推荐零填充：

- 0000.png
- 0001.png
- 0002.png

避免 frame_1、frame_10、frame_2 这类字典序不稳定命名。

## 9. 目录职责与工作流约定

### 9.1 推荐目录职责

- dataset/<scene>/part2/：场景级 RegGS-ready 输入（规范导出物）
- output/part2/：RegGS 原始输出与日志
- results/part2/：汇总后的指标与表格
- plots/part2/：由 results 派生的可视化

### 9.2 Notebook-first 约定

Part2 预处理默认 notebook-first：

1. 在 notebook 内完成检查、导出、校验。
2. 必要时调用脚本化 helper，但 notebook 保持可追踪主流程。
3. 首个 notebook 应覆盖：场景选择、输入检查、导出、校验摘要输出。

## 10. 验收清单（运行前必须通过）

建议每次导出后按以下顺序验收：

1. 帧数一致：images 数、cameras 条目数与预期一致。
2. 名称一致：image_name 集合与图片文件集合一致。
3. 位姿有效：四元数范数接近 1，平移向量有限。
4. 内参有效：焦距为正，主点在分辨率范围内。
5. split 可解释：preview 与 run 参数一致，且 test 采样规律符合公式。

## 11. 实施优先级与里程碑

建议实现顺序：

1. Re10k-1
2. DL3DV-2
3. 405841

理由是先用结构最接近目标格式的数据建立稳定导出模板，再处理需要文本解析和姿态约定检查的复杂场景。

## 12. 附录：关键追溯锚点（最小集合）

为了避免正文过长，关键锚点只保留最小集合：

1. split 公式位置：reggs.py、run_refine.py、evaluator.py。
2. infer/refine/metric 入口：run_infer.py、run_refine.py、run_metric.py。
3. 数据集长度来源与 Re10KDataset 长度实现：datasets.py。
4. 本地协议实证：典型 run 的 eval_test.json 帧号序列。

使用本附录可快速定位“协议定义 -> 执行实现 -> 结果证据”的完整链路。
