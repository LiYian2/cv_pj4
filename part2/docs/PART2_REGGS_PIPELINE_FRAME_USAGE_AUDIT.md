# Part2 RegGS Pipeline 帧数使用审计

更新日期：2026-03-29

## 1. 结论先行

1. 本地当前协议里，测试帧确实是按 sample_rate 子采样，不是所有非训练帧。
2. 这个测试子采样逻辑是 RegGS 仓库原生逻辑，不是我们在 Part2 notebook 新写的评测规则。
3. [part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb](part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb#L12) 只做全序列导出，不做 train/test 划分。
4. 我们在 run notebook 里做了 split preview，但这是可视化核对，真正执行仍以 RegGS 源码为准。

## 2. 全流程帧数使用路径

### 2.1 数据准备阶段（我们自己写）

在 [part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb](part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb#L7) 中，目标明确是保留 full sequence。

关键证据：

1. 目标说明写明 preserve full sequence：[part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb](part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb#L12)
2. 导出时遍历 source frame_paths 全量写入 images：
	[part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb](part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb#L344)
3. 导出后断言图片数、相机条目数、源帧数相等：
	[part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb](part2/notebooks/01a_prepare_reggs_scene_re10k-1.ipynb#L457)

结论：01a 只做格式适配与全量导出，不做稀疏采样划分。

### 2.2 运行编排阶段（我们自己写）

在 run notebook 中，我们做了三类事情：

1. 设定 sample_rate、n_views、new_submap_every，写入 yaml 配置。
2. 复现一份 split preview，提前看到 train/test ids。
3. 调用 run_infer.py、run_refine.py、run_metric.py。

示例证据：

1. 03_run_full 写入 FORMAL_SAMPLE_RATE 和 FORMAL_N_VIEWS：
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L114)
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L115)
2. 03_run_full split preview 公式：
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L297)
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L299)
3. 03_run_full 调用 infer/refine/metric：
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L1574)
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L6900)
	[part2/notebooks/03_run_full.ipynb](part2/notebooks/03_run_full.ipynb#L11576)
4. 03_run_405841_full 也同样设置参数与 split preview：
	[part2/notebooks/03_run_405841_full.ipynb](part2/notebooks/03_run_405841_full.ipynb#L106)
	[part2/notebooks/03_run_405841_full.ipynb](part2/notebooks/03_run_405841_full.ipynb#L107)
	[part2/notebooks/03_run_405841_full.ipynb](part2/notebooks/03_run_405841_full.ipynb#L310)
	[part2/notebooks/03_run_405841_full.ipynb](part2/notebooks/03_run_405841_full.ipynb#L312)

结论：run notebook 负责传参与核对，不负责改写 RegGS 内部采样协议。

### 2.3 RegGS 原生执行阶段（仓库自带逻辑）

在进入 infer/refine/metric 之前，n_frames 的来源是 dataset 的长度。

1. BaseDataset 里有 frame_limit 字段与默认长度逻辑：
	[third_party/RegGS/src/entities/datasets.py](third_party/RegGS/src/entities/datasets.py#L19)
	[third_party/RegGS/src/entities/datasets.py](third_party/RegGS/src/entities/datasets.py#L49)
2. 但 Re10KDataset 覆盖了 __len__，直接返回 poses.shape[0]：
	[third_party/RegGS/src/entities/datasets.py](third_party/RegGS/src/entities/datasets.py#L119)
3. infer/refine/metric 的 split 都使用 n_frames = len(dataset)：
	[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L43)
	[third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L71)
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L69)

这意味着在当前 re10k 数据类实现下，frame_limit 对 split 的影响并不直接生效，split 基数本质上来自导出场景中 cameras.json 对应的总帧数。

#### 2.3.1 infer

入口：
[third_party/RegGS/run_infer.py](third_party/RegGS/run_infer.py#L21)
[third_party/RegGS/run_infer.py](third_party/RegGS/run_infer.py#L22)

实际 split 公式在 RegGS 类初始化：
[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L45)
[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L48)

帧如何被 infer 使用：

1. infer 只在 train_frame_ids 上做相邻配对，配对数约等于 n_views - 1：
	[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L159)
2. 子图切换由 new_submap_every 控制：
	[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L64)
	[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L66)
	[third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L187)

#### 2.3.2 refine

refine 里会再次按同样公式生成 train/test ids：
[third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L73)
[third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L76)

refine 实际训练只采样 train_frame_ids（随机抽取）：
[third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L171)
[third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L172)

#### 2.3.3 metric

metric 入口：
[third_party/RegGS/run_metric.py](third_party/RegGS/run_metric.py#L22)
[third_party/RegGS/run_metric.py](third_party/RegGS/run_metric.py#L24)
[third_party/RegGS/run_metric.py](third_party/RegGS/run_metric.py#L25)

Evaluator 内部再次按同样公式生成 train/test ids：
[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L71)
[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L74)

metric 对 train 与 test 的使用：

1. eval_train_render 遍历 train_frame_ids：
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L107)
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L115)
2. eval_test_render 遍历 test_frame_ids，并对每个 test 帧做 test-time pose 优化：
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L295)
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L309)
	[third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L311)

## 3. 公式与数量关系

RegGS 原生 split 公式：

1. test_ids = frame_ids[int(sample_rate/2)::sample_rate]
2. remain_ids = frame_ids 去掉 test_ids
3. train_ids = 从 remain_ids 上等间隔取 n_views 个索引

这套公式在 infer/refine/metric 三处重复出现，来源一致：

1. [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L45)
2. [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L73)
3. [third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L71)

## 4. 本地运行结果对协议的实证

1. DL3DV 8/30 的 eval_test.json 含 10 个 test 帧，frame_id 为 0015 到 0285（步长 30）：
	[output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json#L7)
	[output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json](output/part2/dl3dv_2/reggs_dl3dv2_sr30_nv8/eval_test.json#L61)
2. Re10k 8/30 的 eval_test.json 含 9 个 test 帧，frame_id 为 0015 到 0255：
	[output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json](output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json#L7)
	[output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json](output/part2/re10k/reggs_re10k1_sr30_nv8/eval_test.json#L55)

这与 RegGS 子采样 test 的实现一致。

## 5. 与论文口径的关系

论文文本写的是：8/16/32-view 场景里训练帧等距采样，测试集包含所有非训练帧：
[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L640)
[part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt](part2/docs/Cheng_RegGS_ICCV2025_paper_extracted.txt#L642)

因此，当前本地 RegGS 默认评测口径与论文口径不完全一致，论文数字不能直接横向对比。

## 6. 我们写的逻辑 vs RegGS 原生逻辑

我们自己写的（Part2 notebook 层）：

1. 数据适配与全序列导出（01a）。
2. 运行参数设定与 yaml 生成（02/03）。
3. split preview（只是复现公式用于核对）。
4. 命令编排调用 infer/refine/metric。

RegGS 原生逻辑（third_party/RegGS）：

1. infer/refine/metric 三阶段的真实 split 计算与执行。
2. infer 的 train 邻接配对与子图切换。
3. refine 只基于 train_frame_ids 的全局优化。
4. metric 对 train 与 test 的渲染评估，以及 test-time pose 优化。

## 7. 额外核对说明

已核对当前本地对 RegGS 的改动 diff，改动点集中在输出兼容与 ply 读取回退，并未改动 split 公式所在行。

相关改动位置：

1. [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L205)
2. [third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L80)
3. [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L231)

split 关键行保持不变：

1. [third_party/RegGS/src/entities/reggs.py](third_party/RegGS/src/entities/reggs.py#L45)
2. [third_party/RegGS/run_refine.py](third_party/RegGS/run_refine.py#L73)
3. [third_party/RegGS/src/evaluation/evaluator.py](third_party/RegGS/src/evaluation/evaluator.py#L71)
