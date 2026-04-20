# hermes.md

> 用途：Part3 BRPO 压缩/重启后的第一入口。先看这份，再按这里列的顺序继续。
> 维护原则：只保留当前真实状态、当前执行顺序、关键文档入口和固定环境信息；不重复写完整 changelog。
> 更新时间：2026-04-20 16:35 (Asia/Shanghai)

## 1. 现在先怎么用这份文档

如果用户让我“先回忆一下现在做到哪了”，按这个顺序：
1. 先看本文件 `docs/agent/hermes.md`
2. 再看 `docs/STATUS.md`
3. 再看 `docs/DESIGN.md`
4. 再看 `docs/CHANGELOG.md`
5. 如果要接 B3，再看 `docs/agent/charles.md`（注意：旧 handoff 在 `docs/agent/charles.md`，不是 `docs/charles.md`）
6. 如果要接 B3，再看 `docs/REFINE_DESIGN.md`
7. 如果要接 A1，再看：
   - `pseudo_branch/brpo_v2_signal/pseudo_observation_brpo_style.py`
   - `scripts/build_brpo_v2_signal_from_internal_cache.py`
   - `scripts/run_pseudo_refinement_v2.py`
   - `scripts/run_a1_brpo_direct_compare.py`

## 2. 当前项目位置（本次更新后的真实状态）

当前最新完成、且比 `docs/agent/charles.md` 更晚的工作，是 **A1 direct BRPO v1**：
1. 已新增 direct builder：`build_brpo_direct_observation(...)`
2. 已在 signal builder 中并行产出 `pseudo_*_brpo_direct_v1`
3. `run_pseudo_refinement_v2.py` 已真实接通 `--pseudo_observation_mode brpo_direct_v1`
4. 已完成 signal smoke、consumer smoke、formal compare
5. compare 结论已经落地：
   - `oldA1 = 24.187551`
   - `newA1 = 24.149272`
   - `brpo_style_v2 = 24.167642`
   - `brpo_direct_v1 = 24.175408`
6. 所以当前结论是：
   - `brpo_direct_v1` 明显优于 current new A1（约 `+0.02614 PSNR`）
   - 也优于 `brpo_style_v2`（约 `+0.00777 PSNR`）
   - 但仍低于 `old A1`（约 `-0.01214 PSNR`）
7. 因此默认主线**仍是**：`old A1 + new T1`
8. 但 A1 的当前研究主线**已经不是** verify proxy，也不是继续磨 `brpo_style_v2`，而是：`brpo_direct_v1`

补充一个重要事实（来自最新 signal meta 聚合）：
- `brpo_direct_v1` 与 `brpo_style_v2` 的 average valid coverage 基本一样
- `brpo_direct_v1` 的 positive confidence mean 已接近 old A1
- 但 old A1 仍然赢

这意味着：
> 当前剩余 gap 更像是 target builder / exact supervision contract / fallback semantics 的问题，而不是把 confidence 再调大一点就能解决。

## 3. 当前固定判断

### 3.1 A1 现在该怎么理解
1. `verify proxy` 已完成职责，不能回头继续磨
2. `brpo_style_v1/v2` 证明了方向对，但它们都只是过渡 probe
3. `brpo_direct_v1` 是当前最强的新 A1 分支，也是最值得继续的 A1 线
4. 但 `old A1` 仍是默认 observation 主线，不能被误写成“已经被 direct BRPO 替代”

### 3.2 B3 现在该怎么理解
1. `docs/agent/charles.md` 对 B3 的 handoff 仍有效，尤其是：**现在不要推进 O2a/b**
2. B3 当前正确下一步仍是 C0-2 的一小步 candidate-law 或 opacity-law 修改
3. 但从时间顺序上说，`charles.md` 早于这次 A1 direct compare；所以它不能覆盖 A1 的最新状态

### 3.3 当前优先级判断
我现在的判断是：
1. 先把 A1 direct BRPO 的 residual gap 看清楚，再决定是否继续 A1 patch
2. B3 仍保留为并行候选线，但**不允许**直接跳 O2a/b
3. 如果用户要求“下一步做什么”，默认先做 A1 residual diagnosis；如果用户明确要切 B3，再按 `charles.md` 的 C0-2 节奏走

## 4. 当前必须按顺序完成的任务

### Task 1：A1 residual-gap diagnosis（当前默认下一步）
先不要再做新的大 patch，也不要再做 conf-only / depth-only toggle。
先做一轮 grounded diagnosis，直接比：`old A1` vs `brpo_direct_v1`。

最少要回答这三件事：
1. 两者 valid / supervision set 的精确差异在哪里？
2. 两者 target depth 的差异主要落在什么像素簇上？
3. 剩余 gap 更像 verifier backend 问题，还是 target/fallback contract 问题？

建议优先产出一个小诊断脚本或报告，读取：
- `joint_meta_v2.json`
- `brpo_direct_observation_meta_v1.json`
- 必要时逐帧对比 `joint_depth_target_v2.npy` vs `pseudo_depth_target_brpo_direct_v1.npy`
- 必要时对比 `joint_confidence_v2.npy` vs `pseudo_confidence_brpo_direct_v1.npy`

### Task 2：只做一侧 direct patch，再重跑 compare
诊断完后，如果继续 A1，只允许做一侧：
1. 要么改 direct verifier backend
2. 要么改 direct target builder / fallback contract

一次只改一侧，不要把 verifier、target、topology、B3 混在一起。
然后再做固定 `new T1 + summary_only` compare。

### Task 3：如果切回 B3，只做 C0-2
如果用户要求切回 B3：
1. 先读 `docs/agent/charles.md`
2. 不推进 O2a/b
3. 一次只改一小步 candidate-law 或 opacity mapping
4. compare 时只重跑新 opacity 臂，复用旧 control

## 5. 明确不要做的事
1. 不把 `brpo_direct_v1` 误写成完整 BRPO implementation
2. 不回去继续磨 `brpo_verify_v1`
3. 不再把 A1 剩余 gap 简化成 confidence-only 或 depth-only 单侧问题
4. 不在 observation compare 里同时改 topology 或 B3
5. 不在 B3 delayed opacity 仍 weak-negative 时直接推进 O2a/b

## 6. 一句话 handoff
现在最准确的状态不是“继续磨 style_v2”，也不是“直接去做 B3 O2”。更准确的状态是：**A1 已经推进到 `brpo_direct_v1`，它是当前最强的新 A1 分支，但仍比 old A1 低约 0.0121 PSNR；所以下一步默认应先做 `old A1` vs `brpo_direct_v1` 的 residual-gap diagnosis，定位 target builder / exact supervision contract 的差异。若用户明确切 B3，则按 `docs/agent/charles.md` 继续 C0-2，小步修改，绝不直接推进 O2a/b。**

## 7. 云服务器环境信息（固定配置）

### SSH alias
- `Group8DDY`（大小写敏感，必须精确匹配）

### Python 环境
s3po 部分：
- Conda env: `s3po-gs`
- 完整路径: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`
- 激活命令: `source ~/.bashrc && conda activate s3po-gs`

### PYTHONPATH（远端执行必须显式设置）
- 必须包含: `/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`

### 项目路径
- Part3 BRPO root: `/home/bzhang512/CV_Project/part3_BRPO`
- 实验输出默认优先: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments`

### 最近关键输出
- A1 direct signal root:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_brpo_direct_signal_full`
- A1 direct compare root:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_brpo_direct_compare_e1`
- A1 direct compare summary:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_a1_brpo_direct_compare_e1/compare_summary.json`
- B3 C0 diagnosis:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_c0_diagnosis/diagnosis_summary.json`
- B3 opacity compare summary:
  - `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260420_b3_opacity_participation_compare_e1/compare_summary.json`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO:$PYTHONPATH && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```
