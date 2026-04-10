# Part3 Stage1：方案 1 与方案 2 的原理规划与落地规划

> 目标：围绕当前 `external pseudo branch + standalone refine` 主线，优先解决“位姿更准但图像质量变差”的问题。本文只规划两条路线：
>
> 1. **方案 1：把 left/right 修复结果做 fused pseudo target**
> 2. **方案 2：控制 pseudo 视角，不先重做数据准备，先改训练采样调度**
>
> 文档重点分两部分：
>
> - **原理规划**：为什么这样设计，数学上如何成立，和 BRPO 以及当前仓库现状如何对齐
> - **落地规划**：具体要改哪些文件、补哪些字段、训练阶段怎么走
>
> 本文默认基于当前仓库结构：
>
> - `part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py`
> - `part3_BRPO/pseudo_branch/build_pseudo_cache.py`
> - `part3_BRPO/pseudo_branch/epipolar_depth.py`
> - `part3_BRPO/pseudo_branch/diag_writer.py`
> - `part3_BRPO/scripts/run_pseudo_refinement.py`
>
> 当前已知事实：
>
> - pseudo 不参与 tracking，只参与 mapping / refine
> - 当前 refine 的 pseudo loss 仍是 **confidence-weighted RGB L1**
> - `target_depth` 已算出来，但尚未进入显式几何 loss
> - 当前最优 E 配置说明：pseudo 不改几何、real 驱动 densify、降低 `lambda_pseudo`、扩大 pseudo coverage 是有效方向
> - 当前主要问题不是位姿，而是 pseudo supervision 将不稳定的外观细节写进高斯外观参数

---

## 1. 当前问题的抽象

当前每个 pseudo sample 已经有两张修复结果：

- `target_rgb_left.png`
- `target_rgb_right.png`

它们分别表示：

- 以左参考帧为条件修复得到的 pseudo 外观
- 以右参考帧为条件修复得到的 pseudo 外观

当前 refine 只会二选一使用 `left` 或 `right`，这意味着：

1. 你已经付出了双向修复的代价；
2. 但训练时只消费了其中单边信息；
3. 单边 diffusion 的 hallucination 一旦落在高置信区域，就可能被直接写回高斯外观。

因此，当前最自然的问题不是“如何让 pseudo 更强”，而是：

**如何把两边修复结果变成一个更保守、更稳定、更几何一致的 fused supervision。**

与此同时，当前 pseudo sample 在训练中是“全量加载 + 均匀采样”的。这样会把：

- 近视角 / 简单样本
- 远视角 / 难样本
- 高置信样本
- 低置信样本

混在一起训练。对于扩散生成的 pseudo，这通常不稳。更合理的做法是把 pseudo 当成一个 **curriculum source**：先学容易的，再逐步放开困难的。

---

# 2. 方案 1：left/right 修复结果做 fused pseudo target

## 2.1 原理目标

方案 1 的目标不是做一个“更漂亮”的均值图，而是构造一个 **几何一致性更强、扩散幻觉更少、可直接进入 refine 的 fused target**。输出应该至少包括：

- `target_rgb_fused.png`
- `confidence_mask_fused.npy`
- `confidence_mask_fused.png`
- 若干 `diag/` 可视化

方案 1 的核心思想应当与 BRPO 对齐：

- diffusion 修复图 **视觉上可能清晰，但几何上可能不可靠**；
- 因此不能把某个 branch 的修复图直接当真值；
- 需要通过 **左右一致性 + 几何置信 + 视角权重**，只保留可信部分。

这和 BRPO 里的三件事是同源的：

1. bidirectional restoration
2. overlap / confidence fusion
3. confidence-guided optimization

你的 hand-crafted 版本不需要复制 BRPO 的全部网络结构，但数学上应保持同样的逻辑：

> **多参考帧条件下，如果两个修复结果在某区域相互一致，并且各自都与几何先验相容，那么该区域的 fused supervision 可信；反之应降权，而不是强行平均。**

---

## 2.2 数学规划：从 branch target 到 fused target

下面给出一个适合你当前仓库的、相对严谨且工程可落地的 hand-crafted 版本。

### 2.2.1 记号

对一个 pseudo sample，定义：

- 原始 render 图：`I_r`
- 左修复图：`I_L`
- 右修复图：`I_R`
- 左侧 EDP 深度与置信度：`D_L, C_L`
- 右侧 EDP 深度与置信度：`D_R, C_R`
- 当前 render depth：`D_r`

其中：

- `I_L, I_R` 已经由 Difix 产出
- `D_L, C_L, D_R, C_R` 可由当前 `compute_edp_depth()` 双边调用得到

### 2.2.2 branch score：每一边先各自打分

对左侧 branch，先定义一个逐像素 branch score：

\[
S_L(p) = C_L(p) \cdot G_L(p) \cdot V_L
\]

对右侧同理：

\[
S_R(p) = C_R(p) \cdot G_R(p) \cdot V_R
\]

各项含义：

- `C_L(p), C_R(p)`：来自 EDP 的逐像素几何置信度
- `G_L(p), G_R(p)`：render depth 与该 branch 深度的一致性 gate
- `V_L, V_R`：view-level scalar，用来编码“该参考帧整体上离 pseudo 有多远 / overlap 多大 / branch 整体有多可靠”

这里 `G` 可以定义为相对深度差的指数衰减：

\[
G_L(p) = \exp\left(- \frac{|D_r(p) - D_L(p)|}{\tau_d (D_r(p)+\epsilon)}\right)
\]

如果某像素没有有效 `D_L`，则 `G_L(p)=0`。

`V_L, V_R` 第一版不需要太复杂，可以来自 sample-level stats，例如：

- `mean_confidence_left`
- `target_valid_left / total_pixels`
- pseudo 和对应参考帧的相对基线长度、索引距离或 placement 难度

所以第一版可以写成：

\[
V_L = \alpha_1 \cdot \overline{C_L} + \alpha_2 \cdot r_L
\]

其中：

- `\overline{C_L}` 是左 branch 的平均有效置信度
- `r_L` 是左 branch 有效 support 比例

右侧同理。

### 2.2.3 agreement term：左右修复图的一致性

仅仅让 `S_L, S_R` 大还不够，因为两边可能各自都“自信”，但在某区域说的是两种不同的内容。这正是 diffusion 容易出错的地方。

因此需要显式加入左右一致性项 `A(p)`：

\[
A_{rgb}(p) = \exp\left(- \frac{\|I_L(p)-I_R(p)\|_1}{\tau_{rgb}} \right)
\]

如果你愿意再保守一点，可以再加一个深度一致性项：

\[
A_{depth}(p) = \exp\left(- \frac{|D_L(p)-D_R(p)|}{\tau_{ld}(D_r(p)+\epsilon)} \right)
\]

最终 agreement 可以取：

\[
A(p) = A_{rgb}(p) \cdot A_{depth}(p)
\]

如果你担心 EDP 太 sparse，第一版可以只用 `A_rgb`，把 `A_depth` 作为可选项。

### 2.2.4 权重归一化

定义左右最终融合权重：

\[
W_L(p) = A(p) \cdot S_L(p), \qquad W_R(p) = A(p) \cdot S_R(p)
\]

再做归一化：

\[
\tilde{W}_L(p) = \frac{W_L(p)}{W_L(p)+W_R(p)+\epsilon}, \qquad
\tilde{W}_R(p) = \frac{W_R(p)}{W_L(p)+W_R(p)+\epsilon}
\]

### 2.2.5 fused RGB：建议使用 residual fusion，而不是直接对 RGB 本体做平均

直接融合 `I_L` 和 `I_R` 会有一个问题：如果两边都带一点 bias，平均后常常得到“糊得很稳定”的图。

对你现在这种任务，更合理的是把两边都看成对原始 render `I_r` 的修正量：

\[
R_L(p) = I_L(p) - I_r(p), \qquad R_R(p)=I_R(p)-I_r(p)
\]

再做 fused residual：

\[
R_F(p) = \tilde{W}_L(p) R_L(p) + \tilde{W}_R(p) R_R(p)
\]

最终 fused image：

\[
I_F(p) = I_r(p) + R_F(p)
\]

这样做的优点是：

1. 如果左右都只在局部修某些区域，未修区域会自然保留原 render；
2. 如果左右都偏离原图但方向相反，不会直接把颜色平均到发灰；
3. 更符合“diffusion 是在修补 render artifact，而不是生成整张新图”的语义。

### 2.2.6 fused confidence：不能只取 max

当前 `build_pseudo_cache.py` 里深度融合本质上是 winner-take-all，再对 confidence 取 `max`。这个做法对 RGB supervision 不够严格。

更合理的 fused confidence 应为：

\[
C_F(p) = \phi\big(W_L(p)+W_R(p)\big) \cdot A(p)
\]

其中 `\phi` 是一个压缩函数，可以取：

\[
\phi(x)=\min(1, x / \tau_c)
\]

这样 `C_F` 同时表达两件事：

- 至少有一边 branch 在此处有较高几何可信度
- 左右两边在此处外观上基本一致

如果某区域只有一边非常自信但另一边严重不同，`A(p)` 会把它压低。这恰好是你想要的保守策略。

### 2.2.7 reject / fallback 机制

如果某像素：

- `W_L + W_R` 太小
- 或 `A(p)` 太小

则该像素不应被当作强 pseudo supervision。建议两种处理方式二选一：

1. **硬拒绝**：令 `C_F(p)=0`
2. **软回退**：令 `I_F(p)=I_r(p)`，但 `C_F(p)` 很低

对当前 Stage1，我更建议：

- 图像上用 `I_r` 回退，保持视觉连续性
- mask 上低置信或 0，避免这部分真正参与 loss

这样 training target 更平滑，但 supervision 仍然保守。

---

## 2.3 和 BRPO 的关系：为什么这个 hand-crafted 版在数学上是合理的

BRPO 的原始思想不是“左右两图平均”，而是：

- 先用 bidirectional restoration 产生候选 pseudo frame
- 再用 overlap / reprojection / feature consistency 得到 confidence
- 最后用 confidence 去指导 Gaussian optimization

你这里的 hand-crafted 版，本质是在用一个更轻量、但同逻辑的版本实现它：

1. **bidirectional restoration**
   - `target_rgb_left`, `target_rgb_right`
2. **overlap / geometry prior**
   - `C_L, C_R, G_L, G_R, V_L, V_R`
3. **cross-view agreement**
   - `A_rgb`, 可选 `A_depth`
4. **confidence-guided optimization**
   - `I_F, C_F` 进入 refine

因此，这不是 ad-hoc 平均，而是一个明确的 **multi-branch confidence fusion** 问题。

如果你想在论文或报告里表述得更正式，可以写成：

> For each pseudo view, we treat the left-conditioned and right-conditioned restorations as two noisy estimators of an unobserved target appearance. We construct a fused pseudo target by confidence-weighted residual fusion, where branch weights are determined jointly by epipolar-depth reliability, render-depth compatibility, and cross-branch appearance agreement.

---

## 2.4 方案 1 的落地规划

### 2.4.1 新增文件

#### A. `part3_BRPO/pseudo_branch/pseudo_fusion.py`

职责：

- 输入：
  - `render_rgb.png`
  - `render_depth.npy`
  - `target_rgb_left.png`
  - `target_rgb_right.png`
  - `depth_left.npy / conf_left.npy`
  - `depth_right.npy / conf_right.npy`
  - sample-level metadata
- 输出：
  - `target_rgb_fused.png`
  - `confidence_mask_fused.npy`
  - `confidence_mask_fused.png`
  - `fusion_meta.json`
  - 可选若干诊断图

建议核心函数：

- `compute_branch_score(...)`
- `compute_agreement_map(...)`
- `fuse_residual_targets(...)`
- `build_fused_confidence(...)`
- `run_fusion_for_sample(...)`

#### B. 可选新增 `part3_BRPO/pseudo_branch/fusion_utils.py`

如果你想把公式和 IO 分开，可以再拆一个 util 文件；如果不想拆太多，全部写进 `pseudo_fusion.py` 也可以。

---

### 2.4.2 修改文件

#### A. `part3_BRPO/pseudo_branch/epipolar_depth.py`

当前问题：

- `compute_edp_depth()` 只返回单边 `depth_epi, confidence_map, stats, diag_extra`
- 双边融合时只返回 fused 结果，没有把左右 branch 的 side outputs 保存下来

建议修改：

1. 保持 `compute_edp_depth()` 接口基本不变
2. 新增一个更适合 cache 构建的接口，例如：
   - `compute_edp_depth_pair(...)`

返回结构中显式包含：

- `depth_left`
- `conf_left`
- `stats_left`
- `depth_right`
- `conf_right`
- `stats_right`
- `target_depth_fused`
- `confidence_fused_depth`

这样后续 `build_pseudo_cache.py` 就能把左右 side outputs 也写盘。

#### B. `part3_BRPO/pseudo_branch/build_pseudo_cache.py`

这是方案 1 最核心的改动点。

当前它已经做了：

- 单边 EDP
- 双边 depth/confidence 融合
- 写 `target_depth.npy` 和 `confidence_mask.npy`

你需要把它扩展成三段：

1. **side branch 构建**
   - 跑 left EDP
   - 跑 right EDP
   - 写：
     - `depth_left.npy`
     - `conf_left.npy`
     - `depth_right.npy`
     - `conf_right.npy`

2. **depth-level fusion**
   - 保留现在的 fused `target_depth.npy`

3. **RGB-level fusion**
   - 调用 `pseudo_fusion.py`
   - 写：
     - `target_rgb_fused.png`
     - `confidence_mask_fused.npy`
     - `confidence_mask_fused.png`
     - `fusion_meta.json`

同时，`stats` 要补 richer sample-level summary，例如：

- `target_valid_left`
- `target_valid_right`
- `target_valid_fused`
- `mean_conf_left`
- `mean_conf_right`
- `mean_conf_fused`
- `mean_rgb_agreement`
- `placement`
- `difficulty_score`

这些字段后面也会被方案 2 使用。

#### C. `part3_BRPO/pseudo_branch/diag_writer.py`

需要新增几类诊断图：

- `rgb_disagreement.png`
- `weight_left.png`
- `weight_right.png`
- `confidence_fused.png`
- `support_left.png`
- `support_right.png`
- `support_fused.png`

这对排查“fused 后是不是只剩很少 support”“哪一边在主导某区域”非常重要。

#### D. `part3_BRPO/scripts/run_pseudo_refinement.py`

当前只有：

- `--use_left`
- `--use_right`

建议改成：

- `--target_side left|right|fused`

并修改 `load_pseudo_viewpoints()`：

- `left` -> `target_rgb_left.png`
- `right` -> `target_rgb_right.png`
- `fused` -> `target_rgb_fused.png`

confidence 读取逻辑也要改：

- `left/right` 时读 side confidence
- `fused` 时优先读 `confidence_mask_fused.npy`
- 若不存在则 fallback 到旧 `confidence_mask.npy`

同时建议把 history 里记录：

- `target_side`
- `pseudo_manifest_version`
- `num_fused_samples_used`

#### E. `part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py`

这个文件不一定要立刻大改，但建议至少更新 schema 生成逻辑，让新 cache 的 manifest 支持 fused 字段。

---

### 2.4.3 数据结构规划（方案 1）

建议把当前 sample 目录扩成这样：

```text
pseudo_cache/
├── manifest.json
└── samples/{frame_id}/
    ├── camera.json
    ├── refs.json
    ├── render_rgb.png
    ├── render_depth.npy
    ├── target_rgb_left.png
    ├── target_rgb_right.png
    ├── target_rgb_fused.png
    ├── depth_left.npy
    ├── depth_right.npy
    ├── conf_left.npy
    ├── conf_right.npy
    ├── target_depth.npy
    ├── confidence_mask.npy              # 旧主字段，可保留
    ├── confidence_mask_fused.npy
    ├── confidence_mask_fused.png
    ├── fusion_meta.json
    └── diag/
        ├── validity_mask.png
        ├── depth_consistency_map.png
        ├── epipolar_distance_left.png
        ├── epipolar_distance_right.png
        ├── rgb_disagreement.png
        ├── weight_left.png
        ├── weight_right.png
        ├── support_left.png
        ├── support_right.png
        ├── support_fused.png
        └── score.json
```

manifest 建议新增字段：

- `target_rgb_fused_path`
- `confidence_mask_fused_path`
- `depth_left_path`
- `depth_right_path`
- `conf_left_path`
- `conf_right_path`
- `fusion_meta_path`
- `sample_metrics`（可选嵌入）

---

## 2.5 方案 1 的实施顺序

建议严格按下面的顺序，不要一开始同时改太多：

### Phase 1
先不改 refine，只把 cache 产物做出来：

1. `epipolar_depth.py` 支持 side outputs
2. 新增 `pseudo_fusion.py`
3. `build_pseudo_cache.py` 写出 fused target 与 fused confidence
4. `diag_writer.py` 补充可视化

### Phase 2
再把 refine 接入 fused：

5. `run_pseudo_refinement.py` 增加 `--target_side fused`
6. 在当前 E 配置上重跑一组 `fused` 对照

### Phase 3
看效果后再决定是否引入更复杂的 agreement 项：

7. 若 fused 仍有脏纹理，再加 `A_depth`
8. 若 support 太少，再考虑保守的 support 扩张

---

# 3. 方案 2：控制 pseudo 视角，先改训练采样调度

## 3.1 原理目标

方案 2 的目标不是“删掉更多 pseudo”，而是把当前 uniform sampling 改成 **quality-aware curriculum sampling**。

原因很直接：

- pseudo 不是普通真实帧；
- 它有明显的质量差异和难度差异；
- 如果一开始就把所有 placement、所有置信度的样本混着训练，pseudo 偏差更容易写进高斯。

因此，方案 2 应该把 pseudo sample 看成有难度分层的训练资源：

- 先用几何更稳、agreement 更强、离 real views 更近的样本
- 再逐步引入更远、更难的样本

这和 DIFIX3D+ 的 progressive update 是一致的：

> 不要一上来让 diffusion 负责最难的外推区域，而要随着 3D 表达逐渐变好，再把更远、更不稳定的 pseudo 放进来。

---

## 3.2 方案 2 的数学抽象：定义 sample quality 与 sample difficulty

方案 2 的关键不是某个 fancy schedule，而是先把每个 sample 的“好坏”和“难度”定义清楚。

建议对每个 pseudo sample 定义以下 sample-level 指标：

### 3.2.1 质量指标 `Q_i`

\[
Q_i = \beta_1 \cdot \overline{C_{F,i}} + \beta_2 \cdot r_i + \beta_3 \cdot \overline{A_i}
\]

其中：

- `\overline{C_{F,i}}`：该 sample fused confidence 的均值
- `r_i`：该 sample fused support ratio，即 `valid pixels / total pixels`
- `\overline{A_i}`：左右 agreement 的均值

`Q_i` 越高，表示这个 pseudo sample 越可靠。

### 3.2.2 难度指标 `H_i`

难度不一定等于低质量。一个远视角样本可能很难，但如果 fused 很稳，也可能仍然可用。

建议定义：

\[
H_i = \gamma_1 \cdot d_i + \gamma_2 \cdot h_{placement,i} + \gamma_3 \cdot (1-Q_i)
\]

其中：

- `d_i`：pseudo 与两侧参考帧的相对距离指标
- `h_{placement,i}`：placement 难度先验，通常 `midpoint < tertile` 或更远 placement 更难
- `1-Q_i`：质量越差，难度越高

### 3.2.3 训练阶段的 sampling set

训练第 `t` 轮，只从某个允许集合 `\Omega_t` 中采样 pseudo：

\[
\Omega_t = \{i \mid H_i \le h(t),\; Q_i \ge q(t)\}
\]

其中：

- `h(t)` 随训练逐渐放宽
- `q(t)` 随训练逐渐降低

这样你就得到了一个 curriculum：

- 前期：只允许低难度、高质量 pseudo
- 后期：逐步引入更难的 pseudo

---

## 3.3 方案 2 的落地规划

方案 2 最重要的一点是：

**第一版不需要先重做数据准备。**

只要在当前 cache 的 manifest / score 里把 sample-level metadata 补齐，就能先在训练脚本里做调度。

### 3.3.1 新增文件

#### A. `part3_BRPO/pseudo_branch/pseudo_schedule.py`

职责：

- 从 pseudo manifest 中读取每个 sample 的 metadata
- 计算 bucket / stage / allowed set
- 给 `run_pseudo_refinement.py` 提供每轮可采样的 sample id 列表

建议核心函数：

- `load_sample_metrics(...)`
- `compute_sample_quality(...)`
- `compute_sample_difficulty(...)`
- `build_curriculum_buckets(...)`
- `sample_pseudo_ids_for_iter(iter_idx, config, bucket_state, rng)`

也可以把它放在 `scripts/` 下，但从职责上看更适合放 `pseudo_branch/`。

---

### 3.3.2 修改文件

#### A. `part3_BRPO/pseudo_branch/build_pseudo_cache.py`

即使不重做数据准备，它也需要补 sample-level metadata。建议在每个 sample 的 `score.json` 和 manifest entry 中写入：

- `placement`
- `target_valid`
- `mean_confidence`
- `mean_rgb_agreement`
- `quality_score`
- `difficulty_score`
- `bucket_id`

如果方案 1 先完成，这些字段可以直接基于 fused outputs 算；这样方案 2 才是真正建立在更稳的 supervision 之上。

#### B. `part3_BRPO/scripts/run_pseudo_refinement.py`

这是方案 2 的主改动点。

建议新增参数：

- `--pseudo_schedule static|curriculum`
- `--pseudo_curriculum_stages 3` 或 4
- `--pseudo_warmup_iters`
- `--pseudo_stage1_until`
- `--pseudo_stage2_until`
- `--pseudo_quality_threshold_stage1`
- `--pseudo_quality_threshold_stage2`
- `--pseudo_bucket_source quality|difficulty|placement`
- `--pseudo_stage1_placements midpoint`
- `--pseudo_stage2_placements midpoint,tertile_left`
- `--pseudo_stage3_placements midpoint,tertile_left,tertile_right`

同时要修改 pseudo 采样逻辑：

当前：

- 从 `pseudo_views` 里 uniform sample `num_pseudo_views`

修改后：

- 每轮先根据当前 iter 决定 stage
- 再从该 stage 允许的 pseudo pool 里采样

这样不需要重做 `prepare_stage1_difix_dataset_s3po.py`。

#### C. 可选修改 `part3_BRPO/docs/DESIGN.md` 与 `STATUS.md`

虽然不是代码必要项，但建议同步记录：

- 新的 schedule 配置
- curriculum 的 stage 定义
- 哪些字段由 manifest 提供

---

## 3.4 数据结构规划（方案 2）

方案 2 不必新增整套 cache，但要在 manifest / score 中新增 metadata。

### 3.4.1 manifest sample entry 新字段

在 `pseudo_cache/manifest.json` 的每个 sample entry 里建议新增：

- `placement`
- `quality_score`
- `difficulty_score`
- `mean_confidence`
- `support_ratio`
- `mean_rgb_agreement`
- `bucket_id`

### 3.4.2 `diag/score.json` 新字段

建议把详细值写到 `diag/score.json`：

- `mean_conf_left`
- `mean_conf_right`
- `mean_conf_fused`
- `support_left`
- `support_right`
- `support_fused`
- `mean_rgb_agreement`
- `quality_score`
- `difficulty_score`
- `placement`

manifest 存摘要，score.json 存细节。

---

## 3.5 refine 过程中如何分阶段

建议第一版采用 **四阶段 curriculum**，非常清楚，也足够好实现。

### Stage 0：warmup real-only

迭代区间：`[0, T0)`

- 不采样 pseudo
- 只做 real branch
- 作用：先让现有高斯在真实 sparse views 上稳定下来

建议：

- `T0 = 200 ~ 400` iter

### Stage 1：easy pseudo only

迭代区间：`[T0, T1)`

只允许：

- `placement = midpoint`
- `quality_score >= q1`
- `support_ratio >= r1`

这是最稳的一批 pseudo。

### Stage 2：moderate pseudo

迭代区间：`[T1, T2)`

允许：

- `midpoint + tertile_left` 或较高质量的 tertile
- `quality_score >= q2`

### Stage 3：full curriculum

迭代区间：`[T2, end)`

允许：

- 全部 placement
- 较低阈值，但仍过滤极差样本

同时，建议把 `lambda_pseudo` 也做分阶段：

- Stage 0: `0`
- Stage 1: `0.2 ~ 0.3`
- Stage 2: `0.4 ~ 0.5`
- Stage 3: `0.5` 或更小

也就是说，方案 2 不只是“采哪些 pseudo”，也包括“什么时候让 pseudo 说得更大声”。

---

## 3.6 方案 2 的实施顺序

### Phase 1
先不改任何 cache 生成流程，只在现有 manifest 基础上试最小版 schedule：

1. `run_pseudo_refinement.py` 里根据 `placement` 做阶段采样
2. 先实现：
   - warmup real-only
   - midpoint first
   - tertile later

### Phase 2
再补 metadata：

3. `build_pseudo_cache.py` 写 sample-level metrics
4. `pseudo_schedule.py` 基于 quality/difficulty 构建 allowed pool

### Phase 3
最后再做更细的 curriculum：

5. stage-wise `lambda_pseudo`
6. 随训练动态调整 `num_pseudo_views`

---

# 4. 两个方案的关系与推荐顺序

这两个方案并不是平行独立的，最好按顺序推进：

## 4.1 为什么先做方案 1，再做方案 2

如果当前 pseudo target 本身就不稳，那么无论 schedule 多聪明，本质上也只是“更聪明地采样脏数据”。

因此更合理的顺序是：

1. 先把 `left/right -> fused` 这件事做对
2. 再在 fused sample 上做 curriculum

这样方案 2 的 quality / difficulty 指标才有意义。

## 4.2 最推荐的工程顺序

### 第一阶段：方案 1 最小闭环

- `epipolar_depth.py`：输出左右 side depth/conf
- 新增 `pseudo_fusion.py`
- `build_pseudo_cache.py`：产出 fused target 和 fused confidence
- `run_pseudo_refinement.py`：支持 `--target_side fused`
- 在当前 E 配置上重跑一组 fused 对照

### 第二阶段：方案 2 最小闭环

- `run_pseudo_refinement.py`：先实现简单 placement-based curriculum
- 先不依赖很多新字段，只做：
  - real warmup
  - midpoint first
  - tertile later

### 第三阶段：方案 2 完整版

- `build_pseudo_cache.py`：写 quality/difficulty metadata
- 新增 `pseudo_schedule.py`
- refine 里接入 quality-aware sampling

---

# 5. 最终建议

如果只从“下一步最值得做什么”来排优先级，我会给出非常明确的顺序：

## Priority 1：方案 1 的 fused target

这是最核心的，因为它直接回答当前主问题：

- 位姿更准了
- 为什么图像还更脏

最可能的原因就是单边 pseudo supervision 仍然不稳。

## Priority 2：方案 2 的最小版 curriculum

先不追求复杂 difficulty score，先做：

- real warmup
- midpoint first
- tertile later

这样可以低工程成本验证“训练调度是否有效”。

## Priority 3：方案 2 的 metadata 完整化

等你确认 curriculum 有价值，再补：

- quality score
- difficulty score
- bucket sampling

这样工程节奏是最稳的。

---

# 6. 一页式实施清单

## 方案 1 需要新增 / 修改的文件

### 新增
- `part3_BRPO/pseudo_branch/pseudo_fusion.py`

### 修改
- `part3_BRPO/pseudo_branch/epipolar_depth.py`
- `part3_BRPO/pseudo_branch/build_pseudo_cache.py`
- `part3_BRPO/pseudo_branch/diag_writer.py`
- `part3_BRPO/scripts/run_pseudo_refinement.py`
- 可选：`part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py`

### 新增数据字段
- `target_rgb_fused.png`
- `confidence_mask_fused.npy`
- `confidence_mask_fused.png`
- `depth_left.npy`
- `depth_right.npy`
- `conf_left.npy`
- `conf_right.npy`
- `fusion_meta.json`
- manifest 中对应 path 与 sample metrics 字段

## 方案 2 需要新增 / 修改的文件

### 新增
- `part3_BRPO/pseudo_branch/pseudo_schedule.py`

### 修改
- `part3_BRPO/pseudo_branch/build_pseudo_cache.py`
- `part3_BRPO/scripts/run_pseudo_refinement.py`
- 可选更新文档：`DESIGN.md`, `STATUS.md`

### 新增数据字段
- `quality_score`
- `difficulty_score`
- `mean_confidence`
- `support_ratio`
- `mean_rgb_agreement`
- `bucket_id`
- `placement`（若 manifest 未完整暴露则补齐）

---

# 7. 结论

这两个方案里，**方案 1 是先决条件，方案 2 是增益器。**

- 方案 1 解决的是：pseudo target 本身是否可信
- 方案 2 解决的是：可信与不可信的 pseudo，训练时应该以什么顺序进入优化

对你当前仓库和当前实验状态，最合理的路线是：

1. **先把 left/right 修复结果做 fused pseudo target**，并把 fused confidence 正式接入 refine。
2. **再把 pseudo 采样从 uniform 改成 curriculum**，先做简单的 placement-based schedule，再升级到 quality-aware sampling。

这两步如果都跑通，Stage1 就会从“只有单边 pseudo RGB loss 的 standalone refine”升级到“有明确 multi-branch fusion 和 curriculum control 的 pseudo-assisted refinement”，而且整个改动仍然保持在当前 repo 的外部 pseudo branch + standalone refine 范围内，不需要立刻深改 S3PO frontend/backend 主循环。
