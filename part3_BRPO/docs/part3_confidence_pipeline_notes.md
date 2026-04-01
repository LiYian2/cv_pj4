# Part 3 参考方案笔记：BRPO 风格的 pseudo-view / confidence / consistency-aware optimization

## 0. 这份文档的定位

这是一份**给后续 agent / 自己继续分析与落地用的工作文档**，目标不是宣称“这就是最终定稿方案”，而是把目前已经明确的判断、推荐的最小实现路线、技术栈候选、以及关键风险点整理清楚。

文档中的方案应被理解为：

- **参考方案**：基于 Project 4 描述、BRPO / DIFIX3D+ / RegGS / S3PO 论文与官方仓库信息做出的工程化整理。
- **需要分析验证**：尤其是不同数据集（Re10K / DL3DV / Waymo）上的匹配器、光流、threshold、mask 权重、以及 pseudo-view 数量，都需要实际实验验证，不能直接视为最终最优配置。
- **目标优先级**：先做一个最小、可闭环、能写进报告的版本；再逐步增强，而不是一开始就追求完整论文复现。

---

## 1. 总体背景与当前判断

### 1.1 Project 4 Part 3 真正要求什么

Project 4 的 Part 3 要求本质上是：

1. 在 Part 2 定义的 sparse-view scenario 上生成 pseudo-views；
2. 把 pseudo-views 加回训练集，做 hybrid optimization；
3. 最好再做 confidence fusion / consistency-aware optimization；
4. 最终比较 **Sparse Views Only** vs **Sparse + Generated Views** 的渲染结果。

这意味着：

- 不要求完整复现某篇论文的全部细节；
- 允许做一个参考 BRPO / ReconX / 其它文献的**简化版工程方案**；
- 重点在于：能否合理生成伪视角、能否控制伪视角误差、能否稳定地回灌进重建。

### 1.2 为什么现在主线更适合走 BRPO 风格，而不是先追求完整 ReconX

目前更现实的路线是 **BRPO 风格的 rendered-image completion + confidence filtering + offline refinement**，而不是完整 ReconX 复现，主要原因：

- BRPO 的整体思路与课程 Part 3 非常贴合：
  - 当前 sparse reconstruction 渲染出 pseudo frame；
  - 用生成模型修复；
  - 做 confidence mask；
  - 再 joint optimize。
- ReconX 的完整方法强依赖：
  - DUSt3R 全局点云条件；
  - 改造过的 DynamiCrafter；
  - 官方仓库当前并未公开完整可运行代码。
- BRPO 本身虽然也没有公开完整 GitHub 实现，但它的工程切入点更容易被“简化复现”。

### 1.3 当前手头模型的定位

当前已经做了 Part 2 的两个候选底座：

- **RegGS**：真正符合“稀疏输入视图 + 未知位姿重建”的定义；输入是外部抽稀后的 sparse views，test 是未参与训练的剩余帧。
- **S3PO**：更偏连续单目视频流的 3DGS-SLAM；虽然论文结果强，尤其在 outdoor 数据上表现好，但其机制本质上是 tracking + mapping 的 SLAM 闭环，而不是固定 sparse view set 的注册式重建。

### 1.4 当前最推荐的主线

从课程要求、工程闭环难度、以及后续 pseudo-view 回灌的便利性来看，当前最推荐的第一主线是：

> **A = RegGS, B = RegGS**
>
> 即：先用 RegGS 做 sparse-only baseline，再基于其当前场景渲染 pseudo-views，经 Difix 修复和置信筛选后，重新作为扩充输入跑 RegGS 的 infer + refine + metric。

原因：

- RegGS 原生就是 sparse-view input，训练/测试定义最接近 Project 4；
- 它天然适合“重新组织输入集后再跑一遍”；
- 不需要碰 tracking thread；
- 能先做出一个最干净的 Part 3 最小闭环。

S3PO 可以作为后续副线实验：

- 验证“更强的 outdoor scene 起点是否会让 pseudo-view 质量更高”；
- 但不建议第一版就作为回灌主模型，因为它没有文档化的 mapping-only augmentation 接口。

---

## 2. 当前推荐的 Part 3 最小闭环

## 2.1 最小闭环的主干思路

建议先做如下最小闭环：

1. 用 **模型 A（优先 RegGS）** 在 sparse train views 上跑出当前重建；
2. 在相邻真实 sparse train views 之间选择少量 pseudo poses；
3. 从当前场景在这些 pseudo poses 上 render 出 RGB（最好也存 depth）；
4. 使用 **Difix_ref**（带参考图）修复这些 rendered pseudo frames；
5. 构造逐像素 confidence map；
6. 把通过 confidence 处理后的 pseudo-views 加入训练集；
7. 用 **模型 B（优先仍为 RegGS）** 重新做 infer + refine；
8. 在 untouched final test set 上做评测。

这条线本质上是：

> sparse-only baseline → pseudo-view generation → confidence-aware re-training/refinement → final evaluation

这是最适合作为 Part 3 stage1 的路线。

---

## 3. pseudo-view 如何选：当前最稳的策略

### 3.1 是否可以直接使用 Part 2 的 test ids 做 render

可以，但要注意一个非常重要的原则：

> **凡是被拿去做 pseudo-view augmentation 的 id，必须从 Part 3 的最终 test set 里永久移除。**

也就是说，Part 2 的 non-train ids 不能再直接整体沿用为 Part 3 最终测试集。

更合理的做法是把 non-train ids 再分成两类：

- **pseudo pool**：允许拿来 render / 修复 / 回灌
- **final untouched test set**：最后只用于评测，不参与任何生成和优化

### 3.2 第一轮最推荐的 pseudo id 选择方式

对于最小版 stage1，不建议一开始就选离 train 很远的视角，也不建议每个间隔里一次加太多 pseudo views。

最推荐的策略是：

> **每个相邻 train gap 只选一个 midpoint pseudo view**

也就是：

- 对每两个相邻真实 sparse train ids；
- 在它们之间选最接近中点的真实 held-out id；
- 用这个 id 对应的 pose 作为 pseudo-view render target。

这样做的优点：

- 比 train 近邻更有 densify 价值；
- 又没远到 render 已经彻底崩坏；
- 最符合“渐进式扩展 novel views”的思路。

### 3.3 对 Re10K 示例的直接建议

如果当前是：

- 196 张图；
- 等间距 9 个 train ids；

那么第一轮最推荐：

- 9 个 train
- 8 个 midpoint pseudo ids（每个 gap 1 个）
- 剩余所有 non-train 且非-pseudo 的帧作为 final untouched test

即：

> **9 train + 8 pseudo + 其余全部 final test**

第一轮不建议直接上 quarter points（1/4, 3/4）或每个 gap 多张 pseudo views。

---

## 4. 对 BRPO 完整 pipeline 的当前理解

为了后续 agent 不混淆，这里明确写出当前理解的 BRPO 流程。

### 4.1 BRPO 完整流程（论文层面）

完整 BRPO 的核心是：

1. 当前 Gaussian scene 在某个 pseudo pose 上 render 当前 pseudo frame；
2. 与前后真实参考帧一起进入一个 **pseudo-view deblur UNet**；
3. UNet 输出再与前/后单独参考帧分别进入 diffusion，得到两份 candidate restorations；
4. 通过 overlap score fusion 进行几何加权融合；
5. 再做 geometric verification / confidence mask inference；
6. 再在 joint optimization 中对 pseudo-view 做 masked RGB-D 监督；
7. 最后做 Gaussian refinement。

### 4.2 deblur UNet 的作用

这个 UNet 不是最终 pseudo-view 的输出器，而是：

> **先把当前 render 中的 ghosting / blending / cross-view inconsistency 过滤掉，给 diffusion 一个更稳定的输入。**

它的输入是三张图：

- 前参考帧
- 当前 render
- 后参考帧

输出是一张 deblurred pseudo frame，供后续 diffusion 使用。

### 4.3 diffusion 的作用

diffusion 不是直接输出最终 pseudo-view，而是：

- 以前参考帧为条件生成一份修复结果；
- 以后参考帧为条件生成另一份修复结果；
- 再通过 overlap / depth / pose consistency 做融合。

所以最终 pseudo-view 不是 UNet 输出，也不是任一单独 diffusion 输出，而是：

> **fusion 后的 fixed / fused pseudo frame**

### 4.4 当前为什么不建议第一版就复现完整 BRPO

原因：

- BRPO 没有公开完整代码；
- deblur UNet 是专门训练的任务网络，不是现成 plug-and-play；
- overlap score fusion、confidence mask inference 都需要自己补实现；
- 一开始就追求 full BRPO 复现，会极大拉长工程周期。

因此当前推荐策略是：

> **先跳过 UNet，只保留 “rendered pseudo frame + reference-conditioned Difix + confidence mask + masked refinement” 这条主线。**

---

## 5. confidence / consistency 这一部分为什么要这么做

Project 4 不只是要求你“生成几张补图”，而是希望你展示：

- 你知道生成视图是有噪声的；
- 你知道不能把它们全图同权地塞回重建；
- 你知道应该对不可靠区域降权；
- 你知道如何把置信图接进重建优化。

BRPO 的论文动机和课程要求在这一点上是高度一致的。

最核心的一句话：

> **confidence 不是判断“整张 pseudo-view 能不能要”，而是逐像素判断“这张图里哪些区域可以信、哪些只能少信、哪些不要信”。**

因此，当前推荐的实现不是训练一个新 mask 网络，而是：

> **几何主导 + patch 级验证 + 像素级平滑**

这是当前最稳、最容易落地、也最容易写进报告的一条线。

---

## 6. 当前推荐的 confidence / consistency 方案

### 6.1 总原则

第一版不要做纯 end-to-end 新模型。建议使用：

- 经典几何（投影 / depth consistency / visibility）
- 现成 matcher（LoFTR / DISK / DeDoDe + LightGlue / MASt3R）
- 可选 optical flow（OpenCV baseline 或 RAFT）
- 自己拼接成脚本式 pipeline

即：

> **库 + 预训练模型 + 自己写的 glue code**

而不是重新训练新网络。

---

## 7. 置信融合（confidence fusion）怎么做

### 7.1 目标

输出一张逐像素 confidence map：

\[
C_m(x) \in [0,1]
\]

表示 pseudo-view 上每个像素的可信程度。

### 7.2 当前推荐的三层结构

当前推荐的 confidence 分成三层：

1. **几何置信（主干）**
2. **特征验证（补强）**
3. **像素级平滑（变成可训练权重图）**

### 7.3 第一层：几何置信（必须做）

输入：

- pseudo pose 下的 render RGB：`I_pseudo_render`
- pseudo pose 下的 render depth：`D_pseudo_render`
- 左右真实参考帧 RGB：`I_left`, `I_right`
- 左右参考帧 pose / intrinsics
- pseudo pose / intrinsics

做法：

把 pseudo pose 视角的像素通过深度反投影到 3D，再投到左右参考帧。

对每个像素计算：

1. **visibility mask**
   - 投影是否落在参考帧图像内
   - 是否没有被遮挡
2. **depth consistency score**
   - 投到参考帧后的深度与参考侧估计深度是否一致
3. **pose consistency scalar**
   - 如果参考帧离 pseudo pose 太远，整体置信下降

可以先构造成：

\[
C_{geom} = O_{vis} \cdot s_{depth} \cdot s_{pose}
\]

这里：

- `O_vis` 是 0/1 或 [0,1] 可见性
- `s_depth` 是深度一致性指数分数
- `s_pose` 是参考帧距离带来的整体衰减

这一步不用训练任何网络，是整个 confidence pipeline 的主干。

### 7.4 第二层：特征验证（推荐做）

目的：

补强几何置信，防止：

- 几何投影看似成立，但图像内容不一致；
- diffusion 修出了“看起来合理但不符合真实参考”的纹理；
- 遮挡边界和纹理less 区域被误信。

当前推荐做法：

- 把 fused pseudo-view 与左参考帧做 matcher
- 把 fused pseudo-view 与右参考帧再做 matcher
- 看某个 patch / 区域是否能被左、右两侧共同支持

对于 matcher 的支持度，可以用如下三档：

- 两侧都支持：高置信
- 只有一侧支持：中置信
- 两侧都不支持：低置信

### 7.5 为什么不直接做纯逐像素 correspondence

没有代码时，直接做纯逐像素 correspondence mask 工程上很脆：

- 匹配点稀疏
- 无纹理区容易失效
- 遮挡边界容易误匹配
- 很难直接转成稳定的 dense weight map

因此当前更推荐：

> **patch 级验证 + 像素级平滑**

即：

- 先切图为 patch（例如 16×16 或 32×32）
- 每个 patch 统计：
  - 有效投影比例
  - 双向匹配点比例
  - depth consistency 均值
- 先得到 patch confidence
- 再上采样回逐像素图并做轻微平滑

### 7.6 第三层：像素级平滑

为了最后能真正作为 loss weight 使用，建议做：

- patch confidence 上采样到原分辨率
- 轻微 Gaussian blur / bilateral smoothing
- 可选在边界区域再做 edge-aware 降权

最终得到可用于 loss 的逐像素 `C_m`。

---

## 8. 光流怎么放进这条链路

### 8.1 当前建议：光流作为 veto / down-weight，不作为主干

Project 4 允许使用 optical flow；但当前最推荐的做法是：

> **光流不是主干，而是额外否决项 / 降权项。**

原因：

- 在大基线、遮挡、多视差场景下，flow 本身会不稳；
- BRPO 的真正核心仍然是 reprojection / overlap / confidence；
- 把 flow 放在主干位置会让工程链更脆弱。

### 8.2 当前推荐的使用方式

对于某些区域：

- 几何投影看起来可行；
- 但 warp 后图像残差很大；
- 或 flow 的 forward-backward consistency 很差；

则对这些区域额外降权。

也就是说，光流更适合这样加入：

\[
C_{final} = C_m \cdot C_{flow}
\]

其中 `C_flow` 只在明显不一致区域把权重打下来。

---

## 9. 一致性感知优化（consistency-aware optimization）怎么接

### 9.1 核心思想

confidence map 的意义在于：

> pseudo-view 可以参与优化，但不能整张图和真实视图同权。

因此，对于 pseudo-view 的 loss，必须改成 **masked loss**。

### 9.2 当前推荐的 masked loss

对于 pseudo-view：

\[
L_{rgb}^{pseudo} = \frac{\|C_m \odot (I_{pseudo}^{target} - I_{pseudo}^{render})\|_1}{\|C_m\|_1}
\]

\[
L_{depth}^{pseudo} = \frac{\|C_m \odot (D_{pseudo}^{target} - D_{pseudo}^{render})\|_1}{\|C_m\|_1}
\]

然后：

\[
L_{pseudo} = \lambda_{rgb} L_{rgb}^{pseudo} + \lambda_d L_{depth}^{pseudo}
\]

真实 train views 保持原来的监督逻辑不变。

### 9.3 当前建议增加两个保险项

#### （1）整图 view-level score

定义：

\[
\bar C = mean(C_m)
\]

然后让整张 pseudo-view 的总 loss 再乘一个 `barC`。

作用：

- 如果一整张 pseudo-view 整体都不可靠，它不会被硬丢掉；
- 但它对全局优化的影响会自动减弱。

#### （2）第一版先固定 pseudo-view pose

当前第一版不建议让 pseudo-view 重新参与 pose optimization。

推荐做法：

- 真实 train views：按原模型正常处理 pose
- pseudo-views：先固定其 pose
- 只让它们提供 scene/map 优化的额外监督

等这条线稳定后，再考虑小的 pose delta optimization。

### 9.4 边界与遮挡区额外降权

建议对以下区域再乘一个 edge penalty：

- depth discontinuity 边界
- visibility 变化剧烈区域
- flow divergence 很大的区域

作用：

减少薄结构、遮挡边界、反光区的错误 pseudo supervision。

---

## 10. 当前推荐技术栈

下面分为最小版和增强版。

### 10.1 最小版（最推荐先做）

目标：不训练任何新模型，先做出完整闭环。

#### 核心组件

- **Difix_ref**：做 reference-conditioned pseudo-view 修复
- **自写几何模块**：reprojection / visibility / depth consistency
- **matcher**：
  - Re10K：LoFTR
  - Waymo / DL3DV：DISK + LightGlue 或 DeDoDe + LightGlue
- **光流（可选）**：OpenCV Farneback
- **优化接法**：把 confidence map 接进 pseudo-view 的 RGB / depth masked loss

#### 适合当前阶段的原因

- 不训练任何新网络
- 工程周期最短
- 足够形成论文式 Part 3 最小结果
- 非常容易写入报告

### 10.2 增强版

在最小版跑通后，可以增强：

- matcher 升级为 MASt3R 风格 correspondence / 更强的 LightGlue pipeline
- 光流升级为 **Torchvision RAFT**
- 融合时加入 forward-backward flow consistency
- loss 中加入 reprojection residual penalty
- 引入更强的 patch-level score aggregation

### 10.3 当前不建议第一版就做的东西

不建议第一版就：

- 自己训练 BRPO 风格 deblur UNet
- 自己训练 confidence 网络
- 直接把 pseudo-view 塞回 S3PO tracking loop
- 同时混搭 A / B 为不同模型

---

## 11. 第一版参考执行顺序（可直接作为实现草案）

下面是当前最推荐的**第一版参考执行顺序**。

### Step 1：确定 train / pseudo / final test 划分

对于每个数据集：

1. 保留现有 sparse train ids
2. 从 train gaps 中选 midpoint pseudo ids
3. final untouched test = 所有剩余非 train、非 pseudo 的帧

### Step 2：用模型 A 跑 sparse-only baseline

优先用 RegGS。

输出：

- 当前 Gaussian scene
- 对应 render 接口
- metric baseline

### Step 3：在 pseudo ids / pseudo poses 上 render

从当前场景输出：

- pseudo render RGB
- pseudo render depth
- 可能还包括 opacity / visibility（若易于获取）

### Step 4：Difix_ref 修复

对于每个 pseudo view：

- 用左参考帧做一次修复
- 用右参考帧再做一次修复

得到两张 candidate pseudo-views。

### Step 5：几何 overlap / fusion

对左、右候选结果：

- 根据 visibility / depth consistency / pose consistency 生成几何权重图
- 做 weighted fusion

得到 fused pseudo-view。

### Step 6：patch 级 confidence 构建

- 对 fused pseudo-view 与左右参考帧分别做 matcher
- 统计 patch-level support
- 与 geometry score 结合
- 可选再乘一个 flow veto
- 上采样并平滑为逐像素 `C_m`

### Step 7：构造 pseudo-view 训练样本

保存：

- fused pseudo-view RGB
- pseudo pose / intrinsics
- confidence map
- 若需要也可保存 pseudo depth target

### Step 8：模型 B 做第二阶段优化

优先仍用 RegGS。

做法：

- 真实 train views 保持原始监督
- pseudo-views 使用 masked RGB/depth loss
- 第一版先固定 pseudo pose

### Step 9：最终评测

只在 untouched final test set 上算：

- PSNR
- SSIM
- LPIPS

同时与 Sparse Views Only baseline 比较。

---

## 12. 当前推荐用到的库 / 模块

### 12.1 Difix 部分

- 官方 Difix / Difix_ref 仓库与预训练权重
- 使用方式：`image + ref_image`

### 12.2 matcher 部分

推荐优先考虑：

- **Kornia**
  - 作为可微分 CV 工具箱
  - 可接入 LoFTR / DISK / LightGlue 等
- **MASt3R**（若后续希望更贴 BRPO）

当前策略：

- Re10K 先 LoFTR
- Waymo / DL3DV 先 DISK + LightGlue 或 DeDoDe + LightGlue

### 12.3 光流部分

最小版：

- **OpenCV Farneback**

增强版：

- **Torchvision RAFT**

### 12.4 图像与几何工具

- OpenCV：图像读写、光流、warp、插值
- NumPy / PyTorch：矩阵运算
- 自己写投影/反投影函数

---

## 13. 环境建议

### 13.1 是否直接在 Difix 环境上补库

当前推荐：

> **Part 3 单独做一个环境，而不是直接在现有 Difix 环境或 RegGS / S3PO 环境上硬补所有库。**

原因：

- Part 3 会同时依赖：
  - Difix
  - 图像匹配/特征库
  - 光流
  - 可能还要接重建模型的渲染脚本
- 直接在已有各自项目环境上叠库，容易出现版本冲突；
- Part 3 本身已经逐渐形成一个独立 experiment pipeline，单独开环境最清楚。

### 13.2 当前推荐的环境结构

建议至少分成：

#### 环境 A：重建环境

- RegGS / S3PO 各自原生环境
- 只负责：
  - baseline infer / refine / slam
  - 导出 render RGB / depth
  - 最终 metric

#### 环境 B：Part 3 生成与置信环境（推荐新建）

包含：

- Difix
- Kornia
- OpenCV
- PyTorch / torchvision（若要 RAFT）
- 自己写的 confidence / fusion 脚本

这样做的优点：

- Part 3 的依赖更可控；
- 可以单独调试生成与 mask；
- 不破坏原有 Part 2 复现实验环境。

### 13.3 什么时候再考虑统一环境

只有当：

- Part 3 pipeline 基本稳定；
- 已经确认哪些库真正需要；
- 不再频繁试不同 matcher / flow 方案；

再考虑是否与某个主模型环境合并。

当前阶段不建议。

---

## 14. 当前最值得强调的风险点

### 14.1 不要污染最终 test

- pseudo ids 不能继续留在 final test 里
- final test 必须保持 untouched

### 14.2 不要一开始就加太多 pseudo views

- 第一轮每个 gap 1 个 midpoint 就够
- 先看趋势，再决定是否加 quarter points

### 14.3 不要让 pseudo-view 第一版直接参与 tracking

尤其是如果后续尝试 S3PO 回灌：

- 第一版只让 pseudo-view 参与 mapping / refinement
- 不要进 tracking loop

### 14.4 不要把光流当主干

- 几何投影和 depth consistency 才是主干
- optical flow 适合做 veto / down-weight

### 14.5 不要第一版就追求 BRPO 完整复现

- 没有 repo
- UNet 是专门训练组件
- full BRPO 代价太高

第一版先做“BRPO-inspired simplification”更合理。

---

## 15. 当前建议的实验优先级

### 第一优先级

做出下面这个闭环：

> RegGS sparse-only baseline → midpoint render → Difix_ref → confidence map → RegGS re-run → final test metrics

### 第二优先级

在第一版跑通后，比较：

- 是否加入光流 veto 有提升
- 不同 matcher 是否对室内/室外差异明显
- 不同 pseudo 数量是否带来收益或破坏

### 第三优先级

再考虑：

- 是否要把 A 改成 S3PO 做更强的 pseudo render 起点
- 是否要尝试 S3PO 作为 B 的 mapping-only refinement
- 是否要做更接近 BRPO 的双候选 overlap fusion / 更细的 correspondence mask

---

## 16. 当前一句话总结

当前最推荐的 Part 3 启动方案是：

> **以 RegGS 为主线，使用 midpoint pseudo views + Difix_ref + “几何主导 + patch 级验证 + 像素级平滑”的 confidence pipeline，先完成一个不训练新模型的 confidence-aware offline refinement 闭环。**

这条线：

- 最符合 Project 4 的 sparse-view 设定；
- 最容易闭环；
- 最容易写进报告；
- 也最适合作为后续 agent 继续深入分析与扩展的起点。

