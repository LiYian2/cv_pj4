# S3PO-GS 原理与调参笔记

## 1. S3PO-GS 的核心思想

S3PO-GS 是一个 **monocular outdoor RGB-only 3DGS-SLAM** 系统。它的目标不是像 RegGS 那样“给一组无位姿稀疏视图，离线做统一注册和重建”，而是面对一个连续视频流，**一边估计相机位姿，一边维护和更新 3D Gaussian 地图**。

这篇方法要解决的核心矛盾有两个：

第一，单目户外场景中，如果只靠 3DGS differentiable rendering 的 photometric error 做 pose optimization，tracking 往往缺少几何锚点，在大位移、复杂纹理、远近层次丰富的 outdoor 场景里容易掉进局部最优。

第二，如果直接引入一个外部预训练几何模块来给深度或点图，虽然几何先验更强，但 monocular 场景天然有尺度不确定性。外部模型和当前地图一旦不在同一个尺度系里，误差就会沿时间累积，最后表现成 **scale drift**，再进一步拖坏位姿和地图。

S3PO-GS 的关键设计就是：

* **使用预训练 pointmap 模型提供几何先验**
* **但不让外部 pointmap 直接定义系统的全局尺度**
* **系统尺度始终锚定在当前 3DGS 地图自身上**

也就是说，它不是“让外部模型接管 SLAM”，而是“让外部模型补充几何，但尺度主权留在当前地图”。

---

## 2. S3PO-GS 的整体工作流程

从系统角度看，S3PO-GS 是一个非常典型的前端 / 后端 SLAM 框架，只是它的 tracking 和 mapping 都围绕 **3DGS-rendered pointmap + pre-trained pointmap prior** 来设计。

可以把它的主流程概括成这样：

1. 初始化第一版 3DGS 地图
2. 对每个新帧做 tracking，估计当前 pose
3. 如果当前帧被选为 keyframe，则进入 mapping，更新 Gaussian 地图
4. 地图更新后，后续 tracking 再基于更新后的地图继续运行
5. 主 SLAM 流程结束后，可选做 color refinement

所以它不是“先重建，再定位”，也不是“先定位完全部轨迹，再建图”，而是一个标准的 **tracking ↔ mapping 闭环**。

---

## 3. 初始 3DGS scene 从哪来

系统开始时不是空地图，也不是从 COLMAP 初始化。S3PO-GS 会先用 **MASt3R** 这样的预训练 pointmap 模型生成初始几何，然后把 pointmap 优化成第一版 3D Gaussian map。

从实现角度理解，可以把它看成：

* 先拿预训练模型给出一个初始 pointmap / geometry prior
* 再把这个 pointmap 转化为高斯地图的第一版结构
* 后续所有 tracking 和 mapping 都在这张地图坐标系下继续进行

因此，这个初始化步骤的意义不是获得最终高质量地图，而是建立一个后续可持续优化、可持续渲染 pointmap 的 **scene anchor**。

---

## 4. 预训练模型是什么，在哪用

S3PO-GS 代码里实际使用的预训练模型是 **MASt3R**。它的作用不是直接输出最终位姿，也不是直接替代 3DGS 地图，而是作为 **pointmap prior / correspondence provider**。

它主要在两个地方用到：

### 4.1 Tracking 中用来建立 correspondence

对相邻关键帧和当前帧，预训练 pointmap 模型会输出两帧的 pointmap 及其匹配关系。系统利用这些关系，帮助建立：

* 关键帧像素 ↔ 当前帧像素
* 关键帧 pointmap ↔ 当前帧 pointmap

但需要强调的是：这里它主要提供的是 **对应关系和局部几何参考**，不是最终 pose 的尺度来源。

### 4.2 Mapping 中用作几何 prior

在 keyframe mapping 阶段，系统会同时拿到：

* 当前 3DGS 地图渲染出来的 pointmap
* 预训练模型输出的 pointmap prior

然后用 patch-based scale alignment 把 prior 对齐到当前地图尺度，再用于 point replacement 和 geometry supervision。

所以它既参与 tracking，也参与 mapping，但在两个阶段里都不是“最终答案提供者”，而是“几何先验提供者”。

---

## 5. 前端和后端分别做什么

### 5.1 前端：以 tracking 为核心

前端的职责是逐帧处理视频，负责当前帧的 pose estimation 和 keyframe decision。

更具体地说，前端做的是：

* 读取当前输入帧
* 找到相邻关键帧
* 基于当前 3DGS 地图，在关键帧视角下渲染 depth / pointmap
* 结合预训练 pointmap 模型建立 correspondence
* 通过 2D-3D correspondence 做 PnP / pose initialization
* 再基于 photometric loss 做 pose refinement
* 判断当前帧是否应该成为 keyframe

可以说，前端完成的是：

**“我现在相机在哪里？”**

这是系统里最偏 online、最偏 frame-by-frame 的部分。

### 5.2 后端：以 mapping 和 map optimization 为核心

后端的职责是维护高斯地图，并在新 keyframe 到来时更新地图。

后端做的事情主要包括：

* 接收关键帧和当前位姿
* 渲染当前视角下的 pointmap
* 取出预训练 pointmap prior
* 做 patch-based scale alignment
* 做 point replacement
* 插入或更新新的 Gaussians
* 在局部窗口内优化地图和部分位姿
* 可选执行 color refinement

可以说，后端完成的是：

**“我应该怎样修改当前地图，让它更符合最新观测？”**

---

## 6. Tracking 的细致流程

S3PO-GS 的 tracking 不是单纯 photometric-only，而是一个两阶段流程：**几何初始化 + 光度细化**。

### 6.1 用当前地图渲染 pointmap

系统从当前 3DGS scene 出发，在相邻关键帧的视角下渲染 depth map 和 pointmap。这个 rendered pointmap 很关键，因为它来自当前地图，所以它天然处于当前 scene scale 中。

这一步的作用是：

* 给 tracking 提供 3D anchor
* 提供当前地图定义下的几何坐标系
* 保证后续 pose estimation 不是在外部模型的漂移尺度里完成

### 6.2 用预训练 pointmap 模型建立跨帧 correspondence

系统把相邻关键帧图像和当前帧图像送进 MASt3R，得到两帧 pointmap 及其对应关系。

这里得到的是一种桥梁信息：

* 关键帧图像里的哪些像素，与当前帧里的哪些像素相对应
* 或者说，当前帧和关键帧之间的局部几何对应

### 6.3 构造 2D-3D correspondences

tracking 真正需要的是：

* 当前帧 2D 像素
* 地图中的 3D 点

这一步是通过两种信息拼接起来的：

* 渲染 pointmap 提供“关键帧像素 ↔ 地图 3D 点”
* 预训练 pointmap 提供“关键帧像素 ↔ 当前帧像素”

于是系统得到：

**当前帧像素 ↔ 当前地图中的 3D 点**

这就是后面 PnP 的输入。

### 6.4 PnP 给出 pose 初值

有了 2D-3D correspondences 后，系统通过 RANSAC + PnP 得到当前帧的 pose initialization。

这里最关键的一点是：

**PnP 里的 3D 点来自当前 3DGS 地图，而不是直接来自预训练模型。**

因此 tracking 的位姿初值天然继承当前地图的尺度，而不是外部 pointmap 的尺度。

### 6.5 Photometric refinement

得到 pose 初值后，系统再通过 3DGS differentiable rendering 的 photometric loss 对当前 pose 做 refinement。

也就是说，tracking 的最终目标函数是 photometric consistency，但它不是从零开始纯靠 photometric 搜索，而是在几何锚定基础上做局部精修。

这一点非常重要，因为如果没有前面的几何初始化，纯 photometric tracking 在复杂 outdoor scene 中很容易不稳。

---

## 7. Mapping 的细致流程

如果当前帧被前端判定为 keyframe，它就会进入 mapping。

S3PO-GS 的 mapping 不是简单把当前帧直接插进地图，而是先做尺度一致化，再做地图更新。

### 7.1 获取两类 pointmap

在当前 keyframe 视角下，系统会拿到两种 pointmap：

* 当前 3DGS 地图渲染得到的 pointmap `Xr`
* 预训练模型输出的 pointmap prior `Xp`

这两者各有优劣：

* `Xr` 的尺度是对的，但局部可能不够准
* `Xp` 的局部几何可能更准，但尺度不一定对

### 7.2 Patch-based scale alignment

S3PO-GS 不是直接把 `Xp` 塞进地图，而是先把 `Xp` 和 `Xr` 切成小 patch，比较这些 patch 的分布，只在分布相似的 patch 上做归一化和对齐，再估计 scale factor。

这一步的本质是：

**先把外部几何 prior 拉回到当前地图尺度系里。**

这就是论文中“global scale-consistent”最关键的实现之一。

### 7.3 Point replacement

对齐后的 pointmap prior 会作为参考，检查当前 rendered pointmap 哪些点是可信的，哪些点是明显错误的。

系统不会全盘替换，而是：

* 保留当前地图中已经可信的点
* 只在错误区域用对齐后的 prior 替换

这样做可以避免 noisy prior 直接破坏已经比较正确的地图部分。

### 7.4 Gaussian insertion 与 map optimization

替换完成后，系统会基于修正后的 pointmap 插入新的 Gaussians，并在局部窗口中联合优化：

* 高斯参数
* 部分关键帧位姿
* photometric consistency
* pointmap geometry consistency
* isotropic regularization

因此 mapping 既是“加点”，也是“修图”和“重整局部一致性”。

---

## 8. Tracking 与 Mapping 的关系

这是理解 S3PO-GS 最重要的部分之一。

tracking 和 mapping 不是两个独立阶段，而是一个强耦合闭环。

一方面，**tracking 依赖 mapping**。因为 tracking 要用当前地图渲染 pointmap 作为几何锚点。如果地图质量差，rendered pointmap 就不准，后面的 correspondence、PnP 和 pose refinement 都会受影响。

另一方面，**mapping 依赖 tracking**。因为 mapping 要知道当前 keyframe 的 pose，才能把新观测写进地图。如果 tracking 给出的 pose 错了，mapping 就会把错误高斯插到错误位置，进一步污染地图。

所以它们的关系可以概括成：

* 地图给 tracking 提供 anchor
* tracking 给地图提供入口
* 地图更新后，下一轮 tracking 再获得更好的 anchor

这就是典型的 SLAM 循环。

### 为什么会 scale drift 和 map collapse

如果 tracking 有一点误差，mapping 就可能把错误的几何写进地图；地图被写坏以后，下一轮 tracking 渲染出来的 pointmap 也会更差，于是 pose 又更容易错。这样误差会不断自我强化。

在 monocular、长序列、稀疏或弱视差场景里，这种问题尤其容易发生，最终表现为：

* 轨迹越来越漂
* 深度和尺度不断累积误差
* 高斯地图发虚、错位、重影
* 渲染结果还能“出图”，但几何已经塌了

S3PO-GS 的很多设计，本质上就是在防止这个闭环朝坏方向滚雪球。

---

## 9. Prior 是怎么来的，又是怎么被使用的

prior 来自 **预训练 pointmap 模型（MASt3R）**。

但它的使用方式非常克制，不是直接把 prior 当最终输出，而是分成两种用途：

在 tracking 中，prior 主要用于 **建立 correspondence**；
在 mapping 中，prior 主要用于 **提供几何参考并经过尺度对齐后做修正与监督**。

所以更准确地说，prior 的角色是：

* 提供几何信息
* 提供匹配关系
* 提供局部结构参考

但它**不直接掌管系统尺度，不直接替代地图，不直接给最终 pose**。

这正是 S3PO-GS 和很多“直接引入外部 depth/pointmap”方法最本质的区别。

---

## 10. Color refinement 在系统里是什么位置

从执行逻辑上，S3PO-GS 可以粗略理解成：

* 主体：几何驱动的 SLAM（tracking + mapping）
* 可选增强：color refinement

Color refinement 不属于 tracking-mapping 主闭环，而是在主 SLAM 跑完之后，对已有高斯地图再做一轮偏外观的优化。

所以如果从“代码执行体验”上分，你可以说系统分成：

* 几何建模主流程
* 颜色/外观 refinement

但如果从论文方法论上讲，主体仍然是 SLAM 闭环，而不是 geometry 和 color 两条并列主线。

---

## 11. 参数调节思路总览

S3PO-GS 的参数很多，但真正高价值的方向，主要集中在以下几组：

1. **tracking 稳定性参数**
2. **keyframe 触发策略**
3. **局部窗口 / pose window**
4. **pointmap scale alignment 相关参数**
5. **photometric vs geometry 权衡**
6. **global BA 与 color refinement**

调参时不要一开始大网格全扫。更合理的方式是：

* 先固定一组较稳的基线
* 再按数据集失败模式做小范围扫描
* 每个数据集只抓最相关的 1 到 2 条轴

---

## 12. General 参数方向与解释

### 12.1 `tracking_itr_num`

表示 tracking 阶段 pose refinement 的迭代次数。

* 增大：tracking 更稳，尤其是 pose 初值不够准时更有帮助，但速度更慢
* 减小：更快，但如果当前场景难，可能 refinement 不够

适合在“pose 初值能跟上，但 refinement 似乎不够稳定”时调。

### 12.2 `kf_interval`

表示关键帧的基本采样密度。

* 更小：keyframe 更密，地图更新更频繁，通常对复杂运动更稳，但计算更重
* 更大：更省算力，但可能让相邻 keyframe 之间运动跨度太大

适合在“关键帧之间变化太大，tracking 容易跳”时调小。

### 12.3 `window_size`

表示 local mapping / local BA 中使用的关键帧窗口大小。

* 更大：局部一致性更强，通常更稳，但更慢、更吃显存
* 更小：更轻，但约束弱

适合在“局部窗口约束不够、后期地图开始发散”时增大。

### 12.4 `pose_window`

更偏向 pose 相关优化参与的最近帧范围。

* 小幅增大通常更值得尝试
* 太大则会拖慢优化，还不一定继续收益

适合“最近几帧 pose 不稳，但不一定需要整体窗口都变大”的情况。

### 12.5 `mapping_itr_num`

表示 mapping 阶段每次局部地图优化跑多少轮。

* 更大：地图更有机会收敛
* 更小：速度更快，但可能写图不充分

适合在“tracking 还行，但地图质量差、渲染细节差”时考虑增加。

### 12.6 `patch_size`

patch-based scale alignment 的 patch 大小。

* 小 patch 更局部，但更容易受噪声影响
* 大 patch 更稳，但可能把局部误差平均掉

这是 S3PO-GS 最值得扫的一组参数之一，尤其在前冲、弱视差场景下。

### 12.7 `alpha`

photometric loss 与 geometry / pointmap supervision 的权衡系数。

* 更大：更相信 photometric fitting
* 更小：更强调几何 prior

如果场景主要问题是尺度和几何不稳，通常更值得尝试**略微降低 alpha**。

### 12.8 `global_BA` 与 `global_BA_itr_num`

是否启用全局 BA，以及全局 BA 的迭代次数。

BA（Bundle Adjustment）的本质是：**联合优化相机位姿和场景几何，使所有观测在图像上的重投影更一致。**

在 S3PO-GS 中，global BA 更像在主 SLAM 完成或阶段性完成后，再做一次全局一致性校正。

* 开启后通常更稳、更一致
* 但耗时明显增加

### 12.9 `color_refinement`

是否在主 SLAM 结束后进一步做颜色/外观优化。

* 开启：最终渲染通常更漂亮
* 关闭：更省时间，更便于只比较几何表现

如果研究重点是 tracking / geometry，本参数可以后置；如果最后要看渲染图质量，则建议保留。

---

## 13. 数据集特定的参数方向

下面的建议基于三个数据集的运动特点，而不是统一地机械套参数。

### 13.1 Re10K：温和场景，适合做“效率边界”扫描

你的统计显示 Re10K 相邻 train：

* 平移约 `0.52`
* 旋转约 `4.05°`

这是相对温和的几何跨度，说明它不一定需要非常激进的 tracking / mapping 设置。Re10K 更适合作为“基线校准数据集”，看哪些参数已经开得偏重。

建议优先扫描：

* `tracking_itr_num`: `200 / 300 / 400`
* `window_size`: `8 / 10 / 12`
* 第二阶段可补 `pose_window`: `3 / 4 / 5`

关注点：

* 是否可以在几乎不掉效果的前提下减轻计算量
* 你当前较重的配置是否其实对 Re10K 已经过保守

### 13.2 DL3DV：旋转主导，优先看 tracking 与 keyframe 组织

你的统计显示 DL3DV：

* 平移约 `2.71`
* 旋转约 `16.61°`

这类数据的主要问题更像：**tracking 注册不稳，而不是地图完全长不出来。**

这里优先看的不是 patch prior，而是前端的关键帧密度和 tracking refinement 强度。

建议优先扫描：

* `kf_interval`: `2 / 3 / 4`
* `tracking_itr_num`: `300 / 400 / 500`
* 第二阶段补 `pose_window`: `4 / 5 / 6`

关注点：

* 更密的 keyframe 是否能显著降低 tracking 跳变
* 更强的 refinement 是否真能解决问题，还是只是变慢
* 最近几帧 pose 约束是否偏弱

### 13.3 Waymo-405841：前向推进主导，优先看尺度一致性

你的统计显示 405841：

* 平移约 `5.63`
* 旋转约 `1.51°`

它的问题不是“大角度旋转”，而是**长段前冲 + 侧向视差弱**。这会使尺度 / 深度歧义更容易沿序列累积，因此重点不应先放在堆 tracking iterations，而应放在 **几何 prior 是否真正以 scale-consistent 的方式进入地图**。

建议优先扫描：

* `patch_size`: `8 / 10 / 12`
* `alpha`: `0.95 / 0.97 / 0.98`
* 第二阶段补 `pose_window`: `4 / 5 / 6`

关注点：

* patch-based scale alignment 是否足够稳
* 是否应该略微提高几何 prior 的权重
* 最近几帧 pose 是否因为尺度累积而相互带偏

---

## 14. 推荐的第一批扫描策略

如果要控制实验量，又想确保每个数据集都扫到真正有意义的方向，建议采用：

* **统一基线配置**：当前 notebook 中已经较稳的这一组
* 每个数据集只额外扫 2 条最相关的轴

建议第一批扫描如下：

### Re10K

* `tracking_itr_num`: `200, 300, 400`
* `window_size`: `8, 10, 12`

### DL3DV

* `kf_interval`: `2, 3, 4`
* `tracking_itr_num`: `300, 400, 500`
* 第二阶段再补 `pose_window`

### 405841

* `patch_size`: `8, 10, 12`
* `alpha`: `0.95, 0.97, 0.98`
* 第二阶段再补 `pose_window`

这样设计的好处是，最后写实验分析时会很清楚：

* Re10K：在看基线是不是过重
* DL3DV：在看 tracking / keyframe 是否是主瓶颈
* 405841：在看尺度一致性和 prior 注入方式是否是主瓶颈

---

## 15. 一个简短总结

S3PO-GS 的本质是：

**用 3DGS 维护一个持续更新的地图，用 pre-trained pointmap 提供几何先验，但始终让系统尺度锚定在当前地图自身上。**

它的前端负责 tracking，后端负责 mapping；tracking 和 mapping 不是独立阶段，而是一个闭环。tracking 依赖地图提供 pointmap 锚点，mapping 又依赖 tracking 提供准确位姿；prior 则在两者之间充当几何补充，但始终不接管尺度。

调参时最重要的不是无差别扫所有参数，而是根据数据集失败模式选参数轴：

* 温和场景看效率边界
* 旋转主导场景看 tracking 与 keyframe
* 前冲主导场景看 scale alignment 与 geometry prior 权重

这比单纯做大网格扫描更容易得到可解释的结论。
