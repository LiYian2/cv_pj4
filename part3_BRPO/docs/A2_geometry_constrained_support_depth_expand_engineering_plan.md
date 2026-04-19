# A2 工程落地方案：Geometry-Constrained Support / Depth Expand

## 1. 目标
在 A1 已把 RGB / depth 的 trusted support 统一之后，把当前过窄的 joint support 向“几何一致区域”做有限扩张，使 depth supervision 从“统一但仍稀疏”推进到“统一且可用”。

一句话目标：扩张的是 `geometry-supported supervision`，不是 raw RGB mask。

## 2. 当前代码事实
当前 repo 已经具备做几何约束扩张所需的大部分输入：
1. `build_brpo_v2_signal_from_internal_cache.py`
   - 能拿到 `projected_depth_left/right.npy`
   - 能拿到 `fusion_weight_left/right.npy`
   - 能拿到 `overlap_mask_left/right.npy`
   - 能拿到基于 MASt3R 的 `raw_rgb_confidence_v2*`
2. `pseudo_branch/flow_matcher.py`
   - live code 直接加载 MASt3R，不是纸面设想
3. 当前设计原则已经固定：
   - 不直接 densify raw RGB
   - 若要扩张，应优先做 geometry-constrained support/depth expand

因此 A2 不是重新找方向，而是把设计原则真正写成模块。

## 3. 理论判断
A2 的核心不是“让 mask 变大”，而是让以下约束成立：
- 只有在 projected depth、overlap、fusion support 共同支持的位置，才允许从高精度 seed 向外扩张
- 扩张后的区域必须仍然能解释成多视角几何一致，而不是单纯 appearance propagation

如果 A1 解决的是“语义统一”，那么 A2 解决的是“统一后仍过窄”。

## 4. 拟修改文件
### 新增
1. `pseudo_branch/brpo_v2_signal/support_expand.py`

### 修改
1. `pseudo_branch/brpo_v2_signal/joint_confidence.py`
2. `scripts/build_brpo_v2_signal_from_internal_cache.py`
3. `pseudo_branch/brpo_v2_signal/__init__.py`

## 5. 设计方案
### 5.1 新增 support expand 模块
建议新增函数：
1. `build_geometry_seed_support(...)`
2. `expand_support_under_geometry_consistency(...)`
3. `build_expanded_joint_confidence(...)`
4. `write_support_expand_outputs(...)`

### 5.2 种子定义
seed 不使用 raw RGB 全量区域，而使用更高精度子集：
- A1 的 `joint_confidence_v2` 高置信度区域
- 双侧投影都有效的区域优先
- `raw_rgb_confidence_cont_v2` 高值位置优先

### 5.3 扩张规则
扩张只能在以下条件满足时发生：
1. 邻域像素在左/右 projected depth 上存在一致支持
2. overlap / projected-validity 有效
3. fusion weight 不过低
4. 与 seed depth 差异在阈值内

推荐第一版规则：
- 先做 conservative region grow，而不是复杂 learned propagation
- 对 both-side support 与 single-side support 分开标记
- 所有扩张结果都写 meta，明确来源是 seed / expanded-both / expanded-single / fallback

### 5.4 输出形式
A2 不覆盖 A1 文件，建议额外写出：
- `joint_confidence_expand_v1.npy`
- `joint_confidence_expand_cont_v1.npy`
- `joint_depth_target_expand_v1.npy`
- `joint_expand_source_map_v1.npy`
- `joint_expand_meta_v1.json`

这样 A1 和 A2 可以并行保留，便于 apples-to-apples compare。

## 6. 输入 / 输出
### 输入
1. A1 的 joint artifacts
2. `projected_depth_left/right.npy`
3. `fusion_weight_left/right.npy`
4. `overlap_mask_left/right.npy`
5. 必要时 `raw_rgb_confidence_cont_v2.npy`

### 输出
1. expanded joint confidence
2. expanded joint depth target
3. source map / meta（标清 seed vs expanded）

### 训练侧预期变化
1. `depth_verified_ratio` 上升
2. `depth_dense_ratio` / `depth_effective_mass` 上升
3. `loss_depth_dense` 的数值和梯度不再长期接近“只在极小区域起作用”

## 7. 实施步骤
1. 先把 A1 joint path 稳定下来
2. 新建 `support_expand.py`
3. 在 builder 中加可开关的 expanded branch
4. 写出 expanded artifacts 和 source map
5. 做帧级可视化 / summary，确认扩张发生在几何一致区域，而不是全局泛化
6. 做小 compare：A1 vs A2，在同一 StageB protocol 下比较

## 8. 验收标准
### 机制验收
1. expanded 区域面积较 A1 增长，但不是无约束暴涨
2. source map 能区分 seed / expanded / fallback
3. 扩张主要集中在 overlap + projected-valid 区域

### 结果验收
1. `depth_effective_mass` 明显高于 A1
2. StageB compare 比 A1 更稳，或至少能解释为“coverage 增加但噪声未显著失控”
3. 若扩大 coverage 后 replay 明显恶化，则要先回退阈值/规则，而不是直接推进更重方案

## 9. 不在 A2 做的事
1. 不做 raw RGB densify
2. 不引入 stochastic propagation
3. 不同时改 SPGM
4. 不重写 depth target 数值生成主逻辑

## 10. 风险点
1. expand 后 coverage 上去了，但 noise 也上去
2. 单侧 support 如果放得太宽，会重新回到“appearance looks denser but geometry unreliable”
3. 如果不保留 source map，很难定位失败来自 seed 还是 expand

## 11. 可交付内容
1. `support_expand.py` 代码
2. expanded joint artifacts
3. source map / meta / summary
4. A1 vs A2 的小 compare 文档
5. 明确的保留 / 回退结论
