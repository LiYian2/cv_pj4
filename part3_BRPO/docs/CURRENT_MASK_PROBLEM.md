# CURRENT_MASK_PROBLEM.md

> 最后更新：2026-04-11 18:02
> 主题：当前 BRPO-style confidence mask 已能跑通，但图像空间 coverage 偏稀；需要先分析“为什么稀、稀到什么程度、下一步该怎么改”，再决定是否继续往 target depth / Stage A / Stage B 推进。

---

## 1. 当前问题一句话

当前 internal route 已经完成：
- `Difix left/right`
- `BRPO-style bidirectional verification`
- `support_left/right/both/single`
- `confidence_mask_brpo_{left,right,fused}`
- `pack → pseudo_cache → refine consumer`

也就是说，**mask 机制本身已经接通**。

但当前问题不是“有没有 mask”，而是：

**这版 BRPO-style mask 在图像空间上偏稀疏。**

所以接下来的关键问题不是继续证明“代码能跑”，而是要分析：
- 这个稀疏性是否是预期内的 BRPO 几何筛选结果；
- 它是否已经稀疏到不适合作为 current refine / target depth 的主监督；
- 如果要继续往 `target_depth_for_refine` 或 Stage A / Stage B 推进，应当怎么处理这种稀疏性。

---

## 2. 当前已经确认的事实

### 2.1 verification 不是坏掉了

当前 verification 不是空转，也不是“几乎没 match 上”。

在 3-frame prototype 上，branch 内部统计是健康的：
- left branch `support_ratio_vs_matches` 可以到 `0.86 ~ 0.94`
- right branch `support_ratio_vs_matches` 也在较高区间
- reprojection / relative depth error 都在可接受范围内

这说明：

**当前 BRPO verification 在“已匹配点内部”的筛选，并没有崩。**

问题不在“几何验证完全失败”，而在于：

**通过几何验证的像素，映射回整张 pseudo 图像后，coverage 仍然偏小。**

### 2.2 当前稀疏性主要体现在图像空间 coverage

当前 3-frame prototype 的平均量级大约是：
- `support_ratio_left ≈ 1.35%`
- `support_ratio_right ≈ 1.00%`
- `support_ratio_both ≈ 0.80%`
- `support_ratio_single ≈ 0.75%`

所以它的问题不是“完全没有可信区域”，而是：

**可信区域在图像空间上太 sparse。**

这和后续任务强相关：
- 如果只把 mask 当成 “RGB loss 的高置信区域加权”，它也许还能用；
- 如果想进一步构造 `BRPO-style target depth`，或者让 Stage A / Stage B 更强地依赖它，这个 coverage 就很可能不够。

---

## 3. 为什么这会变成当前主问题

### 3.1 对 mask-only refine 的影响

前面已经看到：
- 当前 BRPO verification 链是通的；
- 但在 `v1 fixed-pose RGB-only refine` 下，`brpo mask` 还没有优于 `legacy mask`。

一个很自然的解释就是：

**当前 BRPO mask 虽然几何上更“干净”，但它覆盖得太少；在 v1 这种 fixed-pose + RGB-only 的消费方式下，过稀的高置信区域未必比较宽松的 legacy mask 更适合训练。**

### 3.2 对 target depth 的影响

如果后面要把 `target_depth_for_refine` 改成更 BRPO-style 的 depth target，那么它一定会和当前 support / mask 有关系。

这时问题就更敏感：
- 如果只在 `support_both` 上写 depth，可能太 sparse；
- 如果只在少量 verified 区域有 depth，其余全空，那么 Stage A / Stage B 的 depth supervision 会非常脆；
- 如果直接把当前 sparse verified 区域硬当成整图 depth target 主来源，大概率不稳。

所以当前稀疏 mask 问题，不只是“mask 视觉上稀”，而是会直接影响：

**BRPO-style target depth 能不能做、该怎么做。**

---

## 4. 当前最需要分清的判断

现在最重要的是不要把不同问题混在一起。

### 4.1 不能直接得出“BRPO 路线不行”

当前更合理的判断是：
- BRPO verification 机制成立；
- 但其输出更像“高精度、低覆盖”的几何证据；
- 这类信号是否适合直接拿来做 current refine / target depth，还需要进一步分析。

所以当前问题不是：
- `BRPO verification completely failed`

而更像：
- `BRPO verification gives sparse but plausible support; downstream consumption may be mismatched`

### 4.2 不能把 mask 问题和 depth redesign 混为一谈

当前阶段最好把这两个问题拆开：

1. **mask 问题**：
   - 为什么 coverage 稀？
   - 稀疏性来自 matcher / threshold / branch target / view gap，还是 BRPO-style 几何本来就这么严格？

2. **depth 问题**：
   - 如果要做 `target_depth_for_refine`，它应不应该是：
     - 纯 sparse verified depth
     - render-depth fallback
     - mask-aware blended depth

当前不应直接跳到“把 sparse verified depth 硬塞进 Stage A / Stage B”而不先分析 coverage。

---

## 5. 当前最合理的后续分析方向

### 5.1 先分析“为什么稀”

优先应分析：
- `support_both` 稀是因为左右重合本来就小，还是 matcher 覆盖差；
- `support_left/right` 稀是因为阈值太紧，还是 pseudo / ref 本身纹理不足；
- 使用 `Difix left/right` 后，coverage 是提升了，还是只是 cleaner 但仍 sparse；
- 对不同 frame gap（midpoint / tertile / explicit）的 coverage 是否差很多。

### 5.2 评估它适合做什么，不适合做什么

当前更可能的情况是：
- `support_both` 适合作为**最高置信几何区域**；
- `support_left/right` 可以作为次高置信分支；
- 但它们可能都**不适合直接单独撑起整图 depth target**。

所以当前更值得考虑的是：

**mask-aware blended depth target**，而不是纯 sparse BRPO depth。

### 5.3 如果要继续做 target depth，应该怎么想

更稳妥的思路大概率是：
- `support_both`：优先使用双边一致区域的 branch depth
- `support_left/right`：次优先使用单边 branch depth
- unsupported 区域：回退到 `render_depth` 或不监督

也就是说，当前更像要构造：

**BRPO verified depth as correction signal + render depth as fallback**

而不是“用 sparse BRPO verified depth 替换整张图的 target depth”。

---

## 6. 当前结论（供后续工作引用）

当前结论先固定为：

1. **当前 BRPO-style mask 已经做通。**
2. **当前主要问题不是没有 mask，而是 mask coverage 在图像空间上偏稀。**
3. **这种稀疏性会直接影响 target depth 设计。**
4. **因此下一步不能直接把 sparse verified depth 硬塞进 refine；应先分析 coverage，再决定是做纯 sparse depth、还是做 mask-aware blended depth target。**
5. 当前更合理的方向是：
   - 先分析 current mask problem；
   - 再决定 `target_depth_for_refine` 应该如何构造；
   - 然后再推进 Stage A / Stage B 的 depth supervision。

---

## 7. 对后续 agent 的明确提示

如果你接下来要继续推进，请先回答下面三个问题，再决定下一步：

1. 当前稀疏性主要来自哪里：matcher / threshold / overlap / branch target mismatch？
2. 现有 `support_both / left / right / single` 哪些更适合作为 downstream supervision？
3. `target_depth_for_refine` 最终应不应该采用：
   - 纯 sparse verified depth
   - render-depth fallback blended target
   - 只在高置信区域监督、其余区域不使用 depth

在这三个问题没回答清楚前，不建议直接宣称“BRPO-style target depth 已经定义完成”。
