# MASK_DEPTH_CONFIDENCE_PIPELINE_JUDGEMENT_20260414

## 1. 现在的核心判断

基于 `E1 -> E1.5 -> E2` 三轮结果，当前最稳妥的判断不是“这条 pipeline 完全错了”，也不是“再多调一点就会突然转正”，而是：

这条 `mask / depth / confidence` pipeline 已经证明自己能带来真实但有限的上游改善，但它作为主增益来源的边际收益正在快速见顶。

换句话说：
- 它不是伪方向；
- 但它更像一个 **signal filtering / ranking / verification pipeline**；
- 它不太像一个还能持续制造大幅新增 supervision 的主引擎。

## 2. 为什么不是“完全死路”

E1 与 E1.5 说明这条线至少做成了三件真事：

1. `midpoint` 不是稳定最优 pseudo 位置，选点本身确实会改变 signal 质量；
2. `verified_ratio / continuous confidence / agreement` 的改善可以继续传导到 densify；
3. 这种改善虽然不大，但不是假改善，因为它在正式 `fusion -> verify -> pack -> densify -> StageA-20` 链路里仍能保持同方向收益。

这说明 pipeline 不是空转，也不是完全被 evaluation artifact 伪造出来的结果。

## 3. 为什么又说它接近见顶

E2 是最关键的反证。

E2 工程上已经把 multi-pseudo 打通，但方法上没有赢过 E1。它告诉我们：
- 当前最缺的不是 pseudo 数量本身；
- 而是 **高质量 pseudo 的密度不够**；
- 当第二个 pseudo 明显弱于 winner 时，新增样本会把 supervision 从“少量高质量”拉回“更多中等质量”。

这和最初预期不同。原本我们希望“多一个更接近 GT 的 pseudo”会继续增强 signal，但当前事实更像：
- `2/3` 是 winner；
- midpoint 只是次优；
- 把 midpoint 也塞进来后，raw signal 的 aggregate 只轻微变好，densify 与 StageA 反而不如 E1 单 pseudo winner。

因此当前瓶颈不是“mask / confidence 线还没做完”，而是：
**这条线已经越来越像在优化已有弱信号的利用率，而不是创造显著更强的新信号。**

## 4. E2 为什么会比预期差

E2 的负结果不是随机波动，更像两层机制叠加：

### 4.1 upstream 层：第二个 pseudo 没有第一名强

E2 的 top2 实际稳定落成 `1/2 + 2/3`，不是 `1/3 + 2/3`。
其中第一名（基本就是 E1 winner）仍然最好；第二名（多数是 midpoint）虽然不差，但明显更接近 midpoint 层级，而不是新的强 winner。

这意味着 E2 增加的不是“更多强信号”，而是“强信号 + 中等信号”。

### 4.2 consumer 层：StageA 是随机抽样 4 个 pseudo，而不是每轮都吃满 16 个

当前 `run_pseudo_refinement_v2.py` 的 StageA 默认 `num_pseudo_views=4`，每轮从全部 pseudo 中随机抽 4 个。
因此从 8 pseudo 扩到 16 pseudo 后：
- strong winner 被采样到的概率下降；
- second-tier pseudo 占据了一部分训练曝光；
- 结果不是“每轮额外获得 8 个好监督”，而是“同样 4 个名额里混入更多次优样本”。

这会自然导致 short refine 层比 E1 更差。

所以 E2 的结果并不说明“2/3 靠近 GT 这个直觉错了”，而是说明：
**在当前 consumer 机制下，增加次优 pseudo 会稀释 winner，而不是自动叠加收益。**

## 5. 对 E3（multi-anchor verify）的判断

你说得对，anchor 更偏向增强 verify 鲁棒性，而不是直接创造 signal。这正是当前对 E3 的正确预期。

因此 E3 的价值不在于“重新打开大幅提升空间”，而在于回答一个最后的关键问题：

如果我们固定使用 E1 已通过正式对照的 winner pseudo set，再增强 verify 侧的 anchor 语义，是否还能把 winner signal 再往上提一截？

也就是说，E3 更像：
- 对当前 pipeline 上限的一次最终 probing；
- 而不是一个高置信度的大增益来源。

## 6. 我现在对这条 pipeline 的总体结论

当前最合理的定位是：

1. `mask / depth / confidence` pipeline 仍然值得保留；
2. 但它应被视为 **上游 signal gate / ranking / verification 层**；
3. 不应再把它当成“只要继续加 pseudo、加 mask 语义、加 densify 规则，就能持续把结果往上推”的主线；
4. 它的主增益窗口很可能已经被 E1 基本吃到了。

这不是“完全死胡同”，而是“这条支线已经接近可验证上限”。

## 7. 是否已经到了该止损的时候

我不建议现在立刻宣布这条线彻底停止，因为还有一个合理但有限的最后测试：
- 以 E1 `signal-aware-8` 为 winner 基底；
- 做 E3 `nearest2 -> nearest4 / nearest2_plus_context`；
- 明确把它当作一次 **final probe**，而不是新长期主线。

如果 E3 在这个最优基底上：
- 仍然只带来极小 raw/dense 提升；
- 且 StageA short compare 仍不优于 E1；

那么就可以比较有把握地收敛成结论：

**这条 pipeline 已经完成了它能完成的大部分价值，后续不值得继续作为主线重投入。**

## 8. 如果 E3 也不成立，意味着什么

那不代表项目没路了，而是意味着：
- 问题主矛盾不再是 `mask / depth / confidence` 这条线内部；
- 而更可能是更高层的结构问题：
  - weak local supervision 对 global refinement 的失配；
  - consumer 训练对象过大；
  - pseudo-side 局部改善难以转成下游真实价值。

到那一步，继续在这条 pipeline 内卷，只会得到更多“有一点点改善，但没有实质改变”的结果。

## 9. 现在的执行建议

1. 当前 default winner 仍然保持 E1 `signal-aware-8`；
2. E2 不应升为默认方案；
3. 若继续推进，E3 应仅基于 E1 winner 做一次 final probe；
4. 若 E3 仍不成立，就应正式给这条 pipeline 收线，把精力转向更结构性的 consumer / scope / optimization mismatch 问题。

一句话压缩：

**这条 pipeline 不是伪方向，但它大概率已经接近上限；E3 值得做，但应被视为最后一次证伪测试，而不是继续加码投入的开始。**
