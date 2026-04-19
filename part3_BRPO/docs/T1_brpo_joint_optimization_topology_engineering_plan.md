# T1 工程落地方案：BRPO-style Joint Optimization Topology Rewrite

## 1. 文档目标
本文件对应 `BRPO_alignment_unified_RGBD_and_scene_SPGM_plan.md` 中关于 optimization topology 的差距分析，以及补充条目 `0.5.4` 的核心问题。

这次不再讨论“在现有两阶段结构上还能再加什么小修补”，而是直接回答：

> **如果真的要把 pseudo frame 更像 BRPO 那样插入 joint optimization，当前 `StageA -> StageA.5 -> StageB` 的拓扑该怎么改。**

一句话目标：

> **把“pseudo-only micro-tune 再接 joint refine”的两阶段主线，改写成“pseudo + real 在同一个 joint loop 里连续发生、local gating 只控制 pseudo scope 而不再单独切一阶段”的 BRPO-style joint optimization topology。**

---

## 2. 先说清楚：旧拓扑到底哪里不对
### 2.1 旧拓扑是什么
当前主线拓扑是：

```text
StageA (pose/exposure only)
→ StageA.5 (local Gaussian micro-tune, pseudo-only backward)
→ StageB (joint refine: real anchor + pseudo)
```

其中：
- StageA.5 的作用是让 pseudo signal 先在受限 Gaussian scope 上单独写入一轮
- StageB 再把 real 和 pseudo 放到 joint refine 里继续优化

### 2.2 旧拓扑的问题不在“逻辑错”，而在“信号太弱时会被结构稀释”
这个结构在防止 pseudo 污染 real branch 时是合理的，但它有一个明确代价：

1. pseudo signal 先单独跑一段，而且只在 local scope 内 backward
2. 如果 pseudo observation 本来就弱，这段累积出来的结构变化会很小
3. 到 StageB 时，real anchor 与 bounded schedule 很容易把这点变化抹平或稀释
4. 于是系统会表现成：
   - StageA.5 不够强
   - StageB 也看不出明显承接收益
   - 进一步会诱导我们去调 SPGM / opacity / selector 这些后层动作

换句话说，当前拓扑最大的问题不是“StageA.5 一定错”，而是：

> **当 pseudo supervision 本来就弱时，把它单独放在 StageA.5 里跑一小段，再交给 StageB 接着跑，会让信号在结构上传递得过于脆弱。**

### 2.3 这和 BRPO 的差异是什么
更接近 BRPO 的方法语义不是：
- 先 pseudo-only 做一段
- 再 joint refine 做一段

而是：

> **把 pseudo frame 当正常训练帧插进 joint optimization，让 pseudo residual 和 real residual 在同一个训练循环里连续作用。**

差异核心不在“是不是也有 confidence mask”，而在：
- 当前拓扑：**弱信号先单独写，再被 joint 阶段继承**
- BRPO-like 拓扑：**弱信号直接和 real signal 在同一 loop 内共同决定更新方向**

---

## 3. 这次真正要改什么
这次不是继续加强 StageA.5，而是重新定义它在主线里的地位。

### 3.1 旧 StageA.5 降级
旧 StageA.5 的定位改为：

> **可选的局部预热 / 诊断阶段，不再是 BRPO 对齐主线的必要组成部分。**

也就是说：
- 它可以保留，作为老对照或 debug 工具
- 但新版主线不能再依赖“先跑一段 pseudo-only，再看 joint 有没有承接”

### 3.2 新主线
新版 topology 的主线应改成：

```text
StageA (pose/exposure only)
→ Joint Refine Topology v1 (real + pseudo in one loop, local gating only scopes pseudo contribution)
→ replay / compare
```

这里不再单独设一个必须经过的 StageA.5 主阶段。

### 3.3 local gating / SPGM 在新拓扑里的角色
新版 topology 不是把 local gating / SPGM 废掉，而是给它们重新定位：

- **不再作为“先跑一段 pseudo-only”的阶段边界工具**
- 而是作为 **joint loop 内 pseudo branch 的 scope controller**

也就是说：
- pseudo branch 仍然可以只作用于 local Gaussian subset
- 但这个作用发生在 **joint optimization 同一轮循环里**
- real branch 同轮也在提供 anchor / regularization

这才是“pseudo 和 real 在同一 loop 内相互约束”的关键。

---

## 4. 新版拓扑的设计原则（必须严格遵守）
### 4.1 不再让 pseudo-only backward 成为主线必经阶段
新版主线必须允许直接：
- 加载 StageA 输出状态
- 进入 joint loop
- 在 joint loop 内同时处理 real / pseudo

### 4.2 local gating 只控制 pseudo scope，不再定义独立训练阶段
local gating / SPGM 可以保留，但它们只负责回答：

> **当前这一轮 joint refine 中，pseudo supervision 应该作用到哪些 Gaussian。**

而不是回答：

> **是否应该先单独开一个 pseudo-only 阶段。**

### 4.3 pseudo 和 real 的 loss 必须在同一 loop 中共同决定更新
这不是说每一步都必须严格 1:1 混合，而是说：
- optimizer step 应同时看到 real branch 与 pseudo branch 的贡献
- pseudo 不再依赖“先前阶段积累的结构变化”才能在 joint 阶段有效

### 4.4 允许 warmup，但不允许 warmup 偷偷长成主线
如果为了稳定性需要，可以有：
- very short pseudo-warmup
- very short topology transition warmup

但必须满足：
- warmup 是辅助
- compare 主体必须是 joint loop
- 不允许又回到“StageA.5 才是真正承载 pseudo 更新的地方”

### 4.5 topology 验证要优先于 B3 opacity / stochastic
在新版 topology 没验证前，不继续把主要精力投到：
- B3 opacity decay
- stochastic drop
- 更复杂的 population control

因为这些动作都是建立在当前 topology 不错的前提上，而 0.5.4 正是在质疑这个前提。

---

## 5. 对当前代码事实的重新判断
### 5.1 当前 `run_pseudo_refinement_v2.py` 的问题不是功能不够，而是阶段边界太硬
当前脚本已经能：
- 处理 StageA
- 处理 StageA.5
- 处理 StageB
- 在 pseudo branch 后做 local gating / SPGM

问题不在于能力缺少，而在于默认训练流程仍然是“阶段串联”。

如果真要靠近 BRPO，脚本应该新增一种明确模式：

> **joint topology mode**

在这个 mode 下：
- 不需要先跑完 StageA.5 才能开始 joint refine
- pseudo sampling / pseudo loss / local gating 被嵌进 StageB-like joint loop 内

### 5.2 当前 StageB 的问题不是 joint 不存在，而是 pseudo 还不是主循环内的一等公民
当前 StageB 虽然已经同时算 real 和 pseudo，但从拓扑语义上，pseudo 仍然像：
- 从前一阶段带着“先写过一轮”的状态过来
- 在本阶段继续附着在 real refine 上

BRPO-like topology 要求的是：
- pseudo branch 从一开始就是 joint loop 的原生成员
- 它不是 StageA.5 的余波，而是当前 loop 的组成部分

### 5.3 这意味着需要一个新的 topology mode，而不是继续 patch StageA.5 参数
继续调：
- StageA.5 iter
- StageA.5 lr
- StageA.5 gating 范围

都不能解决 topology 是否合理这个问题。那只是在旧拓扑里继续挤牙膏。

---

## 6. 新版拓扑的具体定义
## 6.1 新 mode 名称
建议在 `run_pseudo_refinement_v2.py` 中新增：

```text
--joint_topology_mode brpo_joint_v1
```

该 mode 一旦打开：
- 默认不走旧主线 `StageA -> StageA.5 -> StageB`
- 而是走：

```text
StageA output
→ joint topology refine loop
→ replay / compare
```

### 6.2 joint topology refine loop 的语义
每个 iteration 统一执行：

```text
sample real views
sample pseudo views
forward real branch
forward pseudo branch
apply pseudo local gating / SPGM scope control
assemble joint loss
backward once
optimizer.step once
```

关键点：
- pseudo 和 real 在同一 iter 里共同作用
- gating 只控制 pseudo branch 的 Gaussian scope
- optimizer 看到的是 joint loop 的 combined effect

### 6.3 这和旧 StageB 的区别
区别不是“StageB 也 joint 吗”，而是：
- 旧结构里 joint refine 是在 StageA.5 之后接手
- 新结构里 joint refine 本身就是 pseudo 更新的主发生场所

也就是说，新版 joint loop 必须能在**没有 StageA.5 预热**的前提下直接跑通并形成可解释差异。

---

## 7. local gating / SPGM 在新 topology 里怎么放
### 7.1 旧语义（要降级）
旧语义：
- local gating 帮 StageA.5 限制 pseudo-only backward 的作用范围
- SPGM 也是在 pseudo branch 的单独阶段里做 grad modulation

### 7.2 新语义（主线）
新语义：
- local gating / SPGM 只控制 **joint loop 内 pseudo branch 的优化 scope**
- 它们不再定义“是否需要先单独来一段 pseudo-only 阶段”

### 7.3 一个更清晰的公式化理解
新版 topology 下，每轮更接近：

\[
L_{total}^{(t)} = \lambda_r L_{real}^{(t)} + \lambda_p L_{pseudo}^{(t)}
\]

其中 `L_pseudo` 不是全局无约束地写回，而是在 local gating / SPGM 约束下只影响局部 Gaussian subset。

也就是说：
- **scope control 还在**
- 但 **topology 上的阶段切分没了**

---

## 8. 需要新增 / 修改的文件
### 8.1 修改
1. `scripts/run_pseudo_refinement_v2.py`
   - 新增 `joint_topology_mode=brpo_joint_v1`
   - 新 joint loop
   - 保证在 joint mode 下绕过旧 StageA.5 主流程

2. `pseudo_branch/local_gating/*`
   - 检查是否有逻辑默认依赖“pseudo-only 阶段”
   - 必要时改成“joint loop 中的 pseudo scope control”

3. `pseudo_branch/spgm/*`
   - 保留现有接口，但调用位置要适配 joint loop
   - 如果新版 B3 population manager 开始接入，也应接在 joint topology 下工作

### 8.2 可选新增
1. `docs/T1_brpo_joint_optimization_topology_engineering_plan.md`（本文件）
2. 如有必要，新增小的 helper：
   - `pseudo_branch/joint_topology.py`
   - 用来组织 real/pseudo sampling、loss assemble、scope control

---

## 9. 实施步骤（严格按这个顺序）
### Step 1：冻结旧 StageA.5 作为主线的地位
- StageA.5 保留，但只作为对照臂 / 诊断路径
- 新 compare 不再默认“先跑 StageA.5 再 StageB”

### Step 2：在 `run_pseudo_refinement_v2.py` 中加 `brpo_joint_v1` mode
- joint mode 下从 StageA 输出直接进入 unified joint loop
- 不再依赖 StageA.5 输出的 Gaussian 变化作为前置条件

### Step 3：把 local gating / SPGM 挪到 joint loop 内使用
- pseudo branch 依然可以在 backward 前后做 scope control
- 但这些动作发生在同一 joint iteration 内
- real branch 同轮给 anchor

### Step 4：做机制 smoke
第一轮 smoke 的重点不是指标，而是证明 topology 真的变了：
1. 运行时不经过旧 StageA.5 主流程
2. joint loop 内真实同时处理 real / pseudo
3. local gating / SPGM 仍真实生效
4. optimizer.step 看到的是 joint loss，而不是先 pseudo-only 再 joint

### Step 5：做 topology compare
第一轮 compare 至少三臂：
1. old topology：`StageA -> StageA.5 -> StageB`
2. new topology：`StageA -> brpo_joint_v1 joint loop`
3. new topology + old A1 / new A1（视实现进度选择）

这轮 compare 的目的不是“立刻赢很多”，而是验证：

> **当 pseudo 与 real 在同一 loop 内共同积累时，是否比两阶段串联更稳定、更不容易把弱 pseudo signal 稀释掉。**

---

## 10. 验收标准
### 10.1 方法验收（优先于结果）
只有满足以下条件，才算真正开始靠近 BRPO 的 topology：
1. pseudo-only StageA.5 不再是主线必经阶段
2. pseudo 和 real 在同一 iteration 内共同决定 optimizer.step
3. local gating / SPGM 只扮演 scope controller，而不是阶段边界工具
4. compare 中 new topology 真正绕过旧两阶段主线

### 10.2 结果验收
在方法验收通过之后，再看：
1. 相比 old topology，训练侧至少出现更清晰的 pseudo 信号承接
2. replay 不应更差太多
3. 如果 new topology 比 old topology 更稳，那么说明当前更大的问题在 topology，而不是 B3 细节

---

## 11. 明确不做的事
1. 不继续在旧 StageA.5 上通过扫 iter/lr 伪装成 topology 改进
2. 不把 topology 问题继续误判成 SPGM 细节问题
3. 不在新版 topology 尚未验证前继续主攻 B3 opacity / stochastic
4. 不把“joint loop”写成名字，实际上还偷偷依赖 StageA.5 的承接

---

## 12. 第一轮执行优先级（重启后直接照这个做）
1. **先在 `run_pseudo_refinement_v2.py` 写出 `brpo_joint_v1` mode**
2. **先证明能绕过旧 StageA.5 主线直接跑 joint loop**
3. **先做 topology smoke，再做 topology compare**
4. **在 topology compare 结果出来前，冻结对 B3 opacity / stochastic 的主线投入**

---

## 13. 最后压成一句话
旧拓扑做的是：

> **先让 pseudo signal 在 StageA.5 的 pseudo-only backward 里尝试积累，再把这点变化交给 StageB 继续 joint refine。**

新版拓扑要做的是：

> **让 pseudo 和 real 从一开始就在同一个 joint optimization loop 内共同决定更新，而 local gating / SPGM 只负责控制 pseudo signal 作用到哪些 Gaussian，而不再单独切出一个 pseudo-only 主阶段。**
