# B3 工程落地方案：BRPO-style Scene-level Gaussian Population Manager Rewrite

## 1. 文档地位
本文件**覆盖**旧版 `B3_deterministic_state_management_engineering_plan.md`。旧版 B3 的方法地位降级为：

> **deterministic grad/state probe**

也就是：旧版 B3 只是在已有 pseudo backward 之后，对 active Gaussian 的 `_xyz.grad` 再乘一层 state-dependent scale，最多算“比 repair A 更进一步的 suppressor”。它不是 BRPO 语义里的 population manager。

新版 B3 的唯一目标是：

> **把 SPGM 从“这一轮梯度怎么乘”改写成“这一轮哪些 Gaussian 应该更少参与 pseudo rendering / pseudo update”的 scene-level population control。**

一句话目标：**不再把 `xyz_lr_scale` 当主线，而是直接把 B3 重写成面向 opacity participation / stochastic masking 的 BRPO-style population manager。**

---

## 2. 先说清楚：旧 B1/B2/B3 到底做成了什么、没做成什么
这次不再自欺欺人。

### 2.1 B1/B2/B3 已经做成的部分
- B1 把 SPGM 结构拆成了 `stats -> score -> update policy -> state management -> step`
- B2 把 `ranking_score` 与 `state_score` 分开，并补了 cluster-level diagnostics
- 旧 B3 证明了 manager 可以真实改变 `_xyz.grad`

这些成果都有效，但它们的意义是：

> **我们把“如何给 pseudo branch 的 Gaussian 更新打分/加权/记日志”这件事写清楚了。**

### 2.2 B1/B2/B3 没做成的部分
它们没有解决 BRPO 在 SPGM 上最核心的问题：

> **SPGM 是否真的决定了 Gaussian population 的参与方式，而不是只决定了这一步 backward 的梯度强弱。**

当前 live code 的方法语义仍然是：
1. 先完成 pseudo render
2. 先完成 pseudo backward
3. 再对 active Gaussian 的 gradient 乘权重 / scale
4. 再 optimizer.step

这说明当前 SPGM 主要还是：

> **post-backward update modulator**

而不是：

> **pre-render / in-render population controller**

只要动作发生在 pseudo backward 之后，而且只改 gradient / xyz，不改 opacity participation，那么它就还没有进入 BRPO 的语义核心。

### 2.3 旧 B3 为什么必须降级
旧 B3 的问题不是“做得太保守”这么简单，而是方法对象就没换：
- `state_candidate_quantile` 当前几乎只用于统计，不真正决定 action 子集
- `xyz_lr_scale` 作用在所有 active Gaussian，而不是 BRPO 式的 low-score population subset
- action 发生在 backward 之后，不改变本轮 pseudo rendering 中哪些 Gaussian 真正参与成像
- opacity 不动，render participation 不动，population structure 不动

所以旧 B3 的正确定位只能是：

> **为 population manager rewrite 做诊断性铺路，而不是把 population manager 本身做出来。**

---

## 3. 这次真正要对齐 BRPO 的是什么
这次不是继续研究“state score 如何更好地乘到 grad 上”，而是直接对齐下面这个问题：

> **SPGM 应不应该直接控制低质量 Gaussian 在 pseudo branch 中的参与概率 / 参与强度，而不是只在渲染结束后给它们乘更小的梯度。**

BRPO 方法里，SPGM 更接近：
1. 基于场景级统计给 Gaussian 打分
2. 把低质量、低可信、低优先级的 Gaussian 视为 population subset
3. 通过 opacity attenuation / masking 让这些 Gaussian 在 pseudo optimization 中更少参与 rendering 与 update
4. 因而改变的是“谁在本轮 pseudo supervision 中进入了成像与误差传播”

这和当前旧 B3 的根本差异在于：
- 旧 B3 改的是 **post-backward gradient amplitude**
- 新 B3 要改的是 **pre-render / in-render participation semantics**

换句话说，旧 B3 在回答：

> “这一轮梯度怎么乘？”

新版 B3 要回答：

> “这一轮哪些 Gaussian 应该更少被 pseudo observation 看见、参与渲染和接收 pseudo supervision？”

---

## 4. 新版 B3 的设计原则（必须严格遵守）

### 4.1 state action 必须作用到 rendering participation，而不是只作用到 backward 之后
只要 action 只发生在 backward 之后，它就仍然只是 update modulator，不是 population manager。

### 4.2 action 子集必须真实由 candidate mask 决定
`candidate_quantile` / low-score subset 必须真正影响 action，对象不能再是“所有 active Gaussian 一起轻微缩放”。

### 4.3 manager 的输入必须更接近 current train window population，而不是 accepted pseudo active subset 的后处理
`state_score` 必须面向 population，而不是只面向一轮 accepted pseudo support 的残余影子。

### 4.4 opacity / participation 是第一主轴，xyz grad scaling 降为辅助对照
因为 BRPO 语义里真正接近 population control 的是 opacity participation，而不是 xyz grad 的二次缩放。

### 4.5 stochastic 是主线终点，不再把它当“遥远以后再说”的附属品
但执行上仍然分两步：
- 先做 deterministic participation attenuation，验证 plumbing 和方向
- 再进入 Bernoulli masking

这不是保守拖延，而是为了把随机性和 ranking 误差拆开验收。

---

## 5. 对当前代码事实的重新判断

### 5.1 当前 `stats.py` 的问题
`collect_spgm_stats(...)` 当前虽然已经加入 `population_support_count`，但核心 active mask 仍然与 accepted pseudo support 强绑定，仍带有“这轮 pseudo 看到了谁”的后验色彩。

如果要做 population manager，新版 stats 的中心对象应该是：

> **current train window / current scene-local population**

而不是“accepted pseudo views 的 union visibility 再做一点扩展”。

### 5.2 当前 `score.py` 的问题
`score.py` 现在已经能输出 `weight_score / ranking_score / state_score`，这是对的；但当前 `state_score` 还只是“给 manager 做后续动作的一个连续指标”。

如果要靠近 BRPO，`state_score` 必须直接进入：
- candidate subset 构建
- opacity attenuation / masking probability
- repeated-low-score persistence logic

也就是说，`state_score` 不能只被拿来生成 summary 和 mild scale，它必须进入“谁在 forward 里被弱化”的主决策链。

### 5.3 当前 `policy.py` 的问题
`selector_quantile` 当前仍属于 **update selection**，而不是 **population control**。

它回答的是：
- 哪些 Gaussian 的 gradient 要保留、削弱、删除

但 BRPO-style manager 需要回答的是：
- 哪些 Gaussian 在 pseudo render 中就应该更少参与

所以 `policy.py` 可以保留，但它必须被降级为 manager 的一个子模块，而不是被继续误当成完整 SPGM。

### 5.4 当前 `manager.py` 的问题
当前 manager 最大的问题不是弱，而是动作位置不对：
- 它在 pseudo backward 之后行动
- 它只改 `_xyz.grad`
- `opacity_decay_*` 目前只是 history 字段，没有真实语义闭环

所以新版 manager 必须从：

> **post-backward gradient scaler**

改成：

> **pre-render / branch-scoped opacity participation controller**

---

## 6. 新版 B3 的方法定义：两层角色，主轴换成 population control

### 6.1 仍保留的层：Update Permission Layer
这层保留旧成果，但不再当主角。

职责：
- `ranking_score`
- `selector_quantile`
- `grad weighting`
- `weight_floor / cluster_keep`

它回答的是：

> **这一轮 pseudo update，谁允许被更新、更新多少。**

### 6.2 新主角：Population Participation Layer
这层是新版 B3 的真正主体。

职责：
- 根据 `state_score` / cluster / density / support 生成 low-score population subset
- 对该 subset 的 opacity participation 做 deterministic attenuation 或 stochastic masking
- 让这些 Gaussian 在 pseudo rendering 中更少参与，而不是 render 完再缩梯度

它回答的是：

> **这一轮 pseudo rendering，谁应当更少出现在成像里。**

### 6.3 这层应输出什么
新版 manager 至少要输出：
1. `population_candidate_mask`
2. `participation_scale`（deterministic）
3. `drop_prob`（stochastic）
4. `applied_mask` / `sampled_mask`
5. cluster-wise / source-wise history

而不是只输出：
- `xyz_state_scale`
- `opacity_state_decay` 这种仍停留在“step 前处理”的接口

---

## 7. 新版 manager 应怎么接入代码流

### 7.1 旧代码流（必须放弃作为主线）
当前逻辑：

```text
pseudo backward
→ collect_spgm_stats
→ build_spgm_importance_score
→ build_spgm_update_policy
→ apply_gaussian_grad_mask
→ apply_spgm_state_management
→ optimizer.step
```

这条线的问题是：action 太晚，render participation 已经发生完了。

### 7.2 新代码流（新版主线）
新版 B3 应改成：

```text
iter t start
→ read manager persistent state from last iter (or warmup default)
→ apply population participation control to pseudo branch render only
→ pseudo render / pseudo loss forward
→ pseudo backward
→ optional grad weighting / selector policy
→ optimizer.step
→ collect current-window population stats
→ build ranking/state score
→ update manager state for iter t+1
```

关键变化：
- **manager action 先于 pseudo render**
- 当前迭代的 stats 用于更新下一迭代的 population state
- population control 不是在本轮 render 之后救火，而是影响下一轮谁会被看见

### 7.3 为什么要引入“跨迭代 manager state”
因为要想在 render 前做 action，你不能依赖本轮 render 才知道的 stats。解决办法是：
- manager 在 iter t 结束时写出 `population_state_cache`
- iter t+1 开始时读取并应用

这与 BRPO 的 during-training population control 更接近，也比“本轮 backward 后缩梯度”更像真实 state manager。

---

## 8. 具体落地：从 deterministic opacity attenuation 到 stochastic Bernoulli masking

## 8.1 Phase B3-R1：deterministic branch-scoped opacity participation attenuation
这一步是新版 B3 的第一阶段，也是必须做的过渡层。

### 目标
不是再做 xyz grad scale，而是：

> **在 pseudo branch render 之前，只对 low-score candidate subset 的 opacity 做临时衰减。**

### 方法要求
1. 只作用于 pseudo branch，不改 real branch render
2. 只作用于 selected candidate subset，不是全 active set
3. 使用 branch-scoped temporary attenuation：
   - 进入 pseudo render 前：`opacity_eff = opacity * participation_scale`
   - pseudo render 完 / step 后：恢复原 opacity 参数本体
4. attenuation 不持久写死到参数中，第一版先做 temporary control

### 为什么这一版比旧 B3 更接近 BRPO
因为它改的是：
- 本轮 pseudo render 里哪些 Gaussian 更少出现在成像中

而不是：
- render 完以后它们的 gradient 被再缩一下

### 第一版建议的 candidate 规则
candidate subset 由以下条件共同决定：
1. `population_active_mask` 内
2. 属于 far / optionally mid cluster
3. `state_score` 落在 cluster 内底部 quantile
4. 可选：`ranking_score` 也低于门槛

这意味着 action 必须明确落在：

> **far + low-state-score + low-priority 的 population subset**

而不是“所有 active Gaussian 的 mild scale”。

### 第一版建议的 participation scale
- near: 1.0
- mid candidate: 0.95 ~ 0.98
- far candidate: 0.85 ~ 0.95
- non-candidate: 1.0

注意：这不是继续做 `xyz_lr_scale`，而是直接做 `opacity_participation_scale`。

---

## 8.2 Phase B3-R2：stochastic Bernoulli opacity masking
这是新版 B3 的真正 BRPO 对齐阶段。

### 目标
对 low-score population subset，不再只是 deterministic attenuation，而是：

\[
m_i \sim Bernoulli(1 - p_i^{drop})
\]
\[
\alpha_i^{eff} = \alpha_i \cdot m_i
\]

其中 `m_i` 只作用于 pseudo branch render。

### 第一版原则
1. 只在 pseudo branch 里采样，不影响 real branch
2. 只对 candidate subset 采样，非 candidate 恒为 1
3. `drop_prob` 由 cluster prior 和 `state_score` 决定
4. 采样 mask 必须写 history，保证可追溯
5. 不做硬 prune，不做参数永久删除

### `drop_prob` 的建议形式
可先从保守版本开始：

\[
p_i^{drop} = r \cdot w_{cluster(i)} \cdot (1 - state\_score_i)
\]

其中：
- `r` 是 global drop rate，起步很小
- `w_cluster`：far > mid > near
- `1 - state_score`：低分更容易 drop

### 为什么这一步才是更像 BRPO 的核心
因为它终于把 SPGM 从：
- deterministic suppressor

推进成：
- scene-level stochastic population controller

---

## 9. 需要新增 / 重写的文件

### 9.1 新增
1. `pseudo_branch/spgm/population_manager.py`
   - 新版主 manager，不建议继续在旧 `manager.py` 上硬堆
2. `pseudo_branch/spgm/population_state.py`
   - 持久化当前 iter → 下一 iter 的 manager state cache
3. 可选：`pseudo_branch/spgm/opacity_control.py`
   - 负责 branch-scoped temporary opacity attenuation / masking

### 9.2 修改
1. `scripts/run_pseudo_refinement_v2.py`
   - 重写 pseudo branch render 前后的 SPGM 接入位置
2. `pseudo_branch/spgm/stats.py`
   - 更明确输出 population-level stats / active mask
3. `pseudo_branch/spgm/score.py`
   - state score 进入 drop/participation 决策
4. `pseudo_branch/spgm/policy.py`
   - 降级为 update permission 子层
5. 视需要修改 Gaussian render / temporary parameter patching 相关 helper

### 9.3 旧文件的处理原则
- `manager.py` 保留为旧 B3 对照实现
- `xyz_lr_scale` 保留为对照臂，不再是新主线

---

## 10. 新版 stats / score / manager 的职责划分

### 10.1 `stats.py`
必须输出更明确的 population semantics：
- `population_active_mask`
- `population_support_count`
- `population_visibility_count`
- `population_depth_value`
- `population_density_proxy`
- `population_struct_density`

active mask 不能再主要由 accepted pseudo subset 决定。

### 10.2 `score.py`
必须输出三类不同分数：
1. `update_score`：给 grad weighting / selector 用
2. `state_score`：给 participation / drop probability 用
3. `stability_score`（可选）: 给 persistence / repeated-low-score 用

要求：
- `state_score` 直接进入 population action，不再只是日志摘要

### 10.3 `population_manager.py`
必须包含两类函数：
1. `build_population_action(...)`
   - 生成 candidate subset
   - 生成 deterministic participation scale 或 stochastic drop prob
2. `apply_population_action_for_pseudo_branch(...)`
   - 在 pseudo render 前临时修改 effective opacity / participation mask
   - 在 pseudo branch 结束后恢复

---

## 11. 实施步骤（严格按这个顺序）

### Step 1：冻结旧 B3，不再继续扫 `xyz_lr_scale`
- 旧 B3 保留为 negative / diagnostic baseline
- 不再继续补它的 opacity decay 版本

### Step 2：先把 manager state 改成跨迭代 persistent cache
- 写 `population_state.py`
- 确保 iter t 结束时能更新 state，iter t+1 开始时能读取

### Step 3：实现 deterministic branch-scoped opacity attenuation
- 在 pseudo render 前临时衰减 selected candidate subset 的 effective opacity
- 完成 restore 机制
- 记录 history

### Step 4：完成第一轮机制 smoke
第一轮 smoke 的验收重点不是 replay，而是：
1. 当前 iter 的 pseudo render 真的使用了 modified effective opacity
2. real branch 不受影响
3. candidate subset 真正控制 action 对象
4. action 不是全 active set 平均缩放

### Step 5：最小 compare（deterministic participation vs old B3 vs control）
至少三臂：
1. control：repair A / summary-only
2. old B3：`xyz_lr_scale`
3. new B3-R1：deterministic pseudo-branch opacity participation attenuation

比较重点：
- replay 是否比 old B3 更稳
- history 中 participation object 是否更可解释
- 训练侧是否出现“更像结构筛选而不是纯 suppress”的信号

### Step 6：再进入 stochastic masking
- 先加 `drop_prob`
- 再加 Bernoulli mask
- 再做小 compare

---

## 12. 验收标准

### 12.1 方法验收（比结果更重要）
只有满足以下条件，才算真正开始靠近 BRPO 的 SPGM：
1. manager action 在 pseudo render 前发生，而不是 pseudo backward 后
2. action 对象是 candidate subset，不是全 active set 平均缩放
3. action 直接改变 effective opacity / participation，而不是只改 `_xyz.grad`
4. manager 拥有跨迭代 state cache
5. stochastic 版本的 mask / drop prob 可追溯

### 12.2 结果验收
在方法验收通过之后，再看：
1. 至少优于 old B3 的 replay / stability 表现
2. 不能通过强 suppress 换来表面更小 loss
3. 如果 deterministic participation 已显著负，则先回看 state score / candidate rule，而不是直接否定 population-control 方向

---

## 13. 明确不做的事
1. 不再把 `_xyz.grad` scaling 继续包装成 population manager 主线
2. 不再把 `candidate_quantile` 只当日志统计
3. 不继续在旧 `manager.py` 上堆更多 history 字段伪装“更像 BRPO”
4. 不在新版 B3 尚未改成 pre-render action 之前继续扫 opacity 微调参数
5. 不把 selector-first 误当成完整 SPGM

---

## 14. 第一轮执行优先级（重启后直接照这个做）
1. **先写跨迭代 `population_state` + pre-render action plumbing**
2. **先做 deterministic pseudo-branch opacity participation attenuation，不做 xyz grad scale**
3. **先证明 action 改的是 render participation，不是 backward 后救火**
4. **compare 必须带 old B3，对照“post-backward suppressor”和“pre-render population control”到底差在哪**
5. **deterministic 站住后，再进 stochastic Bernoulli masking**

---

## 15. 最后压成一句话
旧 B3 做的是：

> **在 pseudo backward 之后，对 active Gaussian 的 gradient 再乘一个 state-dependent scale。**

新版 B3 要做的是：

> **在 pseudo rendering 之前，根据 scene-level population score 决定哪些 Gaussian 在这一轮应被 opacity attenuation / stochastic masking，从而直接控制它们参与 pseudo observation 的概率与强度。**
