# current-step joint integration / timing 工程落地文档

> 上位文档：`BRPO_A1_B3_reexploration_master_plan.md`
> 目标：让 A1 verifier 与 B3 participation 不再只是 iter `t` 统计、iter `t+1` 生效，而是尽量在 current step 对当前 pseudo optimization 起作用。

---

## 0. 结论先说

timing 不是 B3 文档里的一个附注，而是独立的一条工程主线。

当前 `run_pseudo_refinement_v2.py` 的 StageB 逻辑，本质上是：

- 用上一次的参与控制 render 当前 pseudo；
- 当前 pseudo backward 完后再统计 gating / SPGM；
- 生成下一轮 action。

这让当前系统更像：

> **delayed controller over optimization trajectory**

而不是：

> **current-step operator over current pseudo observation / participation**

所以这条文档的目标不是“再调一下 update 顺序”，而是设计出一个**current-step 可实现的两段式迭代结构**。

---

## 1. 当前代码事实

当前 StageB 的关键顺序是：

1. sample real / pseudo views
2. 用 `current_pseudo_render_mask` render 当前 pseudo views
3. build pseudo loss / total loss
4. backward
5. `maybe_apply_pseudo_local_gating(...)`
6. 从 `gating_summary` 取 `_spgm_participation_render_mask`
7. 存到 `stageB_spgm_participation_render_mask`
8. 下一轮再消费

如果 manager mode 是 `deterministic_participation`，
当前真正生效的是：

- `iter t` 统计
- `iter t+1` 用 mask render

这就是 analysis 里说的 delayed semantics。

---

## 2. timing 主问题到底是什么

### 2.1 对 B3 来说：action 没有作用到它统计的同一轮 residual landscape

当前 B3 统计依赖本轮 pseudo render 的 visibility / support / depth。  
但统计完成后，action 只作用到下一轮。

这意味着：

- 本轮看到的问题，不是本轮解决；
- 如果 pseudo active set 波动快，controller 会容易振荡。

### 2.2 对 A1 来说：如果 verifier 后续也做成 delayed side-output，会继续弱化 observation semantics

A1 verifier 如果只是 signal precompute 当然没问题；
但如果某些 confidence / participation 要依赖 current render diagnostics，
那就不能默认下一轮再生效。

所以 timing 文档要统一回答：

> **哪些量是 offline/precompute 的，哪些量必须 current-step 生效。**

---

## 3. 建议的总体结构：拆成 probe pass 与 loss pass

### 3.1 新的高层结构

建议 StageB pseudo branch 逐步变成：

1. **probe render pass**
   - 当前 iter 先用 full participation 或 lightweight prior participation 做 probe render
   - 只用于统计 verifier / SPGM 所需量

2. **decision pass**
   - 基于 probe render 的结果计算 current-step confidence / participation action

3. **loss render pass**
   - 用 current-step action 重新 render pseudo views
   - 再组当前 iter 的 pseudo loss / total loss / backward

这样 current-step semantics 变成：

> **iter t probe -> iter t decide -> iter t loss render -> iter t backward**

而不是现在的：

> **iter t render -> iter t backward -> iter t decide -> iter t+1 consume**

### 3.2 为什么要分两次 render

因为 current B3 的统计对象本身依赖 visibility / support / depth，
而这些量必须先经过一次 render 才能拿到。

也就是说，如果想让 action 作用到 current step，几乎不可避免需要：

- 先有一个 probe render
- 再有一个 loss render

这不是浪费，而是 current-step semantics 的代价。

---

## 4. 建议拆成三个阶段

## TMG-0：先把 current loop 中的“统计 / 决策 / 生效” API 分开

### 目标

即使暂时仍是 delayed，也先把代码层职责拆清楚。

### 当前问题

`maybe_apply_pseudo_local_gating(...)` 同时做了：

- stats collection
- score building
- manager action
- diagnostics pack

这让后面无论 A1 verifier 还是 B3 participation，都很难独立插入 current-step pass。

### 建议改造

把当前接口拆成三类函数：

1. `collect_*_stats(...)`
2. `build_*_decision(...)`
3. `apply_*_action(...)` 或 `export_*_action(...)`

即便内部先仍然共用旧代码，也要先把 API 边界拆出来。

---

## TMG-1：实现 B3 current-step probe -> loss render

### 目标

先只让 B3 进入 current-step effective，不同时绑 A1。

### 推荐流程

对每个 sampled pseudo view：

1. probe render（无 action 或用 neutral participation）
2. 收集 visibility / support / depth diagnostics
3. 汇总 scene-level SPGM stats
4. 生成 current-step opacity scale
5. loss render（带 opacity scale）
6. build loss / backward

### 代码落点

主要改：

- `scripts/run_pseudo_refinement_v2.py`

次要改：

- `pseudo_branch/local_gating/*`
- `pseudo_branch/spgm/*`
- renderer（支持 opacity scale）

### 注意事项

- probe render 不应污染最终 loss history；
- log 中要显式区分 probe stats vs loss-pass stats；
- 如果 probe / loss pass 对同一 view 都要 render，必须严格复用 sample set，避免额外随机性。

---

## TMG-2：如果需要，再把 A1 current-step verifier 接进来

### 什么时候需要

只有当 A1 verifier 不再完全是 offline precompute，
而是需要 current render diagnostics 时，才推进这一步。

### 原则

- offline verifier 尽量留在 signal builder；
- current-step verifier 只在 analysis 明确要求时引入；
- 不要把所有 verifier 都搬到训练 loop。

这一步的目标是：

- 让 current-step observation confidence 与 current-step participation action 能在同一轮协同生效；
- 但仍避免把 signal pipeline 和 training loop 完全耦死。

---

## 5. 具体代码建议

### 5.1 `run_pseudo_refinement_v2.py`

建议新增明确的阶段函数，例如：

- `run_stageB_pseudo_probe_pass(...)`
- `build_stageB_current_step_actions(...)`
- `run_stageB_pseudo_loss_pass(...)`

不要继续把 probe / decision / loss render 都塞在同一段长循环里。

### 5.2 gating / spgm 接口

建议将当前 `maybe_apply_pseudo_local_gating(...)` 拆成：

- `collect_pseudo_local_gating_stats(...)`
- `build_pseudo_local_gating_decision(...)`
- `export_pseudo_local_gating_action(...)`

这样 delayed mode 和 current-step mode 可以共享同一套 stats / decision 核心。

### 5.3 history / logging

要新增：

- probe-pass diagnostics
- current-step action summary
- loss-pass effective action summary

否则即使 current-step 改好了，也很难判断 action 是否真的作用到了当前 loss landscape。

---

## 6. compare 顺序

### Compare TMG-C0：API split + delayed mode no-regression

先只拆接口，不改行为，确认结果与现有 delayed mode 一致。

### Compare TMG-C1：delayed opacity vs current-step opacity

在 B3 opacity path 已存在的前提下，比：

1. delayed deterministic opacity
2. current-step deterministic opacity

回答问题：

- current-step 生效本身是否带来更稳定或更接近 BRPO 的行为？

### Compare TMG-C2：A1 + B3 current-step joint integration

最后才做 observation 与 participation 的 current-step 协同 compare。

---

## 7. 暂不做的事

1. 不先把 stochastic Bernoulli 接进 current-step；
2. 不在 current-step 改造前就同时替换 observation / participation / topology 三层；
3. 不把所有 precompute signal 都搬进 training loop。

---

## 8. 一句话版执行目标

> **先把 current loop 的 stats / decision / action API 拆开，再用 probe pass + loss pass 的两段式结构，让 B3 participation 真正作用到当前 iter 的 pseudo render。**
