# B1 工程落地方案：SPGM Manager Shell

## 1. 目标
把当前 `SPGM = stats + score + grad weighting` 的单层实现，拆成：
1. update policy layer
2. state management layer

但第一版只搭结构和日志，不直接改 Gaussian state。

一句话目标：先把“SPGM 不只是 selector / weighting”这件事在代码结构上成立，再决定是否真的动参数。

## 2. 当前代码事实
当前 live call chain 已确认是：
1. `collect_spgm_stats(...)`
2. `build_spgm_importance_score(...)`
3. `build_spgm_grad_weights(...)`
4. `apply_gaussian_grad_mask(...)`
5. `optimizer.step()`

也就是说，SPGM 目前仍然只在 pseudo backward 之后修改 grad，没有 state-level manager。

## 3. 理论判断
B1 不是为立刻提高指标，而是为后续 B2/B3 提供清晰边界：
- `policy.py` 负责“谁被选中、梯度怎么乘”
- `manager.py` 负责“Gaussian state 在场景层面应如何被保守管理”

如果不先拆结构，后面一旦加 per-cluster lr、opacity decay、state score，就会继续堆在 `run_pseudo_refinement_v2.py` 和 `policy.py` 里，难以验证也难以回退。

## 4. 拟修改文件
### 新增
1. `pseudo_branch/spgm/manager.py`

### 修改
1. `scripts/run_pseudo_refinement_v2.py`
2. `pseudo_branch/spgm/__init__.py`
3. 必要时 `pseudo_branch/local_gating/gating_schema.py`
4. 必要时 `pseudo_branch/local_gating/gating_io.py`

## 5. 设计方案
### 5.1 manager.py 的职责
第一版只提供两个函数：
1. `build_spgm_update_policy(...)`
   - 可以先只是薄包装，复用现有 `build_spgm_grad_weights(...)`
2. `apply_spgm_state_management(...)`
   - 第一版只汇总并返回 state-management summary
   - 不直接改 opacity / scale / prune / xyz

### 5.2 run_pseudo_refinement_v2.py 的调用改造
当前：
- stats -> score -> grad_weights -> apply_grad_mask -> step

改成：
- stats -> score -> update_policy -> apply_grad_mask -> state_management -> step

第一版 `state_management` 只做：
- 计算 cluster 级 summary
- 记录 potential state action（例如建议的 lr scale、候选 attenuation）
- 写进 history

### 5.3 历史字段
新增 history 字段建议：
1. `spgm_manager_mode_effective`
2. `spgm_state_action_applied`
3. `spgm_state_lr_scale_near/mid/far`
4. `spgm_state_opacity_decay_near/mid/far`
5. `spgm_state_candidate_count_near/mid/far`

第一版这些字段允许只有 summary，没有真实参数修改。

## 6. 输入 / 输出
### 输入
1. `stats.py` 的输出
2. `score.py` 的输出
3. `policy.py` 的输出
4. 当前 iteration 的 Gaussian state handle

### 输出
1. update policy（沿用当前 weights / selected mask）
2. state management summary
3. history 中可追踪的 manager diagnostics

## 7. 实施步骤
1. 新建 `manager.py`
2. 在 `run_pseudo_refinement_v2.py` 中把 SPGM 路径拆两步
3. 增加 history 字段和 print summary
4. 跑一个 StageB smoke，确认 manager shell 被调用
5. 检查 old behavior 完全兼容：默认配置下结果不应漂移

## 8. 验收标准
### 机制验收
1. `manager.py` 在 live call chain 中真实调用
2. history 新增 manager 字段
3. 默认关闭真实 state action 时，训练结果应与原实现保持近似一致

### 工程验收
1. 旧实验可以原样回放
2. 新结构下 B2/B3 可以只改 manager/stats/score，而不再继续挤压 `policy.py`

## 9. 不在 B1 做的事
1. 不改 opacity
2. 不改 prune priority
3. 不做 stochastic
4. 不改 ranking / density proxy 公式

## 10. 可交付内容
1. `spgm/manager.py`
2. run loop 调整后的 call chain
3. manager 级 history / logging
4. 一次 smoke 证明 manager shell 已接通且兼容旧行为
