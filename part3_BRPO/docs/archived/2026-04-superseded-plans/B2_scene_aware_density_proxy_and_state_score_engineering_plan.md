# B2 工程落地方案：Scene-Aware Density Proxy + State Score

## 1. 目标
在 B1 已经把 update policy 和 state management 结构分开之后，给 manager 提供更像“场景级 Gaussian 管理”而不是“branch-local weighting”的输入统计。

一句话目标：把 `ranking_score` 和 `state_score` 从概念上、实现上都拆开。

## 2. 当前代码事实
当前 live SPGM 统计和打分有两个明显限制：
1. `stats.py`
   - support 主要来自 accepted pseudo views
   - `density_proxy = normalized(opacity) * normalized(support)`
   - 更像 branch-local proxy
2. `score.py`
   - 已经有 `importance_score` / `ranking_score`
   - 但 `ranking_score` 仍只是 selector ordering 的轻度变体
   - 没有明确 `state_score`

因此 B2 的工作重心不是改 policy，而是让 B3 有更像样的 state input。

## 3. 理论判断
如果 observation 还没统一，SPGM 会管理一个“RGB-first / weak-depth”的系统；如果 state score 还不存在，manager 也只能继续复用 selector ranking 充当 state signal。

B2 要解决的是第二个问题：
- selector ordering 需要的是“谁更值得进入更新集合”
- state management 需要的是“谁在场景结构上更该被保守、减 lr、轻衰减”

这两个分数不应长期混为一谈。

## 4. 拟修改文件
### 修改
1. `pseudo_branch/spgm/stats.py`
2. `pseudo_branch/spgm/score.py`
3. `scripts/run_pseudo_refinement_v2.py`
4. 如有需要：`pseudo_branch/spgm/manager.py`

## 5. 设计方案
### 5.1 扩统计范围
`stats.py` 第一版建议从“accepted pseudo subset”扩到“current train window summary”：
- accepted pseudo views
- 当前 real anchor views（如果当前 iteration 可直接拿到）
- 当前 local window 的 visibility summary

目标不是一步做全局 full-scene，而是至少从 branch-local accepted subset 往 current train window 靠。

### 5.2 新 density proxy
当前 `opacity_support` 过于简单。若 Gaussian state 中稳定可读出 scale / covariance 体积，建议增加：
- `density_mode = struct_density`
- 形如 `opacity / volume * support`

第一版可以保留：
- `opacity_support`
- `support`
- `struct_density`（新）

### 5.3 新 score 输出
在 `score.py` 中显式输出：
1. `ranking_score`
   - 给 selector / active-set ordering 用
2. `weight_score`
   - 给 selected-set 上的 soft weighting 用
3. `state_score`
   - 给 manager 的 lr scale / opacity attenuation / prune priority 用

当前 `importance_score` 可以继续保留，但应明确它更接近 `weight_score`。

### 5.4 history / diagnostics
新增 history 建议：
1. `spgm_state_score_mean`
2. `spgm_state_score_p50`
3. `spgm_density_mode_effective`
4. `spgm_struct_density_mean`（如实现）
5. `spgm_population_support_mean`

## 6. 输入 / 输出
### 输入
1. visibility summary
2. support_count
3. opacity
4. 可读的 scale / covariance volume（若可稳定获取）
5. cluster partition

### 输出
1. `ranking_score`
2. `weight_score`
3. `state_score`
4. 更场景化的 density summary

## 7. 实施步骤
1. 先在 `stats.py` 中确认当前 iteration 可以拿到哪些 scene / window 统计
2. 最小化引入 `struct_density` 或等价代理
3. 在 `score.py` 中新增 `state_score`
4. 保持 `ranking_score` 与 `weight_score` 的旧行为兼容
5. 把新字段接到 history 和 manager summary
6. 做 smoke：确认 selector 排序不被意外改变、state score 有独立分布

## 8. 验收标准
### 机制验收
1. `ranking_score` 与 `state_score` 在日志中可同时观察
2. state score 不再只是 ranking score 的简单别名
3. density proxy mode 可以明确切换

### 工程验收
1. 旧 control / selector arm 仍能回放
2. 在不启用 B3 的情况下，B2 只增加可观察性，不引入大漂移

## 9. 不在 B2 做的事
1. 不直接做 opacity attenuation
2. 不直接做 stochastic
3. 不改 StageB protocol
4. 不把统计扩到难以维护的 full-scene global pipeline

## 10. 可交付内容
1. 扩展后的 `stats.py` / `score.py`
2. 新 density proxy mode
3. 独立的 `state_score`
4. history / diagnostics 更新
5. 一次 smoke 或 diagnostics report
