# Pose Optimization 与 Pseudo Effectiveness 审计报告

> 审计时间：2026-05-06 06:30 (Asia/Shanghai)
> 审计范围：D-Series pose optimization 修复、online mapping 链路、pseudo enhancement 无效根因

---

## 1. 审计任务

1. 审计 claude 做的 pose optimization 修复
   - exposure refinement 模块
   - scale regularization 模块
   - Gauss-Newton 集成
   - online mapping 是否真正执行 kf1-kf2-pseudo1.5-kf3 链路

2. 调查 pseudo view enhancement 为何无效
   - 与 baseline 几乎无区别
   - 检查代码流、产物流、bug

---

## 2. 关键发现

### 2.1 配置传递 Bug（严重）

问题：_resolve_brpo_online_mapping_cfg 遗漏读取 GN/scale 参数

证据：
- D4 config.yml 设置：use_gauss_newton: true, lambda_scale: 0.1
- D4 runtime history：use_gauss_newton: false, lambda_scale: 0.01（默认值）

代码位置：slam_backend.py:122-156

遗漏参数：
- use_gauss_newton
- gn_max_iters
- gn_damping
- gn_every_n_steps
- lambda_scale
- max_scale
- depth_loss_mode

影响：D-Series D4 实验声称 GN+Scale 配置效果最好，但实际上 GN 和 Scale 都没有真正启用。

---

### 2.2 Pose Optimization 实现状态

结论：代码实现正确，但配置未传递导致未生效

已正确实现：
1. apply_pose_delta_before_render_() — pose gradient 修复
2. scale_reg_loss() — scale regularization
3. GaussNewtonPoseOptimizer — GN pose optimizer
4. exposure refinement — exposure 参数优化

但未生效原因：use_gauss_newton 和 lambda_scale 配置未传递到 runtime

---

### 2.3 Online Mapping 链路状态

结论：正确执行 kf1-kf2-pseudo1.5-kf3 链路

证据：
- event_kf_0033：kf0 -- kf33 -- pseudo frame16 (midpoint)
- event_kf_0067：kf33 -- kf67 -- pseudo frame50 (midpoint)
- event_kf_0101：kf67 -- kf101 -- pseudo frame84 (midpoint)

这是真正的 online mapping。

---

### 2.4 Pseudo Enhancement 无效根因

结论：不是 bug，而是 SLAM early stage geometry quality 问题

关键数据：
- Event kf_0033: left support=3.3%, reproj=12.17px, both coverage=0.46%
- Event kf_0101: left support=14.2%, reproj=7.76px, both coverage=5.07%

问题：
1. Early SLAM geometry 差（frame 0 是初始帧）
2. 验证阈值严格（tau_reproj_px=16, tau_rel_depth=1.0）
3. Effective coverage 低（15%）
4. RGB loss 极低（6.14e-07）

---

## 3. 修复建议

### 3.1 立即修复：配置传递 Bug（P0）

在 _resolve_brpo_online_mapping_cfg 中添加：
- depth_loss_mode
- lambda_scale
- max_scale
- use_gauss_newton
- gn_max_iters
- gn_damping
- gn_every_n_steps

### 3.2 后续优化：放宽 Early Stage 验证阈值（P1）

### 3.3 后续探索：Pseudo Pose 持久化（P2）

---

## 4. 总结

| 问题 | 状态 | 严重程度 |
|------|------|---------|
| 配置传递 Bug | 遗漏参数 | 严重 |
| Pose gradient 修复 | 已正确实现 | 无 |
| Scale regularization | 已正确实现 | 无 |
| GN integration | 已正确实现 | 无 |
| Online mapping 链路 | 正确执行 | 无 |
| Pseudo coverage 低 | Geometry quality | 中等 |

核心结论：
1. Pose optimization 代码正确，但配置未传递导致未生效
2. Online mapping 链路正确
3. Pseudo 无效根因是 early stage geometry 差
4. 最严重：配置传递 Bug，需立即修复
