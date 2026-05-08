# CHANGELOG.md - Part3 BRPO 历史记录

> 更新时间：2026-05-05 06:00 (Asia/Shanghai)

---

## 2026-05-05 (下午) - GN 集成 + D-Series 规划

### Gauss-Newton 集成到 Online Mapping

**事件**：
- 在 `BRPOMappingConfig` 中添加 GN 参数：`use_gauss_newton`, `gn_max_iters`, `gn_damping`, `gn_every_n_steps`
- 在 `_run_joint_pseudo_engine()` 中集成 GN：
  - GN 在 backward 之后执行（需要 gradient）
  - GN 直接更新 theta/rho，然后 fold 到 R/T
  - 如果使用 GN，跳过 Adam pose optimizer
- 生成 D-series 实验配置（D0-D4）

**修改文件**：
- `third_party/S3PO-GS/utils/slam_backend_brpo.py`：集成 GN
- `scripts/generate_d_series_configs.py`：D-series 配置生成器
- `scripts/run_d_series.sh`：D-series 运行脚本

**D-Series 实验规划**：

| 实验 | 目标 | 关键配置 |
|------|------|---------|
| D0_baseline | 无 pseudo mapping 的 baseline | `pseudo_map_iters=0` |
| D1_pose_fix | 验证 pose gradient 修复 | `pseudo_map_iters=20`, Adam |
| D2_gauss_newton | 验证 GN 效果 | `pseudo_map_iters=20`, GN |
| D3_scale_reg | 验证 scale regularization | `lambda_scale=0.1` |
| D4_gn_scale | 组合验证 | GN + lambda_scale=0.1 |

---

## 2026-05-05 (上午) - Pose Gradient 修复 + Gauss-Newton 实现 + Scale Regularization

**事件**（略，详见上文）

---