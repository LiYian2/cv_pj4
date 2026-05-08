# STATUS.md - Part3 BRPO Online Mapping 当前状态

> 更新时间：2026-05-06 16:30 (Asia/Shanghai)

---

## 0. 最新修改（2026-05-06）

### 0.1 KF0 Pose 跳过逻辑已移除

**文件**：`third_party/S3PO-GS/utils/slam_backend.py`
**修改**：删除第 631-632 行的 `if viewpoint.uid == 0: continue`
**原因**：DL3DV 的 `cameras.json` 来自 COLMAP 重建，不是真实 GT，KF0 也需要优化

**备份**：`slam_backend.py.bak_kf0_skip`

### 0.2 D5 实验配置已生成

**文件**：`/home/bzhang512/CV_Project/part3_BRPO/configs/d5_online_mapping_fix.yaml`

**关键配置**：
| 参数 | D5 值 | 说明 |
|---|---|---|
| placement_mode | quartile | 3 pseudo per gap |
| update_real_pose | true | 允许真实 KF pose 更新 |
| use_gauss_newton | true | 启用 GN |
| lambda_scale | 0.1 | Scale 正则化 |
| lambda_pseudo | 2.0 | 增强 pseudo 权重 |
| dense3d_conf_quantile | 0.15 | 更宽松的 pseudo 选择 |

### 0.3 待解决问题：22:1 Loss 比例失衡

**现状**：
| | 数量 | 像素覆盖 | 总监督 |
|---|---|---|---|
| Real | 10 frames | 100% | 10 单位 |
| Pseudo | 3 frames | ~15% | 0.45 单位 |

**已执行**：
- `lambda_pseudo = 2.0` ✓
- `placement_mode = quartile` (3 pseudo) ✓

**待讨论**：
- 方案 c：分离 pose/scene loss（pose loss 不受 mask 限制）

---

## 1. Config 传递 Bug（已修复 2026-05-05）

**问题**：`_resolve_brpo_online_mapping_cfg()` 遗漏读取关键参数
**修复**：已在 `slam_backend.py` 中补全参数传递

---

## 2. Gaussians 更新范围

**S3PO 主循环**：全部 6 个参数
- `_xyz`, `_features_dc/_rest`, `_opacity`, `_scaling`, `_rotation`

**Pseudo mapping**：
- densify/prune/opacity_reset 默认关闭
- 基本属性通过 gradient backward 更新
- Phase 1 修复后梯度传递完整 ✓

---

## 3. 关键文件

| 文件 | 用途 |
|---|---|
| `configs/d5_online_mapping_fix.yaml` | D5 实验配置 |
| `third_party/S3PO-GS/utils/slam_backend.py` | KF0 跳过已移除 |
| `pseudo_branch/refine/pseudo_camera_state.py` | pose delta 应用 |
| `pseudo_branch/integration/runtime_slot_selector.py` | quartile 模式 |

---

## 4. 下一步

| 优先级 | 任务 | 状态 |
|-------|------|------|
| P0 | 运行 D5 实验 | ❌ 待执行 |
| P1 | 验证 KF0 pose 优化效果 | ❌ 待验证 |
| P2 | 评估 22:1 比例改善 | ❌ 待评估 |

---

## 5. 一句话结论

> **KF0 跳过已移除，D5 配置已生成（quartile + update_real_pose + GN）。下一步运行 D5 实验验证效果。**
