# M~ MASt3R 3D full mechanism validation (2026-04-24)

## 1. 前置清理
- 清理目录：`/home/bzhang512/my_storage_1T/CV_Project/output/part3_BRPO/experiments`
- 保留：
  - `20260424_m3d_live_smoke_single`
  - `20260424_m3d_live_smoke_full`
  - `20260424_m3d_consumer_smoke_q080_v1`
  - `20260424_m3d_consumer_smoke_q080_v2`
- 删除：其余 95 个旧实验目录
- 释放空间：`7085625405 bytes`，约 `6.60 GiB`
- manifest：`/home/bzhang512/.hermes_backups/cleanup_manifests/part3_brpo_my_storage_1T_experiments_cleanup_20260424_0828.txt`
- 清理后 `/data2`：`Avail 32G`（此前约 `26G`）

## 2. 本轮 full mechanism validation 范围
### 2.1 8-frame live full smoke
在已接通的 live builder / live signal 路径上补齐：
- sparse
- dense3d q0.90
- dense3d q0.80
- dense3d q0.70

产物根：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_full`

### 2.2 短程 consumer compare
- compare arms：`sparse / q090 / q080 / q070`
- consumer：`scripts/run_pseudo_refinement_v2.py`
- stage mode：`stageB`
- `stageA_iters=0`, `stageB_iters=20`
- common anchor：`/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80`
  - 说明：原先用于 q0.80 tiny consumer smoke 的 `/data2/.../20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/...` 在清理旧实验目录后不再可用，因此本轮 compare 改用现存的共同 stageA5 gated anchor；四个 arm 共用同一 anchor，比较仍然公平。

产物根：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_compare_stageB20`

## 3. 8-frame live full smoke 结果
### 3.1 backend exact `C_m`
| arm | mean `cm_nonzero_ratio` | vs sparse | mean `support_ratio_both` | mean `support_ratio_single` | mean reproj L/R | mean rel-depth L/R |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 0.015429 | 1.00x | 0.007382 | 0.008047 | 1.442 / 1.134 | 0.0247 / 0.0231 |
| q0.90 | 0.060540 | 3.92x | 0.013358 | 0.047183 | 1.764 / 1.606 | 0.0303 / 0.0306 |
| q0.80 | 0.127471 | 8.26x | 0.040684 | 0.086787 | 1.659 / 1.545 | 0.0291 / 0.0275 |
| q0.70 | 0.190781 | 12.37x | 0.069017 | 0.121764 | 1.619 / 1.550 | 0.0284 / 0.0263 |

### 3.2 live signal / joint observation
| arm | mean `joint_nonzero_ratio` | vs sparse | mean `joint_valid_ratio` | mean `joint_confidence_mean_positive` |
|---|---:|---:|---:|---:|
| sparse | 0.019570 | 1.00x | 0.019558 | 0.4131 |
| q0.90 | 0.079995 | 4.09x | 0.079995 | 0.8659 |
| q0.80 | 0.159125 | 8.13x | 0.159152 | 0.8181 |
| q0.70 | 0.234323 | 11.97x | 0.234375 | 0.7731 |

### 3.3 机制判断
- q0.90 明显偏保守：coverage 只到 q0.80 的一半左右，但 backend reproj / rel-depth 反而是三条 dense arm 里最差。
- q0.80 与 q0.70 都比 sparse 大幅增密；其中 q0.70 在本轮 8-frame 统计里不只 coverage 更高，而且 backend reproj / rel-depth 并没有比 q0.80 更差，反而略好。
- 因此到 mechanism 层为止，`q0.90` 可以先降级为次要下界 control；主候选应收敛到 `q0.80` 与 `q0.70`，其中 `q0.70` 是当前更强的 dense 候选。

## 4. StageB20 consumer compare 结果
### 4.1 compare setup
- common anchor：`20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80`
- stage mode：`stageB`, `stageA_iters=0`, `stageB_iters=20`
- same seed / same topology / same local-gating policy / same real/pseudo sampling budget
- only varying factor：`signal_v2_root`

### 4.2 StageA effective mask coverage seen by consumer
| arm | `mean_mask_cov` |
|---|---:|
| sparse | 0.0154 |
| q0.90 | 0.0605 |
| q0.80 | 0.1275 |
| q0.70 | 0.1908 |

### 4.3 StageB iter 20 snapshot
| arm | `loss_total` @20 | `loss_real` @20 | `loss_pseudo` @20 | `loss_depth` @20 |
|---|---:|---:|---:|---:|
| sparse | 0.1937 | 0.1493 | 0.0444 | 0.0354 |
| q0.90 | 0.2114 | 0.1480 | 0.0634 | 0.0493 |
| q0.80 | 0.2102 | 0.1482 | 0.0619 | 0.0462 |
| q0.70 | 0.2091 | 0.1485 | 0.0607 | 0.0463 |

### 4.4 pose drift summary from run logs
| arm | mean_trans | max_trans | mean_rotF | max_rotF |
|---|---:|---:|---:|---:|
| sparse | 0.002264 | 0.008823 | 0.002711 | 0.004120 |
| q0.90 | 0.001639 | 0.004818 | 0.003639 | 0.010559 |
| q0.80 | 0.001682 | 0.005827 | 0.003071 | 0.007657 |
| q0.70 | 0.001918 | 0.007039 | 0.002540 | 0.004756 |

### 4.5 consumer-layer interpretation
- 这 20 iter compare 说明 dense arms 已经被下游真实消费，且行为稳定，没有接口断裂。
- 但它还没有给出“dense 立即赢 sparse”的证据：到 iter 20 为止，dense arms 的 `loss_total / loss_pseudo / loss_depth` 都仍高于 sparse。
- 在 dense arms 内部，`q0.70` 是当前最好的 early consumer 候选：
  - final `loss_total` 最低（0.2091）
  - final `loss_pseudo` 最低（0.0607）
  - `q0.90` 明显最弱
- 但这仍只是短程 early compare；不能直接据此宣布 dense3d 默认 quantile 已冻结。

## 5. 当前结论
1. `q0.90` 可以从主线候选里降级。它在 full smoke 和 short consumer compare 里都不占优。  
2. `q0.70` 是当前最强 dense 候选。它在 8-frame mechanism compare 里 coverage 最高、backend geometry 统计也没有变坏；在 StageB20 compare 里，它又是 dense arms 里最好的 early consumer arm。  
3. `q0.80` 仍然保留为更保守的主 control。它已经过 live smoke + tiny consumer smoke + short compare 三轮验证，风险感更低。  
4. 当前还不能直接把 dense3d 设成默认主线。因为 dense arm 虽然大幅增加监督覆盖，但在 20-iter consumer compare 中还没有形成对 sparse 的清晰 downstream winner。  

## 6. 建议的下一步
- 丢掉 q0.90，集中做一个更长但仍受控的 compare：`sparse vs q0.80 vs q0.70`
- 保持同一个 common anchor、同一个 seed、同一个 exact-upstream T~ / clean G~ / T1 设置
- 把 compare 长度拉到真正能看出 downstream 质量差异的区间（例如 StageB 80/120 iter 级别，而不是 20 iter）
- 重点看：
  - stageB loss trajectory 是否开始出现 dense arm 真正收敛优势
  - 输出 `refined_gaussians.ply` 的 replay / compare 指标是否开始区分 sparse 与 dense
  - dense q0.70 是否继续稳定优于 q0.80，还是只是 early loss 上略优

一句话：本轮 full 机制验证已经把候选集从 `q0.90/q0.80/q0.70` 收窄到了 `q0.80 vs q0.70`，其中 `q0.70` 当前更强；但要不要真正取代 sparse，还需要下一轮更长程的 downstream compare 来定。