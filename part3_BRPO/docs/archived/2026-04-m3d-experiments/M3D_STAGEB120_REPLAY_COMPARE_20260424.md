# M~ dense3d 3-arm StageB120 + replay compare (2026-04-24)

## 1. 目的
在上一轮 full mechanism validation 之后，候选已经收窄到：
- sparse control
- dense3d q0.80
- dense3d q0.70

这轮 compare 的目标不再是验证“matching coverage 能不能提高”，而是回答更关键的问题：在更长程的 StageB downstream compare + replay evaluation 下，dense3d 能不能真正赢 sparse。

## 2. 为什么选 120 iter
本轮把 StageB 长度固定为 `120 iter`，原因不是拍脑袋，而是沿用项目里已经反复使用过的长程 compare 预算：
- `docs/archived/2026-04-plans-landed/T4_EXACT_UPSTREAM_COMPARE_PLAN_20260421.md` 使用 `--stageB_iters 120`
- 早期 StageB 长程记录里也反复以 `stageB120` 作为正式长程 compare 档位

因此这里用 120 iter，目的是让这轮 dense3d compare 与现有主线 compare 的“长程预算”保持一致。

## 3. 运行设置
### 3.1 common anchor
由于之前 tiny consumer smoke 所用的 `/data2/.../20260416_p2f_stageA5_v2rgbonly_gating_compare_e1/...` 在旧实验目录清理后已不再可用，这轮 compare 统一改用现存公共 anchor：
- anchor root: `/data/bzhang512/CV_Project/output/part3_BRPO/experiments/20260415_p2b_stageA5_local_gating_compare_e1/stageA5_legacy_xyz_gated_80`
- PLY: `refined_gaussians.ply`
- init states: `pseudo_camera_states_final.json`

三个 arm 共用同一个 anchor，因此 compare 本身仍然公平。

### 3.2 3 arms
- `sparse` -> signal root: `20260424_m3d_live_smoke_full/sparse_signal`
- `q080` -> signal root: `20260424_m3d_live_smoke_full/dense3d_q080_signal`
- `q070` -> signal root: `20260424_m3d_live_smoke_full/dense3d_q070_signal`

### 3.3 shared consumer config
- consumer: `scripts/run_pseudo_refinement_v2.py`
- `stage_mode=stageB`
- `stageA_iters=0`
- `stageB_iters=120`
- `stageB_post_switch_iter=40`
- `joint_topology_mode=brpo_joint_v1`
- `pseudo_observation_mode=exact_brpo_upstream_target_v1`
- `stageA_depth_loss_mode=exact_shared_cm_v1`
- same seed / same real+pseudo sampling budget / same SPGM keep policy

### 3.4 replay evaluator
- `scripts/replay_internal_eval.py`
- internal cache root: `/data/bzhang512/CV_Project/output/part2_s3po/re10k-1/s3po_re10k-1_full_internal_cache/Re10k-1_part2_s3po/2026-04-11-05-33-58/internal_eval_cache`
- 额外跑了一个 common anchor replay，方便看每个 arm 相对 anchor 是提升还是退化

### 3.5 产物
- compare root: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_compare_stageB120_replay`
- summary: `compare_summary.json`

## 4. replay 主结果
### 4.1 anchor replay reference
| arm | PSNR | SSIM | LPIPS | ΔPSNR vs internal after_opt |
|---|---:|---:|---:|---:|
| anchor_stageA5_legacy_xyz_gated_80 | 23.8522 | 0.87064 | 0.08093 | -0.0967 |

### 4.2 3-arm replay compare
| arm | PSNR | SSIM | LPIPS | ΔPSNR vs sparse | ΔSSIM vs sparse | ΔLPIPS vs sparse | ΔPSNR vs anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse | 24.0045 | 0.87177 | 0.08247 | 0.0000 | 0.00000 | 0.00000 | +0.1523 |
| q080 | 23.5816 | 0.86296 | 0.08702 | -0.4229 | -0.00881 | +0.00455 | -0.2706 |
| q070 | 23.6660 | 0.86585 | 0.08577 | -0.3385 | -0.00593 | +0.00330 | -0.1862 |

### 4.3 replay 结论
- 这轮最重要的事实是：`sparse` 明确赢了两个 dense arm。
- `q070` 在 dense 内部仍然优于 `q080`，但它依然落后 sparse：
  - PSNR 比 sparse 低 `0.3385`
  - SSIM 比 sparse 低 `0.00593`
  - LPIPS 比 sparse 差 `+0.00330`
- 两个 dense arm 都没有超过 common anchor replay；sparse 则略高于 anchor replay。

因此从 replay 角度看，当前 dense3d 路线虽然能大幅提高 coverage，但还没有转化成 downstream winner。

## 5. StageB120 训练侧结果
### 5.1 final losses @ iter 120
| arm | `loss_total` @120 | `loss_real` @120 | `loss_pseudo` @120 | `loss_depth` @120 |
|---|---:|---:|---:|---:|
| sparse | 0.16891 | 0.12236 | 0.04655 | 0.04105 |
| q080 | 0.16962 | 0.11926 | 0.05036 | 0.04398 |
| q070 | 0.17209 | 0.12024 | 0.05185 | 0.04522 |

### 5.2 selected checkpoints
| arm | iter 40 total | iter 80 total | iter 100 total | iter 120 total |
|---|---:|---:|---:|---:|
| sparse | 0.10994 | 0.06070 | 0.14926 | 0.16891 |
| q080 | 0.11989 | 0.07748 | 0.15505 | 0.16962 |
| q070 | 0.12041 | 0.07672 | 0.15625 | 0.17209 |

### 5.3 pose drift summary from logs
| arm | mean_trans | max_trans | mean_rotF | max_rotF |
|---|---:|---:|---:|---:|
| sparse | 0.004891 | 0.019433 | 0.000698 | 0.002712 |
| q080 | 0.004126 | 0.023616 | 0.001410 | 0.005997 |
| q070 | 0.004468 | 0.023390 | 0.001196 | 0.004915 |

### 5.4 训练侧解释
- 稠密 arms 在 iter 1/20 的 pseudo/depth loss 明显更高，这是 coverage 提高带来的直接结果。
- 到 120 iter 时，dense arms 的 final total / pseudo / depth loss 仍没有优于 sparse。
- q070 依旧比 q080 更像“更好的 dense 候选”，但在 120 iter 这一档，它也没有超过 sparse。

## 6. 总结判断
1. `q0.90` 早已可以排除出主候选。  
2. `q0.70` 仍然是 dense 内部最优候选，但它在真正的长程 replay compare 中仍输给 sparse。  
3. 因此当前不能把 dense3d 默认化，也不能把 q0.70 升成新的主线 quantile。  
4. 当前最可靠的结论是：dense3d 的问题已经不是“matching coverage 不够”，而是“增加的支持区域没有转化成更好的 downstream target / optimization 结果”。  

## 7. 对下一步的含义
下一步不该再继续做 q0.70 / q0.80 的同类长程 sweep，也不该回去测 q0.90。当前更合理的是转向“为什么 dense support 没有转成 replay 增益”的结构性排查，例如：
- dense3d 新增区域里 single/both 的组成是否过于偏 single=0.5
- target depth composition / confidence weighting 是否让新增区域贡献了噪声而不是稳定监督
- dense3d 增加的是 coverage，但不是有效几何约束

一句话：这轮真正回答了问题——当前 3D dense matching 虽然机制上显著增密，但在 120-iter + replay 的 downstream compare 里，效果仍然不如 sparse 主线。