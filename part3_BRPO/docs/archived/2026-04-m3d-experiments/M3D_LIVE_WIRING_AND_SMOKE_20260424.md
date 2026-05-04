# M~ MASt3R 3D live wiring + smoke report (2026-04-24)

## 1. 本次完成内容
- 已把新 matcher factory 真正接入两个 live script：
  - `scripts/brpo_build_mask_from_internal_cache.py`
  - `scripts/build_brpo_v2_signal_from_internal_cache.py`
- 两个脚本现在都支持：`--matcher-mode {sparse_desc_2d,dense_pts3d_3d}`、`--matcher-model-name`、`--matcher-device`、`--dense3d-conf-quantile`，并把 matcher config / matcher meta 写入产物。
- exact backend builder 额外补了 `--sh-degree` override，避免 stage PLY 读取时只能走默认推断。

## 2. 产物路径
- 单帧 live sweep：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_single`
- 8 帧 live smoke：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_live_smoke_full`
- q080 consumer smoke：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_smoke_q080_v2`

## 3. grounded 结果
### 3.1 单帧 frame 23 live sweep
| arm | backend exact `cm_nonzero_ratio` | signal `joint_nonzero_ratio` | vs sparse backend | vs sparse signal |
|---|---:|---:|---:|---:|
| sparse | 0.01644 | 0.02000 | 1.00x | 1.00x |
| dense3d q0.90 | 0.05763 | 0.07538 | 3.51x | 3.77x |
| dense3d q0.80 | 0.12614 | 0.15314 | 7.67x | 7.66x |
| dense3d q0.70 | 0.19215 | 0.22715 | 11.69x | 11.35x |

结论：live path 接通后，coverage 随 quantile 放松单调上升；先前 `~0.05` 只是保守 `q=0.90` 的结果，不是 3D path 的能力上限。

### 3.2 8 帧 full smoke：sparse vs dense3d q0.80
| metric | sparse mean | dense3d q0.80 mean | ratio |
|---|---:|---:|---:|
| backend exact `cm_nonzero_ratio` | 0.015429 | 0.127471 | 8.26x |
| backend `support_ratio_both` | 0.007382 | 0.040684 | 5.51x |
| backend `support_ratio_single` | 0.008047 | 0.086787 | 10.78x |
| signal raw rgb nonzero | 0.019582 | 0.159152 | 8.13x |
| signal joint nonzero | 0.019570 | 0.159125 | 8.13x |
| joint observation valid ratio | 0.019558 | 0.159152 | 8.14x |

结论：q0.80 在真实 live backend + live signal 路径上，不只是单帧 smoke 看起来更密，而是 8 帧平均也稳定地把 raw exact mask / joint signal coverage 从约 `~2%` 拉到约 `~16%`。

## 4. q0.80 consumer smoke
- 命令入口：`/home/bzhang512/run_m3d_consumer_q080.sh`
- 输出根目录：`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260424_m3d_consumer_smoke_q080_v2`
- 真实消费的 signal root：`20260424_m3d_live_smoke_full/dense3d_q080_signal`
- 日志确认：`Loaded 8 pseudo viewpoints ... signal_pipeline=brpo_v2 ... mean_mask_cov=0.1275`
- StageB 1 iter 成功跑通：`total=0.1801, real=0.1135, pseudo=0.0666, rgb=0.0073, depth=0.0614`
- 本次采样的 4 个 pseudo sample (`23,196,260,57`) 全部被 local gating 接受，sample `target_depth_verified_ratio` 分别为：
  - 23: `0.12614`
  - 196: `0.11802`
  - 260: `0.13970`
  - 57: `0.12095`
  - sample mean: `0.12620`
- 日志输出的 true pose delta summary：`mean_trans=0.000866`, `max_trans=0.001732`, `mean_rotF=0.003674`, `max_rotF=0.007348`

结论：dense3d q0.80 不只在 builder/signal 端可落地，也已经被 `run_pseudo_refinement_v2.py` 的 exact-upstream consumer 路径真实消费了一次，没有出现接口断裂。

## 5. 当前判断
- 当前 live exact M~ 已不再是“只能 sparse 2D MASt3R reciprocal matching”；新 wiring 允许在不改 BRPO 离散三档 `C_m ∈ {1.0, 0.5, 0.0}` 语义的前提下切到 dense3d matcher。
- 第一轮 grounded 结果支持把 `dense3d q0.80` 作为当前最合理的主 smoke 配置：比 sparse 明显更密，又比 q0.70 更保守。
- 这一步已经完成“接通 + live smoke + tiny consumer smoke”；若继续做下一轮，应优先补 `q0.70 / q0.90` 的 full 8-frame compare，再决定是否把 consumer compare 扩成多 quantile 正式对比。
