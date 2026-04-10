# CHANGELOG.md - Part3 Stage1 过程记录

> 本文件记录每次工作的过程、发现和结果。
> 更新规则：以日期为单位记录；允许为修正文档一致性而整理旧条目，但不要把“现状”写进本文件。

---

## 2026-04-10

### 协议重估：GT 提升 ≠ pipeline 已验证成功

基于 `external infer` 新结果，重新审视了前面的 Stage1 结论。当前更准确的判断是：
- 之前 A/B/C/D/E 的提升，主要说明 **external GT 固定相机口径**下的渲染更接近 GT；
- 这还不能自动等价于“refine 真正改善了 pose-sensitive protocol”；
- 当前 Stage1 更像 **fixed-pose appearance refinement**，而不是已被充分证明的 pose-aware refinement。

### Re10k-1 sparse：infer 口径 external eval

实验目录：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-10_fused_experiment/E_fused_rgb_only/`

**GT 口径**（`pose_mode=gt`）：
- `avg_psnr = 21.7843`
- `avg_ssim = 0.7984`
- `avg_lpips = 0.2002`
- `num_frames = 270`

**Infer 口径**（`pose_mode=infer`, `infer_init_mode=gt_first`）：
- `avg_psnr = 13.6111`
- `avg_ssim = 0.5067`
- `avg_lpips = 0.4061`
- `ate_rmse = 0.0389 m`
- `pose_success_rate = 67.4%`（`182/270` 帧成功）

与 part2 baseline 对照：
- sparse infer baseline：`13.3265 / 0.5027 / 0.3791`
- full infer baseline：`13.1458 / 0.4960 / 0.3894`

结论：
- relative to baseline，infer 口径只有很小变化；
- 这不足以支持“当前 refine 已显著改善 pose-sensitive 表现”；
- 当前 external infer 更像是在暴露 pose inference 自身的瓶颈，而不是稳定放大 refined PLY 的优势。

### 对 internal cache 方案的审核结论

重新审阅了：
- `0_RENDER_EVAL_PROTOCOL.md`
- `0_INTERNAL_CACHE_REPLY_GPT.md`
- `01_INTERNAL_CACHE.md`

结论：**internal cache / replay 路线可执行，而且当前优先级应高于继续调 E 的细节。** 但执行顺序需要收窄为最小闭环：

1. **先做 Phase 1：导出 internal cache**
   - 最好 before_opt / after_opt 都能导出；
   - 至少保存 final `R/T`、内参、图像尺寸、projection matrix、KF/non-KF 标记。
2. **再做 Phase 3：replay render / replay eval**
   - 用同一组 saved internal camera states 比较 baseline PLY 与 refined PLY；
   - 先回答“refined PLY 在 internal protocol 下是否真的更好”。
3. **Phase 2（internal prepare）与 Phase 4（pose refit）后置**
   - 在 replay 没证明收益前，不急着继续把 internal pseudo 生产线铺长。

### 环境与修复

1. infer eval 误用了 `reggs` 环境（`evo 1.34.3`，新 API），导致 `align_trajectory` 不存在；
2. 切回 **`s3po-gs` 环境**（`evo 1.11.0`，旧 API）后恢复正常；
3. 已恢复 `eval_utils.py` 到原版，用 `s3po-gs` 成功跑通 infer eval。

### 文档层面的重新定性

本次同步调整了三份常用文档：
- `STATUS.md`：不再把 GT 口径结果直接写成“已验证成功”；
- `DESIGN.md`：将当前 Stage1 重新定义为更偏 fixed-pose appearance refinement；
- `CHANGELOG.md`：补充 infer 结果与 internal cache 审核结论，并清理重复 infer 条目。

---

## 2026-04-09

### DL3DV-2 sparse：单组 E 正式实验

数据根目录：
`/home/bzhang512/CV_Project/dataset/DL3DV-2/part3_stage1/dl3dv2__s3po__tertile__sparse_v1/`

输出根目录：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/dl3dv-2/sparse/2026-04-09_joint_refine_tertile_freezegeom_lambda0p5/`

E 组：`E_joint_realdensify_freezegeom_lambda0p5_tertile/`
- pseudo 选择：每个 sparse gap 取两个三分位，共 `18` 个 pseudo frame
- pseudo frame id：`11, 22, 44, 56, 78, 90, 112, 124, 146, 158, 180, 192, 214, 226, 248, 260, 282, 294`
- pseudo cache：`build_pseudo_cache.py` 全部 `18/18` 成功，`Errors = 0`，总有效像素 `59719`
- refine 配置：沿用 E（freeze-geom + real densify + `lambda_pseudo=0.5` + tertile pseudo）
- refine：
  - `final_loss_total = 0.1609`
  - `final_loss_real = 0.0586`
  - `final_loss_pseudo = 0.1170`
  - `final_loss_reg = 0.0439`
  - `final_gaussians = 32041`（初始 `29404`）
- external eval（GT pose）：
  - `avg_psnr = 14.0617`
  - `avg_ssim = 0.4526`
  - `avg_lpips = 0.6893`

### DL3DV-2 sparse：GT-pose baseline 同口径补跑

目录：`_baseline_sparse_gt/`
- `avg_psnr = 8.3061`
- `avg_ssim = 0.2884`
- `avg_lpips = 0.7583`

相对 baseline：
- `PSNR +5.7556`
- `SSIM +0.1642`
- `LPIPS -0.0690`

### 运行环境与修复

1. 系统 `python3` 缺少 `numpy`，不能直接用于 Part3 脚本。
2. 本轮最终采用的环境拆分：
   - `reggs`：`prepare_stage1_difix_dataset_s3po.py` / `build_pseudo_cache.py`
   - `s3po-gs`：`run_pseudo_refinement.py` / `eval_external.py`
3. 运行前统一设置：
   - `TMPDIR=/data/bzhang512/tmp`
4. `reggs` 跑 `eval_external.py` 时缺 `munch`；切到 `s3po-gs` 后正常。
5. Difix 访问 HuggingFace 外网失败，但本地 cache 可用，自动 fallback 后继续完成。

### 本轮补充结论

1. **同一套 E policy 已成功迁移到 DL3DV-2 sparse，并在 GT-pose 同口径下显著优于原始 sparse baseline。**
2. **当前 Re10k-1 / DL3DV-2 的 `target_depth` 不是直接使用 render depth，而是 EDP fused depth；`confidence_mask` 也来自 EDP confidence。**
3. 405841 若不继续使用 GT-reprojection 路线，也可以切到 EDP 路线；但那是“改成 EDP”，不是“直接把 render depth 当 target depth”。

---

## 2026-04-08

### 文档整理与校正

1. 重新核对 `docs/` 下的设计、状态、过程文档与实际代码。
2. 确认本轮之前文档里有两类偏差：
   - 一部分“已完成事项”被提前写到了 2026-04-07；
   - sparse refine / A-B ablation 的正式结果尚未记录。
3. 新增并统一维护：`CHANGELOG.md` / `DESIGN.md` / `STATUS.md` / `charles.md`。

### 代码修改

1. `run_pseudo_refinement.py` 增加 **joint refine**：
   - 支持 `--train_manifest` / `--train_rgb_dir`
   - 每轮同时采样 real sparse train views 与 pseudo views
   - `L_total = λ_real * L_real + λ_pseudo * L_pseudo + L_reg`
2. 增加 densify 统计来源开关：
   - `--densify_stats_source pseudo|real|mixed`
3. 增加 pseudo 参数更新控制：
   - `--freeze_geometry_for_pseudo`
   - `--pseudo_trainable_params auto|all|appearance|geometry|none|csv-list`
4. backward 拆分为 pseudo branch 与 real branch：
   - pseudo branch 先反传；
   - 对 pseudo 不允许更新的参数组清梯度；
   - real branch 再反传，从而保证 real anchor 仍能更新几何。

### 已完成实验

#### 1. Smoke checks

- `joint_smoke_check/`：确认 real + pseudo 双路 loss、history、输出路径都正常。
- `joint_real_densify_smoke/`：确认 `densify_stats_source=real` 真正生效。
- `joint_freeze_geom_smoke/`：确认 `freeze_geometry_for_pseudo` 生效。
- `joint_opacity_only_smoke/`：确认 `pseudo_trainable_params` 的细粒度控制生效。

#### 2. 正式 A/B 实验（Re10k-1 sparse）

根目录：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_joint_refine_freezegeom_ablation/`

统一配置：
- 数据：Re10k-1 sparse PLY + sparse pseudo cache
- `num_iterations=2000`
- `lambda_real=1.0`
- `lambda_pseudo=1.0`
- `num_real_views=2`
- `num_pseudo_views=2`
- `densify_stats_source=real`
- `densify_from_iter=800`
- `densify_interval=200`
- `densify_until_iter=1600`
- `densify_grad_threshold=0.0002`
- `min_opacity=0.01`
- eval 统一使用 `pose_mode=gt`, `origin_mode=test_to_sparse_first`

A 组：`A_joint_realdensify_allparams/`
- pseudo 可更新参数：`xyz, f_dc, f_rest, opacity, scaling, rotation`
- refine 结果：
  - `final_gaussians = 48951`
  - `final_loss_total = 0.0661`
  - `final_loss_real = 0.0289`
  - `final_loss_pseudo = 0.0096`
  - `final_loss_reg = 0.0276`
- external eval：
  - `avg_psnr = 20.1598`
  - `avg_ssim = 0.7284`
  - `avg_lpips = 0.2960`

B 组：`B_joint_realdensify_freezegeom/`
- pseudo 可更新参数：`f_dc, f_rest, opacity`
- pseudo 冻结参数：`xyz, scaling, rotation`
- refine 结果：
  - `final_gaussians = 39916`
  - `final_loss_total = 0.1027`
  - `final_loss_real = 0.0321`
  - `final_loss_pseudo = 0.0535`
  - `final_loss_reg = 0.0171`
- external eval：
  - `avg_psnr = 20.9542`
  - `avg_ssim = 0.7575`
  - `avg_lpips = 0.2613`

#### 3. GT pose 原始 sparse baseline（同口径补跑）

目录：`_baseline_sparse_gt/`
- `avg_psnr = 16.9367`
- `avg_ssim = 0.6526`
- `avg_lpips = 0.1722`

#### 4. Follow-up C/D 对照（Re10k-1 sparse）

根目录：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_B_followup_lambda_vs_nodensify/`

C 组：`C_joint_realdensify_freezegeom_lambda0p5/`
- 配置：freeze-geom + real densify + `lambda_pseudo=0.5`
- refine：
  - `final_loss_total = 0.0725`
  - `final_loss_real = 0.0283`
  - `final_loss_pseudo = 0.0546`
  - `final_loss_reg = 0.0169`
  - `final_gaussians = 41544`
- external eval（GT pose）：
  - `avg_psnr = 21.5173`
  - `avg_ssim = 0.7797`
  - `avg_lpips = 0.2242`

D 组：`D_joint_nodensify_freezegeom_lambda1p0/`
- 配置：freeze-geom + `disable_densify` + `lambda_pseudo=1.0`
- refine：
  - `final_loss_total = 0.1158`
  - `final_loss_real = 0.0325`
  - `final_loss_pseudo = 0.0560`
  - `final_loss_reg = 0.0274`
  - `final_gaussians = 35639`
- external eval（GT pose）：
  - `avg_psnr = 20.6968`
  - `avg_ssim = 0.7518`
  - `avg_lpips = 0.2484`

#### 5. 运行环境修复（同日）

在正式跑 C/D 时遇到两类系统问题并已绕过：
1. `LD_LIBRARY_PATH` 未定义导致 conda activate 脚本报错；
2. 根分区 `/` 无可用临时目录导致 Python tempfile / open3d 初始化失败。

处理方式：
- 运行前显式设置 `LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"`；
- 将 `TMPDIR` 指向 `/data/bzhang512/tmp` 后恢复正常。

#### 6. tertile pseudo 正式实验（Re10k-1 sparse）

数据根目录：
`/home/bzhang512/my_storage_500G/CV_Project/dataset/Re10k-1/part3_stage1/re10k1__s3po__tertile__sparse_v1/`

输出根目录：
`/home/bzhang512/my_storage_500G/CV_Project/output/part3_stage1/re10k-1/sparse/2026-04-08_joint_refine_tertile_freezegeom_lambda0p5/`

E 组：`E_joint_realdensify_freezegeom_lambda0p5_tertile/`
- pseudo 选择：每个 sparse gap 取两个三分位，共 `16` 个 pseudo frame
- pseudo frame id：`11, 23, 46, 57, 81, 92, 116, 127, 150, 162, 185, 196, 220, 231, 255, 266`
- refine 配置：沿用 C 组（freeze-geom + real densify + `lambda_pseudo=0.5`）
- refine：
  - `final_loss_total = 0.0709`
  - `final_loss_real = 0.0255`
  - `final_loss_pseudo = 0.0551`
  - `final_loss_reg = 0.0178`
  - `final_gaussians = 42454`
- external eval（GT pose）：
  - `avg_psnr = 21.7658`
  - `avg_ssim = 0.7879`
  - `avg_lpips = 0.2110`


### 本轮结论

1. **freeze geometry for pseudo** 是稳定有效方向（A→B）。
2. **在 freeze-geom + real-densify 框架下，下调 `lambda_pseudo` 到 0.5（B→C）显著改善 PSNR/SSIM/LPIPS。**
3. **在保持 C 策略不变时，将 pseudo 从 midpoint 扩展到 tertile（E）进一步提升三项指标。**
4. `disable_densify`（D）会略改善 LPIPS 相对 B，但总体不如 C/E。
5. 当前下一步默认应从 **E 组**出发继续做更保守 pseudo 对照（如 `lambda_pseudo=0.25`）。

---

## 2026-04-07

### 工程修复与排查

1. 确认 `reggs` 环境可跑 Difix；`difix3d` 环境 CUDA / torch 不兼容。
2. 修复 EDP confidence 尺度问题：
   - `combined_conf = match_conf * epipolar_conf`
   - 再做 95 分位裁剪并归一化到 `[0,1]`
3. 规范化 `Re10k-1 sparse` pseudo cache：
   - split/run timestamp 一致
   - symlink 正常
   - schema 完整

### 关键发现

- EDP 的主要问题更像 **support 太 sparse**，而不是“数值略有误差”。
- sparse pseudo cache 终于具备了做正式 sparse refine 的条件。

---

## 2026-04-06

### Full refine 测试

1. 完成 Re10k-1 full split refine（3000 iter）。
2. 完成 external eval。

### 结果

- refine 前后：
  - `PSNR: 13.15 -> 14.17`
  - `SSIM: 0.496 -> 0.486`
  - `LPIPS: 0.389 -> 0.554`
- Gaussian 数量：
  - `~35k -> ~430k`

### 结论

- 这是第一次明确暴露出：
  - 像素误差可能下降；
  - 但感知质量与结构一致性会恶化；
  - densify / 几何漂移是核心嫌疑项。

---

## 2026-04-05

### 初始 refine 排错

1. 定位到 `create_viewpoint()` 的旋转矩阵使用错误：
   - 错误：`R = pose_w2c[:3, :3].T`
   - 正确：`R = pose_w2c[:3, :3]`
2. 修复后，早期 sparse refine sanity check 不再出现明显异常崩坏。

---

## 2026-04-04

### Pseudo branch 首次搭建

1. 实现 EDP 相关模块：
   - `flow_matcher.py`
   - `epipolar_depth.py`
   - `build_pseudo_cache.py`
2. 形成 GT route / EDP route 双路线。
3. 首次生成多个数据集的 pseudo cache。

### 关键发现

- GT 重投影路线只适合局部可见性较强的数据；
- Re10k-1 / DL3DV-2 更依赖 EDP；
- MASt3R `desc_conf` 不能直接当 pixel weight 使用。

---
