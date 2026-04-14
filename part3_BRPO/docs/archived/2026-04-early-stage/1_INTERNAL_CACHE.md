# 01_INTERNAL_CACHE.md — Internal Protocol Cache 与 Replay 工程规划

> 目标：建立不影响 S3PO 原有主流程的 internal 路线，使以下链路可执行：
> S3PO(after_opt) 保存 internal cache → internal prepare → pseudo cache → refine → replay eval → 可选 pose refit
> 
> 创建时间：2026-04-10

---

## 0. Agent执行须知（先看）

这份文档现在的角色，不是“立刻把整条 internal 路线一次性做完”，而是**为一个最小闭环 implementation 提供边界与监督**。

当前最重要的判断是：
- 我们已经知道 `external GT` 下 refined PLY 往往更好；
- 但 `external infer` 没有给出足够强的改进证据；
- 因此眼下最急迫的问题不是继续扩展 pseudo pipeline，而是：**refined PLY 在 internal tracked-camera protocol 下到底有没有真实收益。**

所以，执行优先级必须收敛到：
1. **先导出 internal cache**；
2. **再做 replay render / replay eval**；
3. **先回答 baseline vs refined 在 internal protocol 下谁更好**；
4. 只有这一步成立，才继续 internal prepare / pseudo cache / pose refit。

### 0.1 本文档的 v1 目标边界

v1 目标不是把四个 Phase 全做完，而是建立下面这个**最小闭环**：

```text
S3PO run (before_opt / after_opt)
    ↓
export internal cache
    ↓
replay baseline/refined PLY under saved internal camera states
    ↓
compare replay render metrics
```

也就是说，当前必须优先完成的是：
- **Phase 1：保存 internal cache**
- **Phase 3：replay render / replay eval**

而不是：
- 一上来就写完整 `internal prepare → pseudo cache → refine` 生产线；
- 一上来就做 pose refit。

### 0.2 当前推荐执行顺序

请严格按下面顺序推进，不要跳步：

1. **Phase A：S3PO 侧导出 internal cache**
   - 最好 `before_opt` 和 `after_opt` 都支持导出；
   - 先确认导出的 camera state 足够 replay render。

2. **Phase B：part3 侧实现 replay_internal_eval.py**
   - 用 saved camera states 对给定 PLY 回放渲染；
   - 至少能比较 baseline PLY 与已有 refined PLY。

3. **Phase C：先做协议验证实验**
   - 比较 baseline sparse PLY vs 当前 E refined PLY；
   - 回答 internal protocol 下是否真的有收益。

4. **Phase D：只有 replay 明确成立后，再推进 internal prepare**
   - 包括 internal pseudo cache / internal refine / pose refit。

### 0.3 不要把当前任务做宽

当前最常见的跑偏风险是：
- 还没证明 replay 值得做，就提前铺 `internal prepare` 的大生产线；
- 还没证明 refined PLY 在 internal protocol 下更好，就提前做 pose refit；
- 把“导出 cache”与“重做训练范式”混到同一轮提交里。

因此本轮的硬约束是：
- **优先回答协议问题，不优先扩展训练系统。**
- **优先做 replay，不优先做新 pseudo 数据生产。**
- **优先比较 baseline vs refined，不优先调新超参。**

### 0.4 对 S3PO 主流程的改动要求

internal cache 必须做成**附加功能**，不要改写原有 part2 / S3PO 主流程。

执行时请遵守：
- 不改变已有 `init → keyframe → map → color_refinement → eval_rendering` 主逻辑；
- 新增导出调用时，尽量只挂在 `after_opt` / `before_opt` 评测附近；
- 不要让没有开启 internal cache 导出的普通 run 行为发生变化。

一句话：**默认行为不变，internal cache 是 opt-in 附加导出。**

### 0.5 v1 必须保存哪些字段

第一版不要贪大求全。只要能保证 replay render 成立即可。

**硬依赖：**
- `frame_id / uid`
- `is_keyframe`
- `R, T`
- `fx, fy, cx, cy`
- `FoVx, FoVy`
- `image_height, image_width`
- `projection_matrix`
- `kf_indices / non_kf_indices`

**建议保存但不是 replay 第一版硬依赖：**
- `cam_rot_delta, cam_trans_delta`
- `exposure_a, exposure_b`
- `R_gt, T_gt`
根据s3po的代码查看以下这些是否拿得到
   - pose_c2w 显式保存，不只拆成 R/T
	•	image_path 或 canonical image name
	•	render_rgb_path / render_depth_path 的 manifest 映射
	•	nearest_left_kf_id / nearest_right_kf_id 或至少能直接恢复邻接关系
	•	stage_tag：before_opt / after_opt
	•	exposure_a/b 最终值
	•	如果拿得到，pre_clean 版本的 residual / exposure 快照

注意：当前代码里 `camera.clean()` 会把 residual / exposure 置空，因此**不能把这些字段当 replay 第一版的硬依赖**。第一版 replay 应以最终 `R/T + intrinsics + image size + projection_matrix` 为核心。

### 0.6 before_opt / after_opt 的要求

如果工程量允许，建议 **before_opt 和 after_opt 都导出**。原因有两个：
1. 这样可以区分 **tracking/mapping 阶段地图** 和 **color refinement 后地图** 的贡献；
2. 后面 replay 时可以回答：提升来自 refine、还是主要来自 color_refinement。

如果本轮只能先做一个，优先顺序是：
- **先做 after_opt**（因为它对应当前 internal 最强地图）
- 再补 before_opt。

### 0.7 replay 阶段的硬规则

第一版 `replay_internal_eval.py` 必须遵守：
- **不依赖 runtime frontend.cameras**；
- **只依赖 saved internal camera states + 指定 PLY**；
- **不使用 GT pose 做渲染**；
- 渲染时 residual 可直接设为 0；
- metrics 可以继续使用 GT image，但不要把这件事误写成“用了 GT pose”。

目标是：
**用同一组 internal tracked camera states，公平比较不同 PLY。**

### 0.8 修改代码前的检查与回退建议

因为这次会碰 `third_party/S3PO-GS/`，执行前务必：

1. 先检查是否有相关 run 正在进行，避免改到运行中的文件；
2. 先记录 `git status`；
3. 修改 `third_party/S3PO-GS/slam.py` 前，至少保证：
   - 能用 git 一键回退；或
   - 先保存一份 patch / 临时分支。

建议：
- **小改动优先走 git 提交**；
- **涉及 `slam.py` / `eval_utils.py` / renderer 调用的大改动，建议额外保留 patch**。

### 0.9 实验要求

v1 完成后，实验不要发散，先做这两类：

1. **协议验证实验（必须）**
   - baseline sparse PLY
   - 当前 E refined PLY
   - 同一组 saved internal camera states replay

2. **阶段分离实验（强烈建议）**
   - before_opt internal cache replay
   - after_opt internal cache replay

这样至少能回答三件事：
- refined PLY 是否优于 baseline
- 这种优势是否只在 after_opt 下存在
- color_refinement 在 internal protocol 中占多大比重

### 0.10 暂不优先做的内容

在 replay 结论出来之前，暂不建议优先：
- internal pseudo cache 全链路
- pose refit
- 新一轮 fused / lambda / densify 大量调参
- 把 internal 路线和 backend pseudo integration 混到一起
###：我们目前只根据这个文档做internal cache和replay的验证，不修改refine等

### 0.11 文档更新要求

实现与实验完成后，至少同步更新：
- `STATUS.md`：写清 internal replay 是否已建立，以及当前结论是否改变
- `DESIGN.md`：写清 internal cache / replay 在协议上的角色
- `CHANGELOG.md`：记录导出字段、脚本新增、实验结果与结论

如果只做了 Phase 1 + 3，也要明确写清：
- 哪些内容已完成
- 哪些内容刻意后置
- 为什么后置

---

## 1. 协议差异核心

两套评测协议不能直接比较：

- **Internal protocol**：用 frontend.cameras 中 tracked 的相机状态，对 non-KF 帧渲染评测。相机与地图在同一 SLAM 过程中共同形成，更自洽。
- **External protocol**：离线读取 PLY，重建相机（GT pose 或 infer pose），在 test split 上渲染。相机与地图不是共同演化形成的。

关键事实：internal before_opt 远大于 external gt，说明 internal tracked camera state 与地图的自洽性是决定性因素，而非 pose 离 GT 多近。

---

## 2. 四阶段工程路线

### Phase 1：保存 internal cache（S3PO 侧）

在 after_opt 阶段附加导出 internal cache，不改 S3PO 主流程。

**新增文件**：third_party/S3PO-GS/utils/internal_eval_utils.py

**核心函数**：

- export_camera_states(frames, kf_indices, save_dir, config, iteration) — 导出相机状态
- export_internal_render_cache(frames, gaussians, dataset, save_dir, pipe, background, kf_indices, iteration) — 导出渲染缓存

**camera_states.json 每帧字段**（完整保存）：

- frame_id / uid：帧标识
- is_keyframe：是否为 KF
- R, T：最终估计位姿
- fx, fy, cx, cy：内参
- FoVx, FoVy：视场角
- image_height, image_width：图像尺寸
- projection_matrix：投影矩阵
- cam_rot_delta, cam_trans_delta：SE(3) 残差（预留 pose refit）
- exposure_a, exposure_b：曝光参数（预留 pose refit）

**manifest.json 顶层字段**：

- iteration：after_opt
- kf_indices, non_kf_indices：KF 与 non-KF 列表
- num_frames：总帧数
- background：背景颜色
- pipeline_params：渲染参数

**渲染缓存目录**：

- render_rgb/：non-KF 渲染 RGB
- render_depth_npy/：原始深度矩阵
- render_depth_vis/：可视化（可选）

**slam.py 改动**：在 after_opt 的 eval_rendering() 后新增一行调用 export_internal_render_cache()。

---

### Phase 2：internal prepare → pseudo cache → refine（part3 侧）

**新增文件**：part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po_internal.py

**输入参数**：

- internal-cache-root：指向 save_dir/internal_eval_cache/after_opt
- scene-name, run-key
- dataset-root, rgb-dir
- placement：midpoint / tertile / both
- limit, difix-model-*

**逻辑步骤**：

1. 读取 internal cache：manifest.json, camera_states.json, render_rgb/, render_depth_npy/
2. 确定 pseudo 候选帧：non_kf_indices，左右参考为邻近 keyframes（沿用 KF 邻接语义）
3. Sample selection：按 placement 在 non-KF 中选择
4. Difix：当前帧 internal render 作为 degraded input，左右 KF RGB 作为 reference，产出 target_rgb_left.png, target_rgb_right.png
5. Pack：写出 internal 版 pseudo_cache/

**camera.json 字段来源**：

- pose_c2w：来自 internal camera_states.json 的 saved R/T
- intrinsics_px：来自 internal saved fx/fy/cx/cy
- image_size：来自 internal saved image_height/image_width
- 不再依赖 external trj_est

**refs.json 字段来源**：

- left_ref_frame_id, right_ref_frame_id：邻近 KF 的 frame_id
- left_ref_pose, right_ref_pose：来自 internal camera_states.json 对应 KF 的 saved pose
- ref_pose_source：标记为 internal_eval_cache.after_opt

**render_depth 用途**：internal render depth 可能较准确，可作为 target depth 的候选来源（替代 EDP），但第一版留作实验对照。

**refine**：沿用现有 run_pseudo_refinement.py，对 after_opt ply 做 refine。

---

### Phase 3：replay render / replay eval（part3 侧）

**新增文件**：part3_BRPO/scripts/replay_internal_eval.py

**输入参数**：

- ply-path：refined PLY
- internal-cache-root：saved camera states
- output-dir
- dataset-root（可选，用于算 metrics）
- use-gt-metrics（可选 bool）

**replay render 行为**：

对 camera_states.json 中每个 non-KF：

1. 从 saved 字段构造 ReplayCamera（不依赖 runtime frontend.cameras）
2. 用 refined ply 调用 render()
3. 保存 replay_render_rgb/, replay_render_depth_npy/

**ReplayCamera 构造**：

- R, T：saved internal pose
- fx, fy, cx, cy, FoVx, FoVy, image_height, image_width：saved 内参
- projection_matrix：saved 投影矩阵
- cam_rot_delta = 0, cam_trans_delta = 0：第一版 residual 设为零（after_opt 已并入 R/T）
- exposure_a, exposure_b：saved 值或零

**原则**：不使用 GT pose，不依赖旧 runtime frontend.cameras，只使用 saved internal camera states。

**metrics 计算**：replay 渲染不需要 GT pose，但 PSNR/SSIM/LPIPS 计算仍需 GT image。

---

### Phase 4（可选）：pose refit

**新增文件**：part3_BRPO/scripts/replay_internal_eval_with_pose_refit.py（或在 replay 脚本中增加 pose-refit 模式）

**输入参数**：

- ply-path
- internal-cache-root
- dataset-root
- refit-iters
- optimize-exposure（可选）

**pose refit 逻辑**：

对每个 non-KF：

1. 从 dataset 加载真实图像
2. 从 camera_states.json 读取 saved internal R/T 作为初值
3. 构造 ReplayCameraForRefit
4. 固定 refined ply，仅优化 cam_rot_delta, cam_trans_delta（可选 exposure_a/b）
5. 小步数优化后，按官方 update_pose() 逻辑：将 residual 并入 R/T 并清零
6. 保存 camera_states_refit.json
7. 再做 replay render

**注意**：residual 并入后需确认合理性，后续若有问题再改。

---

## 3. 数据结构

### 3.1 internal cache 目录

save_dir/internal_eval_cache/after_opt/
├── manifest.json
├── camera_states.json
├── render_rgb/
├── render_depth_npy/
└── render_depth_vis/  # 可选

### 3.2 internal pseudo prepare 输出

dataset/<scene>/part3_stage1/<run_key>_internal/
├── inputs/
├── difix/
├── pseudo_cache/
└   ├── manifest.json
└   └── samples/{frame_id}/
└       ├── camera.json
└       ├── refs.json
└       ├── render_rgb.png
└       ├── render_depth.npy
└       ├── target_rgb_left.png
└       └── diag/
└── manifests/

### 3.3 replay output 目录

output/part3_internal_replay/<scene>/<run_name>/
├── replay_render_rgb/
├── replay_render_depth_npy/
├── metrics.json                # 可选
├── camera_states_used.json
└── camera_states_refit.json    # 可选

---

## 4. 文件改动清单

| 文件 | 类型 | 说明 |
|------|------|------|
| third_party/S3PO-GS/utils/internal_eval_utils.py | 新增 | 导出 internal cache |
| third_party/S3PO-GS/slam.py | 修改 | after_opt 后新增导出调用 |
| part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po_internal.py | 新增 | internal prepare |
| part3_BRPO/scripts/replay_internal_eval.py | 新增 | replay render / eval |
| part3_BRPO/scripts/replay_internal_eval_with_pose_refit.py | 新增（可选） | pose refit |

---

## 5. 实施顺序

1. S3PO 侧新增 internal_eval_utils.py，在 slam.py after_opt 分支导出 internal cache
2. part3 侧新增 prepare_stage1_difix_dataset_s3po_internal.py，建立 internal prepare 链路
3. 沿用 run_pseudo_refinement.py 对 after_opt ply 做 refine
4. 新增 replay_internal_eval.py，实现 refined ply + saved internal camera states → replay render / eval
5. 可选：新增 pose refit 分支

---

## 6. 注意事项

- **after_opt 为准**：internal baseline 应以 after-opt 为准，saved ply 也是 after-opt
- **replay render 与 metric eval 区分**：replay 不需 GT pose，但 metrics 需 GT image
- **不依赖 runtime frontend.cameras**：non-KF frame 在 frontend 会被 clean() 清掉若干字段，应以 saved camera_states.json + dataset 原图为输入
- **residual 第一版只保存不强依赖**：after_opt replay 默认 zero residual，residual 仅作为 pose refit 的可选初始化
