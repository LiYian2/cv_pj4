# Part3 Internal Cache / Replay Engineering Plan

> 目标：建立一条**不影响 S3PO 原有 part2 主流程**的 internal 路线，使以下链路可执行：  
> **S3PO(after_opt) 保存 internal cache → part3 internal prepare → pseudo cache → refine ply → replay render / replay eval → 可选 pose refit**

---

## 1. 当前已经确认的关键事实

### 1.1 当前 pseudo 改动不会破坏普通 part2 / S3PO 主流程

当前本地改动中：

- `utils/slam_utils.py` 只是**新增**了 `get_loss_pseudo()`，原有 `get_loss_tracking*()` 与 `get_loss_mapping*()` 仍然走原逻辑；
- `utils/slam_backend.py` 虽然新增了 `pseudo_refinement()`，但 `run()` 里**没有**新增对应消息分支；
- `slam.py` 也**没有**向 backend queue 发 `"pseudo_refinement"` 相关消息。

因此：

- 当前代码仍支持**不带 pseudo 的 part2 / S3PO 正常运行**；
- internal cache 保存链路应尽量做成**附加功能**，不要改写现有 `init → keyframe → map → color_refinement` 主流程。

---

### 1.2 官方 internal eval 当前不会自动把渲染缓存完整落盘

S3PO 官方 `utils/eval_utils.py::eval_rendering()` 当前虽然创建了：

- `render_rgb/`
- `render_depth/`
- `render_depth_npy/`

但真正保存 `pred png` 和 `depth npy` 的两行代码是注释掉的。  
因此，若要为 Part3 internal 路线提供稳定输入，**必须显式新增 internal cache 导出逻辑**，不能只依赖现有 `eval_rendering()`。

---

### 1.3 渲染真正依赖哪些 camera / viewpoint 信息

从官方 `gaussian_splatting/gaussian_renderer/__init__.py::render()` 可知，渲染时 rasterizer 直接依赖：

- `viewpoint_camera.FoVx`
- `viewpoint_camera.FoVy`
- `viewpoint_camera.image_height`
- `viewpoint_camera.image_width`
- `viewpoint_camera.world_view_transform`
- `viewpoint_camera.full_proj_transform`
- `viewpoint_camera.projection_matrix`
- `viewpoint_camera.camera_center`
- `viewpoint_camera.cam_rot_delta`
- `viewpoint_camera.cam_trans_delta`

其中：

- `world_view_transform`
- `full_proj_transform`
- `camera_center`

都可以由以下基础量重新构造：

- `R`
- `T`
- `projection_matrix`
- `fx, fy, cx, cy`
- `FoVx, FoVy`
- `image_height, image_width`

因此，**用于 replay render 的核心保存集**不是运行时对象本身，而是足够重建这些属性的相机状态。

---

### 1.4 residual 是否必须保存

官方 `utils/pose_utils.py::update_pose()` 的行为是：

1. 用 `cam_rot_delta / cam_trans_delta` 构造增量 SE(3)
2. 更新 `camera.R / camera.T`
3. 将 `cam_rot_delta / cam_trans_delta` 清零

这说明对于**最终收敛后的 after_opt 渲染 replay**：

- 只要保存的是最终 `R/T`，通常就已经足够复现 render；
- `cam_rot_delta / cam_trans_delta` 对于 **final replay render 不是硬依赖**。

但是，为了后续可能做 **pose refit**，仍建议把这几个量一起保存：

- `cam_rot_delta`
- `cam_trans_delta`
- `exposure_a`
- `exposure_b`

保存成本很低，后续扩展空间大。

---

## 2. 需要保存的 internal cache：完整清单

建议按两个层次保存：

---

### 2.1 Level A：replay render 必须保存的信息

这些是 replay 渲染的最小必要集。

#### 每帧 camera state

建议保存到：

```text
save_dir/internal_eval_cache/after_opt/camera_states.json
```

每帧字段建议如下：

```json
{
  "frame_id": 132,
  "uid": 132,
  "is_keyframe": false,
  "R": [[...], [...], [...]],
  "T": [...],
  "fx": ...,
  "fy": ...,
  "cx": ...,
  "cy": ...,
  "FoVx": ...,
  "FoVy": ...,
  "image_height": 512,
  "image_width": 512,
  "projection_matrix": [[...], [...], [...], [...]]
}
```

#### 顶层 manifest

建议保存到：

```text
save_dir/internal_eval_cache/after_opt/manifest.json
```

顶层字段建议如下：

```json
{
  "iteration": "after_opt",
  "num_frames": ...,
  "kf_indices": [...],
  "non_kf_indices": [...],
  "background": [0.0, 0.0, 0.0],
  "pipeline_params": {
    "compute_cov3D_python": false,
    "convert_SHs_python": false,
    "debug": false
  }
}
```

#### 渲染缓存

建议保存：

```text
save_dir/internal_eval_cache/after_opt/
├── render_rgb/
├── render_depth_npy/
└── render_depth_vis/
```

说明：

- `render_rgb/`：保存 after_opt internal 非 keyframe 渲染结果
- `render_depth_npy/`：保存原始深度矩阵
- `render_depth_vis/`：可选，仅供可视化/debug

---

### 2.2 Level B：二阶段 pose refit 建议保存的信息

这些对 replay render 不是硬依赖，但对后续 pose refit 有帮助。

建议在 `camera_states.json` 每帧附带：

```json
{
  "cam_rot_delta": [...],
  "cam_trans_delta": [...],
  "exposure_a": ...,
  "exposure_b": ...
}
```

说明：

- after_opt replay 第一版可以把 residual 直接视作 0；
- 但如果后面要做 `refined ply + saved pose init → per-frame pose refit`，这些字段最好预留。

---

### 2.3 Level C：调试 / 对照可选保存的信息

以下信息不是 replay 的必要条件，但建议可选保存，便于后期核对：

- `R_gt`
- `T_gt`
- `dataset_idx`
- `original_image_path`（如果路径固定可不存）
- `mono_depth_path`（通常不是 replay render 必需）

---

## 3. 整体工程路线

本工程建议分为四段：

1. **在 S3PO 官方库侧保存 internal cache**
2. **在 part3_BRPO 侧做 internal prepare 链路**
3. **在 refine 之后做 replay render / replay eval**
4. **再加一个可选的二阶段 pose refit**

下面按模块分别说明。

---

## 4. 模块一：在 S3PO 官方库侧保存 internal cache

### 4.1 目标

在 **不改变现有 S3PO 主流程逻辑**的前提下，在 `after_opt` 阶段附加导出：

- `camera_states.json`
- `manifest.json`
- `render_rgb/`
- `render_depth_npy/`
- `render_depth_vis/`

该导出结果将成为后续 Part3 internal 路线的唯一标准输入。

---

### 4.2 建议新增文件

新增文件：

```text
third_party/S3PO-GS/utils/internal_eval_utils.py
```

该文件建议包含两个主函数。

---

### 4.3 函数一：导出 camera states

建议函数：

```python
def export_camera_states(frames, kf_indices, save_dir, config, iteration="after_opt"):
    ...
```

#### 输入

- `frames`：`self.frontend.cameras`
- `kf_indices`：`self.frontend.kf_indices`
- `save_dir`
- `config`
- `iteration`：`after_opt`

#### 输出

```text
save_dir/internal_eval_cache/after_opt/
├── camera_states.json
└── manifest.json
```

#### camera state 记录逻辑

对每个 `frame_id`：

- 读取 `frame.R`, `frame.T`
- 读取 `frame.fx/fy/cx/cy`
- 读取 `frame.FoVx/FoVy`
- 读取 `frame.image_height/image_width`
- 读取 `frame.projection_matrix`
- 可选记录 `frame.cam_rot_delta`, `frame.cam_trans_delta`
- 可选记录 `frame.exposure_a`, `frame.exposure_b`
- 通过 `frame_id in kf_indices` 标记 `is_keyframe`

#### 顶层 manifest 记录逻辑

保存：

- `iteration`
- `kf_indices`
- `non_kf_indices`
- `num_frames`
- `background`
- `pipeline_params`
- `source = "frontend.cameras"`

---

### 4.4 函数二：导出 internal render cache

建议函数：

```python
def export_internal_render_cache(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="after_opt",
):
    ...
```

#### 行为

模仿官方 `eval_rendering()` 的渲染循环，但做以下改动：

1. 只遍历 `non_kf_indices`
2. 真的保存：
   - `render_rgb/{idx}_pred.png`
   - `render_depth_npy/{idx}.npy`
   - `render_depth_vis/{idx}.png`
3. 同时调用 `export_camera_states(...)`

#### 注意

这个函数主要是**导出 internal replay cache**，不必承担 metric 评测职责。  
建议保持职责单一，不要把它直接塞进原始 `eval_rendering()` 本体中。

---

### 4.5 `slam.py` 的修改方案

官方 `slam.py` 在 `after_opt` 时已经有：

- `color_refinement`
- `eval_rendering(..., iteration="after_opt")`
- `save_gaussians(..., "final_after_opt")`

建议在该位置**新增一行调用**，而不替换原逻辑。

#### 建议修改位置

在 `after_opt` 的 `eval_rendering(...)` 完成后新增：

```python
from utils.internal_eval_utils import export_internal_render_cache
```

然后：

```python
export_internal_render_cache(
    self.frontend.cameras,
    self.gaussians,
    self.dataset,
    self.save_dir,
    self.pipeline_params,
    self.background,
    self.frontend.kf_indices,
    iteration="after_opt",
)
```

#### 为什么这不会影响 S3PO 主流程

因为这只是：

- 读取现有结果
- 额外导出缓存

不改变：

- tracking
- keyframe selection
- mapping
- backend queue
- color_refinement 逻辑

---

## 5. 模块二：在 part3_BRPO 侧做 internal prepare 链路

### 5.1 目标

建立 internal 路线的数据准备链：

```text
S3PO internal_eval_cache(after_opt)
    → pseudo sample selection
    → difix
    → pack
    → pseudo_cache
```

这条链不能继续依赖 external 路线的：

- `trj_json`
- `render_rgb_dir`
- `render_depth_dir`

因为 internal 路线的 pose 与 render source 都来自 S3PO 保存的 internal cache。

---

### 5.2 为什么不要硬改现有 external prepare 脚本

当前已有脚本：

```text
part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py
```

它是按 external 假设设计的：

- 读 external `trj_json`
- 读 external `render_rgb_dir`
- 读 external `render_depth_dir`

如果强行塞 internal 分支，会让脚本条件分支过多，后续 external / internal 两条线难维护。

因此建议：

- **保留 external 版本不动**
- **新增 internal 版本脚本**

---

### 5.3 建议新增文件

新增：

```text
part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po_internal.py
```

---

### 5.4 internal prepare 脚本输入

建议参数：

- `--internal-cache-root`
  - 指向：`save_dir/internal_eval_cache/after_opt`
- `--scene-name`
- `--run-key`
- `--dataset-root`
- `--rgb-dir`
- `--full-manifest` 或 `--split-manifest`
- `--placement`
- `--limit`
- `--difix-model-*`

---

### 5.5 internal prepare 逻辑

#### 步骤 1：读取 internal cache

从 `internal_cache_root` 读取：

- `manifest.json`
- `camera_states.json`
- `render_rgb/`
- `render_depth_npy/`

#### 步骤 2：确定 pseudo 候选帧

internal 路线建议默认：

- pseudo 候选 = `non_kf_indices`
- 左右参考 = 邻近 keyframes

第一版不建议改成“所有普通帧都能当左右参考”，先沿用 keyframe 邻接语义。

#### 步骤 3：sample selection

依据 `placement`（如 midpoint / tertile / internal_subsample_rule）  
在 `non_kf_indices` 中选择 pseudo sample。

#### 步骤 4：difix

对每个 pseudo sample：

- 当前帧 internal render 作为 degraded input
- 左右 keyframe RGB 作为 reference
- 产生：
  - `target_rgb_left.png`
  - `target_rgb_right.png`

#### 步骤 5：pack

写出 internal 版 `pseudo_cache/`：

- `camera.json`
- `refs.json`
- `render_rgb.png`
- `render_depth.npy`
- `target_rgb_left.png`
- `target_rgb_right.png`

并保留后续 additional 阶段可用的字段。

---

### 5.6 internal pseudo_cache 中 `camera.json` 的来源

这次 `camera.json` 的核心字段不再来自 external `trj_est`，而是来自 internal `camera_states.json`。

即：

- `pose_c2w`：来自 internal saved pose
- `intrinsics_px`：来自 internal saved intrinsics
- `image_size`：来自 internal saved image size

---

### 5.7 internal pseudo_cache 中 `refs.json` 的来源

左右参考帧应优先来自 internal keyframe states，而不是 external trajectory。

建议字段：

```json
{
  "left_ref_frame_id": ...,
  "right_ref_frame_id": ...,
  "left_ref_rgb_path": "...",
  "right_ref_rgb_path": "...",
  "left_ref_pose": ...,
  "right_ref_pose": ...,
  "ref_pose_source": "internal_eval_cache.after_opt"
}
```

---

## 6. 模块三：refine 之后做 replay render / replay eval

### 6.1 目标

在 refine 得到新 ply 之后，用 **保存的 internal camera states** 做 replay render，回答：

> 在固定 internal accurate pose 的前提下，refined ply 是否真的提升了 internal 渲染质量？

这一步是 internal 路线的核心评测。

---

### 6.2 建议新增文件

新增：

```text
part3_BRPO/scripts/replay_internal_eval.py
```

---

### 6.3 输入

建议参数：

- `--ply-path`
- `--internal-cache-root`
- `--output-dir`
- `--dataset-root`（可选，用于算 metrics）
- `--use-gt-metrics`（可选 bool）

---

### 6.4 replay render 的行为

对于 `internal_cache_root/after_opt/camera_states.json` 中的每个 `non_kf`：

1. 构造 replay camera
2. 用 refined ply 调用 `render()`
3. 保存：
   - `render_rgb/`
   - `render_depth_npy/`
   - `render_depth_vis/`

#### 重要原则

- **不使用 GT pose**
- **不依赖旧 runtime `frontend.cameras`**
- **只使用保存下来的 internal camera states**

---

### 6.5 replay eval 是否需要 GT

需要区分两件事：

#### replay 渲染
不需要 GT pose，只需要 internal camera states。

#### metric 评测（PSNR/SSIM/LPIPS）
如果要算这些指标，仍然需要 GT image。  
因此建议：

- 脚本支持 `render-only`
- 也支持 `render + metrics`

但 metrics 使用的是真实图像，不是 GT pose。

---

### 6.6 replay camera 应如何构造

建议不要依赖原 `frontend.cameras` 运行时对象。  
而是新建一个轻量对象，例如：

```python
class ReplayCamera:
    ...
```

需要字段：

- `R`
- `T`
- `fx`, `fy`, `cx`, `cy`
- `FoVx`, `FoVy`
- `image_height`, `image_width`
- `projection_matrix`
- `cam_rot_delta = 0`
- `cam_trans_delta = 0`

可选：
- `original_image`
- `exposure_a`
- `exposure_b`

#### 为什么 residual 默认设为 0

因为 after_opt replay 目标是复现最终状态渲染，  
而官方 `update_pose()` 已经把优化后的 residual bake 进了 `R/T`。  
因此第一版 replay 建议直接：

- `cam_rot_delta = 0`
- `cam_trans_delta = 0`

---

## 7. 模块四：可选的二阶段 pose refit

### 7.1 为什么需要这一步

虽然 internal saved poses 是 after_opt 时最适合原始 ply 的 pose，  
但 refine 之后 ply 已经改了：

- 外观可能变了
- opacity 可能变了
- 局部几何 / 可见性组织也可能变了

因此，原来最优的 internal pose，不一定仍是 refined ply 下的最优 pose。  
所以需要一个可选的第二阶段：

> 用 saved internal pose 作为 init，在 refined ply 上对每个 non-KF frame 做小步数 pose refinement。

---

### 7.2 建议新增文件

新增：

```text
part3_BRPO/scripts/replay_internal_eval_with_pose_refit.py
```

或者在 `replay_internal_eval.py` 中增加 `--pose-refit` 模式。

---

### 7.3 输入

- `--ply-path`
- `--internal-cache-root`
- `--dataset-root`
- `--output-dir`
- `--refit-iters`
- `--optimize-exposure`

---

### 7.4 pose refit 的核心逻辑

对每个 `non_kf`：

1. 从 dataset 加载真实图像
2. 从 `camera_states.json` 读取 saved internal `R/T`
3. 构造 `ReplayCameraForRefit`
4. 初始化：
   - `R/T` = saved internal pose
   - `cam_rot_delta = 0`
   - `cam_trans_delta = 0`
   - `exposure_a / exposure_b` = saved or 0
5. 固定 refined ply，不更新 gaussians
6. 仅优化：
   - `cam_rot_delta`
   - `cam_trans_delta`
   - 可选 `exposure_a / exposure_b`
7. 小步数优化后，保存：
   - `camera_states_refit.json`
8. 再做 replay render

---

### 7.5 为什么这不会丢掉 internal pose 优势

这一步不是重新用 dataset pose，而是：

- 用 dataset 提供图像张量
- 用 saved internal pose 作为优化初值

因此它本质上还是：

> internal pose → refined map 下的小范围局部修正

而不是回退到外部推断或 GT pose。

---

## 8. 文件级修改清单

### 8.1 官方 S3PO 侧

#### 新增文件

```text
third_party/S3PO-GS/utils/internal_eval_utils.py
```

建议包含：

- `export_camera_states(...)`
- `export_internal_render_cache(...)`

#### 修改文件

```text
third_party/S3PO-GS/slam.py
```

修改点：

- 在 `after_opt` 的 `eval_rendering(...)` 后新增 `export_internal_render_cache(...)`

原则：

- 只读现有结果
- 不改变 tracking / mapping / backend queue
- 不影响普通 S3PO 流程

---

### 8.2 part3_BRPO 侧：internal prepare

#### 新增文件

```text
part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po_internal.py
```

职责：

- 读取 internal cache
- 选 pseudo sample
- 跑 Difix
- pack 成 internal pseudo_cache

#### 可复用文件

- `part3_BRPO/pseudo_branch/build_pseudo_cache.py`
- `part3_BRPO/pseudo_branch/epipolar_depth.py`
- `part3_BRPO/pseudo_branch/diag_writer.py`

说明：

这些 additional 阶段文件本身仍可复用，只是其输入来源会变成 internal pseudo_cache。

---

### 8.3 part3_BRPO 侧：replay render / eval

#### 新增文件

```text
part3_BRPO/scripts/replay_internal_eval.py
```

职责：

- 读取 refined ply
- 读取 internal camera states
- replay render
- 可选 metrics

#### 新增文件（可选）

```text
part3_BRPO/scripts/replay_internal_eval_with_pose_refit.py
```

职责：

- 在 replay 前做 per-frame pose refit
- 保存 `camera_states_refit.json`
- 再 replay render

---

## 9. 建议的数据结构

### 9.1 internal cache 目录

```text
save_dir/internal_eval_cache/after_opt/
├── manifest.json
├── camera_states.json
├── render_rgb/
│   ├── 0001_pred.png
│   └── ...
├── render_depth_npy/
│   ├── 0001.npy
│   └── ...
└── render_depth_vis/
    ├── 0001.png
    └── ...
```

---

### 9.2 internal pseudo prepare 输出目录

```text
dataset/<scene>/part3_stage1/<run_key>_internal/
├── inputs/
├── difix/
├── pseudo_cache/
└── manifests/
```

---

### 9.3 replay output 目录

```text
output/part3_internal_replay/<scene>/<run_name>/
├── replay_render_rgb/
├── replay_render_depth_npy/
├── replay_render_depth_vis/
├── metrics.json                # optional
├── camera_states_used.json
└── camera_states_refit.json    # optional
```

---

## 10. 推荐实施顺序

### 第一步
在 S3PO 官方库中新增 `internal_eval_utils.py`  
并在 `slam.py` 的 after_opt 分支导出 internal cache。

### 第二步
新增 `prepare_stage1_difix_dataset_s3po_internal.py`  
建立 internal prepare 链路。

### 第三步
沿用现有 `run_pseudo_refinement.py`  
对 after_opt ply 做 refine。

### 第四步
新增 `replay_internal_eval.py`  
实现 `refined ply + saved internal camera states → replay render / eval`。

### 第五步（可选增强）
新增 pose refit 分支  
实现 `saved internal pose → refined map 下局部修正 → replay render`。

---

## 11. 风险与注意事项

### 11.1 不要混淆 before_opt 与 after_opt

当前 internal baseline 应以 **after_opt** 为准，因为保存的 ply 也是 after-opt。  
所以：

- internal cache 应导出 after-opt
- replay baseline 也必须和 after-opt 同口径

---

### 11.2 replay render 与 metric eval 要区分清楚

- replay render：不需要 GT pose
- metric eval：仍然需要 GT image

不要把“无需 GT pose”误写成“完全不需要 GT”。

---

### 11.3 第一版 replay 不建议依赖 runtime `frontend.cameras`

因为 non-KF frame 在 frontend 里会被 `clean()` 清掉若干字段。  
应始终以：

- `camera_states.json`
- dataset 原图

作为 replay / refit 的重建输入。

---

### 11.4 residual 第一版建议只保存，不强依赖

对于 after_opt replay：

- 第一版默认 zero residual
- residual 仅作为 pose refit 的可选初始化信息

---

## 12. 最终结论

这套 internal 路线的工程核心不是“改动 S3PO 主流程”，而是**附加一条标准化缓存导出链**。  
一旦这条链跑通，你就可以稳定建立以下流程：

```text
S3PO(after_opt)
    -> export internal_eval_cache
    -> prepare_stage1_difix_dataset_s3po_internal.py
    -> pseudo_cache
    -> run_pseudo_refinement.py
    -> refined ply
    -> replay_internal_eval.py
    -> optional pose_refit
```

这条路线满足你当前老师给的要求：

- external+sparse 作为稳妥底线
- full+internal 作为更难但更贴近 S3PO / PseudoView 语境的增强路线

并且它不需要改写 S3PO 的 tracking / mapping 主逻辑，只需要：

- 新增 internal cache 导出
- 新增 internal prepare
- 新增 replay / refit 脚本

这是目前最稳、最清晰、最容易逐步验证的工程方案。
