# PART2 RegGS → Part3 Internal Cache Implementation Plan

更新日期：2026-04-22

## 1. 文档目标

本文档给出一份可执行的工程落地方案，使 RegGS 能产出 Part3 当前可直接消费的 internal cache。目标不是把 RegGS 整体改写成 S3PO，而是让 RegGS 在 **固定 sparse anchors + all non-train pose 导出 + 单 stage render/depth 导出** 的前提下，输出一份与 Part3 当前 `internal_eval_cache` 口径兼容的最小数据包。

本文档同时回答两个工程问题：

1. 是否可以复用此前 `all_non_train_subset_v1` 测试脚本的逻辑来导出 test pose。
2. 需要改哪些文件、增加哪些字段/函数、输出文件名如何约定，以及如何验收。

---

## 2. 先给结论

### 2.1 关于 test pose 导出逻辑

可以。**第一版（v0）应直接复用此前额外测试脚本/Notebook 中的 `all_non_train` pose 导出逻辑**，也就是沿用：

- `compute_estimated_test_c2ws`
- `optimize_estimated_c2w`

这套链路已经在 `third_party/RegGS/src/evaluation/evaluator.py` 中存在，在 `part2/notebooks/03_test.ipynb` 中也已经有过 `all_non_train_subset_v1` 的实际落地和输出证据。第一版不应另起炉灶，而应将这套逻辑**从 notebook/测试入口抽成正式可复用导出单元**。

但要明确语义：

- RegGS 建图主链（infer/refine）本身不依赖 GT pose 做训练初始化；
- 这条 dense test pose 导出链是 **GT-assisted init + per-frame photometric pose refine**；
- 因此它适合做 **Part3 接线版 / exporter v0**，不应误标为“纯 GT-free dense pose export”。

### 2.2 关于 Part3 真正需要的信息

Part3 当前真正消费的是：

1. `kf_indices`（定义 gap / sparse anchors）
2. 每帧 `pose_c2w + intrinsics + image_path/image_name + is_keyframe`
3. all non-train frame 的 `render_rgb` 与 `render_depth_npy`
4. 当前 stage 的 `point_cloud.ply`

当前 Part3 **不依赖** upstream 的 `exposure_a/b`、`cam_rot_delta`、`cam_trans_delta`、residual history 作为硬输入。

### 2.3 工程主张

第一版建议采用如下策略：

1. **sparse anchors 完全对齐 S3PO 当前 `kf_indices`**，不要试图通过 `sample_rate / n_views` 近似逼近。
2. **dense non-train pose 复用 evaluator / 03_test.ipynb 中已存在的 `all_non_train` 导出逻辑**。
3. **只导出单 stage internal cache**，统一映射到 `after_opt/`，避免在 RegGS 上引入多 stage 复杂度。
4. 先保证现有 Part3 `select -> verify -> pack` 可以零改动或极少改动读取，再考虑更纯的 non-GT exporter v1。

---

## 3. Part3 当前消费的最小上游契约

目标输出口径：

```text
<run_root>/internal_eval_cache/
├── manifest.json
├── camera_states.json
└── after_opt/
    ├── point_cloud/point_cloud.ply
    ├── render_rgb/<frame_id>_pred.png
    ├── render_depth_npy/<frame_id>_pred.npy
    └── stage_meta.json
```

### 3.1 manifest.json 最小字段

必须包含：

- `schema_version`
- `kf_indices`
- `non_kf_indices`
- `stages.after_opt`
- `background`（如果渲染路径需要；若 RegGS 无等价字段，可写固定默认背景并在文档中注明）

建议保持与 S3PO internal cache 接近：

```json
{
  "schema_version": "internal-eval-cache-v1",
  "camera_layout": "shared_across_stage",
  "num_frames": 279,
  "kf_indices": [0, 34, 69, 104, 139, 173, 208, 243, 278],
  "non_kf_indices": [...],
  "export_source": "reggs_part3_export_v0",
  "stages": {
    "after_opt": {
      "stage_tag": "after_opt",
      "camera_states_file": "../camera_states.json",
      "point_cloud": "point_cloud/point_cloud.ply",
      "render_rgb_dir": "render_rgb",
      "render_depth_npy_dir": "render_depth_npy",
      "rendered_non_kf_frames": [...]
    }
  }
}
```

### 3.2 camera_states.json 每帧最小字段

必须包含：

- `frame_id`
- `uid`
- `is_keyframe`
- `pose_c2w`
- `fx / fy / cx / cy`
- `image_height / image_width`
- `image_path`
- `image_name`

可选但建议补齐（即使先写 null）：

- `R`
- `T`
- `projection_matrix`
- `cam_rot_delta`
- `cam_trans_delta`
- `exposure_a`
- `exposure_b`

原则：**Part3 当前不依赖这些可选字段，但保留字段位有利于兼容与后续扩展。**

### 3.3 stage_meta.json 最小字段

- `stage_tag`
- `camera_states_file`
- `point_cloud`
- `render_rgb_dir`
- `render_depth_npy_dir`
- `rendered_non_kf_frames`
- `num_rendered_non_kf_frames`

---

## 4. Sparse 对齐策略

## 4.1 为什么不能继续依赖 RegGS 现有 split 公式

RegGS 当前 split 逻辑是：

1. `test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]`
2. `remain_frame_ids = frame_ids \ test_frame_ids`
3. `train_frame_ids = remain_frame_ids[np.linspace(..., n_views)]`

该逻辑同时出现在：

- `third_party/RegGS/src/entities/reggs.py`
- `third_party/RegGS/run_refine.py`
- `third_party/RegGS/src/evaluation/evaluator.py`

它不是“直接在全序列上均匀选 sparse train”，而是“先切 test，再从剩余帧里等距抽 train”。因此即使 sparse 程度相近，也不会自然对齐到 S3PO 当前 Part3 所用的 `kf_indices`。

## 4.2 需要对齐到哪组 anchors

当前 Re10k-1 / Part3 使用的 anchors 应固定为：

```text
[0, 34, 69, 104, 139, 173, 208, 243, 278]
```

它们来自：

- `part2_s3po/configs/s3po_re10k1_full_full.yaml`
- S3PO run 生成的 `internal_eval_cache/manifest.json::kf_indices`

## 4.3 正确改法

不要通过调参逼近；要在 RegGS 中增加**显式 frame split 覆盖机制**。

建议新增配置字段：

```yaml
frame_split:
  mode: explicit
  train_frame_ids: [0, 34, 69, 104, 139, 173, 208, 243, 278]
  test_protocol: all_non_train
  test_frame_ids: null
```

规则：

1. 若 `frame_split.mode == explicit`：
   - `train_frame_ids` 直接使用配置
   - 若 `test_frame_ids` 为 null，默认 `all_non_train = all frames not in train`
2. 若 `frame_split.mode == legacy` 或配置缺失：
   - 回退到原始 `sample_rate + n_views` 逻辑

这样可确保 RegGS train anchors 与 S3PO sparse anchors **完全一致**。

---

## 5. Dense test pose 获取策略

## 5.1 v0：复用已有 all_non_train 逻辑（推荐）

第一版 exporter 直接基于：

- `compute_estimated_test_c2ws`
- `optimize_estimated_c2w`

对所有 non-train frame 求 pose。

这也是你此前 `all_non_train_subset_v1` 的实际做法，只是当时逻辑放在 notebook/测试脚本中。

## 5.2 v0 输出语义

对每个 non-train frame：

1. 根据邻近 train frames 和 GT-assisted 几何关系得到初值 `est_test_c2w_init`
2. 使用该 frame 的真实 RGB 图像做 pose optimization
3. 记录最终 refined `est_test_c2w_final`
4. 使用最终 pose 在当前 RegGS 最终地图上重渲 RGB / depth

## 5.3 v0 必须新增的显式导出

当前 evaluator 只做 test render + metric，并不总是把 dense test pose 作为正式 exporter 产物约定清楚。

需要正式导出：

- `estimated_test_c2w_<tag>.ckpt`
- 或等价 `estimated_test_c2w_<tag>.json`

建议默认命名：

- `estimated_test_c2w_all_non_train_part3_v0.ckpt`

如果是 subset/smoke：

- `estimated_test_c2w_all_non_train_subset_v1.ckpt`
- `estimated_test_c2w_part3_smoke_v0.ckpt`

原则：**文件名中必须编码 test 协议与导出标签。**

---

## 6. 需要改动的文件与职责

本节给出建议最小改动集。

### 6.1 `third_party/RegGS/src/utils/frame_split.py`（新文件）

**新增**一个公共 split 解析模块，统一 infer / refine / metric 三处逻辑。

建议函数：

```python
def resolve_frame_split(config: dict, n_frames: int) -> dict:
    """
    return {
        train_frame_ids: np.ndarray,
        test_frame_ids: np.ndarray,
        test_protocol: str,
        split_mode: str,
    }
    """
```

职责：

1. 解析 `frame_split.mode == explicit`
2. 回退 legacy `sample_rate + n_views`
3. 保证 train/test 不重叠
4. 产出统一的 train/test ids

### 6.2 `third_party/RegGS/src/entities/reggs.py`

**修改**：将现有硬编码 split 替换为 `resolve_frame_split(...)`。

当前职责变化：

- 使用 `train_frame_ids` 作为建图 sparse anchors
- 保存 `self.train_frame_ids / self.test_frame_ids`
- 保证 `estimated_c2ws` 的索引语义与 `train_frame_ids` 一一对应

### 6.3 `third_party/RegGS/run_refine.py`

**修改**：复用 `resolve_frame_split(...)`，不要再本地重复 split 公式。

职责：

- refine 阶段 train frame 使用与 infer 完全一致的 `train_frame_ids`
- 保证与后续 metric/export 一致

### 6.4 `third_party/RegGS/src/evaluation/evaluator.py`

**修改重点最多**。

需要做三件事：

1. split 来源统一走 `resolve_frame_split(...)`
2. 将 `eval_test_render()` 中的 test pose 计算/优化逻辑抽成可导出的函数
3. 增加 internal cache export 所需的 pose / render 输出

建议新增函数：

```python
def build_test_pose_bank(self) -> dict:
    """返回 all test/non-train frame 的初值与最终 refined c2w"""


def export_test_pose_bank(self, pose_bank: dict, output_tag: str) -> Path:
    """保存 estimated_test_c2w_<tag>.ckpt/json"""


def render_views_with_pose_bank(self, pose_bank: dict, frame_ids: list, render_dir: Path, save_depth: bool = True):
    """按给定 pose bank 渲染 RGB/depth"""
```

建议保留旧 `eval_test_render()`，但内部调用新函数，避免破坏现有 notebook/metric 入口。

### 6.5 `third_party/RegGS/src/utils/internal_cache_export.py`（新文件）

**新增** internal cache exporter。

建议函数：

```python
def export_internal_cache(
    *,
    run_root: Path,
    dataset,
    train_frame_ids,
    test_frame_ids,
    train_c2ws,
    test_c2ws,
    stage_ply_path: Path,
    render_rgb_dir: Path,
    render_depth_dir: Path,
    output_root: Path,
    stage_tag: str = after_opt,
    export_label: str = reggs_part3_v0,
):
    ...
```

职责：

1. 组织 `manifest.json`
2. 组织 `camera_states.json`
3. 将 stage ply / render 结果链接或复制到 `internal_eval_cache/after_opt/`
4. 写 `stage_meta.json`

### 6.6 `third_party/RegGS/run_metric.py`

**轻改或不改都可**。

如果维持当前入口，可以只新增两个可选参数：

- `--export_test_pose_bank`
- `--export_internal_cache_root`

如果希望保持 `run_metric.py` 简洁，也可新增独立入口：

### 6.7 `third_party/RegGS/run_export_internal_cache.py`（新文件，推荐）

推荐新增一个专门导出入口，职责单一，避免 `run_metric.py` 过载。

建议参数：

```bash
python run_export_internal_cache.py \
  --checkpoint_path <run_output> \
  --config_path <config.yaml> \
  --stage_ply global_refined_gs \
  --test_protocol all_non_train \
  --output_tag part3_v0 \
  --internal_cache_root <run_output>/internal_eval_cache
```

职责：

1. 读取 run output
2. 生成 all non-train pose bank
3. 渲染 non-train RGB/depth
4. 导出 internal cache

### 6.8 `part2/notebooks/03_test.ipynb`

**不建议继续作为主实现承载点**。

它可以保留作 smoke/debug notebook，但应将当前 `all_non_train_subset_v1` 逻辑迁回正式 python 函数/脚本，notebook 只负责调用和观察输出。

---

## 7. 输出文件命名约定

为避免历史结果与新 exporter 混淆，建议统一命名规则。

### 7.1 pose bank

- `estimated_c2w.ckpt`：train sparse poses（保持现状）
- `estimated_test_c2w_all_non_train_part3_v0.ckpt`：all non-train refined test poses
- `estimated_test_c2w_all_non_train_subset_v1.ckpt`：历史 notebook subset/test 输出（保留）

### 7.2 test metrics

- `eval_test.json`：legacy sampled_test
- `eval_test_all_non_train_part3_v0.json`：all non-train exporter 对应测试指标

### 7.3 rendered views

- `test_all_non_train_part3_v0/`：all non-train RGB 渲染目录
- `depth_all_non_train_part3_v0/`：如果选择单独保存 depth，可显式分目录

### 7.4 internal cache

推荐固定为：

```text
<run_output>/internal_eval_cache_part3_v0/
```

或若必须复用现有名：

```text
<run_output>/internal_eval_cache/
```

建议第一版先用带 tag 的根目录，避免覆盖历史结果：

- `internal_eval_cache_part3_v0/`

根内 stage 固定：

- `after_opt/`

---

## 8. 推荐实施顺序

### Phase 0：对齐配置与 split

目标：让 RegGS train anchors 与 S3PO `kf_indices` 完全一致。

动作：

1. 新增 `frame_split` 配置结构
2. 实现 `resolve_frame_split()`
3. 替换 `reggs.py / run_refine.py / evaluator.py` 中重复 split 逻辑
4. 输出日志中明确打印 `split_mode / train_frame_ids / test_frame_ids[:N]`

验收：

- `train_frame_ids == [0, 34, 69, 104, 139, 173, 208, 243, 278]`
- `test_frame_ids == all frames not in train_frame_ids`
- infer/refine/metric 三处打印一致

### Phase 1：正式化 dense test pose export

目标：将 notebook 中已有 `all_non_train_subset_v1` 能力收回正式 evaluator/exporter。

动作：

1. 抽取 pose bank 构建与保存函数
2. 对 all non-train 生成 `estimated_test_c2w_all_non_train_part3_v0.ckpt`
3. 可选保留 subset mode 做 smoke

验收：

- exporter 可无 notebook 独立运行
- pose bank 文件存在
- pose bank 中 frame_id 覆盖所有 non-train frame

### Phase 2：渲染 non-train RGB/depth

目标：为 Part3 提供 pseudo candidates 对应的 render cache。

动作：

1. 基于最终地图（优先 `global_refined_gs.ply`）渲染 all non-train frame
2. 保存 RGB / depth
3. 输出渲染 frame 列表

验收：

- `render_rgb/<frame_id>_pred.png` 完整覆盖 non-train
- `render_depth_npy/<frame_id>_pred.npy` 完整覆盖 non-train
- 数量与 `test_frame_ids` 一致

### Phase 3：导出 internal cache

目标：产出 Part3 可直接读取的 cache root。

动作：

1. 写 `manifest.json`
2. 写 `camera_states.json`
3. 写 `after_opt/stage_meta.json`
4. 复制/链接最终 ply
5. 复制/链接 render RGB/depth

验收：

- `select_signal_aware_pseudos.py` 可以成功读取该 root
- `brpo_verify_single_branch.py` 可以成功读取该 root
- `prepare_stage1_difix_dataset_s3po_internal.py` 可以成功读取该 root

### Phase 4：Part3 烟雾测试

目标：证明 RegGS → Part3 链路打通。

动作：

1. 用少量 frame ids / 少量 gaps 跑 `select`
2. 跑 `verify`
3. 跑 `pack`
4. 检查 `pseudo_cache/samples/*/camera.json, refs.json, render_rgb.png, render_depth.npy`

验收：

- `camera.json.pose_c2w` / `refs.json.left/right_ref_pose` 完整
- Part3 无 schema mismatch
- smoke 成功后再进入正式 compare

---

## 9. 运行与验收清单

### 9.1 Split 验收

- [ ] train sparse ids 与 S3PO `kf_indices` 完全一致
- [ ] all non-train test 集大小正确
- [ ] infer/refine/metric 使用同一套 split

### 9.2 Pose bank 验收

- [ ] `estimated_test_c2w_all_non_train_part3_v0.ckpt` 存在
- [ ] non-train 每个 frame 都有 pose
- [ ] train pose 与 test pose 的 frame id 索引语义清晰、不混淆

### 9.3 Render cache 验收

- [ ] non-train 每帧都有 `render_rgb`
- [ ] non-train 每帧都有 `render_depth_npy`
- [ ] RGB/depth 数量与 `test_frame_ids` 一致

### 9.4 Internal cache 验收

- [ ] `manifest.json` 有 `kf_indices/non_kf_indices`
- [ ] `camera_states.json` 每帧有 `pose_c2w + intrinsics + image_path`
- [ ] `stage_meta.json` 有 `rendered_non_kf_frames`
- [ ] `after_opt/point_cloud/point_cloud.ply` 存在

### 9.5 Part3 消费验收

- [ ] `select_signal_aware_pseudos.py` 成功
- [ ] `brpo_verify_single_branch.py` 成功
- [ ] `prepare_stage1_difix_dataset_s3po_internal.py` 成功
- [ ] 形成可读 `pseudo_cache/manifest.json`

---

## 10. 不在第一版范围内的内容

以下内容明确不纳入 v0：

1. 不在 RegGS 上构造 before/after 多 stage 体系
2. 不尝试让 dense test pose export 摆脱 GT-assisted init
3. 不把 RegGS camera state 扩展到完全复刻 S3PO 的 exposure/cam_delta 语义
4. 不在 Part3 `select/verify/pack` 主体上做大改动

v0 的目标只有一个：**让 RegGS 成为 Part3 当前 pipeline 可直接接入的另一个上游 source。**

---

## 11. 最终建议

执行时应坚持两条工程纪律：

1. **先做严格 sparse 对齐，再做 dense test pose/export。** 否则 Part3 compare 里混入的将不只是“上游方法差异”，还包括“anchors 统计差异”。
2. **先把 notebook 中已有能力抽回正式代码，再谈新 exporter。** 不要在 notebook 和正式脚本里各维护一套 all_non_train 逻辑。

对当前任务而言，最优路线是：

> 用 S3PO 当前 `kf_indices` 作为 RegGS 显式 `train_frame_ids`；
> 复用已有 `all_non_train_subset_v1` 所在逻辑，整理成正式的 pose bank/export 函数；
> 导出单 stage `internal_eval_cache_part3_v0/after_opt/`；
> 以现有 Part3 `select -> verify -> pack` 作为最终验收。


---

## 12. 实施进度记录

### 2026-04-22 Phase 1 完成 ✅

**目标：固定 sparse + 正式化 dense test pose 导出**

**已完成改动：**

1. **新增 `third_party/RegGS/src/utils/frame_split.py`**
   - 提供 `resolve_frame_split()` 公共函数
   - 支持 `frame_split.mode=explicit` 显式指定 train_frame_ids
   - 支持 `test_protocol=sampled_test / all_non_train`
   - 统一 infer/refine/metric 三处 split 口径

2. **修改 `third_party/RegGS/src/entities/reggs.py`**
   - 训练入口改用 `resolve_frame_split()`
   - 打印 `Frame split mode` 和统一 train/test ids

3. **修改 `third_party/RegGS/run_refine.py`**
   - Refinement 入口改用 `resolve_frame_split()`
   - 增加 `estimated_c2ws` 与 `train_frame_ids` 数量一致性检查

4. **修改 `third_party/RegGS/src/evaluation/evaluator.py`**
   - Evaluator 初始化改用 `resolve_frame_split()`
   - 新增 `build_test_pose_bank(frame_ids, do_pose_opt)` 方法
   - 新增 `export_test_pose_bank(pose_bank, output_tag)` 方法
   - `eval_test_render()` 支持 `save_test_poses=True` 参数

5. **修改 `third_party/RegGS/run_metric.py`**
   - 新增 CLI 参数：`--test_protocol`, `--test_output_tag`, `--save_test_poses`

6. **新增配置示例**
   - `/home/bzhang512/CV_Project/part2/configs/reggs_re10k1_re10k-ckpt_exact_sparse_part3_v0.yaml`
   - 固定 `train_frame_ids=[0,34,69,104,139,173,208,243,278]`
   - `test_protocol=all_non_train`

**自主验收结果：**

- ✅ resolver split 输出 train ids 完全对齐目标 `[0,34,69,104,139,173,208,243,278]`
- ✅ Refinement 与 Evaluator 使用同一套 split（打印一致）
- ✅ smoke test（2 帧）pose bank 成功导出：
  - `estimated_test_c2w_phase1_smoke_opt2.ckpt`
  - frame_ids=[1,2], pose shape=(2,4,4)
- ✅ `py_compile` 全部通过

**结论：Phase 1 通过，sparse 对齐与 formal pose export 基础已就位。**

---

### 2026-04-22 Phase 2 完成 ✅

**目标：导出 Part3-compatible internal cache**

**已完成改动：**

1. **新增 `third_party/RegGS/run_export_internal_cache.py`**
   - 单独 exporter 入口
   - CLI 参数：`--checkpoint-path`, `--test-protocol`, `--output-tag`, `--stage-tag`, `--ply-name`, `--frame-ids`, `--do-pose-opt/--no-pose-opt`, `--overwrite`
   - 输出目录：`<checkpoint>/internal_eval_cache_<tag>/`

2. **导出内容：**
   - `manifest.json`（schema_version, kf_indices, non_kf_indices, background, stages）
   - `camera_states.json`（每帧 pose_c2w, intrinsics, image_path/image_name, is_keyframe, projection_matrix 等）
   - `after_opt/` 单 stage：
     - `point_cloud/point_cloud.ply`（复制指定 ply）
     - `render_rgb/<fid>_pred.png`
     - `render_depth_npy/<fid>_pred.npy`
     - `stage_meta.json`（rendered_non_kf_frames, metrics）

**自主验收结果（smoke, 3 帧）：**

- ✅ 输出目录结构符合 S3PO 口径
- ✅ `manifest.json`：kf_indices 完全对齐，270 个 non_kf_indices
- ✅ `camera_states.json`：279 个帧，每个包含必需字段
- ✅ render_rgb/render_depth_npy：3 个 PNG + 3 个 NPY
- ✅ stage_meta.json：metrics 字段存在（PSNR=27.67, SSIM=0.931, LPIPS=0.058）
- ✅ point_cloud.ply 复制成功

**输出路径示例：**
```
/home/bzhang512/CV_Project/output/part2/re10k_1/reggs_re10k1_re10k-ckpt_sr50_nv9_sm2_comparison_check/internal_eval_cache_phase2_smoke/
```

**结论：Phase 2 通过，RegGS -> S3PO-compatible internal cache exporter 可用。**

---

### 2026-04-22 Phase 3 验证进行中 ⏳

**目标：用 Part3 现有脚本消费导出的 internal cache**

**已尝试：**
- 运行 `select_signal_aware_pseudos.py` 对 phase2_smoke cache

**遇到问题：**
1. **缺少 `config.yml`**：Part3 的 `select_signal_aware_pseudos.py` 依赖 `<cache_root>/../config.yml`（S3PO run 的配置）
   - 已临时补写一个最小 `config.yml` 在 RegGS checkpoint 目录下

2. **PLY 格式兼容性问题**：RegGS PLY 包含完整 SH coefficients（f_dc_0/1/2 + 45 个 f_rest_*），而 S3PO 的 `GaussianModel.load_ply()` 期望特定格式并做 SH degree 校验（assertion failed）
   - S3PO PLY：仅 f_dc + opacity + scale + rotation（无 f_rest）
   - RegGS PLY：包含 f_dc + 45 个 f_rest（对应 SH degree=3）

**当前状态：**
- ❌ `select_signal_aware_pseudos.py` 因 PLY 加载断言失败无法继续
- Part3 消费路径依赖 S3PO 的 `GaussianModel`，而 RegGS PLY 格式与其不兼容

**下一步计划：**
- **方案 A（短期）**：修改 Part3 的 `load_gaussians_from_ply` / 相关函数，让它能识别 RegGS PLY 格式，或使用 RegGS 的 `GaussianModel.load_ply()`
- **方案 B（更干净）**：在 exporter 中转换 PLY 格式，只保留 DC + opacity + scale + rotation，去掉 f_rest，让输出 PLY 与 S3PO 的 GaussianModel 兼容
- **方案 C（绕过）**：在 Part3 脚本中提供 `--skip-gaussian-load` 或 `--use-reggs-gaussian` 参数，允许用 RegGS 自己的加载逻辑

**优先级判断：**
- 方案 B 最干净（exporter 负责格式转换，下游不需要改动）
- 方案 A 最快（改下游一个函数）
- 方案 C 是中间路线

**建议：**
- 先用方案 A 快速打通 Phase 3 验证，证明 cache 结构正确
- 再考虑是否在 exporter 中做格式标准化（方案 B）

---

## 13. 阶段验收清单

### Phase 1 验收 ✅

- [x] `resolve_frame_split()` 实现
- [x] explicit split 配置生效
- [x] infer/refine/metric 三处 split 一致
- [x] pose bank 导出方法可用
- [x] smoke test pose bank 落盘

### Phase 2 验收 ✅

- [x] exporter 入口可用
- [x] manifest.json 结构符合 S3PO 口径
- [x] camera_states.json 字段完整
- [x] render_rgb / render_depth_npy 导出成功
- [x] point_cloud.ply 复制成功
- [x] stage_meta.json 包含 metrics

### Phase 3 验收 ⏳

- [ ] `select_signal_aware_pseudos.py` 能加载 cache
- [ ] `prepare_stage1_difix_dataset_s3po_internal.py` 能消费 cache
- [ ] `build_brpo_v2_signal_from_internal_cache.py` 能消费 cache
- [ ] 最终能形成 `pseudo_cache/samples/*/camera.json, refs.json`
- [ ] 证明 RegGS cache 能被 Part3 现有 pipeline 读取

---

**更新日期：2026-04-22 17:26 GMT+8**
