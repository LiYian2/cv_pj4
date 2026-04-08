# Pseudo Cache Schema 改造建议与迁移计划

## 1. 目标

本文档用于给出 pseudo cache 从现状结构升级到 `pseudo-cache-v1.1` 的实施建议，覆盖以下内容：

1. 现有结构与目标规范的差距。
2. 数据结构改造项。
3. 是否需要新增脚本及其职责边界。
4. 推荐执行顺序与风险控制。

本文件用于工程落地，不包含 Stage2 full EDP 或 backend 训练改造细节。

## 2. 现状摘要

当前主数据构建脚本：

1. `part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py`

当前产物具备以下能力：

1. `inputs/raw_render`、`inputs/left_ref`、`inputs/right_ref`。
2. `difix/left_fixed`、`difix/right_fixed`。
3. `pseudo_cache/samples/<frame_id>/camera.json`、`render_rgb.png`、`render_depth.npy`。
4. `pseudo_cache/manifest.json` 的基础样本索引。

当前主要缺口：

1. 缺少 `schema_version`、`backend`、`image_size` 等顶层规范字段。
2. 缺少 `refs.json` 作为 additional 几何构造上下文。
3. `camera.json` 未统一包含 `intrinsics_px` 与 `intrinsics_source`。
4. `samples[]` 未按 additional 阶段统一暴露目标路径字段（`target_depth_path`、`confidence_mask_path`、`diag_dir`）。
5. 缺少通用 schema 校验流程。

## 3. 结构改造建议

## 3.1 顶层清单（`pseudo_cache/manifest.json`）

建议新增并固定以下字段：

1. `schema_version`：当前值 `pseudo-cache-v1.1`。
2. `backend`：当前值 `s3po`。
3. `source_run_root`：external eval 运行目录。
4. `image_size`：`{"width": 512, "height": 512}`。

`samples[]` 建议统一字段（保持相对路径语义）：

1. `sample_id`
2. `frame_id`
3. `test_idx`
4. `placement`
5. `camera_path`
6. `refs_path`
7. `render_rgb_path`
8. `render_depth_path`
9. `target_rgb_left_path`
10. `target_rgb_right_path`
11. `target_depth_path`
12. `confidence_mask_path`
13. `diag_dir`

## 3.2 样本级文件

### 3.2.1 `camera.json`

建议固定最小字段：

1. `camera_schema`
2. `frame_id`
3. `test_idx`
4. `pose_c2w`
5. `intrinsics_px`（`fx, fy, cx, cy`）
6. `image_size`
7. `intrinsics_source`

### 3.2.2 `refs.json`

建议新增 `refs.json`，用于 additional 阶段几何构造：

1. `left_ref_frame_id`
2. `right_ref_frame_id`
3. `left_ref_rgb_path`
4. `right_ref_rgb_path`
5. `left_ref_pose`
6. `right_ref_pose`
7. `ref_pose_source`

说明：

1. `refs.json` 属于数据生产与几何构造上下文，不要求 backend 训练阶段直接消费。

## 4. 跨数据集相机来源策略

### 4.1 405841

建议策略：

1. pseudo 位姿从 `trj_external_infer.json` 读取。
2. resize 后内参从 `part2_s3po/*/split_manifest.json` 的 `calibration_sync` 读取。
3. 左右参考位姿从 `FRONT/gt/<frame_id>.txt` 读取。

### 4.2 Re10k-1 / DL3DV-2

建议策略：

1. 优先读取 `part2_s3po/*/cameras.json` 与 `intrinsics_px.json`。
2. 文件缺失时回退到 split manifest 中可用字段。

## 5. 是否需要新增脚本

结论：需要，且建议新增两类脚本。

### 5.1 迁移脚本（必需）

建议文件：

1. `part3_BRPO/scripts/migrate_pseudo_cache_schema_v11.py`

职责：

1. 对既有 run 执行结构迁移，不重跑 Difix。
2. 补齐 `manifest.json` 新字段。
3. 生成每个 sample 的 `refs.json`。
4. 升级 `camera.json`，补齐 `intrinsics_px` 与来源信息。

### 5.2 校验脚本（必需）

建议文件：

1. `part3_BRPO/scripts/validate_pseudo_cache_schema.py`

职责：

1. 校验字段完整性与类型。
2. 校验路径存在性与相对路径语义。
3. 校验样本计数一致性。
4. 输出结构化统计与错误列表。

### 5.3 现有生产脚本改造（必需）

1. 在 `prepare_stage1_difix_dataset_s3po.py` 的 `pack` 阶段直接产出 v1.1 格式。
2. 避免后续所有新 run 都依赖迁移脚本二次处理。

## 6. 推荐执行顺序

1. 冻结 `pseudo-cache-v1.1` 字段定义（以 schema 规范文档为准）。
2. 实现 `validate_pseudo_cache_schema.py`。
3. 实现 `migrate_pseudo_cache_schema_v11.py`。
4. 在单场景（建议 405841）执行迁移与校验。
5. 批量迁移 Re10k-1 与 DL3DV-2。
6. 修改 `prepare_stage1_difix_dataset_s3po.py`，使新产物原生满足 v1.1。
7. 全部通过后进入 additional 模块实现（target depth / confidence / diag）。

## 7. 风险与控制

1. 历史 run 目录字段不一致：通过迁移脚本统一。
2. 路径混用绝对/相对：校验脚本强约束。
3. 405841 相机来源分散：在迁移脚本中固定读取优先级并记录 `intrinsics_source`。
4. 新旧字段并存导致解析歧义：以 `schema_version` 控制分支。

## 8. 交付清单

1. 规范文档：`part3_BRPO/docs/pseudo_cache_schema_spec.md`
2. 迁移方案文档：`part3_BRPO/docs/pseudo_cache_schema_migration_plan.md`
3. 迁移脚本：`part3_BRPO/scripts/migrate_pseudo_cache_schema_v11.py`
4. 校验脚本：`part3_BRPO/scripts/validate_pseudo_cache_schema.py`
5. 更新后的生产脚本：`part3_BRPO/scripts/prepare_stage1_difix_dataset_s3po.py`
