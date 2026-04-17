# Part1 VGGT+BA 实验落地方案

## 1. 目标与背景

### 问题
Part1 报告（PART1_UNIFIED_EXPERIMENT_REPORT.md §11）指出：
- VGGT-108（无 BA）比 COLMAP-108 差约 4.8 dB PSNR
- 原因：feedforward export 导致 track length=1.0，无多视图几何约束
- 无法形成真正的 triangulation，只能依赖 depth prediction

### 目标
验证开启 BA 后，VGGT 能否缩小与 COLMAP-108 的差距。

### 实验矩阵
| Scene | Model | Variant | 组合数 |
|-------|-------|---------|--------|
| Re10k-1 | 3dgs | vggt_ba_108 | 1 |
| Re10k-1 | scaffold | vggt_ba_108 | 1 |
| DL3DV-2 | 3dgs | vggt_ba_108 | 1 |
| DL3DV-2 | scaffold | vggt_ba_108 | 1 |
| 405841 | 3dgs | vggt_ba_108 | 1 |
| 405841 | scaffold | vggt_ba_108 | 1 |
| **总计** | | | **6 组** |

---

## 2. VGGT BA 参数设置

### demo_colmap.py 关键参数
```bash
python demo_colmap.py \
    --scene_dir <scene_dir> \
    --use_ba                    # 开启 BA（关键）\
    --shared_camera             # 可选：共享相机（默认 False）\
    --camera_type SIMPLE_PINHOLE # 可选：相机类型（默认 SIMPLE_PINHOLE）\
    --query_frame_num 8         # BA 分支默认 8（之前无 BA 用的是 4）\
    --max_query_pts 4096        # BA 分支默认 4096（之前无 BA 用的是 1024）\
    --fine_tracking             # 可选：精细追踪（可能显存爆炸）\
    --vis_thresh 0.2            # visibility threshold
```

### 推荐配置（保守方案）
基于现有 notebook 和显存限制，推荐：
```python
VGGT_BA_ARGS = {
    'use_ba': True,
    'shared_camera': False,      # 默认 False，可尝试 True
    'camera_type': 'SIMPLE_PINHOLE',  # 默认
    'query_frame_num': 8,        # BA 分支建议用默认值
    'max_query_pts': 4096,       # BA 分支建议用默认值
    'fine_tracking': False,      # 保守：避免显存爆炸
    'vis_thresh': 0.2,
    'max_reproj_error': 8.0,
}
```

### 显存风险
- BA 分支需要 `predict_tracks`，显存占用更高
- fine_tracking 可能导致 OOM
- 建议：先用 conservative config 试一个 scene，观察显存

---

## 3. 输入输出路径规划

### 输入数据（已存在）
```
/home/bzhang512/CV_Project/dataset/<SCENE>/part1/shared/
  ├── images_512/           # 512x512 图片
  └── subsets/subset_108.txt # 108 图列表
```

### VGGT BA 工作目录（新建）
```
/home/bzhang512/CV_Project/dataset/<SCENE>/part1/planB/vggt_ba_108/
  ├── inputs/scene/images/  # VGGT 输入（从 subset_108 复制）
  ├── raw_vggt/             # VGGT 原始输出（含 BA）
  ├── converted_colmap/     # COLMAP 格式转换
  ├── eval/                 # 评估数据
  └── gs_scene/             # 最终 3DGS 输入
      ├── images/
      └── sparse/0/
```

### 训练输出目录
```
/home/bzhang512/CV_Project/output/part1/<SCENE>/planB/vggt_ba_108/
  ├── 3dgs/default_40k_eval2k/
  │   └ results.json
     └ logs/
  └── scaffold/final_40k_eval2k/
      ├── results.json
         └ logs/
```

---

## 4. 实验流程

### Phase 1：准备 VGGT BA gs_scene（手动 notebook）
参考 `03_planB_vggt.ipynb`，创建新 notebook 或修改现有：

**步骤**：
1. 从 subset_108 复制图片到 `vggt_ba_108/inputs/scene/images/`
2. 运行 `demo_colmap.py --use_ba`（BA 分支）
3. 检查 sparse 输出（track length 应 >1.0）
4. 整理成 `gs_scene/sparse/0` 结构

**关键验证**：
- track length > 1.0（vs 无 BA 的 1.0）
- observations > points（vs 无 BA 的 100k/100k）
- reprojection error > 0（vs 无 BA 的 0.0）

### Phase 2：训练脚本（sh）
创建 `run_part1_vggt_ba_6_experiments.sh`：
- 参照 `run_part1_12_experiments.sh` 格式
- 只跑 planB + vggt_ba_108 + 2 models
- 40k iterations, eval every 2k

### Phase 3：结果解析
- 用现有 `04_read_results.ipynb` 解析
- 对比 vggt_ba_108 vs vggt_108 vs colmap_108

---

## 5. sh 脚本模板

```bash
#!/usr/bin/env bash
set -euo pipefail

# Part 1 VGGT+BA Experiments: 6 configurations
# 3 scenes x 1 init (vggt_ba_108) x 2 trainers (3dgs, scaffold)

CONDA=/home/bzhang512/miniconda3/bin/conda
GS3D_ROOT=/home/bzhang512/CV_Project/third_party/gaussian-splatting
SCAFFOLD_ROOT=/home/bzhang512/CV_Project/third_party/Scaffold-GS
OUTROOT=/home/bzhang512/CV_Project/output/part1
DATA_ROOT=/home/bzhang512/CV_Project/dataset

ITERS="$(seq 2000 2000 40000 | xargs)"

run_3dgs_post () {
    local gpu="$1"; shift
    local model_dir="$1"; shift
    local log_file="$1"; shift

    cd "$GS3D_ROOT"
    echo "[$(date '+%F %T')] POST 3dgs render.py model=$model_dir" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python render.py \
        -m "$model_dir" \
        --skip_train \
        2>&1 | tee -a "$log_file"

    echo "[$(date '+%F %T')] POST 3dgs metrics.py model=$model_dir" | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python metrics.py \
        -m "$model_dir" \
        2>&1 | tee -a "$log_file"
}

run_3dgs () {
    local gpu="$1"; shift
    local port="$1"; shift
    local scene="$1"; shift
    local plan="$1"; shift
    local variant="$1"; shift
    local source="$1"; shift

    local tag="${scene}__${plan}__${variant}__3dgs_default_40k_eval2k"
    local model_dir="$OUTROOT/${scene}/${plan}/${variant}/3dgs/default_40k_eval2k"
    local log_dir="$OUTROOT/${scene}/${plan}/${variant}/3dgs/logs"
    local log_file="$log_dir/${tag}.log"

    mkdir -p "$log_dir"
    rm -rf "$model_dir"

    echo "[$(date '+%F %T')] START $tag gpu=$gpu port=$port" | tee "$log_file"
    echo "source=$source" | tee -a "$log_file"
    echo "model_dir=$model_dir" | tee -a "$log_file"

    cd "$GS3D_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n 3dgs python train.py \
        -s "$source" \
        -m "$model_dir" \
        --eval \
        --port "$port" \
        --iterations 40000 \
        --test_iterations $ITERS \
        --save_iterations 40000 \
        2>&1 | tee -a "$log_file"

    run_3dgs_post "$gpu" "$model_dir" "$log_file"

    echo "[$(date '+%F %T')] END $tag" | tee -a "$log_file"
}

run_scaffold () {
    local gpu="$1"; shift
    local port="$1"; shift
    local scene="$1"; shift
    local plan="$1"; shift
    local variant="$1"; shift
    local source="$1"; shift

    local tag="${scene}__${plan}__${variant}__scaffold_final_40k_eval2k"
    local model_dir="$OUTROOT/${scene}/${plan}/${variant}/scaffold/final_40k_eval2k"
    local log_dir="$OUTROOT/${scene}/${plan}/${variant}/scaffold/logs"
    local log_file="$log_dir/${tag}.log"

    mkdir -p "$log_dir"
    rm -rf "$model_dir"

    echo "[$(date '+%F %T')] START $tag gpu=$gpu port=$port" | tee "$log_file"
    echo "source=$source" | tee -a "$log_file"
    echo "model_dir=$model_dir" | tee -a "$log_file"

    cd "$SCAFFOLD_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run -n scaffold_gs_cu124 python train.py \
        -s "$source" \
        -m "$model_dir" \
        --eval \
        --port "$port" \
        --iterations 40000 \
        --test_iterations $ITERS \
        --save_iterations 40000 \
        --voxel_size 0 --update_init_factor 16 --appearance_dim 0 \
        2>&1 | tee -a "$log_file"

    echo "[$(date '+%F %T')] END $tag" | tee -a "$log_file"
}

source_path () {
    local scene="$1"; shift
    local plan="$1"; shift
    local variant="$1"; shift
    echo "$DATA_ROOT/${scene}/part1/${plan}/${variant}/gs_scene"
}

# GPU 0: Re10k-1 + DL3DV-2
worker_gpu0 () {
    run_3dgs     0 6220 Re10k-1 planB vggt_ba_108 "$(source_path Re10k-1 planB vggt_ba_108)"
    run_scaffold 0 6220 Re10k-1 planB vggt_ba_108 "$(source_path Re10k-1 planB vggt_ba_108)"

    run_3dgs     0 6220 DL3DV-2 planB vggt_ba_108 "$(source_path DL3DV-2 planB vggt_ba_108)"
    run_scaffold 0 6220 DL3DV-2 planB vggt_ba_108 "$(source_path DL3DV-2 planB vggt_ba_108)"
}

# GPU 1: 405841
worker_gpu1 () {
    run_3dgs     1 6221 405841 planB vggt_ba_108 "$(source_path 405841 planB vggt_ba_108)"
    run_scaffold 1 6221 405841 planB vggt_ba_108 "$(source_path 405841 planB vggt_ba_108)"
}

worker_gpu0 &
PID0=$!

worker_gpu1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "[$(date '+%F %T')] ALL DONE - 6 VGGT+BA experiments completed"
```

---

## 6. Notebook 修改要点

### 关键修改（相对于 03_planB_vggt.ipynb）

1. **variant 名称**：`vggt_108` → `vggt_ba_108`

2. **VGGT 参数**：
```python
# 原配置（无 BA）
VGGT_ARGS = {
    'use_ba': False,
    'query_frame_num': 4,
    'max_query_pts': 1024,
    ...
}

# 新配置（开 BA）
VGGT_BA_ARGS = {
    'use_ba': True,              # 关键
    'shared_camera': False,      # 可尝试 True
    'camera_type': 'SIMPLE_PINHOLE',
    'query_frame_num': 8,        # BA 默认
    'max_query_pts': 4096,       # BA 默认
    'fine_tracking': False,      # 保守
    'vis_thresh': 0.2,
}
```

3. **命令行调用**：
```python
cmd = [
    'python', 'demo_colmap.py',
    '--scene_dir', str(scene_dir),
    '--use_ba',                   # 关键
    '--query_frame_num', '8',
    '--max_query_pts', '4096',
]
```

---

## 7. 可行性审查

### 检查项

| 检查项 | 状态 | 说明 |
|--------|------|------|
| subset_108.txt 存在 | ✓ 已确认 | 3 个 scene 都有 subset_108 |
| images_512 存在 | ✓ 已确认 | 3 个 scene 都有 images_512 |
| vggt 环境可用 | ✓ 已确认 | conda env `vggt` (torch 2.7.0+cu128) |
| 3dgs 环境可用 | ✓ 已确认 | conda env `3dgs` |
| scaffold 环境可用 | ✓ 已确认 | conda env `scaffold_gs_cu124` |
| demo_colmap.py 支持 BA | ✓ 已确认 | `--use_ba` 参数存在 |
| GPU 可用 | 待确认 | 需检查当前 GPU 状态 |

### 显存预估
- VGGT BA：~35-40 GB（比无 BA 高，因为 predict_tracks）
- 3DGS 训练：~10-15 GB
- Scaffold 训练：~10-15 GB

### 风险点
1. **BA 分支显存**：可能超出单卡容量，建议先单场景测试
2. **fine_tracking**：保守关闭，避免 OOM
3. **query_frame_num/max_query_pts**：建议用默认值（8/4096），比无 BA 的 4/1024 更合理

---

## 8. 执行顺序

1. **检查 GPU 状态**：`nvidia-smi`，确认空闲
2. **单场景测试**：先跑一个 scene（如 405841）的 VGGT BA
3. **验证 sparse 质量**：检查 track length >1.0
4. **确认无误后**：跑全部 6 组训练
5. **结果解析**：对比 vggt_ba_108 vs vggt_108 vs colmap_108

---

## 9. 预期结果

如果 BA 有效，预期：
- **track length**：从 1.0 → >5.0（接近 COLMAP-108 的 9-18）
- **observations**：从 100k → >points（真正多视图约束）
- **PSNR**：从 ~27.5 dB → 接近 ~32 dB（COLMAP-108 水平）
- **收敛轨迹**：late-stage test slope 更稳定

如果 BA 无效或显存不足：
- 考虑减少 query_frame_num 或 max_query_pts
- 或尝试 shared_camera=True（简化相机模型）
- 或降级为 fine_tracking=False + conservative params

---

## 10. 后续扩展

如果 BA 实验成功：
- 可考虑 vggt_ba_108 + COLMAP-full 对比（全量 BA）
- 可考虑 VGGT BA + COLMAP BA 对比（不同 BA 实现）
- 可考虑 different camera_type（PINHOLE vs SIMPLE_PINHOLE）

---

## 附录：文件清单

### 需创建的文件
1. `part1/notebooks/03_planB_vggt_ba.ipynb`（或修改现有）
2. `part1/scripts/train/run_part1_vggt_ba_6_experiments.sh`

### 需创建的目录（每个 scene）
1. `dataset/<SCENE>/part1/planB/vggt_ba_108/`
2. `output/part1/<SCENE>/planB/vggt_ba_108/`

---

**文档状态**：已完成，待审查
**审查人**：Noah（self-check）
**下一步**：按方案执行实验