# AGENT_TODO.md

## 0. 目的

这份清单服务于 **Project 4 Part 3 的 Stage 1 / Stage 2 落地**。目标不是继续讨论抽象路线，而是把接下来要做的工程任务、文件位置、输入输出、测试方式和实验计划明确下来，方便后续 agent 直接执行。

当前共识：
- **Stage 1** 先走最稳路线：`external pseudo branch + backend-only pseudo refinement`
- **pseudo view 只参与 mapping，不参与 tracking**
- **full EDP**（pseudo + 左右真实参考帧 → 几何深度约束 → 鲁棒融合 → bad-point filtering → target_depth + C_geom）先放到 **Stage 2**
- Stage 1 允许先做 **简化版 target_depth / confidence_mask**，先验证 pseudo supervision 链路能闭环、能做 ablation
- 需要保留 **独立开关**，因为最终要做实验对比

---

## 1. 当前已知现状

### 1.1 已有结果

当前已经有：
- real-only 的 Part 2 / RegGS / S3PO 结果
- Re10k-1 的 stage1 Difix 数据准备结果：
  - `/home/bzhang512/CV_Project/dataset/Re10k-1/part3_stage1/re10k1__reggs__midpoint__v1`
- 对应 notebook：
  - `/home/bzhang512/CV_Project/part3_BRPO/notebooks/prepare_stage1_difix_dataset_re10k1_reggs.ipynb`
- 对应脚本：
  - `/home/bzhang512/CV_Project/part3_BRPO/scripts/prepare_stage1_difix_dataset.py`

### 1.2 当前 stage1 目录结构（已存在）

以 `Re10k-1 / re10k1__reggs__midpoint__v1` 为例，当前已有：
- `inputs/raw_render/`
- `inputs/left_ref/`
- `inputs/right_ref/`
- `difix/left_fixed/`
- `difix/right_fixed/`
- `augmented_train_left/rgb/`
- `augmented_train_right/rgb/`
- `manifests/source_manifest.json`
- `manifests/pseudo_selection_manifest.json`
- `manifests/difix_run_manifest.json`
- `manifests/pack_manifest.json`

% 此处需要更新，我们目前先不用reggs做，用s3po做。
### 1.3 当前缺口

当前还**没有**：
- pseudo pose 的显式保存
- 完整 pseudo cache
- `target_depth`
- `confidence_mask`
- `diag`
- fused pseudo RGB
- S3PO backend-only pseudo refinement 接口

---

## 2. Stage 划分（先定死）

## Stage 1：先做最小闭环

核心目标：
- 不碰 frontend tracking 主流程
- 先把 pseudo supervision 做成可用缓存
- 在 backend 中加一条 **mapping-only pseudo refinement** 路径
- 能跑通以下 ablation：
  - baseline real-only
  - + pseudo RGB only
  - + pseudo RGB + simplified target_depth
  - + confidence mask

### Stage 1 暂不做
- full EDP
- online RAP
- pseudo online virtual keyframe
- pseudo 相机 pose optimization
- 深改 `FrontEnd.add_new_keyframe(...) / process_depth(...)`
- 把 pseudo 冒充成普通 `"keyframe"`

## Stage 2：增强版
- full EDP
- pseudo RGB fuse（若决定走几何/质量引导融合）
- RAP offline audit
- 把 EDP 推进到 S3PO real-keyframe depth initialization 链路

---

## 3. 总体执行顺序（建议按这个顺序干）

1. **补齐 pseudo cache 设计与目录规范**
2. **扩展 stage1 数据准备脚本**，把 pseudo pose / metadata 真正落盘
3. **实现 external pseudo branch（简化版）**：生成 target_depth、confidence_mask、diag
4. **定义 pseudo sample manifest**，让 backend 有稳定输入
5. **在 S3PO repo 中加 pseudo_refinement 开关和消息分支**
6. **新增 pseudo loss**（masked RGB / masked depth）
7. **先跑小规模 integration test**（少量 pseudo、少迭代）
8. **再跑完整 Re10k-1 stage1 实验**
9. **整理 ablation 输出结构**
10. **Stage 1 稳定后，再进入 Stage 2 / full EDP**

---

## 4. 文件级任务清单

## 4.1 先补 stage1 数据结构（必须先做）

### 需要修改
- `/home/bzhang512/CV_Project/part3_BRPO/scripts/prepare_stage1_difix_dataset.py`
- `/home/bzhang512/CV_Project/part3_BRPO/notebooks/prepare_stage1_difix_dataset_re10k1_reggs.ipynb`

### 要做什么
1. 在 `PseudoRecord` / manifest 中新增 pseudo pose 信息位：
   - `pseudo_pose_src`（来源说明）
   - `camera_path` 或可序列化相机参数路径
   - 最少先能落盘 `frame_id -> pose` 的关系
2. 补一个新的输出层级，不再只停留在 left/right difix 和 augmented train：
   - 为后续 pseudo cache 预留目录
3. 把 `run_root` 下的 manifest 变成真正可被后续脚本复用的标准输入

### 建议新增字段
在 `pseudo_selection_manifest.json` 每条记录中增加：
- `pseudo_camera_path`
- `left_ref_camera_path`
- `right_ref_camera_path`
- `scene_id`
- `backend`
- `run_key`

### 准备怎么做
- 如果当前 RegGS 路径没有现成 camera json，则先从原始数据或 part2 运行时可恢复的 pose 文件中导出
- 第一版不追求统一到所有 backend；先把 Re10k-1 / RegGS 这条链打通

### 测试
- 跑一次 `--stage select`，确认 manifest 中出现 pseudo pose 路径/字段
- 随机抽 1 个 pseudo sample，检查 `frame_id / left_ref / right_ref / pose` 是否一致

---

## 4.2 新建 external pseudo branch（Stage 1 简化版）

完整的 pseudo cache，可以把它理解成“一份给后续 pseudo refinement 直接使用的样本包”。最核心的内容通常是：pseudo_camera、target_rgb、target_depth、confidence_mask，外加左右参考帧信息和一些调试产物。也就是说，它不只是几张图，而是一套围绕某个 pseudo view 的完整监督数据。

从 part2 直接获得 的，主要是这类“原始几何和视角信息”：pseudo pose / pseudo_camera，当前场景在该 pseudo 视角下 render 出来的 render_rgb，以及 render_depth / render_depth_npy。如果是 S3PO 新的 external eval 输出，那么 trj_external_infer.json、render_rgb/、render_depth_npy/ 基本就属于这一层。另外，train split 里的左右真实参考帧是谁，这个也通常可以结合 split_manifest.json 和数据集相机文件直接推出来，所以 refs 的索引与相机信息本质上也属于“part2 体系可直接恢复”的内容。

从 Difix 获得 的，核心就是 target_rgb 相关内容。也就是：以 pseudo view 的 render 作为待修复输入，再结合左/右真实参考帧，产出的 pseudo 视角修复图。当前你的 RegGS 版本里对应的是 left_fixed、right_fixed，以后如果做融合，还会有 target_rgb_fused。这些都不是 part2 原生产物，而是 Difix 这一步生成的外观监督。

从 额外模块获得 的，就是为了把 pseudo view 变成“可安全使用的监督”而补出来的几何与置信信息。最典型的是 target_depth、confidence_mask、以及 diag。target_depth 是 pseudo 视角下更可信的目标深度，可能先用简化版几何过滤得到，后面再升级成 full EDP；confidence_mask 是告诉 backend 哪些区域能信、哪些该降权；diag 则是调试和分析用的，比如 visibility map、depth consistency map、epipolar distance、view score 之类。

所以你可以这样记：part2 提供“这个 pseudo view 在哪、当前渲染长什么样”，Difix 提供“这个 pseudo view 应该长成什么样”，额外模块提供“这个 pseudo view 哪些地方可信、几何上该对齐到什么程度”。三者合起来，才是完整 pseudo cache。

### 建议新增目录
- `/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/`

### 建议新增文件
- `part3_BRPO/pseudo_branch/__init__.py`
- `part3_BRPO/pseudo_branch/camera_io.py`
- `part3_BRPO/pseudo_branch/pseudo_cache_schema.py`
- `part3_BRPO/pseudo_branch/pseudo_renderer.py`
- `part3_BRPO/pseudo_branch/depth_target_builder.py`
- `part3_BRPO/pseudo_branch/confidence_builder.py`
- `part3_BRPO/pseudo_branch/diag_writer.py`
- `part3_BRPO/pseudo_branch/build_pseudo_cache.py`

### 要做什么

#### a) `camera_io.py`
负责：
- 统一读取 / 写出 pseudo / left / right camera
- 把后续需要的字段整理成固定 schema

#### b) `pseudo_cache_schema.py`
定义：
- pseudo sample manifest 的 Python schema / dataclass
- 避免后续脚本各写各的字段

#### c) `pseudo_renderer.py`
负责：
- 给定 `pseudo_camera` 和当前 Gaussian scene / 已保存结果，render 出：
  - `I_t_gs`
  - `D_t_gs`
- 第一版允许只支持离线读取已有 scene + render，不要求接进 frontend

#### d) `depth_target_builder.py`
Stage 1 先做 **simplified target_depth**：
- 输入：`pseudo render depth`、左右参考帧、相机、可见性信息
- 输出：`target_depth.npy`
- 第一版不要求 full EDP
- 建议先做：
  - visibility / in-image 检查
  - 与参考帧的几何一致性过滤
  - 输出可用的 depth mask + filtered depth

> 备注：full EDP 放 Stage 2，在这里预留接口即可。

#### e) `confidence_builder.py`
Stage 1 先做简化版 `C_geom`：
- `C_geom = O_vis * s_depth * s_pose`
- 第一版优先几何项，不上 matcher-heavy 版本
- 可选留一个 `flow_veto=False/True` 开关，但默认先关或只做 very weak 版本

#### f) `diag_writer.py`
输出调试图：
- `visibility_map`
- `depth_consistency_map`
- `pose_score.json`
- `confidence_mask.png / .npy`
- 以后如果上 full EDP，再加：
  - `epipolar_distance`
  - `left_score_map`
  - `right_score_map`

#### g) `build_pseudo_cache.py`
把所有东西串起来，输出完整 pseudo cache。

### 测试
- 先只跑 1 个 pseudo sample
- 输出目录里必须能看到：
  - `target_rgb`
  - `render_rgb`
  - `render_depth`
  - `target_depth`
  - `confidence_mask`
  - `diag/`
  - `camera.json`
- 肉眼检查 3~5 个样本

---

## 4.3 统一 pseudo cache 路径设计（需要先钉住）

### 建议路径
以 scene+run_key 为粒度：

```text
/home/bzhang512/CV_Project/dataset/Re10k-1/part3_stage1/re10k1__reggs__midpoint__v1/
    pseudo_cache/
        manifest.json
        samples/
            0017/
                camera.json
                refs.json
                target_rgb_left.png
                target_rgb_right.png
                target_rgb_fused.png        # Stage 1 可先缺省
                render_rgb.png
                render_depth.npy
                target_depth.npy
                confidence_mask.npy
                confidence_mask.png
                diag/
                    visibility_map.png
                    depth_consistency_map.png
                    pose_score.json
            0052/
            ...
```

### Stage 1 关于 left/right/fuse 的决定
- 当前 **左右分开保留**
- `fuse` 暂不作为 Stage 1 必须项
- Stage 1 可以先使用以下二选一方案：
  1. `left` / `right` 各自产生 sample，后续分别试验
  2. 先做一个简单融合版本（如按 confidence/visibility 加权或规则选择）

### 当前建议
先采用 **“保留 left/right + 额外预留 fused 占位”**：
- 文件结构先设计好
- Stage 1 主实验可以先只吃 `left_fixed` 或 `right_fixed` 的某一侧，或者对两侧独立跑小试验
- full EDP / 几何融合成熟后，再真正写入 `target_rgb_fused.png`

---

## 4.4 S3PO backend 接口改造（Stage 1 核心）

### 需要修改
- `/home/bzhang512/CV_Project/third_party/S3PO-GS/slam.py`
- `/home/bzhang512/CV_Project/third_party/S3PO-GS/utils/slam_backend.py`
- `/home/bzhang512/CV_Project/third_party/S3PO-GS/utils/slam_utils.py`

### a) 修改 `slam.py`

#### 要做什么
新增一个 **独立开关触发** 的 pseudo refinement stage。

#### 建议配置项
在 config 中新增：
- `Results.pseudo_refinement: bool`
- `Results.pseudo_cache_manifest: str`
- `Results.pseudo_refinement_tag: str`

#### 调用时机
放在：
- `frontend.run()` 结束之后
- `color_refinement` 之前或并列

#### 准备怎么做
仿照当前 `color_refinement` 的调用方式：
- 向 backend queue 发 `[pseudo_refinement, pseudo_cfg]`
- 等 backend 返回 `sync_backend`
- 保存一个独立 iteration tag（例如 `after_pseudo_refine`）

### b) 修改 `slam_backend.py`

#### 要做什么
新增：
- 新 message handler：`"pseudo_refinement"`
- 新函数：`pseudo_refinement(...)`
- 新的 pseudo sample loader

#### 核心约束
- **不把 pseudo 相机放进 `keyframe_optimizers`**
- **不修改 `current_window` 语义**
- **不调用 `add_next_kf(...)` 把 pseudo 当高斯初始化关键帧插入**
- pseudo 只作为渲染监督视角

#### 建议实现方式
- 读取 pseudo cache manifest
- 每次迭代：
  - 采样少量真实 anchor view
  - 采样少量 pseudo sample
  - 对 pseudo_camera 做 render
  - 计算 `get_loss_mapping_pseudo(...)`
  - 与 real anchor loss 加权求和
  - 只更新 gaussians / exposure（是否更新 exposure 可单独做开关）

#### 需要新增的 config
- `Training.lambda_pseudo`
- `Training.lambda_pseudo_rgb`
- `Training.lambda_pseudo_depth`
- `Training.pseudo_refine_iters`
- `Training.pseudo_batch_size`
- `Training.real_anchor_batch_size`
- `Training.pseudo_use_depth`
- `Training.pseudo_use_confidence`

### c) 修改 `slam_utils.py`

#### 要做什么
新增：
- `get_loss_mapping_pseudo(...)`

#### 建议定义
输入：
- render 出来的 `image`, `depth`
- `target_rgb`
- `target_depth`
- `confidence_mask`
- 若干开关 / 权重

输出：
- `loss_pseudo_total`
- 可选返回分项日志：`loss_rgb`, `loss_depth`, `valid_pixels`

#### 形式
建议先做：
- masked RGB L1
- masked depth L1

公式：
- `L_rgb_masked = || C * (I_hat - I_target) ||_1 / (sum(C)+eps)`
- `L_depth_masked = || C * (D_hat - D_target) ||_1 / (sum(C)+eps)`

### 测试
- 用 1~2 个 pseudo sample 跑 10~20 iter，确认：
  - 代码能通
  - loss 有数值
  - 不报 shape/device 错
  - gaussians 能成功同步回 frontend

---

## 4.5 配置文件与开关（为 ablation 服务）

### 需要新增或修改
- S3PO 使用的 config yaml（具体 scene config 后续补）

### 必须有的开关
- `Results.pseudo_refinement`
- `Results.pseudo_cache_manifest`
- `Training.lambda_pseudo`
- `Training.lambda_pseudo_rgb`
- `Training.lambda_pseudo_depth`
- `Training.pseudo_refine_iters`
- `Training.pseudo_use_depth`
- `Training.pseudo_use_confidence`
- `Training.pseudo_target_rgb_mode`：`left | right | fused`
- `Training.pseudo_target_depth_mode`：`render_filtered | simplified_geom | full_edp`

### 准备怎么做
Stage 1 先支持：
- `pseudo_target_rgb_mode = left/right`
- `pseudo_target_depth_mode = simplified_geom`
- `pseudo_use_confidence = true/false`

这样后面 ablation 很方便。

---

## 5. 接口设计（先钉住）

## 5.1 pseudo cache manifest 接口

### 顶层 manifest 建议字段
- `scene_name`
- `run_key`
- `backend`
- `source_run_root`
- `pseudo_target_rgb_mode`
- `pseudo_target_depth_mode`
- `num_samples`
- `sample_ids`
- `sample_dirs`

### 每个 sample 建议字段
- `frame_id`
- `placement`
- `pseudo_camera_path`
- `left_ref_frame_id`
- `right_ref_frame_id`
- `left_ref_camera_path`
- `right_ref_camera_path`
- `render_rgb_path`
- `render_depth_path`
- `target_rgb_left_path`
- `target_rgb_right_path`
- `target_rgb_fused_path`（可为空）
- `target_depth_path`
- `confidence_mask_path`
- `diag_dir`

---

## 5.2 backend pseudo_refinement 接口

建议 backend handler 接受：

```python
[
  "pseudo_refinement",
  {
    "manifest_path": ".../pseudo_cache/manifest.json",
    "iters": 1000,
    "lambda_pseudo": 0.2,
    "lambda_rgb": 1.0,
    "lambda_depth": 1.0,
    "target_rgb_mode": "left",
    "target_depth_mode": "simplified_geom",
    "use_confidence": True,
  }
]
```

---

## 6. 测试清单（必须分层做）

## 6.1 数据层测试
- manifest 是否完整
- pseudo pose 是否落盘
- sample 目录是否齐全
- target_depth / confidence_mask 是否有有效像素
- left/right target_rgb 是否对应正确参考帧

## 6.2 可视化 sanity check
至少抽 3~5 个 sample，看：
- `render_rgb`
- `left_fixed`
- `right_fixed`
- `target_depth`
- `confidence_mask`
- `diag`

## 6.3 S3PO integration test
小规模测试：
- 只用 1~2 个 pseudo sample
- 只跑 10~50 iter
- 检查 loss / 显存 / 同步机制

## 6.4 Stage 1 小实验
- scene: `Re10k-1`
- pseudo ids: 当前 8 个 midpoint
- 先只试 `left` 或 `right`
- 跑短版 pseudo refinement
- 比较 before/after 渲染结果

## 6.5 正式实验前检查
- baseline real-only 指标保存完整
- pseudo refinement 结果单独保存 tag
- config 与 manifest 一一对应，能复现实验

---

## 7. 实验计划（简版）

### 7.1 Stage 1 最小实验

#### 实验 A
- baseline：real-only S3PO / RegGS

#### 实验 B
- + pseudo RGB only
- `target_depth` 关闭
- `confidence_mask` 可先关闭或设全 1

#### 实验 C
- + pseudo RGB + simplified target_depth

#### 实验 D
- + pseudo RGB + simplified target_depth + confidence_mask

对比：
- PSNR / SSIM / LPIPS
- 关键可视化结果
- geometry 是否被拖坏

### 7.2 Stage 2 实验
- full EDP 替换 simplified target_depth
- 比较 `simplified_geom` vs `full_edp`
- 再考虑 left/right/fused 的影响

---

## 8. full EDP 的位置（先记账，不在 Stage 1 强上）

当前对 full EDP 的定义：
- 用 pseudo view 和左右真实参考帧算出几何约束深度
- 对多来源深度做鲁棒融合
- 过滤掉不满足极线 / flow 一致性的坏点
- 输出更可信的 `target_depth`
- 把一致性信息并入 `C_geom`

### 当前决定
- **full EDP 放 Stage 2**
- Stage 1 先预留 `target_depth_mode = full_edp` 接口，不立即实现

### Stage 2 建议新增文件
- `part3_BRPO/pseudo_branch/epipolar_depth.py`
- `part3_BRPO/pseudo_branch/depth_fusion.py`
- `part3_BRPO/pseudo_branch/flow_veto.py`

---

## 9. 目前最优先的 TODO（按执行顺序）

1. **修改 `prepare_stage1_difix_dataset.py`，把 pseudo pose 显式保存到 manifest / camera 文件**
2. **新建 `part3_BRPO/pseudo_branch/`，先把 pseudo cache schema 和 build 脚本立起来**
3. **定义并实现 Stage 1 simplified target_depth**
4. **定义并实现 Stage 1 simplified confidence_mask (`C_geom`)**
5. **生成完整 pseudo cache（Re10k-1 / reggs / midpoint）**
6. **改 `third_party/S3PO-GS/slam.py` 增加 pseudo_refinement 开关与调用**
7. **改 `third_party/S3PO-GS/utils/slam_backend.py` 增加 `pseudo_refinement` 分支**
8. **改 `third_party/S3PO-GS/utils/slam_utils.py` 增加 `get_loss_mapping_pseudo(...)`**
9. **跑小规模 integration test**
10. **跑 Stage 1 Re10k-1 正式实验与 ablation**

---

## 10. 备注

- 当前最关键的工程边界不是“做不做 pseudo”，而是**不要把 pseudo 混进 tracking / keyframe 主语义**。
- 第一版最重要的是**形成稳定的数据生产线 + 稳定的 backend-only refinement 接口**。
- 只要这两件事成了，Stage 2 的 full EDP、fuse、RAP 都是增强件，而不是系统重写。

