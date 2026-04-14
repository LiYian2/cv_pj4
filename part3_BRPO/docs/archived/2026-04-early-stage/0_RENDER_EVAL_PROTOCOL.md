# RENDER_EVAL_PROTOCOL.md

> 最后更新：2026-04-08 23:00
> 目的：记录 Part2/Part3 当前涉及的渲染评测协议差异，避免把不同协议下的分数直接混比。

## 1. 现状：当前其实有两套 render-eval protocol

当前 Re10k 上至少同时存在两套协议：

- **Internal protocol（S3PO native）**：在一次完整 S3PO 运行内部，用 `frontend.cameras` 的当前相机状态对 **non-KF** 做渲染评测。
- **External protocol（eval_external）**：离线读取一张给定 PLY，在 test split 上重新构建相机并渲染评测。

两套协议不能直接横向比较，因为它们的 **相机状态来源、坐标系关系、是否与当前地图共同优化过** 都不同。

当前 Re10k-1 代表性结果：

| 结果 | 协议 | PLY 来源 | 相机来源 | 指标 |
|------|------|----------|----------|------|
| A1 | external infer | sparse final PLY | sequential infer pose | PSNR 13.3265 |
| B | external infer | full final PLY | sequential infer pose | PSNR 13.1458 |
| A2 | external gt | sparse final PLY | dataset GT pose | PSNR 16.9367 |
| C1 | internal before_opt | full internal map | internal tracked non-KF cameras | PSNR 18.4422 |
| C2 | internal after_opt | full internal map after color refinement | internal tracked non-KF cameras | PSNR 24.2260 |

当前最重要的观察不是单个数值，而是：
- `A1 ≈ B`：external infer 下，full tracking 带来的地图差异基本被外部 pose infer 误差淹没；
- `A2 > A1/B`：GT pose 会改善 external eval，但提升仍有限；
- `C1 >> A2`，`C2 >> C1`：internal protocol 下的 non-KF render eval 明显更强，且 after_opt 进一步提升很多。

不能把 `C1/C2` 理解成“只是 pose 更接近 GT”。它们更准确地表示：**internal tracked camera state 与当前地图更自洽。**

## 2. Internal protocol：S3PO 本身的 render/test 流程

### 2.1 相机状态来源

Internal protocol 使用的是 `slam.py` 运行过程中保存在 `frontend.cameras` 里的相机对象，而不是离线重建的新相机。每个 `Camera` 对象至少包含：
- 当前估计位姿：`R/T`
- GT 位姿：`R_gt/T_gt`
- 局部位姿残差：`cam_rot_delta / cam_trans_delta`
- 曝光参数：`exposure_a / exposure_b`

其中：
- 第一帧初始化会直接使用 `R_gt/T_gt`；
- 后续帧 tracking 不是用 GT 渲染，而是先通过 `get_pose()` 给一个相对位姿初值，再通过 photometric optimization 更新 `R/T`；
- `cam_rot_delta / cam_trans_delta` 不是“与 GT 的差值”，而是当前估计位姿的局部 SE(3) 残差参数；更新后会被并入 `R/T` 并清零。

### 2.2 渲染评测流程

Internal render eval 在 `slam.py -> eval_rendering()` 中完成：
- 遍历当前 run 中的全部帧；
- **跳过 KF**；
- 对所有 **non-KF** 帧渲染并计算 PSNR/SSIM/LPIPS；
- 使用的是 `frontend.cameras[idx]` 当前保存的内部相机状态。

这意味着 internal render eval 的渲染对象是：
- **270 个 non-KF 帧**（以 Re10k-1 full 为例）；
- 在当前 SLAM 地图坐标系下，用内部 tracked camera state 直接渲染。

### 2.3 ATE 流程与 render 流程不是同一批帧

`eval_ate()` 当前只看 **KF**，而不是全部 non-KF。Re10k-1 full run 的 `plot/trj_final.json` 里只有 9 个 id，对应固定 KF：
`0, 34, 69, 104, 139, 173, 208, 243, 278`

因此当前 internal protocol 实际上是：
- **轨迹误差（ATE）看 KF**；
- **渲染指标看 non-KF**。

这不是 bug，但它是一个需要被明确记录的协议事实。

### 2.4 before_opt 与 after_opt 的区别

- **before_opt**：使用 tracking/mapping 后的当前地图，直接做 non-KF 渲染评测；
- **after_opt**：先运行 `color_refinement()`，进一步优化 gaussians 的 appearance/map，再做同样的 non-KF 渲染评测。

`after_opt` 的提升主要来自 **地图本身被进一步优化**，不是重新做了 pose 优化。

## 3. External protocol：eval_external 的离线评测流程

### 3.1 输入与输出

`eval_external.py` 的输入是一张现成的 `ply_path` 和一份 test split config。它会：
- 重新加载 test split dataset；
- 重新为每个 test 帧构建 `Camera`；
- 按 `pose_mode` 决定使用 GT 还是 infer pose；
- 输出 `eval_external.json / eval_external_meta.json / psnr/{pose_mode}/final_result.json`。

### 3.2 gt 与 infer 的真实含义

- `pose_mode=gt`：确实调用 `apply_gt_pose_to_cameras()`，直接把每个 camera 的 `R/T` 设为 `R_gt/T_gt`；
- `pose_mode=infer`：调用 `infer_poses_sequential()`，以第一帧 GT 或 identity 初始化，然后逐帧顺序估相对位姿。

所以 external 的 gt/infer 模式都是真实生效的，不是“表面切换、实则同一套 pose”。

### 3.3 为什么 external gt 仍然显著低于 internal before_opt

因为 external gt 不是 internal tracked camera state。它的特点是：
- 相机是离线新建的；
- 地图与相机不是在同一轮 SLAM 过程中共同形成的；
- 当前对齐只用了很弱的 `origin_mode=test_to_sparse_first`（首帧平移偏移），不是完整的 map-to-test 刚体/相似变换对齐。

因此，`external gt` 不能被直接当作 internal protocol 下的 oracle upper bound。

## 4. 对当前结果的解释

当前更合理的解释是：

1. `A1 ≈ B` 正常：external infer pose 是主要瓶颈，因此 sparse/full 的地图差异在该协议下不明显；
2. `A2 > A1/B` 也正常：GT pose 帮助 external eval 变好，但仍然不是 internal camera protocol；
3. `C1` 高于所有 external 结果，说明 **internal tracked camera state 与地图更自洽**；
4. `C2` 又显著高于 `C1`，说明 color refinement 在 internal protocol 下可以大幅提升地图外观质量。

所以当前不能把：
- internal before/after
- external infer
- external gt

当成“同一种 test 的三个版本”。它们测的是不同层面的系统表现。

## 5. 如果后续切换到 full/internal protocol，pipeline 怎么改

这里分三档改动，按可行性递增排列。

### 方案 A：最小改动 —— 增加 internal replay eval

目标：让任意给定 PLY 都能在 **internal tracked camera states** 下复现 non-KF 渲染评测。

需要 Part2 额外保存：
- 所有 non-KF 帧最终 `R/T`
- 对应 `frame_idx / uid`
- 相机内参、图像尺寸
- 最好同时保存 `R_gt/T_gt` 与 `kf flag`

然后新增一个类似 `eval_internal_replay.py` 的入口：
- 输入：`ply_path + tracked_camera_states.json`
- 输出：internal protocol 下的 non-KF render eval 指标

优点：
- 不改 Stage1 refine 核心；
- 立刻能比较 baseline/refined PLY 在 internal protocol 下的差异；
- 是当前最值得先做的补充。

### 方案 B：中等改动 —— pseudo cache 切到 internal source

目标：让 Part3 的 pseudo supervision 也来自 internal protocol，而不是 external infer protocol。

需要 Part2 额外保存：
- non-KF tracked camera states
- internal render RGB/depth（至少 before_opt 一版；必要时 after_opt 一版）

然后修改 `prepare_stage1_difix_dataset_s3po.py`：
- 不再默认从 external infer 的 `render_rgb / render_depth / trj_external_infer.json` 取 pseudo 输入；
- 改为从 full run 的 internal non-KF render + tracked poses 构建 pseudo cache。

优点：
- standalone refine 仍可保留；
- pseudo 生成协议与最终 internal eval 更一致。

### 方案 C：重改动 —— 把 pseudo refine 集成到 S3PO backend

目标：让 full split 全量 tracking、sparse mapping 的同时，把 pseudo supervision 直接接进 backend/mapping 主流程。

需要改动：
- frontend/backend 交互
- pseudo 数据加载与调度
- mapping loss 组织方式
- densify/pruning 与 pseudo 的交互
- 最终结果保存与评测流程

优点：
- pose、map、pseudo supervision 在同一系统中共同演化；
- 最接近真正的 S3PO-native 扩展。

缺点：
- 工程量最大；
- 当前不适合在协议未定前立刻做。

## 6. 当前建议

在老师明确认可哪套协议之前，不要继续基于“external 分数”盲目推进 Part3 结论。当前更稳的顺序是：

1. 先把 **internal / external 协议差异**说清楚；
2. 若老师认可 full-tracking + sparse-mapping + internal non-KF eval，则优先做 **方案 A**；
3. 若老师进一步认可 pseudo 也应遵循 internal protocol，再推进 **方案 B**；
4. **方案 C** 只在确认研究目标需要系统级集成时再做。
