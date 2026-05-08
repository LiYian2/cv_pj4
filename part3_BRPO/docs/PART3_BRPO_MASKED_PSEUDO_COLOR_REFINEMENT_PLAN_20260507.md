# Part3 BRPO Final Color Refinement：接入 Masked Pseudo 的落地规划

**日期**: 2026-05-07
**状态**: 待实施
**优先级**: 高
**目标**: 把当前 real-only final color refinement，改成“real + masked pseudo”的 appearance refinement，同时保持 pose 不更新。

---

## 0. 结论摘要

当前 final color refinement 与作者口述 BRPO 语义之间，也存在明确的 pipeline 差异：

1. live `color_refinement()` 只从 `self.viewpoints` 中采样 real keyframes。
2. runtime pseudo records 从未进入 final color refinement。
3. 当前 color refinement loss 是 full-image `L1 + SSIM`，没有任何 pseudo mask / `C_m` 加权语义。
4. internal eval artifact 已证明 `after_opt` 的 camera states 与 `before_opt` 共用同一个 `camera_states.json`，即 color refinement 不改 pose，只改 gaussians。

因此，如果作者要求“结束时的 color refinement 也应包含带 mask 的 pseudo”，那么当前 Part3 live pipeline 仍未对齐。

---

## 1. 当前 live 行为：代码级证据

### 1.1 `color_refinement()` 只消费 real viewpoints

**文件**: `third_party/S3PO-GS/utils/slam_backend.py`

`color_refinement()`（约 683-717 行）当前逻辑是：

```python
viewpoint_idx_stack = list(self.viewpoints.keys())
viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(...))
viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
render_pkg = render(viewpoint_cam, ...)
gt_image = viewpoint_cam.original_image.cuda()
loss = (1-lambda_dssim) * Ll1 + lambda_dssim * (1-ssim(image, gt_image))
```

当前没有任何一处从 `self.brpo_runtime_pseudo_records` 读取 pseudo supervision。

### 1.2 runtime pseudo records 实际上已经在 backend 中常驻缓存

**文件**: `third_party/S3PO-GS/utils/slam_backend.py`

backend 初始化时已有：

- `self.brpo_runtime_pseudo_records = {}`

在 runtime pseudo build 时会持续写入：

- `self.brpo_runtime_pseudo_records[int(slot.frame_id)] = record_bundle.record`

reset 时再统一清空。

这意味着：

- color refinement 所需的 pseudo 监督数据并不缺
- 问题不在于“没有 pseudo artifact”
- 而在于 `color_refinement()` 压根没消费它们

### 1.3 pseudo record 已经携带 final color refinement 所需的监督信息

**文件**:
- `part3_BRPO/pseudo_branch/refine/backend_pseudo_view_loader.py`
- `part3_BRPO/pseudo_branch/integration/runtime_pseudo_builder.py`

`BackendPseudoViewRecord` 当前已有：

- `target_rgb`
- `target_depth`
- `confidence_mask`
- `valid_mask`
- `target_confidence`
- `support_both_mask`
- `viewpoint`

对 final color refinement 来说，最核心的是：

- `viewpoint`：用于 render
- `target_rgb`：pseudo 监督图像
- `confidence_mask`：离散 `C_m`

也就是说，接入 masked pseudo 所需的数据 contract 已经基本存在。

### 1.4 当前 loss utils 只有 full-image SSIM，没有 masked SSIM

**文件**: `third_party/S3PO-GS/gaussian_splatting/utils/loss_utils.py`

当前已有：

- `l1_loss(...)`
- `ssim(...)`

但没有：

- `masked_l1_loss(...)`
- `masked_ssim(...)`

论文作者要求“L1 和 SSIM 都要带 `C_m` 权重”，所以 final color refinement 若要对齐，就不能简单复用当前 full-image `ssim()`。

### 1.5 现有 artifact 已证明 color refinement 不更新 pose

**文件**: `third_party/S3PO-GS/utils/internal_eval_utils.py`

manifest 当前固定写：

```json
"color_refinement_updates_pose": false
```

且 E2 的 `internal_eval_cache/manifest.json` 明确说明：

- before/after 共用 `camera_states.json`
- after_opt 改善来自 Gaussian / appearance，不来自 pose

这意味着：

- 将 masked pseudo 接进 color refinement，天然会主要影响 PSNR/SSIM/LPIPS
- 不应把它误解为 pose 改善手段

---

## 2. 为什么必须改

作者明确补充：

1. final color refinement 不只使用 keyframes，也使用 pseudo；
2. pseudo supervision 仍然带 mask 限制；
3. 也就是说，最终 appearance refinement 应该是 `real + masked pseudo`，而不是 `real-only`。

当前 live pipeline 不满足这一点，所以当前 after_opt 指标实际上是：

**online mapping + real-only final appearance cleanup**

而不是：

**online mapping + real + masked pseudo final appearance refinement**

对 PSNR/SSIM 的解释影响很大：

- 如果 pseudo 只在前面的 online mapping 起作用、但最后一大段 color refinement 又完全忽略 pseudo，pseudo 对 appearance 的贡献很可能被冲淡或洗掉。

---

## 3. 目标语义与边界条件

### 3.1 必须满足

1. final color refinement 仍然 **只更新 gaussians**，不更新任何 pose/exposure。
2. real view 继续走 full-image appearance supervision。
3. pseudo view 进入同一 color refinement stage，但 loss 必须带 mask。
4. 默认 mask 应该对齐作者说的 `C_m`，即优先使用 `record.confidence_mask`。
5. pseudo 不需要、也不应该插入 `self.viewpoints` 变成 tracking camera。

### 3.2 第一版明确不做

1. 不在 final color refinement 阶段优化 pseudo pose。
2. 不在 final color refinement 阶段优化 real pose。
3. 不把 pseudo 用作新 keyframe / fusion / camera-state export 对象。

---

## 4. 建议的落地策略：保留 real-only path，新增可开关的 masked-pseudo refinement

不建议直接替换现有 `color_refinement()`。更稳妥的方式是：

1. `color_refinement_use_pseudo=false` 时，保持当前 real-only 行为；
2. `color_refinement_use_pseudo=true` 时，进入 mixed sampling path；
3. 所有 compare 都明确标注 real-only vs real+pseudo appearance refinement。

这样做的好处是：

- 可以直接隔离“pseudo final refinement 对 PSNR 的净贡献”；
- 出现异常时能回退到当前稳定路径。

---

## 5. 具体改动方案（按文件拆解）

### 5.1 `third_party/S3PO-GS/gaussian_splatting/utils/loss_utils.py`

需要新增 masked loss helper，而不是在 `slam_backend.py` 里直接硬写数学。

建议新增：

```python
def masked_l1_loss(network_output, gt, mask):
    ...

def masked_ssim(img1, img2, mask, window_size=11, size_average=True):
    ...
```

#### 5.1.1 `masked_l1_loss`

语义：

- `gt` 是 pseudo `target_rgb`
- `mask` 默认使用 `C_m`
- 按 mask 加权后归一化

#### 5.1.2 `masked_ssim`

不能偷懒做成“先把图像乘 mask，再调用现有 `ssim()`”。

原因：

- 那会把未监督区硬置零，改变局部均值/方差统计
- 与“L1、SSIM 都由 `C_m` 加权”的语义不一致

更合理的第一版实现是：

1. 先用现有 `_ssim(...)` 生成 `ssim_map`
2. 再用 `mask` 对 `ssim_map` 做加权平均
3. 返回 weighted SSIM 标量

即：

```python
weighted_ssim = (ssim_map * mask).sum() / (mask.sum() + eps)
```

然后 pseudo 分支 loss 使用：

```python
(1-lambda_dssim) * masked_l1 + lambda_dssim * (1-weighted_ssim)
```

### 5.2 `third_party/S3PO-GS/utils/slam_backend.py`

#### 5.2.1 扩展 `Results` 配置面

建议新增：

```yaml
Results:
  color_refinement: true
  color_refinement_use_pseudo: true
  color_refinement_pseudo_ratio: 0.5
  color_refinement_pseudo_weight: 1.0
  color_refinement_pseudo_mask_source: confidence_mask   # confidence_mask | support_both_mask | valid_mask
  color_refinement_log_every: 200
```

推荐默认：

- `pseudo_mask_source = confidence_mask`
- 因为作者语义是 `C_m` 加权

#### 5.2.2 抽象出 view-sampling 逻辑

当前 `color_refinement()` 每步只从 `self.viewpoints` 采样 1 个 real camera。

新版建议变成两类 member：

1. `real_member`
   - 来源：`self.viewpoints`
   - supervision：`original_image`
   - loss：full-image `L1 + SSIM`
2. `pseudo_member`
   - 来源：`self.brpo_runtime_pseudo_records.values()`
   - supervision：`record.target_rgb`
   - mask：`record.confidence_mask`（默认）
   - loss：masked `L1 + SSIM`

每步可以：

- 按 `color_refinement_pseudo_ratio` 决定采 real 还是 pseudo
- 或每步固定 real/pseudo 各一条后合并 loss

第一版推荐 **单步单样本 + 随机比例采样**，侵入性最低。

#### 5.2.3 给 pseudo path 单独封装一个 helper

建议新增：

- `_sample_color_refinement_member()`
- `_color_refinement_loss_real(viewpoint_cam, render_pkg)`
- `_color_refinement_loss_pseudo(record, render_pkg, mask_source)`

这样可以避免 `color_refinement()` 本体过度膨胀，也方便日后单独审计 pseudo path。

#### 5.2.4 不要把 pseudo 塞进 `self.viewpoints`

这是非常重要的边界。

原因：

- pseudo `viewpoint` 是 runtime/refine 视角，不是 tracking 主相机
- `self.viewpoints` 目前承载的是 SLAM/frontend 相机集合
- 若直接塞进去，会污染 camera export、随机 real sampling、甚至后续别的逻辑

正确做法是：

- color refinement 单独读取 `self.brpo_runtime_pseudo_records`
- 把 pseudo 当成 appearance supervision member，而不是 frontend camera

#### 5.2.5 加入 debug summary

当前 color refinement 结束后只打：

- `Map refinement done`

这不足以证明 pseudo 是否真的被消费。

建议新增：

- `color_refinement_summary.json`

至少记录：

- `use_pseudo`
- `pseudo_pool_size`
- `pseudo_mask_source`
- `num_real_steps`
- `num_pseudo_steps`
- `mean_real_loss`
- `mean_pseudo_loss`
- `mean_pseudo_mask_nonzero_ratio`

有了这个 summary，后续就能直接判定“这次 after_opt 到底是不是含 pseudo 的 color refinement”。

### 5.3 `third_party/S3PO-GS/utils/internal_eval_utils.py`

建议同步扩展 manifest 元数据，至少写入：

- `color_refinement_use_pseudo`
- `color_refinement_pseudo_mask_source`
- `color_refinement_pseudo_pool_size`
- `color_refinement_updates_pose = false`（继续保留）

否则未来再看 artifact 时，无法从 after_opt 产物直接知道这次 color refinement 到底走的是 real-only 还是 real+pseudo。

### 5.4 `part3_BRPO/configs/*.yaml`

应新建独立 config，而不是静默改掉 E2/E3：

例如：

- `e6_jointprimary_rgbonly_pseudo_colorrefine.yaml`
- `e7_jointprimary_rgbonly_realpose_pseudo_colorrefine.yaml`

推荐第一版设置：

```yaml
Results:
  color_refinement: true
  color_refinement_use_pseudo: true
  color_refinement_pseudo_ratio: 0.5
  color_refinement_pseudo_weight: 1.0
  color_refinement_pseudo_mask_source: confidence_mask
```

### 5.5 不建议做的错误改法

1. **不要**把 pseudo image 乘 mask 后直接喂原始 `ssim()` 作为“masked SSIM”。
2. **不要**把 pseudo 注册进 `self.viewpoints` 伪装成普通 real camera。
3. **不要**在 color refinement 阶段偷偷打开 pose/exposure 优化；这会把 appearance refinement 和 pose refinement 再次混在一起。

---

## 6. 第一版建议的实施顺序

### Phase A：loss helper 落地

1. 在 `loss_utils.py` 新增 `masked_l1_loss` / `masked_ssim`
2. 写一个最小单元验证，确认：
   - 全 1 mask 时接近原始 full-image loss
   - 稀疏 mask 时只在有效区贡献 loss

### Phase B：backend color refinement 接 pseudo

1. 扩展 `Results` 配置
2. 给 `color_refinement()` 增加 mixed sampling path
3. 输出 `color_refinement_summary.json`

### Phase C：正式 compare

建议实验顺序：

1. `E4/E5` real-only color refinement
2. `E6/E7` 在相同 mapping 拓扑上开启 pseudo color refinement

这样可以单独回答：

- final masked pseudo appearance refinement 本身，是否能把 after_opt 的 PSNR/SSIM 再往上推。

---

## 7. 审计方案：如何证明 masked pseudo 已接进 final color refinement

### 7.1 静态代码审计

必查点：

1. `color_refinement()` 必须存在读取 `self.brpo_runtime_pseudo_records` 的路径。
2. pseudo path 必须调用 `masked_l1_loss` / `masked_ssim`，而不是 full-image `l1_loss` / `ssim`。
3. pseudo mask source 必须显式可配，默认是 `confidence_mask`。
4. color refinement 不能创建 pose optimizer，也不能调用 `update_pose(...)`。

### 7.2 运行时 smoke 审计

建议先把 `S3PO_COLOR_REFINEMENT_ITERS` 暂时降到一个小值（例如 200），做短程验证。

需要看到的证据：

1. `color_refinement_summary.json` 中：
   - `use_pseudo = true`
   - `pseudo_pool_size > 0`
   - `num_pseudo_steps > 0`
2. 日志里明确区分：
   - real refinement steps
   - pseudo refinement steps
3. `mean_pseudo_mask_nonzero_ratio > 0`

### 7.3 产物流审计

重点看：

1. `internal_eval_cache/manifest.json` 是否记录了 pseudo color refinement metadata。
2. `camera_states.json` 是否保持不变（仍只对应 real frames）。
3. `color_refinement_updates_pose` 是否仍为 `false`。

### 7.4 隔离 effect 的 compare 审计

要证明“after_opt 改善来自 masked pseudo color refinement”，必须尽量固定 mapping 前状态。

推荐做法：

1. 以同一份 mapping 结束时的 gaussians / camera_states 作为输入
2. 跑两次 final color refinement：
   - `real-only`
   - `real + masked pseudo`
3. 比较最终 after_opt：
   - PSNR
   - SSIM
   - LPIPS

只有这样，才能把 pseudo final refinement 的贡献从前面的 online mapping 贡献中剥离出来。

### 7.5 正式宣称标准

只有同时满足以下条件，才能说“masked pseudo 已进入 final color refinement”：

1. 静态代码路径明确读取 runtime pseudo records
2. pseudo loss 使用 masked L1 + masked SSIM
3. runtime summary 里 `num_pseudo_steps > 0`
4. `camera_states.json` 与 pose metadata 保持 real-only、不更新 pose

少任何一条，都不能说“已经对齐作者语义”。

---

## 8. 预期影响与风险

### 8.1 预期收益

1. 让 pseudo 对最终 appearance 结果的贡献不再止步于 online mapping 阶段。
2. 更接近作者描述的“最终 color refinement 也使用 mask 限制的 pseudo”。
3. 如果当前 PSNR 提升被 real-only color refinement 冲淡，这一步有机会直接释放 pseudo 的 appearance 增益。

### 8.2 风险

1. 若 pseudo target 质量本身不稳，final appearance refinement 也可能把 noise 写进 gaussians。
2. 若 masked SSIM 实现不正确，可能出现看似“加入 pseudo 后效果变坏”，但实际是 loss 定义错了。
3. 如果 mixed sampling 比例过大，pseudo 有可能压过 real GT appearance supervision。

因此第一版建议：

- `pseudo_ratio` 保守起步
- `pseudo_weight` 先与 real 相同或更低
- 必须保留 real-only control

---

## 9. 一句话版实施原则

**不要把 pseudo 当成新的真实相机；要把 pseudo 当成 final appearance refinement 里的 masked supervision member。**

这条边界和 online mapping 改动一一样重要。
