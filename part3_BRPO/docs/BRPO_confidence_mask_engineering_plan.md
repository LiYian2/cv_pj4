# BRPO-style Confidence Mask 工程落地方案

> 目标：在你当前 `part3_BRPO/` 结构上，落地一版 **RPO-style confidence mask**，核心是“只信任有双向几何证据的 pseudo 像素”。
> 
> 本文默认：
> 1. `internal cache` 会先补齐，pseudo 相机优先来自 internal tracked camera states；
> 2. 本阶段**先不用 EDP 作为主几何来源**，后续可再叠加；
> 3. 本文只讨论 **confidence mask**，不展开两阶段 refine 本身。

---

## 1. 当前实现 vs BRPO：差异先说清楚

### 1.1 你当前 repo 里已经有的东西

当前 `part3_BRPO/` 已经有：
- `prepare_stage1_difix_dataset_s3po.py`：select / difix / pack 三阶段数据准备
- `pseudo_branch/build_pseudo_cache.py`：构建 `target_depth.npy`、`confidence_mask.npy`
- `pseudo_branch/epipolar_depth.py`：单参考帧 EDP depth + soft confidence
- `pseudo_branch/flow_matcher.py`：MASt3R reciprocal matches
- `pseudo_branch/pseudo_fusion.py`：left/right repaired RGB fusion
- `scripts/run_pseudo_refinement.py`：消费 pseudo cache 做 standalone refine

当前系统已经能做：
- pseudo sample 选择
- left/right pseudo restoration
- soft confidence / fused confidence
- standalone pseudo refinement

### 1.2 你当前实现和 BRPO 的关键不同

当前实现更像：
- **soft confidence prior**
- 以 `match_conf * epipolar_conf` 或 heuristic fusion 为核心
- left/right 两支是“谁更大信谁”或“加权融合”
- 最终没有显式回答：**某个 pseudo 像素是否被左右真实帧共同几何验证**

BRPO 要的是：
- **bidirectional geometric verification**
- 先建立 `pseudo ↔ left_ref`、`pseudo ↔ right_ref` 两组几何支持集
- 再把 pseudo 像素分成：
  - 双边都支持：高置信
  - 单边支持：中置信
  - 两边都不支持：拒收

也就是说，BRPO-style confidence mask 的本质不是“连续 confidence 更精细”，而是：
**先判定 support set，再把 support set 转成 mask。**

### 1.3 为什么你旧版几何监督容易崩

`run_pseudo_refinement.py.bak_before_fix` 里旧几何监督的核心问题不是“loss 写错”，而是：
- 有效 depth support 很稀时，loss 会很脆；
- target depth 本身不够稳时，直接作为监督会拉崩优化；
- 那一版还没有后来的 real-anchor / pseudo-freeze / split-backward 稳定器；
- 更重要的是，它没有先做 BRPO 式 verified support mask。

所以这一轮应该先补的是 **mask 机制**，而不是重新把“弱 depth target”硬塞回 loss。

---

## 2. 本文目标边界

本阶段只做下面这件事：

```text
internal cache / pseudo cache 已有
    ↓
构建 BRPO-style verified support mask
    ↓
输出新的 confidence_mask_brpo.npy / png
    ↓
供后续 refine 使用
```

**本阶段不做：**
- 不改 tracking 主流程
- 不重写 Difix
- 不先接两阶段 refine
- 不先用 EDP 做主监督
- 不先做 scene perception Gaussian management

一句话：
**先把“哪些 pseudo 像素可信”这件事做对。**

---

## 3. BRPO-style confidence mask：工程化定义

### 3.1 输入

对每个 pseudo sample，需要：
- `camera.json`：pseudo 相机
- `refs.json`：left/right 参考帧 id 与 pose
- `render_rgb.png`
- `render_depth.npy`
- `target_rgb_left.png`
- `target_rgb_right.png`
- left/right 真实参考帧 RGB
- left/right 参考深度来源（第一版可来自 internal render depth / pointmap，不强依赖 EDP）

### 3.2 输出

每个 sample 新增：
- `confidence_mask_brpo.npy`
- `confidence_mask_brpo.png`
- `support_left.npy`
- `support_right.npy`
- `support_both.npy`
- `verification_meta.json`
- `diag/` 下若干可视化

### 3.3 核心逻辑

定义两个支持集：
- `M_L`：pseudo 像素中，被左参考帧几何验证通过的像素集合
- `M_R`：pseudo 像素中，被右参考帧几何验证通过的像素集合

然后定义最终 mask：
- `p ∈ M_L ∩ M_R` → `C(p) = 1.0`
- `p ∈ M_L xor M_R` → `C(p) = 0.5`
- `p ∉ M_L ∪ M_R` → `C(p) = 0.0`

这就是第一版 BRPO-style confidence mask。

---

## 4. “几何验证通过”怎么定义

第一版建议不要写得太复杂，先用四条规则。

### 4.1 匹配存在

从 pseudo frame 到 ref frame 必须存在 reciprocal match：
- pseudo repaired / fused frame 与 ref RGB 跑一次 matcher
- 对每个 pseudo 像素，若没有足够稳定的 reciprocal match，直接 reject

### 4.2 重投影有效

把 ref 像素对应的 3D 点重投影回 pseudo 相机后，要求：
- 在 pseudo 图像范围内
- 深度为正
- 对应像素不是 NaN / invalid

### 4.3 重投影残差小

重投影回 pseudo 后，要求：
- 与原 pseudo 匹配像素距离小于阈值 `tau_reproj_px`

建议第一版阈值：
- `tau_reproj_px = 3 ~ 5 px`

### 4.4 深度一致性通过

若当前 pseudo 有 `render_depth.npy`，则要求：
- `|D_reproj - D_render| / max(D_render, eps) < tau_rel_depth`

建议第一版阈值：
- `tau_rel_depth = 0.1 ~ 0.2`

### 4.5 最终定义

对某一条 ref 分支，某 pseudo 像素通过验证当且仅当：

```text
has_reciprocal_match
AND reprojection_in_bounds
AND reprojection_error < tau_reproj_px
AND relative_depth_error < tau_rel_depth
```

这样得到 `M_L` 或 `M_R`。

---

## 5. 第一版建议：不用 EDP，改用 reprojection-based verification

你这轮明确说先不用 EDP，我赞成。

第一版建议用下面这条链：

```text
pseudo frame + left/right ref RGB
    ↓ matcher
得到 pseudo↔ref 2D match
    ↓ ref pose + ref depth/pointmap
把 ref match 反投影到 3D
    ↓ 投到 pseudo 相机
检查重投影误差 + 深度一致性
    ↓
得到 support_left / support_right
```

### 5.1 ref depth 从哪里来

第一版优先级：
1. **internal cache 的 ref render depth / pointmap**
2. 若已有更可靠的 ref pointmap，可直接用
3. 没有则退回当前已有 render depth

这里的关键是：
- 先保证 ref 的深度/点图和 ref pose 自洽
- 不要求一开始就通过 EDP 在 pseudo 侧造新深度

### 5.2 为什么这比旧 EDP-first 更稳

因为这里的 mask 只是决定“信不信这个 pseudo 像素”，不是直接生成一张全图 depth target 去压优化器。

它的失败模式更温和：
- 最多 support 少
- 不会像旧版那样把错误 depth 直接喂进 loss

---

## 6. 建议的数据结构改动

### 6.1 pseudo_cache 结构扩展

在现有 `samples/{frame_id}/` 下新增：

```text
samples/{frame_id}/
├── camera.json
├── refs.json
├── render_rgb.png
├── render_depth.npy
├── target_rgb_left.png
├── target_rgb_right.png
├── target_rgb_fused.png              # 若已有 fusion
├── confidence_mask_brpo.npy          # 新增
├── confidence_mask_brpo.png          # 新增
├── support_left.npy                  # 新增
├── support_right.npy                 # 新增
├── support_both.npy                  # 新增
├── verification_meta.json            # 新增
└── diag/
    ├── support_left.png
    ├── support_right.png
    ├── support_both.png
    ├── reproj_error_left.png
    ├── reproj_error_right.png
    ├── rel_depth_error_left.png
    ├── rel_depth_error_right.png
    └── match_density.png
```

### 6.2 manifest.json 扩展

每个 sample 可新增：
- `confidence_mask_brpo_path`
- `support_left_path`
- `support_right_path`
- `support_both_path`
- `verification_meta_path`
- `verification_version`

---

## 7. 建议新增 / 修改的文件

### 7.1 新增文件

建议新增：

```text
part3_BRPO/pseudo_branch/confidence_mask_inference.py
part3_BRPO/pseudo_branch/reprojection_verify.py
```

#### `reprojection_verify.py`
职责：
- 给定 `pseudo_camera`, `ref_camera`, `ref_depth/pointmap`, `pseudo↔ref matches`
- 输出：
  - `support_mask`
  - `reproj_error_map`
  - `rel_depth_error_map`
  - `stats`

#### `confidence_mask_inference.py`
职责：
- 调用 left/right 两次 verification
- 合成 `M_L`, `M_R`, `M_L∩M_R`, `M_L xor M_R`
- 输出最终 `confidence_mask_brpo`
- 写 `verification_meta.json`

### 7.2 修改文件

#### `pseudo_branch/build_pseudo_cache.py`
修改方式：
- 保留现有 GT / EDP / fusion 逻辑
- 在其后面新增一个 `verification` 阶段
- 第一版允许通过参数控制：
  - `--build-brpo-mask`
  - `--use-internal-ref-depth`
  - `--tau-reproj-px`
  - `--tau-rel-depth`

职责变化：
- 现有 `target_depth / confidence_mask` 逻辑不删
- 新增 `confidence_mask_brpo` 输出

#### `scripts/run_pseudo_refinement.py`
本阶段只做最小改动：
- 增加参数选择 mask 来源
- 例如：`--confidence_mask_source legacy|brpo`

先不要在本文件内混入新的 verification 逻辑。

---

## 8. 推荐的实现顺序

### Phase A：先打通单分支验证

先只做 `pseudo ↔ left_ref`：
- matcher
- ref 3D 回投 pseudo
- 生成 `support_left.npy`
- 生成诊断图

目的：
- 验证几何检查链本身没问题

### Phase B：补右分支

得到 `support_right.npy`。

### Phase C：合成 BRPO-style mask

按 1.0 / 0.5 / 0.0 规则生成：
- `confidence_mask_brpo.npy`
- `support_both.npy`

### Phase D：在 refine 中只替换 mask 来源

此时暂不改 loss，只做：
- 同样的 pseudo RGB-only refine
- 但 mask 从 `legacy confidence` 改成 `brpo mask`

这样能回答：
- 仅仅换 mask，视觉结果是否更稳
- support 稀疏度是否合理

---

## 9. 建议的第一版阈值

先给一组保守默认值，后续再调：

```text
tau_reproj_px      = 4.0
tau_rel_depth      = 0.15
min_match_conf     = 0.0   # 第一版可先不额外阈值
mask_value_both    = 1.0
mask_value_single  = 0.5
mask_value_none    = 0.0
```

如果 support 太 sparse，再按顺序放宽：
1. `tau_reproj_px: 4 -> 6`
2. `tau_rel_depth: 0.15 -> 0.2`
3. 再考虑让单边 support 区域扩大

不要一上来就把阈值放得很松。

---

## 10. 评估与诊断

这一阶段最重要的不是最终 PSNR，而是 support 质量。

每个 sample 建议至少记录：
- `num_support_left`
- `num_support_right`
- `num_support_both`
- `support_ratio_left`
- `support_ratio_right`
- `support_ratio_both`
- `mean_reproj_error_left`
- `mean_reproj_error_right`
- `mean_rel_depth_error_left`
- `mean_rel_depth_error_right`

场景级 summary 建议记录：
- average support ratio
- average both-support ratio
- support 过低的 sample 列表

### 第一版成功标准

不看最终分数，先看这三条：
1. `support_both` 不是几乎全空
2. `confidence_mask_brpo` 空间分布和可见结构基本吻合
3. 用它替换旧 mask 后，refine 不再明显糊化或发散

---

## 11. 和 internal cache 的关系

这个模块强依赖 internal cache，但不需要你在本文件里重新展开 internal cache 实现。

本模块对 internal cache 的最小要求是：
- 能拿到 pseudo frame 的 internal pose
- 能拿到 left/right ref frame 的 internal pose
- 能拿到 left/right ref 的 render depth 或 pointmap
- 帧 id 与相机状态能稳定映射

一句话：
**internal cache 提供稳定的相机与参考深度源；confidence mask 模块负责把它变成“哪些 pseudo 像素可信”。**

---

## 12. 本阶段完成后，下一步怎么接

本阶段完成后，下一步不是先上 EDP，而是优先做：

1. `run_pseudo_refinement.py` 支持 `confidence_mask_brpo`
2. 做一轮 **mask-only replacement ablation**
3. 如果结果稳定，再进入两阶段 refine
4. 两阶段 refine 跑稳后，再考虑把 EDP 作为 depth target 候选叠加进来

也就是说，推荐顺序是：

```text
internal cache
→ BRPO-style confidence mask
→ 两阶段 refine
→ 再尝试 EDP / richer depth target
```

这样最稳。
