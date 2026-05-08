# Difix Restoration + RGB Fusion 集成规划

> 规划日期: 2026-05-06
> 目标: 将 Difix 双向修复和 RGB Fusion 加入 Online Mapping Pipeline

---

## 1. 当前流程 vs 目标流程

### 1.1 当前流程 (无 Difix)

```
build_runtime_exact_backend_bundle()
    ↓
render_rgb_depth_from_state() → coarse RGB + depth
    ↓
left_ref_img = load(left_state.image_path)  ← 直接用 GT RGB
right_ref_img = load(right_state.image_path)  ← 直接用 GT RGB
    ↓
matcher.match_pair(pseudo_rgb, left_ref_rgb) → MASt3R matching
matcher.match_pair(pseudo_rgb, right_ref_rgb) → MASt3R matching
    ↓
verify_single_branch_exact() → projected depth + confidence
    ↓
创建 RuntimeExactBackendBundle
```

### 1.2 目标流程 (加入 Difix + Fusion)

```
build_runtime_exact_backend_bundle_with_difix()
    ↓
render_rgb_depth_from_state() → coarse RGB + depth
    ↓
⭐ load_difix_model() (Backend 初始化时加载一次)
    ↓
⭐ run_single_difix(coarse_rgb, left_ref_rgb) → left_fixed_rgb
⭐ run_single_difix(coarse_rgb, right_ref_rgb) → right_fixed_rgb
    ↓
⭐ compute_overlap_confidence_map() → w_left, w_right
⭐ fuse_residual_targets() → fused_rgb
    ↓
⭐ matcher.match_pair(fused_rgb, left_ref_rgb) → MASt3R matching
⭐ matcher.match_pair(fused_rgb, right_ref_rgb) → MASt3R matching
    ↓
verify_single_branch_exact() → projected depth + confidence
    ↓
创建 RuntimeExactBackendBundle (使用 fused RGB)
```

---

## 2. 集成方案

### 2.1 架构变更

| 位置 | 变更 |
|------|------|
| `slam_backend.py` | 添加 difix_model 初始化 |
| `runtime_exact_backend.py` | 添加 difix restoration + fusion |
| `BRPOOnlineMappingConfig` | 添加 difix 相关配置参数 |
| `D5 config` | 启用 difix |

### 2.2 新增配置参数

```yaml
Results:
  brpo_online_mapping:
    # 新增 Difix 参数
    use_difix_restoration: true
    difix_model_name: "nvidia/difix_ref"
    difix_model_path: null  # 可选本地路径
    difix_timestep: 100
    difix_prompt: ""
    difix_height: 512
    difix_width: 512
    difix_fusion_mode: "brpo_overlap_confidence"
    depth_consistency_tau: 0.15
    translation_scale_tau: 1.0
```

### 2.3 Backend 初始化变更

**文件**: `slam_backend.py`

```python
def set_hyperparams(self):
    # ... existing code ...

    # ⭐ 新增: 初始化 Difix model (只加载一次)
    self.brpo_difix_model = None
    if self.brpo_online_mapping_cfg is not None:
        difix_cfg = self.brpo_online_mapping_cfg
        if bool(difix_cfg.get("use_difix_restoration", False)):
            from scripts.legacy_prepare.prepare_stage1_difix_dataset_s3po import load_difix_model
            self.brpo_difix_model = load_difix_model(
                model_name=str(difix_cfg.get("difix_model_name", "nvidia/difix_ref")),
                model_path=difix_cfg.get("difix_model_path"),
                timestep=int(difix_cfg.get("difix_timestep", 100)),
            )
            Log("[BRPOOnlineMapping] Difix model loaded")
```

### 2.4 runtime_exact_backend.py 变更

**文件**: `runtime_exact_backend.py`

**新增函数**:

```python
def run_difix_restoration(
    model_bundle: dict,
    pseudo_rgb: np.ndarray,
    left_ref_rgb: np.ndarray,
    right_ref_rgb: np.ndarray,
    cfg: RuntimeExactBackendConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """执行 Difix 双向修复.

    Args:
        model_bundle: Difix model bundle
        pseudo_rgb: Coarse render RGB (H, W, 3)
        left_ref_rgb: Left reference RGB (H, W, 3)
        right_ref_rgb: Right reference RGB (H, W, 3)
        cfg: Config with prompt, height, width

    Returns:
        left_fixed_rgb: Left-branch restored RGB
        right_fixed_rgb: Right-branch restored RGB
    """
    from PIL import Image

    pseudo_img = Image.fromarray(pseudo_rgb.astype(np.uint8))
    left_ref_img = Image.fromarray(left_ref_rgb.astype(np.uint8))
    right_ref_img = Image.fromarray(right_ref_rgb.astype(np.uint8))

    # 左侧修复
    left_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=left_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )
    # 右侧修复
    right_fixed = run_single_difix_pil(
        model_bundle=model_bundle,
        image=pseudo_img,
        ref_image=right_ref_img,
        prompt=str(cfg.difix_prompt or ""),
        height=int(cfg.difix_height or 512),
        width=int(cfg.difix_width or 512),
    )

    return np.array(left_fixed), np.array(right_fixed)


def run_single_difix_pil(model_bundle, image, ref_image, prompt, height, width):
    """Difix 修复单个分支 (PIL 输入)."""
    if model_bundle["kind"] == "hf_pipeline":
        pipe = model_bundle["obj"]
        out = pipe(
            prompt,
            image=image,
            ref_image=ref_image,
            height=height,
            width=width,
            num_inference_steps=1,
            timesteps=[model_bundle["timestep"]],
            guidance_scale=0.0,
        ).images[0]
    else:
        model = model_bundle["obj"]
        out = model.sample(image=image, ref_image=ref_image, prompt=prompt, height=height, width=width)
    if out.size != image.size:
        out = out.resize(image.size, Image.LANCZOS)
    return out
```

**修改 build_runtime_exact_backend_bundle**:

```python
def build_runtime_exact_backend_bundle(
    *,
    slot: RuntimePseudoSlot,
    states_by_id: dict[int, dict[str, Any]],
    gaussians,
    pipe,
    background,
    frame_root: str | Path,
    cfg: RuntimeExactBackendConfig,
    matcher=None,
    difix_model=None,  # ⭐ 新增参数
) -> RuntimeExactBackendBundle:
    pseudo_state = dict(states_by_id[int(slot.frame_id)])
    left_state = dict(states_by_id[int(slot.left_ref_frame_id)])
    right_state = dict(states_by_id[int(slot.right_ref_frame_id)])

    # 1. Render coarse pseudo view
    pseudo_render_rgb, pseudo_render_depth = render_rgb_depth_from_state(
        state=pseudo_state,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        device=cfg.matcher_device,
    )

    # 2. Load reference images (GT RGB)
    left_ref_rgb = np.asarray(Image.open(Path(left_state["image_path"])).convert("RGB"), dtype=np.float32) / 255.0
    right_ref_rgb = np.asarray(Image.open(Path(right_state["image_path"])).convert("RGB"), dtype=np.float32) / 255.0

    # ⭐ 3. Difix Restoration (如果启用)
    if difix_model is not None:
        left_fixed_rgb, right_fixed_rgb = run_difix_restoration(
            model_bundle=difix_model,
            pseudo_rgb=pseudo_render_rgb,
            left_ref_rgb=(left_ref_rgb * 255).astype(np.uint8),
            right_ref_rgb=(right_ref_rgb * 255).astype(np.uint8),
            cfg=cfg,
        )
        # 保存修复结果
        save_rgb_png(frame_root / "difix" / "left_fixed.png", left_fixed_rgb)
        save_rgb_png(frame_root / "difix" / "right_fixed.png", right_fixed_rgb)
    else:
        # 无 Difix，直接使用 coarse render
        left_fixed_rgb = pseudo_render_rgb.astype(np.uint8)
        right_fixed_rgb = pseudo_render_rgb.astype(np.uint8)

    # ⭐ 4. RGB Fusion (如果启用)
    if difix_model is not None:
        from pseudo_branch.observation.pseudo_fusion import (
            compute_overlap_confidence_map,
            normalize_branch_weights,
            fuse_residual_targets,
        )

        # Render ref depth for fusion weight computation
        left_ref_depth = render_rgb_depth_from_state(
            state=left_state, gaussians=gaussians, pipe=pipe, background=background, device=cfg.matcher_device
        )[1]
        right_ref_depth = render_rgb_depth_from_state(
            state=right_state, gaussians=gaussians, pipe=pipe, background=background, device=cfg.matcher_device
        )[1]

        # Compute overlap confidence (depth-based)
        left_geom = compute_overlap_confidence_map(
            pseudo_state=pseudo_state,
            ref_state=left_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=left_ref_depth,
            depth_consistency_tau=float(cfg.depth_consistency_tau or 0.15),
            translation_scale_tau=float(cfg.translation_scale_tau or 1.0),
        )
        right_geom = compute_overlap_confidence_map(
            pseudo_state=pseudo_state,
            ref_state=right_state,
            pseudo_depth=pseudo_render_depth,
            ref_depth=right_ref_depth,
            depth_consistency_tau=float(cfg.depth_consistency_tau or 0.15),
            translation_scale_tau=float(cfg.translation_scale_tau or 1.0),
        )

        # Fusion weights
        w_left, w_right, fused_conf = normalize_branch_weights(
            left_geom["overlap_confidence"],
            right_geom["overlap_confidence"],
        )

        # Fuse RGB
        fused_rgb = fuse_residual_targets(
            I_render=pseudo_render_rgb.astype(np.uint8),
            I_L=left_fixed_rgb,
            I_R=right_fixed_rgb,
            W_L=w_left,
            W_R=w_right,
        )

        # 保存融合结果
        save_rgb_png(frame_root / "fusion" / "fused_rgb.png", fused_rgb)
        np.save(frame_root / "fusion" / "fusion_weight_left.npy", w_left)
        np.save(frame_root / "fusion" / "fusion_weight_right.npy", w_right)
        np.save(frame_root / "fusion" / "confidence_mask_fused.npy", fused_conf)
    else:
        fused_rgb = pseudo_render_rgb.astype(np.uint8)
        w_left = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)
        w_right = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)
        fused_conf = np.ones(pseudo_render_rgb.shape[:2], dtype=np.float32)

    # ⭐ 5. MASt3R Matching (使用 fused RGB)
    fused_rgb_path = frame_root / "inputs" / "pseudo_fused_rgb.png"
    save_rgb_png(fused_rgb_path, fused_rgb)

    # 用 fused RGB 做 matching
    pts_pseudo, pts_ref_left, _ = matcher.match_pair(str(fused_rgb_path), str(left_state["image_path"]), size=int(pseudo_state["image_width"]))
    left_result = verify_single_branch_exact(...)

    pts_pseudo, pts_ref_right, _ = matcher.match_pair(str(fused_rgb_path), str(right_state["image_path"]), size=int(pseudo_state["image_width"]))
    right_result = verify_single_branch_exact(...)

    # ⭐ 6. 融合 MASt3R confidence 到 fusion weight
    # 可选：用 exact backend confidence 替换 proxy overlap confidence
    if difix_model is not None:
        exact_conf_left = _confidence_fusion_weight(left_result)
        exact_conf_right = _confidence_fusion_weight(right_result)
        w_left_final, w_right_final, fused_conf_final = normalize_branch_weights(
            exact_conf_left,
            exact_conf_right,
        )
    else:
        w_left_final = _confidence_fusion_weight(left_result)
        w_right_final = _confidence_fusion_weight(right_result)
        fused_conf_final = np.clip(w_left_final + w_right_final, 0.0, 1.0)

    # 创建 bundle
    return RuntimeExactBackendBundle(
        ...
        pseudo_render_rgb=fused_rgb,  # ⭐ 使用 fused RGB
        fusion_weight_left=w_left_final,
        fusion_weight_right=w_right_final,
        difix_enabled=(difix_model is not None),
        ...
    )
```

---

## 3. 调用链变更

### 3.1 Backend 主循环

```python
# slam_backend.py
def run(self):
    while True:
        if data[0] == "keyframe":
            # ...

            # ⭐ 准备 pseudo slots (传入 difix_model)
            prepare_payload = self._maybe_prepare_brpo_runtime_slots(
                cur_frame_idx,
                difix_model=self.brpo_difix_model,  # ⭐ 新增
            )

            self._run_brpo_runtime_pseudo_mapping(...)
```

### 3.2 _maybe_prepare_brpo_runtime_slots 变更

```python
def _maybe_prepare_brpo_runtime_slots(self, cur_frame_idx, difix_model=None):
    # ...

    for slot in slots:
        # ⭐ 传入 difix_model
        exact_bundle = build_runtime_exact_backend_bundle(
            slot=slot,
            states_by_id=self.brpo_runtime_camera_states,
            gaussians=self.gaussians,
            pipe=self.pipeline_params,
            background=self.background,
            frame_root=frame_root,
            cfg=exact_cfg,
            matcher=matcher,
            difix_model=difix_model,  # ⭐ 新增
        )
```

---

## 4. 数据流对比

### 4.1 无 Difix (当前)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Coarse      │     │  Left Ref    │     │  Right Ref   │
│  Render RGB  │────►│  GT RGB      │────►│  GT RGB      │
│  (noisy)     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │                    │
        │                    │                    │
        └──────┬─────────────┴────────────────────┘
               │
               ▼
        ┌─────────────────┐
        │  MASt3R Match   │ (coarse ↔ GT)
        │  (noisy input)  │
        └─────────────────┘
               │
               ▼
        ┌─────────────────┐
        │  Projected      │
        │  Depth + Conf   │
        └─────────────────┘
```

### 4.2 有 Difix (目标)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Coarse      │     │  Left Ref    │     │  Right Ref   │
│  Render RGB  │     │  GT RGB      │     │  GT RGB      │
│  (noisy)     │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │                    │
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Difix       │────►│  Difix       │────►│  Difix       │
│  Left Fix    │     │  Right Fix   │     │              │
│  (ref=left)  │     │  (ref=right) │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │                    │
        │                    │                    │
        └──────┬─────────────┴────────────────────┘
               │
               ▼ (depth-guided fusion)
        ┌─────────────────┐
        │  Fused RGB      │
        │  (restored)     │
        └─────────────────┘
               │
               ▼
        ┌─────────────────┐
        │  MASt3R Match   │ (fused ↔ GT)
        │  (clean input)  │
        └─────────────────┘
               │
               ▼
        ┌─────────────────┐
        │  Projected      │
        │  Depth + Conf   │
        └─────────────────┘
```

---

## 5. 实施步骤

### Phase 1: Difix 模型加载 (Backend 初始化)

| 步骤 | 文件 | 任务 |
|------|------|------|
| 1.1 | `slam_backend.py` | 添加 `brpo_difix_model` 属性 |
| 1.2 | `slam_backend.py` | `set_hyperparams()` 中加载模型 |
| 1.3 | `slam_backend.py` | `_maybe_prepare_brpo_runtime_slots()` 传递 model |

### Phase 2: Difix Restoration 函数

| 步骤 | 文件 | 任务 |
|------|------|------|
| 2.1 | `runtime_exact_backend.py` | 新增 `run_difix_restoration()` |
| 2.2 | `runtime_exact_backend.py` | 新增 `run_single_difix_pil()` |
| 2.3 | `runtime_exact_backend.py` | 修改 `build_runtime_exact_backend_bundle()` |

### Phase 3: RGB Fusion 函数

| 步骤 | 文件 | 任务 |
|------|------|------|
| 3.1 | `runtime_exact_backend.py` | 集成 `compute_overlap_confidence_map()` |
| 3.2 | `runtime_exact_backend.py` | 集成 `fuse_residual_targets()` |
| 3.3 | `runtime_exact_backend.py` | 融合 exact backend confidence |

### Phase 4: 配置和测试

| 步骤 | 文件 | 任务 |
|------|------|------|
| 4.1 | `BRPOOnlineMappingConfig` | 添加 difix 配置参数 |
| 4.2 | `d5_online_mapping_fix.yaml` | 启用 difix |
| 4.3 | 测试 | 运行 D5 验证 difix 效果 |

---

## 6. 关键问题

### 6.1 Difix 模型加载时机

**问题**: Difix 模型需要 GPU memory，何时加载？

**方案**: Backend 进程启动时加载（`set_hyperparams()`），只加载一次，所有 pseudo slots 共享。

**GPU Memory 影响**: Difix 模型约 2-3 GB，需要确保 A6000 (48GB) 有足够空间。

### 6.2 Difix 输入尺寸

**问题**: Difix 需要 512x512 输入？

**方案**: 配置中设置 `difix_height=512, difix_width=512`，与 DL3DV 数据集一致。

### 6.3 Fusion 权重来源

**问题**: Fusion 权重用什么？

**方案**:
- Primary: Depth-guided overlap confidence (`compute_overlap_confidence_map`)
- Secondary: Exact backend confidence (MASt3R matching confidence)

**Priority**: Exact backend confidence > Proxy overlap confidence

### 6.4 性能影响

**问题**: Difix restoration 会增加多少时间？

**预估**: 每个 pseudo slot 约 0.5-1 秒（单步 diffusion）。3 slots = 3-6 秒。

**建议**: 可配置 `difix_enabled` 来开关，对比有无 difix 的效果。

---

## 7. 预期效果

| 效果 | 描述 |
|------|------|
| **Pseudo RGB 质量** | Difix 修复后减少 coarse render artifacts |
| **MASt3R matching** | Fused RGB 作为输入，匹配更准确 |
| **Confidence mask** | 更可靠的 mask，减少 false positive |
| **ATE/PSNR** | 预期进一步改善（待验证） |

---

## 8. 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| GPU memory 不足 | Difix 模型加载失败 | 使用 A6000 48GB，空间足够 |
| Difix 推理慢 | Online mapping 延迟增加 | 可配置开关，对比效果 |
| Difix 修复失败 | RGB 质量未改善 | fallback 到 coarse render |
| Fusion 权重不准 | Mask 不准确 | 多级 confidence 融合 |

---

## 9. 下一步

1. **确认 Difix 模型路径**: 检查 `nvidia/difix_ref` 是否可用
2. **实施 Phase 1-3**: 按步骤修改代码
3. **生成 D6 配置**: D5 + difix enabled
4. **运行验证**: 对比 D5 (no difix) vs D6 (with difix)