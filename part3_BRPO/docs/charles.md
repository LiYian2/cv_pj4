# charles.md

> 记录时间：2026-04-19 05:16
> 主题：B3 已真正开始执行。第一版 deterministic participation controller 已接入主 loop 并完成首轮 formal compare，但当前是 weak-negative。下一次 session 的任务不是再证明 B3 能不能跑，而是沿着 **更保守 participation schedule** 做第二轮调整。

## 这次 session 到底做成了什么
这次不是只改文档，而是真把 B3 往 BRPO-style population manager 推了一步。

### 1) 当前主线没有变
A1/T1 主线仍然是：
- observation：`old A1`（`joint_confidence_v2 + joint_depth_v2`）
- topology：`new T1`（`joint_topology_mode=brpo_joint_v1`）
- StageA.5：optional warmup / control

不要下一次又回去重新争 `new A1 + new T1`。
当前默认候选主线仍然是：

> **`old A1 + new T1`**

### 2) B3 第一版已经不再是旧 grad scaler
这次已经把 B3 的第一版动作接成：

```text
iter t:
  统计 state / candidate subset
  生成 participation_render_mask
iter t+1:
  pseudo render 在 forward 前消费这个 mask
```

也就是说：
- 旧 B3：post-backward `_xyz.grad` scale
- 新 B3 v1：pre-render deterministic participation control

这点已经成立，不要下次再把它说回“只是旧 B3 小修小补”。

### 3) 首轮 compare 结果
正式 compare 路径：
`/data2/bzhang512/CV_Project/output/part3_BRPO/experiments/20260419_b3_det_participation_compare_e1/compare_summary.json`

结果：
- `oldA1_newT1_summary_only = 24.185744 / 0.875419 / 0.080386`
- `oldA1_newT1_b3_det_participation = 24.182511 / 0.875360 / 0.080516`
- delta = `-0.003232 PSNR / -0.000059 SSIM / +0.000130 LPIPS`

结论：
- **B3 的方法对象已经接对**
- 但 **第一版 keep 配置是 weak-negative / no-go for landing**

所以，不要下次误判成“B3 没价值”。真正更准确的判断是：

> **当前 action 位置已经对了，但 action 强度还过重。**

---

## 这次实际踩到的两个 wiring bug（下次别忘）
这两个 bug 是这次真正把 participation control 接进主 loop 后才暴露的：

1. `gaussian_renderer.render(mask=...)` 的 masked 分支返回值签名和 unmasked 分支不一致；
2. masked 分支的 `visibility_filter` 是子集长度，不是 full-length Gaussian mask，导致 SPGM stats 报尺寸不一致。

这两处都已经修过。下次如果你再沿着 B3 participation 路线改代码，先记得：

> **renderer 的 masked branch contract 已经是 B3 的一部分，不是无关底层细节。**

---

## 下一次 session 必须先读的文档（严格按顺序）
1. `/home/bzhang512/CV_Project/part3_BRPO/docs/STATUS.md`
2. `/home/bzhang512/CV_Project/part3_BRPO/docs/DESIGN.md`
3. `/home/bzhang512/CV_Project/part3_BRPO/docs/CHANGELOG.md`
4. `/home/bzhang512/CV_Project/part3_BRPO/docs/B3_deterministic_state_management_engineering_plan.md`
5. 这一份 `charles.md`

如果需要重新进入代码事实，再看：
6. `/home/bzhang512/CV_Project/part3_BRPO/scripts/run_pseudo_refinement_v2.py`
7. `/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/spgm/manager.py`
8. `/home/bzhang512/CV_Project/part3_BRPO/pseudo_branch/spgm/stats.py`
9. `/home/bzhang512/CV_Project/third_party/S3PO-GS/gaussian_splatting/gaussian_renderer/__init__.py`

记住：
- 这次要跟进进度，**先看文档再看代码**；
- 这次最重要的不是“B3 有没有开始做”，而是“B3 第一版为什么 weak-negative”。

---

## 我对下一轮 B3 调整方案的分析
### 结论先说
**下一轮不要改动作位置，先改动作强度。**

动作位置已经对了：现在 B3 已经是 pre-render participation control。当前问题不是“做早了还是做晚了”，而是：

> **第一次 keep 过猛，导致 low-score subset 被削得太多。**

### 我现在的判断
当前 first try 用的是：
- near = `1.0`
- mid = `0.9`
- far = `0.75`
- `state_candidate_quantile = 0.5`

从 log 看，far cluster 一直稳定出现：
- `part_far ≈ 0.875`
- `drop_far` 在几百到一千级
- `cand_far` 也一直很大

这说明什么？
不是代码没生效，而是：
- candidate subset 很大；
- far 侧长期持续 drop；
- 对 pseudo supervision 来说，这个收缩强度已经足够大到开始伤 replay。

### 所以下一轮该怎么调
我倾向的顺序是：

1. **先保守收缩 far keep**
   - 从 `0.75` 提到 `0.9`，必要时再看 `0.95`
   - 目的：先验证“轻度 participation attenuation”是否还能保留 B3 方向的好处，但不再伤主线

2. **mid keep 也更保守**
   - 可以先保持 `1.0` 或最多 `0.95`
   - 当前没必要同时对 mid/far 都做明显 drop

3. **candidate 先不大改**
   - 暂时保留 `state_candidate_quantile=0.5`
   - 因为这次已经能证明 action 对象是 low-score subset；当前优先问题是 action 强度，不是 candidate 口径

4. **暂不进 stochastic masking**
   - deterministic participation 还没转正之前，不该把随机性再加进来

### 我建议的下一轮最小对照
下次 session 直接做这个最小 compare：
- `summary_only`
- `det_participation_far090_mid100`
- 如果有余力，再补：`det_participation_far095_mid100`

也就是：
- **先只收 far**
- **mid 暂时不动或几乎不动**
- 不要一上来又做三四个超参数联动 sweep

### 为什么这样更合理
因为现在我们已经知道：
- B3 路径可跑；
- participation control 已经真正接入；
- 当前问题更像 **over-suppression**；
- 所以下一步应该是 **减少 suppression**，而不是再扩 action 范围。

---

## 明确不要做什么
1. 不要把当前 weak-negative 误读成 B3 方向失败；
2. 不要回退到旧 `xyz_lr_scale` 当主线；
3. 不要立刻跳 stochastic masking；
4. 不要重新去争 old A1 / new A1 谁才是 observation 主线；
5. 不要在下一次 session 一上来就做大 sweep，把因果又搅混。

---

## 给下一个 Charles 的一句话

**你现在不需要再证明 B3 能不能接进主 loop，这件事已经做完了。下一步只做一件事：在 `old A1 + new T1` 主线上，把 deterministic participation 的 action 强度收窄，验证它能不能从 weak-negative 变成至少不伤 replay。**
