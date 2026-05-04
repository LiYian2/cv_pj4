# S3PO DL3DV-2 → BRPO Compare Result (Sparse Phase-1 + Dense Add-on, 2026-05-04)

> 目标：先在 **S3PO DL3DV-2 internal-cache root** 上完成一轮 `sparse_desc_2d` 的 old / exact-upstream / paper-style 受控 compare，再补一轮代表性 `dense_pts3d_3d (q=0.70)` compare，判断 dense matching 到底是在救 exact 线、paper 线，还是谁都没有真正转正。

---

## 1. 一句话结论

这轮结果已经比较清楚了：

- **Sparse phase-1**：`old baseline` 仍然最好；exact-upstream 与 paper-style 都跑通了，但都没转正。
- **Dense add-on (q0.70)**：dense matching **显著救了 exact-upstream**，把它从“明显偏稀、明显落后”拉到“几乎追平 old baseline”；但 **paper target 线并没有被 dense 救起来**。

因此，当前最准确的判断是：

> **主线暂时仍保留 old baseline；下一条最值得继续推进的分支，是 `exact_upstream + split-depthconf + dense q0.70`。paper target 这条线虽然 coverage 很大，但到目前为止还没有把这种 coverage 兑现成 replay 收益。**

---

## 2. 运行身份

### 2.1 上游 root（统一使用 S3PO）
- run root：`part2_s3po` 的 DL3DV-2 internal-cache run
- 数据形态：`internal_eval_cache + internal_prepare + pseudo_cache`
- 训练/评估都在 `part3_BRPO` 上执行

### 2.2 Sparse phase-1
- matcher：`sparse_desc_2d`
- frames：`22 50 84 118 158 192 226 248 294`
- schedule：`stageA80 + stageB120 + post40_lr03`
- compare arms：7 条

### 2.3 Dense add-on
- matcher：`dense_pts3d_3d`
- quantile：`q=0.70`
- same frame set / same StageA/StageB schedule
- compare arms：5 条代表臂

---

## 3. Sparse phase-1 结果回顾

### 3.1 replay 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs old |
| --- | ---: | ---: | ---: | ---: |
| `s0_old_joint_sourceaware` | `18.232956` | `0.629209` | `0.322842` | `0` |
| `s1_exact_upstream_coupled` | `18.218182` | `0.627931` | `0.324254` | `-0.014774` |
| `s2_exact_upstream_cm_only` | `18.217217` | `0.627958` | `0.324315` | `-0.015738` |
| `s3_exact_upstream_split_depthconf` | `18.219379` | `0.628093` | `0.324252` | `-0.013577` |
| `s4_paper_cm_old_split` | `18.215819` | `0.628362` | `0.323458` | `-0.017137` |
| `s5_paper_target_split` | `18.216969` | `0.628218` | `0.323887` | `-0.015986` |
| `s6_paper_target_split_depthconf` | `18.215398` | `0.628138` | `0.323867` | `-0.017558` |

### 3.2 关键信号统计

| signal family | `C_m` nonzero ratio | both ratio | single ratio | depth target fill | avg target confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| old joint | `0.01904` | `0.00883` | - | `0.01904` | - |
| exact upstream | `0.00527` | `0.00074` | `0.00453` | `0.00527` | `0.50170` |
| paper cm + old target | `0.01904` | `0.00275` | `0.01629` | `1.00000` | - |
| paper target | `0.01904` | `0.00275` | `0.01629` | `0.81291` | `0.67390` |

### 3.3 Sparse 结论

Sparse 这一轮的结论很直接：

- **old baseline 最稳**；
- **exact-upstream 已经真实接通，但 support 太稀**，所以 replay 还没追上；
- **paper-style C_m / paper target 也都真实接通了**，而且 paper target 的 depth fill 很大，但这些更“宽”的 supervision 还没有转成更好的 replay。

如果只看 sparse，旧主线还不能切。

---

## 4. Dense add-on（q0.70）结果

### 4.1 5 条代表臂 replay 主指标

| arm | PSNR | SSIM | LPIPS | ΔPSNR vs dense old |
| --- | ---: | ---: | ---: | ---: |
| `d0_old_joint_sourceaware_dense_q070` | `18.231776` | `0.628295` | `0.323402` | `0` |
| `d1_exact_upstream_coupled_dense_q070` | `18.229089` | `0.627952` | `0.324760` | `-0.002687` |
| `d2_exact_upstream_split_depthconf_dense_q070` | `18.229831` | `0.628005` | `0.324670` | `-0.001945` |
| `d3_paper_target_split_dense_q070` | `18.216159` | `0.627530` | `0.324252` | `-0.015617` |
| `d4_paper_target_split_depthconf_dense_q070` | `18.216586` | `0.627537` | `0.324261` | `-0.015190` |

### 4.2 Dense 关键信号统计

| signal family | `C_m` nonzero ratio | both ratio | single ratio | depth target fill | avg target confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| old joint (dense) | `0.11973` | `0.01229` | - | `0.11973` | - |
| exact upstream (dense) | `0.02693` | `0.00296` | `0.02397` | `0.02693` | `0.41235` |
| paper target (dense) | `0.11973` | `0.02129` | `0.09844` | `0.81291` | `0.67390` |

### 4.3 Dense 相比 sparse 到底改变了什么

最关键的变化，不在于 old baseline 变好，而在于 **dense 把 exact 那条线从“过稀”拉回来了**。

- exact `C_m` nonzero ratio：`0.00527 -> 0.02693`，提升约 **5.1x**
- old / paper `C_m` coverage 也从 `0.01904 -> 0.11973`，提升约 **6.3x**

但 replay 的反应并不一样：

- **old baseline**：`18.23296 -> 18.23178`，几乎不变
- **exact coupled**：`18.21818 -> 18.22909`，明显回升
- **exact split-depthconf**：`18.21938 -> 18.22983`，也明显回升，而且是 exact dense 两条里最好的一条
- **paper target split**：`18.21697 -> 18.21616`，几乎没提升
- **paper target split-depthconf**：`18.21540 -> 18.21659`，只有极小回升

所以 dense 并不是“把所有线都一起救起来”，而是**主要在救 exact 线**。

---

## 5. 怎么解读 sparse + dense 合起来的结果

### 5.1 exact-upstream：dense 确实有效

这是这轮最重要的正面结果。

Sparse 阶段，exact-upstream 最大的问题是 support 太稀；dense q0.70 后，这个问题被明显缓解了。更重要的是，replay 也跟着抬起来了，说明 dense 并不是只把 support 数量堆高，而是真的让 exact 线接近可用。

而且在 exact dense 两条里：
- `d1 exact coupled` = `18.22909`
- `d2 exact split-depthconf` = `18.22983`

`split-depthconf` 还是更好一点，虽然幅度不大，但方向是一致的：**在 exact 线上，consumer 解耦比完全 coupled 更有希望。**

### 5.2 paper target：coverage 很大，但收益没有兑现

paper target 线的问题现在也更清楚了。

无论 sparse 还是 dense，它的 coverage 都很宽，尤其 dense 下 `C_m` 已经到 `0.11973`，depth fill 还是 `0.81291`。这说明它不是没有 supervision，而是：

> **当前 paper target 线的问题更像“这些 supervision 没有被当前 refine/consumer 吸收成更好的最终结果”。**

也就是说，现在 paper 线的瓶颈已经不太像是 matcher coverage 本身，而更像 consumer contract 或 target semantics 怎么真正落到优化里。

### 5.3 old baseline：仍是当前主线锚点，但优势已经缩小

old baseline 目前还是第一名，但 dense 后 exact 已经追得很近了：
- `d0 old baseline` = `18.231776`
- `d2 exact split-depthconf` = `18.229831`
- 差距只有 **`0.00195 PSNR`** 量级

这意味着：

- 现在还**不能**说 exact 已经赢了；
- 但也已经**不能**再把 exact 视作明显落后线。

它现在更像是：**最有希望在下一轮超车的分支。**

---

## 6. 当前结论

当前可以把结论压成三句：

1. **主线暂时仍保留 old baseline。** 因为到目前为止，它仍是 replay 最优。
2. **最值得继续推进的是 `exact_upstream + split-depthconf + dense q0.70`。** dense 已经把它救到几乎追平 old baseline。
3. **paper target 线暂时不继续大规模扩张。** 它的 coverage 很大，但 sparse 和 dense 两轮都没把这种 coverage 转成 replay 收益，说明问题不在“有没有更多点”，而在“这些点怎么被消费”。

---

## 7. 下一步建议

下一步不应该再把全矩阵重复一遍，而应该更聚焦：

- 以 **exact dense split-depthconf** 为主候选，做下一轮更细的 dense follow-up；
- dense 参数如果要扩展，优先围绕 exact 线做，而不是 paper 线全量复制；
- paper 线如果继续看，更该回头看 consumer / target-confidence contract，而不是继续只堆 matcher coverage。

---

## 8. 文档状态

- 规划文档已归档：
  `docs/archived/2026-05-experiments/S3PO_DL3DV2_BRPO_EXPERIMENT_PLAN_20260504.md`
- 本文件已从“仅 sparse 结果记录”更新为“**sparse phase-1 + dense add-on 的合并结果记录**”。