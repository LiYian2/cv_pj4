# hermes.md

> 用途：Part3 BRPO 压缩/重启后的第一入口。先看这份，再按这里列的顺序继续。
> 维护原则：只保留当前真实状态、当前执行顺序、关键文档入口和固定环境信息。
> 更新时间：2026-04-20 19:15 (Asia/Shanghai)

---

## 口径说明（重要）

**当前统一口径**：
- **M~** = Mask（confidence），监督域 + 监督强度
- **T~** = Target，监督目标数值
- **G~** = Gaussian Management，per-Gaussian gating
- **R~** = Joint Refine，topology joint loop

历史文档中的 **A1 → M~ + T~**（分离记载），**B3 → G~**，**T1 → R~**。

详细设计文档位置：
- `docs/design/MASK_DESIGN.md` — M~ 详细分析
- `docs/design/TARGET_DESIGN.md` — T~ 详细分析
- `docs/design/GAUSSIAN_MANAGEMENT_DESIGN.md` — G~ 详细分析
- `docs/design/REFINE_DESIGN.md` — R~ 详细分析

---

## 1. 现在先怎么用这份文档

如果用户让我"先回忆一下现在做到哪了"，按这个顺序：
1. 先看本文件 `docs/agent/hermes.md`
2. 再看 `docs/current/STATUS.md`
3. 再看 `docs/current/DESIGN.md`
4. 再看 `docs/current/CHANGELOG.md`
5. 如果要看详细设计，再看 `docs/design/` 下的四个文件

---

## 2. 当前项目位置（真实状态）

### M~ 状态：已完成 BRPO 对齐

M~（mask）已完成 BRPO semantics 对齐：
- exact M~ + old T~ 组合达到 24.1875 PSNR
- 与 old M~ + old T~ (24.1877) 差距仅 0.0002 PSNR
- 说明 M~ 侧已经成功对齐

### T~ 状态：residual gap 需 upstream 改进

T~（target）仍有 residual gap：
- exact M~ + exact T~ = 24.1745 PSNR，比 old 组合低约 0.013
- hybrid/stable T~ 都比 old T~ 差
- 说明 T~ 的瓶颈在上游 proxy backend，不是 consumer blend 微调

### G~ 状态：已诊断，待推进

G~（Gaussian Management）opacity compare 显示：
- opacity vs summary = -0.000573 PSNR（weak-negative）
- 三模式（boolean/opacity/summary）无区别
- 原因：分数相关、动作轻、delayed、动作域窄

### R~ 状态：稳定

R~（Topology）T1 = brpo_joint_v1 是稳定主线。

---

## 3. 当前最佳组合

`exact M~ + old T~ + summary G~ + T1` ≈ old A1 + new T1

---

## 4. 下一步：推进 G~ BRPO 对齐

用户明确要求推进 G~ 的 BRPO 对齐。

### G~ BRPO 对齐的核心方向

**Phase 1：action law 改进**
- 当前 opacity scale mean ~0.98 太保守
- 目标：让 low participation score 有更明显的 opacity reduction

**Phase 2：action domain 扩展**
- 当前 candidate_part_only_ratio ~10% 太窄
- 目标：让更多 Gaussians 受影响

**Phase 3：score semantics separation**
- 当前 state vs participation correlation 高
- 目标：让 participation score 有更独立语义

### 重要约束
- 每次只改一小步
- 改完必须跑 compare 验证
- 不把 G~ 和 M~/T~/R~ 混在一起改
- 不直接跳到 O2a/b

---

## 5. 相关代码路径

G~ 核心文件：
- `pseudo_branch/spgm/manager.py` — action generation
- `pseudo_branch/spgm/score.py` — score definitions
- `scripts/run_pseudo_refinement_v2.py` — consumer side

---

## 6. 明确不要做的事

1. 不偷偷做文档"安全小改动"
2. 不把 G~ 和 M~/T~/R~ 同时改
3. 不在没跑 compare 时声称改动有效
4. 不继续优先投入 T~ residual gap（太小不值得）

---

## 7. 一句话 handoff

M~ 已完成 BRPO 对齐，T~ residual gap 太小不值得继续。现在重心转向 G~ BRPO 对齐：action law 太保守、action domain 太窄、score semantics 太相似——三个方向依次推进，每次只改一小步，必须跑 compare。

---

## 8. 云服务器环境信息

### SSH alias
- `Group8DDY`

### Python 环境
- Conda env: `s3po-gs`
- 路径: `/home/bzhang512/miniconda3/envs/s3po-gs/bin/python`

### PYTHONPATH
- `/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO`

### 项目路径
- Part3 root: `/home/bzhang512/CV_Project/part3_BRPO`
- 输出: `/data2/bzhang512/CV_Project/output/part3_BRPO/experiments`

### 执行模板
```bash
ssh Group8DDY "cd /home/bzhang512/CV_Project/part3_BRPO && export PYTHONPATH=/home/bzhang512/CV_Project/third_party/S3PO-GS:/home/bzhang512/CV_Project/part3_BRPO && /home/bzhang512/miniconda3/envs/s3po-gs/bin/python scripts/xxx.py --args"
```