# 运行调试信息整理

说明：  
- `01` 脚本跑的是 **405841 / waymo_405841**  
- `02` 脚本跑的是 **re10k1 / re10k_1**  
- `03` 脚本跑的是 **dl3dv2 / dl3dv_2**

---

## 图 1：关于 `color_refinement` 的判断

你这个问题问得很关键。结论先说：

1. `color_refinement` 是一个“SLAM主流程结束后的”后处理优化开关，不是前端跟踪主循环本身。  
2. 设为 `false` 会跳过这一整段后处理与队列同步等待。  
3. 关掉它通常会明显缩短收尾时间，并降低你现在这种“高显存低功耗长时间不动”的风险。  
4. 代价是少了 `after_opt` 这轮渲染评估与可能的外观质量提升。

证据链如下：

1. 触发位置在主流程后半段。  
   见 `third_party/S3PO-GS/slam.py` 到 `third_party/S3PO-GS/slam.py`。  
   这里在 `self.color_refinement` 为 `true` 时会：
   - 清空 `frontend_queue`
   - 给 `backend_queue` 发送 `["color_refinement"]`
   - 然后 `while True` 等 `frontend_queue` 回 `sync_backend`

2. `backend` 收到 `color_refinement` 后会跑一个非常重的循环。  
   见 `third_party/S3PO-GS/utils/slam_backend.py` 到 `third_party/S3PO-GS/utils/slam_backend.py`。  
   核心是 `iteration_total = 26000`，每轮 `render + loss.backward + optimizer.step`。

3. 跑完后 `backend` 会 `push_to_frontend` 做同步返回。  
   见 `third_party/S3PO-GS/utils/slam_backend.py` 到 `third_party/S3PO-GS/utils/slam_backend.py`。

关掉它会具体影响什么：

### 1. 会被跳过的内容

- `backend` 的 26000 次 refinement 迭代。
- 主进程等待 `sync_backend` 的这段同步。
- `after_opt` 的渲染评估  
  （`eval_rendering(iteration="after_opt")`）。  
  对应位置 `third_party/S3PO-GS/slam.py` 到 `third_party/S3PO-GS/slam.py`。

### 2. 仍会保留的内容

- `before_opt` 的 ATE 和渲染评估。  
  对应 `third_party/S3PO-GS/slam.py` 到 `third_party/S3PO-GS/slam.py`。
- 正常 `stop backend` 并退出。  
  对应 `third_party/S3PO-GS/slam.py` 到 `third_party/S3PO-GS/slam.py`。

### 3. 结果文件层面的差异

- 关掉后通常只有 `psnr/before_opt/final_result.json`，不会有  
  `psnr/after_opt/final_result.json`。  
  评估结果写盘逻辑见 `third_party/S3PO-GS/utils/eval_utils.py#L263`。
- 质量上通常表现为：外观优化机会减少，可能  
  `PSNR/SSIM/LPIPS` 略差于开启时；但主流程可用结果一般仍可得到。

---

## 图 2：关于“已有结果但进程不退出”的判断（对应 03 / dl3dv2）

结论也出来了，你这个现象不是错觉，确实有“结果已落盘但进程不退出”的问题，而且我已经定位到非常可疑的点。

### 当前实测状态

1. `DL3DV` 这次 run 目录已经有最终产物：  
   - `2026-03-31-01-09-59`
   - 里面有 `point_cloud/final/point_cloud.ply`、  
     `psnr/before_opt/final_result.json`、轨迹图等。

2. 但对应进程仍在：
   - 主进程 `slam.py` 在 `do_wait`
   - 子进程在 `futex_wait_queue_me`  
   这就是“主进程在等子进程退出，但子进程没退出”的经典形态。

3. `re10k` 那个进程还在跑中，不是同一阶段卡住。

### 最可疑根因（高概率）

不是你 notebook 本身，而是 `S3PO` 进程间队列轮询写法有潜在死等风险，尤其大量使用 `Queue.empty()` 判空分支。  
关键代码：

- 前端/主流程等待同步：`third_party/S3PO-GS/slam.py` 到 `third_party/S3PO-GS/slam.py`
- `backend` 主循环不断做 `backend_queue.empty()`：  
  `third_party/S3PO-GS/utils/slam_backend.py` 到 `third_party/S3PO-GS/utils/slam_backend.py`

`multiprocessing.Queue.empty()` 在多进程里本来就不可靠。  
可能出现“队列里有消息但判断为空”的时序问题，从而导致 `stop/sync` 消息被错过，进程不退出。

### 为什么你 interrupt 后 GPU 进程还在

你在 notebook 里用的是阻塞式 `subprocess.run(...)`  
（见 `part2_s3po/notebooks/02_run_full_03.ipynb` 到 `part2_s3po/notebooks/02_run_full_03.ipynb`）。  
Jupyter 的中断不一定会把子进程树（`slam.py` + backend 子进程）一起干净终止，所以会出现你看到的“cell 停了但 `nvidia-smi` 还有进程，占显存，需要手动 `pkill`”。

### 你同学提到 `color_refinement` 的判断

- 你同学说“关掉就好”在工程上是有效缓解手段。
- 但从当前代码看，`color_refinement` 不是主 SLAM 的核心必需环节，更像后处理优化阶段（耗时很长），关掉会减少卡在收尾同步阶段的概率。
- 它是性能/质量增强项，不是“没有就不能跑”的主流程项。

如果你愿意，我下一步可以继续“不改代码”帮你做一件事：  
给你一个只读脚本，自动识别“已出最终结果但仍僵死”的 run，并仅打印建议清理的 PID 列表（不执行 kill）。

---

## 图 3：关于 02 / re10k1 的判断

`02` 现在的问题不是 `color_refinement`。你这次 `Re10k` 失败的直接原因是 **ATE 对齐阶段抛了几何退化异常**，进程以 `returncode=1` 退出，所以后面的 `PSNR` 和 `final ply` 都不会生成。

### 关键证据

1. `02` 和 `03` 的实际运行配置里，`color_refinement` 都是 `false`  
   - `config.yml:27`
   - `config.yml:27`

2. `02` 失败产物特征非常明确  
   `s3po_re10k1_sparse_full_result.json:10-16` 显示只有 `trj_final`，没有 `stats/psnr/final ply`，且 `returncode=1`。  
   这和 ATE 阶段崩溃完全一致。

3. 代码路径对应关系吻合  
   在 `slam.py:128-150` 先走 `eval_ate`，再走 `eval_rendering`。  
   `eval_ate` 会先写 `trj_final`，再调用 `evaluate_evo`：
   - `eval_utils.py:112-117`
   - `evaluate_evo` 在对齐处调用 `Umeyama`：
   - `eval_utils.py:33`

   如果这里抛异常，就不会写 `stats`，也不会进入后续  
   `PSNR/final ply` 写出逻辑（分别在：
   - `eval_utils.py:45-50`
   - `eval_utils.py:256-263`）。

4. 我做了同环境复现实验  
   直接用 `02` 的 `trj_final` 复跑 `evaluate_evo`，报错为：  
   `Degenerate covariance rank, Umeyama alignment is not possible.`

   所以 `02` 的核心失败点已经定位，不是 `color_refinement`。

5. 你看到“像卡住/子进程堆积”的原因也有独立因素  
   `02/03` 的 run cell 都在用缓冲式调用：
   - `02_run_full_02.ipynb:572-573`
   - `02_run_full_03.ipynb:559-560`

   这会导致运行中几乎不实时刷新日志，看起来像“没动静”。  
   如果中途手动中断，容易留下清理不完整的进程树。你之前 `02` 里也有中断痕迹：
   - `02_run_full_02.ipynb:534`

### 补充对比 03

- `03` 对应 run 目录已经有完整结果，包括 `final ply`，说明主流程实际上跑完了：
  - `point_cloud.ply`
- 所以 `03` 与 `02` 不是同一种失败。  
  `02` 是评估阶段异常退出，`03` 更像收尾同步/退出阶段表现不稳定。

### 如果要做最小改动补丁

1. 在 ATE 对齐失败时捕获异常并跳过 ATE，不让整次 run 失败。  
2. 保留后续 `eval_rendering` 和 `final ply` 输出。  
3. 顺手把 notebook run cell 改成流式输出，避免“假卡住”误判。
