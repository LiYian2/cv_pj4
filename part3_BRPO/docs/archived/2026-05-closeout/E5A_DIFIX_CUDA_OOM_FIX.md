# E5a Difix CUDA OOM 问题分析与修复方案

## 问题现象

E5a 实验配置 CUDA_VISIBLE_DEVICES=1 后，主进程正确运行在 GPU 1，但 multiprocessing spawn 的 BackEnd 子进程在加载和运行 Difix 模型时，CUDA device context 没有正确继承 CUDA_VISIBLE_DEVICES 映射，导致 Difix VAE decode 推理时仍在物理 GPU 0 上分配内存并 OOM。

GPU 状态（2026-05-08 12:13）：
- GPU 0: 47359MiB / 49140MiB（几乎被其他用户进程占满）
- GPU 1: 41104MiB / 49140MiB（本实验目标 GPU）

## 问题根源

### 代码链路分析

1. **slam.py 第 71 行**：
```python
self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
```
父进程创建 CUDA tensor，device="cuda" 应该在 CUDA_VISIBLE_DEVICES=1 环境下映射到物理 GPU 1。

2. **slam.py 第 95 行**：
```python
self.backend.background = self.background
```
父进程把 CUDA tensor 赋值给 BackEnd 对象。

3. **slam.py 第 385 行**：
```python
mp.set_start_method("spawn")
```
使用 spawn 方式创建子进程。

4. **BackEnd 继承自 mp.Process**（utils/slam_backend.py 第 33 行）：
```python
class BackEnd(mp.Process):
```

5. **BackEnd.run() 第 1192-1195 行**：
```python
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if cuda_visible is not None:
    torch.cuda.set_device(0)
```
子进程 run() 开头设置 cuda:0（应该映射到物理 GPU 1）。

### spawn 方式的问题

**关键问题**：spawn 方式创建子进程时，CUDA context 的继承机制与 fork 方式不同。

spawn 工作流程：
1. 父进程创建 BackEnd 对象（包含 CUDA tensor 属性）
2. 父进程调用 `backend.start()`
3. spawn 创建新的子进程（全新的 Python 解释器）
4. 子进程 unpickle BackEnd 对象
5. **unpickle CUDA tensor 时，会触发 CUDA context 的初始化**
6. 子进程调用 `BackEnd.run()`

**问题在于**：
- unpickle CUDA tensor 时，CUDA context 可能被初始化到错误的物理 GPU
- 这个初始化发生在 `BackEnd.run()` 的 `torch.cuda.set_device(0)` 被调用之前
- 结果：Difix VAE decode 在物理 GPU 0 上运行，而不是物理 GPU 1

### 为什么 unpickle 会触发错误的 CUDA context

假设：
1. spawn 子进程继承了 CUDA_VISIBLE_DEVICES=1 环境变量
2. 但 unpickle CUDA tensor 时，PyTorch 的 CUDA context 初始化可能不遵循 CUDA_VISIBLE_DEVICES
3. 或者 unpickle 过程在 BackEnd.run() 的 device 设置之前就触发了 CUDA context

实际观察到：
- Difix VAE decode 在物理 GPU 0 上分配内存
- 说明 CUDA context 被初始化到了物理 GPU 0

## 修复方案

### 方案 A：在 BackEnd.run() 开头强制重建 CUDA context

在 `utils/slam_backend.py` 的 `BackEnd.run()` 开头：

```python
def run(self):
    # 强制在正确的 device 上初始化 CUDA context
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        # 先销毁可能存在的错误 CUDA context
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 强制设置 device
        torch.cuda.set_device(0)
        # 确认 CUDA context 已初始化
        _ = torch.zeros(1, device="cuda:0")
        Log(f"[BackEnd.run] CUDA initialized on cuda:0 (physical GPU {cuda_visible})")
    
    # 重建 background tensor（而不是使用 unpickle 的）
    if self.background is not None:
        bg_color = self.background.cpu().numpy().tolist()
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda:0")
    
    # ... 继续原有逻辑
```

### 方案 B：修改 BackEnd.__init__ 不存储 CUDA tensor

在 `utils/slam_backend.py` 的 `BackEnd.__init__` 中：

```python
def __init__(self, ...):
    # 不直接存储 CUDA tensor，存储 CPU tensor 或 numpy
    self.background_cpu = None  # 存储 CPU 版本
    # ... 其他初始化
```

在 `BackEnd.run()` 中：

```python
def run(self):
    # ... CUDA context 初始化
    
    # 从 CPU 版本重建 CUDA tensor
    if self.background_cpu is not None:
        self.background = torch.tensor(self.background_cpu, dtype=torch.float32, device="cuda:0")
    
    # ... 继续原有逻辑
```

### 方案 C：修改 slam.py 不传递 CUDA tensor 给 BackEnd

在 `slam.py` 中：

```python
# 不传递 CUDA tensor
self.backend.background = bg_color  # 传递 list/numpy，而不是 tensor
```

在 `BackEnd.run()` 中：

```python
def run(self):
    # ... CUDA context 初始化
    
    # 从原始数据创建 CUDA tensor
    if self.background is not None and not isinstance(self.background, torch.Tensor):
        self.background = torch.tensor(self.background, dtype=torch.float32, device="cuda:0")
```

### 方案 D：使用 fork 代替 spawn（如果支持）

修改 `slam.py`：

```python
mp.set_start_method("fork")  # fork 方式继承 CUDA context
```

注意：fork 方式在某些 CUDA 操作中可能有其他问题，需要测试。

## 推荐方案

**推荐方案 A + C 组合**：

1. 修改 `slam.py`，传递 CPU 数据而不是 CUDA tensor
2. 在 `BackEnd.run()` 开头强制初始化 CUDA context
3. 在 `BackEnd.run()` 中重建 CUDA tensor

这样最安全，避免任何 unpickle CUDA tensor 的问题。

## 具体修改

### 修改 1：slam.py

```python
# 第 71 行改为存储 CPU 版本
self.background_cpu = torch.tensor(bg_color, dtype=torch.float32, device="cpu")

# 第 95 行改为传递 CPU 版本
self.backend.background = self.background_cpu
```

### 修改 2：utils/slam_backend.py BackEnd.run()

```python
def run(self):
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        # 强制在正确的 device 上初始化 CUDA context
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        # 确认 CUDA context 已初始化
        dummy = torch.zeros(1, device="cuda:0")
        Log(f"[BackEnd.run] CUDA context initialized on cuda:0 (physical GPU {cuda_visible})")
    
    # 重建 background tensor
    if self.background is not None:
        if isinstance(self.background, torch.Tensor):
            # 如果是 tensor，先移到 CPU 再重建
            bg_data = self.background.cpu().numpy()
        else:
            bg_data = self.background
        self.background = torch.tensor(bg_data, dtype=torch.float32, device="cuda:0")
    
    # ... 继续原有逻辑
```

### 修改 3：load_difix_model 中确保 device 正确

```python
def load_difix_model(...):
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        torch.cuda.set_device(0)
    
    # ... 加载模型
    
    # 使用 cuda:0 而不是 cuda
    pipe = pipe.to(torch.device("cuda:0"))  # 明确指定 cuda:0
    
    return model_bundle
```

## 验证方法

运行修复后的代码，检查：
```bash
nvidia-smi
```
应该看到 BackEnd 子进程和 Difix 进程都在 GPU 1 上分配内存。

## 参考

- PyTorch multiprocessing spawn CUDA context：https://pytorch.org/docs/stable/notes/multiprocessing.html
- CUDA_VISIBLE_DEVICES 行为：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars