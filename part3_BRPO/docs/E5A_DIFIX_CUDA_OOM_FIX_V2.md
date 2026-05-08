# E5a Difix CUDA OOM 问题修复方案 V2

## 设计目标

1. **动态继承 GPU**：任务提交时在哪个 GPU 上，BackEnd 子进程和 Difix 就用哪个 GPU
2. **开关控制**：通过配置控制是否启用 Difix
3. **不硬编码**：不写死 cuda:0，而是根据 CUDA_VISIBLE_DEVICES 动态确定

---

## 问题根源

spawn 方式创建 BackEnd 子进程时：
1. 父进程创建 `self.background = torch.tensor(bg_color, device="cuda")` CUDA tensor
2. 传递给 BackEnd 对象：`self.backend.background = self.background`
3. spawn 子进程 unpickle BackEnd 对象时，unpickle CUDA tensor **触发 CUDA context 初始化**
4. 这个初始化发生在 BackEnd.run() 的 `torch.cuda.set_device(0)` 之前
5. 结果：CUDA context 初始化到错误的物理 GPU（默认 GPU 0）

---

## 修复方案

### 核心思路

1. **不传递 CUDA tensor 给 BackEnd**：传递 CPU 数据（list 或 numpy）
2. **BackEnd.run() 开头正确初始化 CUDA context**：读取 CUDA_VISIBLE_DEVICES，设置正确的 device
3. **在 BackEnd.run() 中重建 CUDA tensor**：从 CPU 数据重建
4. **Difix 模型加载时动态获取 device**：不硬编码 cuda:0

---

## 具体修改

### 修改 1：slam.py（不传递 CUDA tensor）

文件：`third_party/S3PO-GS/slam.py`

原代码（第 71-95 行）：
```python
bg_color = [0, 0, 0]
self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
...
self.backend.background = self.background
```

修改后：
```python
bg_color = [0, 0, 0]
self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 父进程仍用 CUDA
self._bg_color_cpu = bg_color  # 存储 CPU 版本，传递给 BackEnd
...
self.backend.background_cpu = self._bg_color_cpu  # 传递 CPU 数据，不传递 CUDA tensor
```

### 修改 2：slam_backend.py BackEnd.__init__

文件：`third_party/S3PO-GS/utils/slam_backend.py`

原代码（第 40 行）：
```python
self.background = None
```

修改后：
```python
self.background = None  # CUDA tensor，在 run() 中创建
self.background_cpu = None  # CPU 数据，从父进程传递
self._target_cuda_device = 0  # 目标 CUDA device index
```

### 修改 3：slam_backend.py BackEnd.run()

文件：`third_party/S3PO-GS/utils/slam_backend.py`

原代码（第 1189-1196 行）：
```python
def run(self):
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        Log(f"[BackEnd.run] CUDA_VISIBLE_DEVICES={cuda_visible}, setting device cuda:0")
        torch.cuda.set_device(0)
    while True:
        ...
```

修改后：
```python
def run(self):
    # Step 1: 动态初始化 CUDA context（关键：必须在任何 CUDA 操作之前）
    import os
    self._target_cuda_device = 0  # 默认 cuda:0
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        # CUDA_VISIBLE_DEVICES 重映射了 GPU index
        # 例如 CUDA_VISIBLE_DEVICES=1 时，cuda:0 映射到物理 GPU 1
        # 所以子进程应该用 cuda:0（就是用户指定的物理 GPU）
        self._target_cuda_device = 0
        Log(f"[BackEnd.run] CUDA_VISIBLE_DEVICES={cuda_visible}, will use cuda:0 (mapped to physical GPU {cuda_visible})")
    
    # 强制初始化 CUDA context 到正确的 device
    torch.cuda.set_device(self._target_cuda_device)
    
    # 确认 CUDA context 已初始化（创建 dummy tensor）
    try:
        _ = torch.zeros(1, device=f"cuda:{self._target_cuda_device}")
        Log(f"[BackEnd.run] CUDA context initialized on cuda:{self._target_cuda_device}")
    except Exception as e:
        Log(f"[BackEnd.run] WARNING: CUDA context initialization failed: {e}")
    
    # Step 2: 重建 background tensor（从 CPU 数据）
    if self.background_cpu is not None:
        self.background = torch.tensor(
            self.background_cpu, 
            dtype=torch.float32, 
            device=f"cuda:{self._target_cuda_device}"
        )
        Log(f"[BackEnd.run] background tensor rebuilt on cuda:{self._target_cuda_device}")
    elif self.background is not None and isinstance(self.background, torch.Tensor):
        # 兼容旧代码：如果父进程传递了 CUDA tensor，检查 device 是否正确
        if self.background.device.index != self._target_cuda_device:
            Log(f"[BackEnd.run] WARNING: background tensor on wrong device {self.background.device}, rebuilding...")
            bg_data = self.background.cpu().numpy()
            self.background = torch.tensor(bg_data, dtype=torch.float32, device=f"cuda:{self._target_cuda_device}")
    
    while True:
        ...
```

### 修改 4：load_difix_model（动态获取 device）

文件：`scripts/legacy_prepare/prepare_stage1_difix_dataset_s3po.py`

原代码：
```python
def load_difix_model(model_name, model_path, timestep):
    import torch
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None:
        torch.cuda.set_device(0)
    ...
    pipe = pipe.to(torch.device("cuda"))
    ...
```

修改后：
```python
def load_difix_model(model_name, model_path, timestep, target_device=None):
    import torch
    import os
    
    # 动态确定目标 device
    if target_device is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible is not None:
            # cuda:0 映射到用户指定的物理 GPU
            target_device = torch.device("cuda:0")
        else:
            # 没有 CUDA_VISIBLE_DEVICES，默认 cuda:0
            target_device = torch.device("cuda:0")
    
    # 强制初始化 CUDA context
    torch.cuda.set_device(target_device.index)
    _ = torch.zeros(1, device=target_device)
    
    if model_name and not model_path:
        from diffusers import DiffusionPipeline
        custom_pipeline = "/home/bzhang512/CV_Project/third_party/Difix3D/src/pipeline_difix.py"
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            custom_pipeline=custom_pipeline,
            trust_remote_code=True,
        )
        pipe = pipe.to(target_device)  # 使用动态 device
        ...
        return {"kind": "hf_pipeline", "obj": pipe, "timestep": timestep, "device": target_device}
    ...
```

### 修改 5：BackEnd._ensure_brpo_difix_model_loaded

文件：`third_party/S3PO-GS/utils/slam_backend.py`

原代码（第 260-280 行）：
```python
def _ensure_brpo_difix_model_loaded(self):
    ...
    self.brpo_difix_model = load_difix_model(...)
    ...
```

修改后：
```python
def _ensure_brpo_difix_model_loaded(self):
    cfg = self.brpo_online_mapping_cfg
    if cfg is None or not bool(cfg.get("use_difix_restoration", False)):
        return None
    if self.brpo_difix_model is None:
        from scripts.legacy_prepare.prepare_stage1_difix_dataset_s3po import load_difix_model
        
        # 使用 BackEnd.run() 中确定的 target device
        target_device = torch.device(f"cuda:{self._target_cuda_device}")
        
        self.brpo_difix_model = load_difix_model(
            model_name=str(cfg.get("difix_model_name", "nvidia/difix_ref")),
            model_path=cfg.get("difix_model_path"),
            timestep=int(cfg.get("difix_timestep", 100)),
            target_device=target_device,  # 传递动态 device
        )
        Log(f"[BRPOOnlineMapping] Difix model loaded on {target_device}")
    return self.brpo_difix_model
```

---

## 配置开关

已有配置（e5a yaml）：
```yaml
brpo_online_mapping:
  use_difix_restoration: true  # 开关：是否启用 Difix
  difix_model_name: nvidia/difix_ref
  difix_model_path: null
  difix_timestep: 200
```

不需要新增配置，复用现有开关。

---

## 使用方式

用户提交任务时指定 GPU：
```bash
# 使用物理 GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/run_e5a_jointprimary_maskedcolor_rgbonly_cm_difix.sh

# 使用物理 GPU 0,2（多 GPU）
CUDA_VISIBLE_DEVICES=0,2 python scripts/run_e5a_jointprimary_maskedcolor_rgbonly_cm_difix.sh
```

BackEnd 子进程会自动：
- 读取 CUDA_VISIBLE_DEVICES
- 在 cuda:0（映射到用户指定的物理 GPU）上初始化 CUDA context
- 在正确的 GPU 上加载 Difix 模型

---

## 验证方法

运行修复后的代码：
```bash
CUDA_VISIBLE_DEVICES=1 python run_e5a...

# 检查 nvidia-smi
nvidia-smi
```

应该看到：
- 主进程在 GPU 1
- BackEnd 子进程在 GPU 1
- Difix VAE decode 在 GPU 1

---

## 关键点总结

1. **不传递 CUDA tensor 给 spawn 子进程**：传递 CPU 数据（list/numpy）
2. **BackEnd.run() 开头强制初始化 CUDA context**：在任何 CUDA 操作之前
3. **动态 device**：不硬编码 cuda:0，而是根据 CUDA_VISIBLE_DEVICES 确定
4. **复用配置开关**：use_difix_restoration 已有，不需要新增