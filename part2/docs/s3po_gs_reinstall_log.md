# S3PO-GS 环境重装记录（截至当前）

> 目标：在 `s3po-gs` conda 环境中，把 **Python / PyTorch / NumPy / pip 构建链** 先稳定下来，并把 `simple-knn` 的编译问题定位到当前这一步。  
> 原则：**只记录已验证有效的步骤**；已经证明无效或不稳定的尝试不写入。

---

## 1. 删除旧环境并重建干净环境

### 命令
```bash
conda deactivate
conda remove -n s3po-gs --all -y

conda create -n s3po-gs python=3.11 -y
conda activate s3po-gs
python -m pip install --upgrade pip setuptools wheel
```

### 说明
这一组命令的目的，是把之前已经被污染或状态不明的环境彻底删除，然后用 **Python 3.11** 重建一个干净的 `s3po-gs` 环境。  
后面的 `pip / setuptools / wheel` 升级，是为了让后续 Python 包安装和源码构建更稳定。

---

## 2. 用 PyTorch 官方 CUDA 11.8 wheel 安装 torch

### 命令
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 说明
这一步是关键。  
我们没有继续走原先那条复杂的 conda 求解路线，而是直接使用 **PyTorch 官方提供的 CUDA 11.8 wheel**，这样能更稳定地拿到：

- `torch==2.1.0`
- `torchvision==0.16.0`
- `torchaudio==2.1.0`

并且它们自带的 CUDA 版本与项目官方环境更加接近。

---

## 3. 将 NumPy 固定回 1.x

### 命令
```bash
pip install numpy==1.26.4
```

### 说明
最初环境里出现了 **NumPy 2.x 与部分已编译模块不兼容** 的警告。  
而 S3PO-GS 官方 `environment.yml` 本来也固定了 `numpy=1.24.4` 一类的 1.x 版本，因此把 NumPy 降回 **1.26.4** 是正确且有效的修复。

---

## 4. 验证当前基础环境状态

### 命令
```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 当前已验证结果
- `numpy == 1.26.4`
- `torch == 2.1.0+cu118`
- `torch.version.cuda == 11.8`
- `torch.cuda.is_available() == True`

### 说明
这一步确认了最重要的底座已经成立：

1. Python 环境正常  
2. PyTorch 可以成功导入  
3. PyTorch 看到的 CUDA 是 **11.8**  
4. GPU 可用

也就是说，**PyTorch 底座已经修好**。

---

## 5. 解决 `pkg_resources` 缺失问题（为源码构建做准备）

### 命令
```bash
pip install "setuptools<81"
```

### 说明
在安装 `submodules/simple-knn` 时，构建过程经过 `torch.utils.cpp_extension`，它仍然依赖 `pkg_resources`。  
而较新的 `setuptools` 已经不再保留这一接口，因此需要把 `setuptools` 降到较旧版本，让 `pkg_resources` 可用。

这是后续源码编译链能够继续工作的前提之一。

---

## 6. 在 conda 环境内安装 CUDA 11.8 的 nvcc

### 命令
```bash
conda install -y -c nvidia cuda-nvcc=11.8.89
```

### 说明
最开始系统默认的 `nvcc` 指向的是系统 CUDA 12.4。  
为了让源码编译尽量贴近当前 `torch==2.1.0+cu118`，我们把 **CUDA 11.8 的 nvcc** 装进当前 conda 环境里。

---

## 7. 固定当前 shell 的 CUDA 编译路径

### 命令
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 说明
这一步的作用是让当前 shell 优先使用 **conda 环境里的 CUDA 11.8 工具链**，而不是系统全局的 CUDA 12.4。

这样做之后，`nvcc` 的来源已经被确认切换到了环境内。

---

## 8. 验证当前 `nvcc` 已切换到环境内 11.8

### 命令
```bash
echo $CONDA_PREFIX
which nvcc
nvcc --version
```

### 当前已验证结果
- `CONDA_PREFIX=/home/bzhang512/miniconda3/envs/s3po-gs`
- `which nvcc -> /home/bzhang512/miniconda3/envs/s3po-gs/bin/nvcc`
- `nvcc release 11.8, V11.8.89`

### 说明
这一步说明：  
**编译器路径已经修正成功**，后续源码编译不再使用系统 12.4 的 `nvcc`，而是使用 conda 环境内的 **11.8 nvcc**。

---

## 9. 尝试编译 `simple-knn` 的正确方式

### 命令
```bash
pip install --no-build-isolation submodules/simple-knn
```

### 说明
这里加 `--no-build-isolation` 是必须的。  
因为 `simple-knn` 的构建过程在 `setup.py` 中会直接依赖 `torch`；如果使用 pip 默认的隔离构建环境，那个临时环境里看不到当前 conda 环境中的 `torch`，会直接报错。

所以这里的正确安装方式是 **关闭 build isolation**。

---

## 10. 当前已经定位出的最新问题：缺少 CUDA 头文件

### 命令
```bash
ls $CONDA_PREFIX/include/cuda_runtime.h
ls $CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h
```

### 当前已验证结果
这两个路径都不存在 `cuda_runtime.h`。

### 说明
这说明虽然现在已经有了：

- 环境内的 `nvcc 11.8`
- 正常可用的 `torch 2.1.0+cu118`

但是 **CUDA 11.8 的开发头文件还没有装全**。  
因此 `simple-knn` 在真正开始编译 `.cu` 文件时，会因为找不到：

- `cuda_runtime.h`

而失败。

这也是截至当前已经明确定位出的**最新、单一、核心问题**。

---

# 当前结论

截至目前，已经被确认有效的结论是：

1. `s3po-gs` 环境已经重建成功  
2. `torch==2.1.0+cu118` 安装成功  
3. `numpy==1.26.4` 安装成功  
4. GPU 可被 torch 正常识别  
5. `setuptools` 降级后，源码构建链的前置问题已解决  
6. 环境内 `nvcc` 已切换为 **11.8.89**  
7. 当前唯一剩余的明确问题是：**缺少 CUDA 11.8 头文件（例如 `cuda_runtime.h`）**

---

# 下一步（尚未执行，只是当前计划）

下一步预计要做的是：  
给当前 conda 环境补齐 **CUDA 11.8 toolkit / runtime dev headers**，然后再重新尝试：

```bash
pip install --no-build-isolation submodules/simple-knn
```

> 注意：这一步还没有真正执行完成，因此这里只作为“下一步计划”保留，不算入“已验证有效的步骤”。
