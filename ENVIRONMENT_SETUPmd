# Environment Setup

This document records our environment setup process for Project 4.

For Part 1, we use two separate conda environments:

- `3dgs`: for COLMAP and Gaussian Splatting
- `vggt`: for VGGT

We keep them separate because they may require different PyTorch versions and dependency settings.

---

## 1. Install Miniconda

We use Miniconda to manage the environments.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda --version
```
## 2. Create the 3dgs environment
```bash
conda create -n 3dgs python=3.10.0 -y
conda activate 3dgs
```
Install torch (the version should accord with cuda 12.1)
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```
Clone the Gaussian Splatting Repository from github and install the submodules manuallly
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('OK')"
python -c "import torch; from simple_knn._C import distCUDA2; print(' OK :', torch.__version__)"
```


Install the rest libraries and Colmap
```bash
pip install plyfile tqdm scipy matplotlib ipywidgets scikit-learn
conda install -c conda-forge colmap -y
which colmap
```

## 3. Create the vggt environment
```bash
conda create -n vggt python=3.10 -y
conda activate vggt

pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```
Clone the vggt repository and install the dependencies
```bash

git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
```

4. Verify 
