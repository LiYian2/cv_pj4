# Part 1 Setup

This document summarizes the Part 1 environment setup for three pipelines:

- **3DGS / COLMAP**
- **VGGT**
- **Scaffold-GS**

For Part 1, we use separate conda environments because the dependency stacks are different.

---

## 1. 3DGS / COLMAP Setup

### Environment

Create the environment:

```bash
conda create -n 3dgs python=3.10.0 -y
conda activate 3dgs
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

### Gaussian Splatting repository and CUDA extensions

Clone the repository with submodules:

```bash
cd ~/CV_Project/third_party
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
```

Install the required CUDA extensions:

```bash
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

Verify the extension installs:

```bash
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print(diff-gaussian-rasterization OK)"
python -c "import torch; from simple_knn._C import distCUDA2; print(simple-knn OK:, torch.__version__)"
```

### Remaining packages and COLMAP

```bash
pip install plyfile tqdm scipy matplotlib ipywidgets scikit-learn
conda install -c conda-forge colmap -y
```

Check the environment:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import plyfile; print(plyfile OK)"
which colmap
colmap -h
python train.py --help
```

### Small validation: COLMAP to 3DGS

Download and extract the sample dataset:

```bash
mkdir -p ~/CV_Project/part1
cd ~/CV_Project
wget https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip south-building.zip -d part1/colmap_test/
```

Run automatic reconstruction:

```bash
cd ~/CV_Project/part1/colmap_test/south-building/south-building

colmap automatic_reconstructor \
  --workspace_path . \
  --image_path images
```

Organize the sparse model into `sparse/0` and convert to binary:

```bash
mkdir -p sparse/0
mv sparse/cameras.txt sparse/0/
mv sparse/images.txt sparse/0/
mv sparse/points3D.txt sparse/0/

colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse/0 \
  --output_type BIN
```

Inspect the model:

```bash
colmap model_analyzer --path sparse/0
```

Validation result recorded in this test:

- Cameras: `1`
- Images: `128`
- Registered images: `128`
- Points: `61514`
- Mean reprojection error: `0.497032 px`

Undistort the scene:

```bash
mkdir -p distorted/sparse/0
cp sparse/0/* distorted/sparse/0/

colmap image_undistorter \
  --image_path images \
  --input_path distorted/sparse/0 \
  --output_path undistorted \
  --output_type COLMAP
```

Prepare the final 3DGS scene:

```bash
mkdir -p gs_scene/sparse/0
cp -r undistorted/images gs_scene/
cp undistorted/sparse/* gs_scene/sparse/0/
```

Expected final structure:

```text
gs_scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

---

## 2. VGGT Setup

### Environment

Create the environment:

```bash
conda create -n vggt python=3.10 -y
conda activate vggt
```

Install PyTorch:

```bash
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Repository and dependencies

Clone the repository:

```bash
cd ~/CV_Project/third_party
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
```

### Verification

```bash
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy OK)"
```

If the repository provides a demo script or help command, test that as well.

---

## 3. Scaffold-GS Setup

### Goal and issue

The official Scaffold-GS environment assumes an older stack, but the server uses a newer system CUDA toolchain. The main setup problem was a CUDA / PyTorch mismatch during CUDA extension builds.

Observed situation:

- official environment torch: `1.12.1 + cu116`
- system `nvcc`: `12.4`
- no local CUDA `11.6` installation available

So the official `environment.yml` route was not suitable on this machine.

### Initial failed route

The original attempt was:

```bash
cd ~/CV_Project/third_party/Scaffold-GS
conda env create --file environment.yml
```

Useful checks during diagnosis:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
which nvcc
echo $CUDA_HOME
ls /usr/local | grep cuda
```

Key conclusion:

- the problem was **not** command order
- the problem was **PyTorch CUDA version != system compiler CUDA version**
- installing CUDA `12.6` would not help if torch stayed on `cu116`

### Final workable route

Create a new environment:

```bash
cd ~/CV_Project/third_party/Scaffold-GS
conda create -n scaffold_gs_cu124 python=3.10 -y
conda activate scaffold_gs_cu124
pip install --upgrade pip setuptools wheel ninja
```

Install newer PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Set CUDA-related environment variables:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Build-isolation issue and fix

A separate issue appeared when installing packages that need `torch` during build.

Example failure:

```bash
pip install torch-scatter
```

and similarly for CUDA submodules. The error was effectively:

```text
ModuleNotFoundError: No module named torch
```

This was caused by **pip build isolation**, not by torch being absent from the conda environment.

The fix was to use:

```bash
--no-build-isolation
```

### Install Scaffold-GS dependencies

Install `torch-scatter`:

```bash
pip install --no-build-isolation torch-scatter
```

Install the CUDA submodules:

```bash
pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/simple-knn
```

### Minimal validation checks

Check torch / CUDA:

```bash
python -c "import torch; print(torch, torch.__version__); print(cuda in torch, torch.version.cuda); print(cuda available, torch.cuda.is_available()); print(device count, torch.cuda.device_count())"
```

Check extension imports:

```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print(core imports ok)"
```

Check main script startup:

```bash
python train.py -h
```

### Missing Python dependencies encountered later

After the CUDA-related issues were fixed, `python train.py -h` exposed missing pure Python packages.

Install fixes used in this setup:

```bash
pip install colorama
pip install opencv-python-headless
```

Alternative for OpenCV:

```bash
pip install opencv-python
```

For a headless cloud server, `opencv-python-headless` is preferred.

### Main commands used in the final workable route

```bash
cd ~/CV_Project/third_party/Scaffold-GS

conda create -n scaffold_gs_cu124 python=3.10 -y
conda activate scaffold_gs_cu124

pip install --upgrade pip setuptools wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install --no-build-isolation torch-scatter

pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/simple-knn

pip install colorama
pip install opencv-python-headless

python -c "import torch; import diff_gaussian_rasterization; import simple_knn; print(core imports ok)"
python train.py -h
```

### Current practical state

At this stage, the following had been achieved:

- Scaffold-GS repository cloned correctly with submodules
- environment `scaffold_gs_cu124` created
- newer torch installed
- `torch-scatter` installed
- `diff-gaussian-rasterization` installed
- `simple-knn` installed
- missing Python dependencies fixed incrementally

Recommended next step:

- continue checking `python train.py -h` until no dependency error remains
- then test on a very small real scene before running full experiments
