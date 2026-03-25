# Environment Setup

This document records our environment setup process for Project 4 Part 1.

For Part 1, we use two separate conda environments:

- `3dgs`: for COLMAP and Gaussian Splatting
- `vggt`: for VGGT

We keep them separate because they may require different PyTorch versions and different dependency configurations.

---

## 1. Install Miniconda

We use Miniconda to manage all environments in this project.

```bash
mkdir -p ~/CV_Project
cd ~/CV_Project
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After installation, restart the shell, then verify that Conda is available:

```bash
source ~/.bashrc
conda --version
```

---

## 2. Create the `3dgs` Environment

First, create a clean conda environment for COLMAP and Gaussian Splatting:

```bash
conda create -n 3dgs python=3.10.0 -y
conda activate 3dgs
```

Install PyTorch. In our setup, the version is chosen to match CUDA 12.1:

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Install Gaussian Splatting Dependencies

Clone the official Gaussian Splatting repository with submodules:

```bash
cd ~/CV_Project
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
```

Then install the required submodules manually:

```bash
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
```

Check whether the two CUDA extensions were installed successfully:

```bash
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('diff-gaussian-rasterization OK')"
python -c "import torch; from simple_knn._C import distCUDA2; print('simple-knn OK:', torch.__version__)"
```

---

## 4. Install the Remaining Libraries and COLMAP

Install the remaining Python libraries used in our setup:

```bash
pip install plyfile tqdm scipy matplotlib ipywidgets scikit-learn
```

Then install COLMAP in the same `3dgs` environment:

```bash
conda install -c conda-forge colmap -y
```

Check whether COLMAP is available:

```bash
which colmap
```

---

## 5. Verify the `3dgs` Environment

Run a few basic checks to verify that the `3dgs` environment is working:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import plyfile; print('plyfile OK')"
colmap -h
python train.py --help
```

If these commands run correctly, then the basic `3dgs` environment setup is complete.

---

## 6. Create the `vggt` Environment

Create another clean conda environment for VGGT:

```bash
conda create -n vggt python=3.10 -y
conda activate vggt
```

Install PyTorch. In our setup, we use the CUDA 12.8 build:

```bash
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

---

## 7. Install VGGT Dependencies

Clone the official VGGT repository and install its dependencies:

```bash
cd ~/CV_Project
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
```

---

## 8. Verify the `vggt` Environment

Run basic checks to verify that the `vggt` environment is working:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print('numpy OK')"
```

If the repository provides a demo script or help command, it can also be tested here.

---

## 9. Small-Scale Validation: COLMAP to 3DGS

This section records a small-scale validation pipeline from COLMAP reconstruction to Gaussian Splatting training.

### 9.1 Download and Extract the Sample Dataset

First, download the official COLMAP sample dataset `south-building.zip` and extract it:

```bash
mkdir -p ~/CV_Project/part1
cd ~/CV_Project
wget https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip south-building.zip -d part1/colmap_test/
```

### 9.2 Run COLMAP Automatic Reconstruction

Go to the extracted dataset directory and run COLMAP automatic reconstruction:

```bash
cd ~/CV_Project/part1/colmap_test/south-building/south-building

colmap automatic_reconstructor \
  --workspace_path . \
  --image_path images
```

### 9.3 Organize the Sparse Model into `sparse/0`

After reconstruction, organize the sparse model files into the standard COLMAP structure expected by many downstream pipelines:

```bash
mkdir -p sparse/0
mv sparse/cameras.txt sparse/0/
mv sparse/images.txt sparse/0/
mv sparse/points3D.txt sparse/0/
```

Then convert the sparse model to binary format:

```bash
colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse/0 \
  --output_type BIN
```

Verify the reconstruction result:

```bash
colmap model_analyzer --path sparse/0
```

The reconstruction result in this validation was:

- Cameras: 1
- Images: 128
- Registered images: 128
- Points: 61514
- Mean reprojection error: 0.497032 px

### 9.4 Undistort the COLMAP Scene for 3DGS

The original COLMAP reconstruction used a camera model that is not directly supported by the official Gaussian Splatting training code. Therefore, the scene needed to be undistorted first.

First, back up the original sparse model and run image undistortion:

```bash
mkdir -p distorted/sparse/0
cp sparse/0/* distorted/sparse/0/

colmap image_undistorter \
  --image_path images \
  --input_path distorted/sparse/0 \
  --output_path undistorted \
  --output_type COLMAP
```

### 9.5 Prepare the Final Scene Directory for 3DGS

Create a new scene folder for Gaussian Splatting input:

```bash
mkdir -p gs_scene/sparse/0
cp -r undistorted/images gs_scene/
cp undistorted/sparse/* gs_scene/sparse/0/
```

At this point, the final scene structure becomes:

```text
gs_scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

This is the format expected by the official Gaussian Splatting training pipeline.

### 9.6 Run 3D Gaussian Splatting Training

Go to the `gaussian-splatting` repository and run a short validation training with evaluation split enabled and only 3000 iterations:

```bash
cd ~/CV_Project/gaussian-splatting
python train.py \
  -s ~/CV_Project/part1/colmap_test/south-building/south-building/gs_scene \
  -m ~/CV_Project/part1/colmap_test/south-building/south-building/gs_scene/output_3dgs \
  --eval \
  --iterations 3000
```

Notes:

- `-s` specifies the input scene directory.
- `-m` specifies the output model directory.
- `--eval` enables the train/test split mode.
- `--iterations 3000` runs only a short validation training instead of full optimization.

Example final log:

```text
Training progress: 100%|███████████████████████████| 3000/3000 [02:42<00:00, 18.51it/s, Loss=0.1659867, Depth Loss=0.0000000]
[ITER 3000] Saving Gaussians
```

This confirms that the small-scale validation pipeline from COLMAP reconstruction to 3D Gaussian Splatting training ran successfully.