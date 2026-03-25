# Part 2 Setup Log (RegGS)

This document records the setup process, decisions, fixes, and current blocker for Project 4 Part 2 using **RegGS**.

---

## 1. Part 2 Workspace Layout

We created a dedicated Part 2 workspace under:

```text
/home/bzhang512/CV_Project/part2
```

Current intended layout:

```text
part2/
├── README.md
├── SETUP.md
├── data/
│   └── re10k_1_sparse/
├── outputs/
│   ├── gs/
│   └── reggs_init/
└── scripts/
```

Originally, the RegGS repository was extracted under:

```text
/home/bzhang512/CV_Project/part2/reggs
```

However, this is **not ideal** for long-term project organization, because the user wants `part2/` to contain only personal project files that may later be uploaded to GitHub. Third-party repositories should instead live directly under `CV_Project/`.

Recommended later cleanup target:

```text
/home/bzhang512/CV_Project/RegGS
```

---

## 2. Sparse Re10k-1 Input Preparation

We prepared a sparse/unposed input subset for Part 2 from:

```text
/home/bzhang512/CV_Project/part1/part1_data/re10k_1/images
```

A sparse subset was created at:

```text
/home/bzhang512/CV_Project/part2/data/re10k_1_sparse
```

The selection rule was:

- take every 30th frame from the sorted image list
- append the last frame if it is not already included

Summary:

- original image count: `279`
- selected image count: `11`

Selected filenames:

```text
00000.png
00030.png
00060.png
00090.png
00120.png
00150.png
00180.png
00210.png
00240.png
00270.png
00278.png
```

Files created for this dataset slice:

```text
part2/data/re10k_1_sparse/
├── images/
├── image_list.txt
├── meta.json
└── README.md
```

A helper script was also created:

```text
/home/bzhang512/CV_Project/part2/scripts/prepare_re10k1_sparse.py
```

---

## 3. RegGS Repository Retrieval

We identified the repository as:

```text
https://github.com/3DAgentWorld/RegGS
```

The repository was cloned **locally first** with submodules, because direct cloud-side cloning was expected to be slow:

```bash
git clone --recursive https://github.com/3DAgentWorld/RegGS.git
```

Submodules confirmed during clone:

- `thirdparty/diff-gaussian-rasterization-w-pose`
- `thirdparty/gaussian_rasterizer`
- `thirdparty/simple-knn`
- nested glm submodule under diff-gaussian-rasterization-w-pose

Then the repo was compressed locally and uploaded to the cloud.

Archive created locally:

```text
RegGS.tar.gz
```

Archive size:

- about `166 MB`

Then it was uploaded and extracted on cloud under:

```text
/home/bzhang512/CV_Project/part2/reggs
```

---

## 4. What RegGS Expects as Input

After reading the repository, README, config, and dataset loader, we confirmed that RegGS does **not** consume only an image folder.

For `dataset_name: re10k`, it expects this structure:

```text
<input_path>/<scene_name>/
├── images/
├── intrinsics.json
└── cameras.json
```

This was verified from:

- `config/re10k.yaml`
- `src/entities/datasets.py`
- `sample_data/000c3ab189999a83`

This means our current sparse input under `part2/data/re10k_1_sparse/` is still **incomplete** for RegGS, because it currently has:

- `images/`
- `image_list.txt`
- `meta.json`

but does **not yet** have:

- `intrinsics.json`
- `cameras.json`

So before running actual Part 2 experiments, we still need a conversion/preparation step to build a proper RegGS-compatible Re10K scene directory.

---

## 5. Environment Setup Attempt

RegGS README states that the tested environment is:

- Ubuntu 22.04
- Python 3.10
- CUDA 11.8
- PyTorch 2.5.1

We created a dedicated conda environment:

```bash
conda env create -f environment.yaml
```

This successfully created:

```text
/home/bzhang512/miniconda3/envs/reggs
```

Initial verification:

- Python available
- environment exists

However, the first conda-solved PyTorch installation turned out to be **CPU-only**, even though `pytorch-cuda` packages were present.

Observed state in the first attempt:

- `pytorch 2.5.1 cpu_openblas`
- `torchvision 0.20.1 cpu`
- `torch.cuda.is_available() == False`
- `torch.version.cuda == None`
- `nvcc` not in PATH

This caused CUDA extensions to fail.

---

## 6. Python Requirements Installation

After the environment was created, we installed:

```bash
pip install -r requirements.txt
```

This completed successfully.

Main installed packages included:

- `open3d`
- `lightning`
- `lpips`
- `opencv-python==4.11.0.86`
- `wandb`
- `scikit-image`
- `scikit-video`
- `e3nn`
- `omegaconf`
- `trimesh`
- `evo`

So the pure Python dependency layer is working.

---

## 7. Fixing PyTorch to a GPU Build

Because the conda-resolved installation gave a CPU build, we switched to the same style already used successfully in Part 1 environments: install PyTorch explicitly via pip from a CUDA wheel index.

### First GPU attempt

We replaced the CPU torch with:

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

This succeeded, and verification showed:

- `torch 2.7.0+cu128`
- `torch.version.cuda = 12.8`
- `torch.cuda.is_available() = True`
- GPU count = 2

So this solved the GPU availability issue.

However, this later introduced compatibility trouble for `curope`.

### Second GPU attempt (closer to author environment)

To get closer to the RegGS README, we later downgraded to a GPU build of PyTorch 2.5.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

This also succeeded.

Verification showed:

- `torch 2.5.1+cu124`
- `torch.version.cuda = 12.4`
- `torch.cuda.is_available() = True`
- GPU count = 2

This is the **current** PyTorch state of the `reggs` environment.

---

## 8. Third-Party CUDA Extensions

RegGS requires several third-party extensions.

### First issue: build isolation could not see torch

When installing:

```bash
pip install thirdparty/diff-gaussian-rasterization-w-pose
```

it initially failed with:

```text
ModuleNotFoundError: No module named torch
```

This came from pip build isolation. The README itself suggests using `--no-build-isolation` if torch import fails during extension build.

So we switched to:

```bash
pip install --no-build-isolation thirdparty/diff-gaussian-rasterization-w-pose
```

### Second issue: CUDA_HOME not set

After fixing build isolation, the next error became:

```text
OSError: CUDA_HOME environment variable is not set.
```

To fix this, we explicitly used:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Current result for third-party modules

After fixing torch and CUDA environment issues, the following all installed successfully:

- `diff-gaussian-rasterization-w-pose`
- `gaussian_rasterizer`
- `simple-knn`

So all three RegGS third-party rasterization / KNN extensions are now installed.

---

## 9. CroCo `curope` Kernel Compilation

This is the **only remaining setup blocker** at the moment.

The kernel lives under:

```text
/home/bzhang512/CV_Project/part2/reggs/src/noposplat/model/encoder/backbone/croco/curope
```

### 9.1 Initial failure: unsupported GPU architectures

At first, the build system attempted to compile for many architectures automatically, including:

- `sm_75`
- `sm_80`
- `sm_86`
- `sm_90`
- `sm_100`
- `sm_120`

Because `nvcc 12.4` does not support `compute_100`, it failed with:

```text
nvcc fatal: Unsupported gpu architecture compute_100
```

To address this, we modified `setup.py` so that the extension only targets the actual GPU in this machine (RTX A6000), i.e. `sm_86`.

Current patched line in `setup.py`:

```python
all_cuda_archs = [-gencode, arch=compute_86,code=sm_86]
```

This successfully removed the unsupported-architecture error.

### 9.2 Next failure with torch 2.7.0

After the architecture issue was fixed, compilation with torch 2.7.0 produced a different error in `kernels.cu`, indicating API incompatibility with newer PyTorch C++ internals.

This motivated the downgrade back toward a 2.5.1 GPU build.

### 9.3 Current failure with torch 2.5.1+cu124

With torch downgraded to `2.5.1+cu124`, the kernel now **compiles successfully** and generates:

```text
curope.cpython-310-x86_64-linux-gnu.so
```

However, import still fails:

```text
ImportError: undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKSs
```

This means the problem is no longer in the CUDA architecture flags or torch availability. It is now a **binary compatibility / ABI / symbol resolution** issue at import time.

Important observation:

- the extension builds
- the `.so` file is produced
- but Python cannot load it because the compiled binary expects a torch C++ symbol that is not resolved at runtime

We also tested rebuilding after cleaning old build artifacts:

```bash
rm -rf build *.so
python setup.py build_ext --inplace
```

This did **not** fix the problem.

So the remaining issue is likely one of:

- ABI mismatch in the extension build
- a source-level incompatibility in `curope` with the currently used torch binary
- a build configuration mismatch in how the extension links against PyTorch

At the moment, this is the **last unresolved setup issue**.

---

## 10. Current Overall Status

### Confirmed working

- cloud SSH workflow
- Part 2 workspace scaffold
- sparse Re10k-1 subset generation
- RegGS repository clone/upload/extract
- dedicated `reggs` conda environment
- GPU-enabled PyTorch in `reggs`
- `requirements.txt`
- `diff-gaussian-rasterization-w-pose`
- `gaussian_rasterizer`
- `simple-knn`

### Confirmed incomplete

- RegGS-compatible sparse scene formatting for our own `Re10k-1` subset (`intrinsics.json` and `cameras.json` still need to be prepared)
- `curope` importability
- actual RegGS inference / refinement / evaluation on our own data

### Single active blocker

The only remaining environment blocker is:

```text
curope import fails with undefined symbol torchCheckFail
```

---

## 11. Recommended Next Step

The next step should **not** be redoing the whole environment. Most of the environment is now correct.

The next focused task should be:

1. inspect `curope.cpp`, `kernels.cu`, and `setup.py`
2. identify why the built extension still imports against an unresolved torch symbol
3. test whether a small source/build patch can restore compatibility
4. once `curope` is fixed, proceed to:
   - build a proper RegGS-compatible sparse Re10k scene
   - adjust `config/re10k.yaml` or create a new custom config for our own sparse input
   - run `run_infer.py`, `run_refine.py`, and `run_metric.py`

