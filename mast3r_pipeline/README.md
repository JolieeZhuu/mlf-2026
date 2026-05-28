# MASt3R 3D Reconstruction Pipeline

Dense 3D reconstruction using [MASt3R](https://github.com/naver/mast3r) (Matching And Stereo 3D Reconstruction) with TSDF-based watertight meshing.

## Why MASt3R?

The previous COLMAP-based pipeline often produced meshes that were **not watertight**, making volume estimates unreliable. MASt3R + TSDF solves this:

| | COLMAP Pipeline | MASt3R Pipeline |
|---|---|---|
| **Camera Poses** | Feature extraction → matching → incremental SfM | MASt3R predicts directly |
| **Dense Geometry** | Patch-match stereo → fusion | MASt3R dense pointmaps |
| **Meshing** | Poisson / Delaunay (often not watertight) | TSDF + Marching Cubes (**always watertight**) |
| **Scale** | Arbitrary (up-to-scale) | **Metric scale** |
| **Dependencies** | COLMAP CLI + pycolmap | MASt3R + Open3D |

## Quick Start

### 1. Setup (run once)

```powershell
cd mast3r_pipeline
.\setup_mast3r.ps1
```

This will:
- Clone `naver/mast3r` into `mast3r/`
- Create a `mast3r` conda environment
- Install PyTorch with CUDA, MASt3R, Open3D, and other dependencies
- Download the MASt3R model checkpoint (~700 MB)

### 2. Run Reconstruction

```powershell
conda activate mast3r

# Use the existing Example images
python reconstruct_mast3r.py --image_dir ..\Example\images

# Or use masked images (background removed)
python reconstruct_mast3r.py --image_dir ..\Example\masked_images

# Limit to fewer images (faster, uses less GPU memory)
python reconstruct_mast3r.py --image_dir ..\Example\images --max_images 15

# Adjust voxel size for finer/coarser mesh
python reconstruct_mast3r.py --image_dir ..\Example\images --voxel_length 0.003

# Only export point cloud (skip meshing)
python reconstruct_mast3r.py --image_dir ..\Example\images --skip_tsdf
```

### 3. Inspect Results

```powershell
# Inspect all output meshes
python inspect_mesh.py

# Inspect a specific mesh
python inspect_mesh.py --mesh output\mesh_tsdf.ply
```

## Output Files

After running `reconstruct_mast3r.py`, the `output/` directory will contain:

| File | Description |
|------|-------------|
| `point_cloud.ply` | Raw colored point cloud from MASt3R |
| `mesh_tsdf.ply` | Watertight mesh from TSDF + Marching Cubes |
| `mesh_tsdf.obj` | Same mesh in OBJ format |
| `mesh_poisson.ply` | Poisson reconstruction (for comparison) |
| `mesh_poisson.obj` | Same mesh in OBJ format |

## Pipeline Architecture

```
Input Images
     │
     ▼
┌─────────────────────┐
│  MASt3R Inference    │  Pairwise dense pointmaps
│  (ViT-Large model)  │  + confidence maps
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Global Alignment    │  All views → common coordinate frame
│  (gradient descent)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  TSDF Integration    │  Fuse depth maps into voxel grid
│  (Open3D)            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Marching Cubes      │  Extract watertight mesh
│  → Volume Estimate   │
└─────────────────────┘
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_images` | 30 | Max images to use (subsamples uniformly if more) |
| `--image_size` | 512 | MASt3R input resolution |
| `--voxel_length` | 0.005 | TSDF voxel size (smaller = finer mesh, more memory) |
| `--confidence_threshold` | 1.5 | Min MASt3R confidence to include a point |
| `--niter` | 300 | Global alignment iterations |
| `--scene_graph` | swin-5 | Pair selection strategy |
| `--device` | cuda | Use `cpu` if no GPU (very slow) |

## Requirements

- **GPU:** NVIDIA GPU with ≥8 GB VRAM (RTX 3070 or better recommended)
- **CUDA:** 12.1 (or match your PyTorch install)
- **Python:** 3.11
- **OS:** Windows (with WSL2) or Linux

## File Structure

```
mast3r_pipeline/
├── environment.yml          # Conda environment definition
├── setup_mast3r.ps1         # One-click setup script
├── reconstruct_mast3r.py    # Main reconstruction pipeline
├── inspect_mesh.py          # Mesh inspection utility
├── README.md                # This file
├── mast3r/                  # (created by setup) MASt3R repo clone
│   ├── checkpoints/         # Model weights
│   └── dust3r/              # DUSt3R submodule
└── output/                  # (created at runtime) Results
    ├── point_cloud.ply
    ├── mesh_tsdf.ply
    ├── mesh_tsdf.obj
    ├── mesh_poisson.ply
    └── mesh_poisson.obj
```

## License

MASt3R is licensed under **CC BY-NC-SA 4.0** (non-commercial use only).
See the [MASt3R repository](https://github.com/naver/mast3r) for details.
