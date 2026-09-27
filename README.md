# MLF 2026

3D reconstruction pipeline for volume estimation from video/images.

## Pipelines

### MASt3R Pipeline (Recommended)
Dense 3D reconstruction using MASt3R with TSDF-based watertight meshing.
Produces guaranteed watertight meshes for reliable volume estimates.

→ See [`mast3r_pipeline/README.md`](mast3r_pipeline/README.md) for setup and usage.

### COLMAP Pipeline (Legacy)
Original COLMAP-based SfM + dense stereo pipeline.
Produces Poisson/Delaunay meshes (may not be watertight).

→ See [`automatic_reconstruction.py`](automatic_reconstruction.py) and [`Example/`](Example/).

## Project Structure

```
mlf-2026/
├── mast3r_pipeline/             # NEW: MASt3R-based reconstruction
│   ├── reconstruct_mast3r.py    # Main pipeline
│   ├── inspect_mesh.py          # Mesh analysis
│   ├── setup_mast3r.ps1         # Setup script
│   └── README.md                # Detailed docs
├── Example/                     # Example data + legacy COLMAP results
│   ├── images/                  # Extracted video frames (168 frames)
│   ├── masked_images/           # Background-removed frames
│   └── dense_model/             # COLMAP dense reconstruction outputs
├── automatic_reconstruction.py  # Legacy COLMAP pipeline
├── watertightness.py            # Legacy mesh watertightness checker
└── Notes/                       # Project notes and diagrams
```
