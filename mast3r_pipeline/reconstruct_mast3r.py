"""
MASt3R-based 3D Reconstruction Pipeline
========================================

Replaces the COLMAP-based pipeline with MASt3R for dense 3D reconstruction.
Uses Open3D TSDF Volume Integration to produce watertight meshes.

Pipeline:
    1. Load images from a directory
    2. Run MASt3R pairwise inference → dense pointmaps + confidence
    3. Global alignment → all views in one coordinate frame
    4. TSDF integration → fuse pointmaps into a volumetric grid
    5. Marching Cubes → extract watertight mesh
    6. Compute volume and export

Usage:
    conda activate mast3r
    python reconstruct_mast3r.py --image_dir ../Example/images
    python reconstruct_mast3r.py --image_dir ../Example/masked_images --max_images 30

Prerequisites:
    Run setup_mast3r.ps1 first to install MASt3R and download checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path

# Enable CPU fallback for any unsupported MPS operators on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

# ── Resolve MASt3R imports ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MAST3R_DIR = SCRIPT_DIR / "mast3r"

if not MAST3R_DIR.exists():
    print(f"ERROR: MASt3R not found at {MAST3R_DIR}")
    print("Run setup_mast3r.sh (or setup_mast3r.ps1) first to clone and install MASt3R.")
    sys.exit(1)

# Add MASt3R and DUSt3R to Python path
sys.path.insert(0, str(MAST3R_DIR))
sys.path.insert(0, str(MAST3R_DIR / "dust3r"))


# ── Configuration ───────────────────────────────────────────────────

DEFAULT_CHECKPOINT = (
    MAST3R_DIR
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)

DEFAULT_IMAGE_SIZE = 512      # MASt3R input resolution
DEFAULT_NITER = 300           # Global alignment iterations
DEFAULT_LR = 0.01             # Global alignment learning rate
DEFAULT_VOXEL_LENGTH = 0.005  # 5 mm voxels for TSDF
DEFAULT_SDF_TRUNC_MULT = 5    # sdf_trunc = voxel_length * this
DEFAULT_CONFIDENCE_THR = 1.5  # Min confidence to include a point
DEFAULT_MAX_IMAGES = 30       # Cap on number of images (GPU memory)
DEFAULT_SCENE_GRAPH = "swin-5" # Scene graph type for pair selection


def get_default_device() -> str:
    """Auto-detect available acceleration device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    default_device = get_default_device()
    parser = argparse.ArgumentParser(
        description="MASt3R 3D Reconstruction → TSDF → Watertight Mesh → Volume"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing input images (PNG/JPG).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ./output/ next to this script.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to MASt3R model checkpoint.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size for MASt3R (default: 512).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="Maximum number of images to use (default: 30).",
    )
    parser.add_argument(
        "--scene_graph",
        type=str,
        default=DEFAULT_SCENE_GRAPH,
        help="Scene graph type for pair selection (default: swin-5).",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=DEFAULT_NITER,
        help="Number of global alignment iterations (default: 300).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for global alignment (default: 0.01).",
    )
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=DEFAULT_VOXEL_LENGTH,
        help="TSDF voxel size in scene units (default: 0.005).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THR,
        help="Minimum confidence to include a 3D point (default: 1.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["cuda", "mps", "cpu", "auto"],
        help=f"Device to use: 'cuda', 'mps', 'cpu', or 'auto' (default: {default_device}).",
    )
    parser.add_argument(
        "--skip_tsdf",
        action="store_true",
        help="Skip TSDF meshing; only export the raw point cloud.",
    )
    args = parser.parse_args()
    if args.device == "auto":
        args.device = default_device
    return args


# ── Lazy imports (only after sys.path is set) ───────────────────────

def import_mast3r():
    """Import MASt3R modules. Called after sys.path setup."""
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    return {
        "torch": torch,
        "AsymmetricMASt3R": AsymmetricMASt3R,
        "inference": inference,
        "load_images": load_images,
        "make_pairs": make_pairs,
        "global_aligner": global_aligner,
        "GlobalAlignerMode": GlobalAlignerMode,
    }


def import_open3d():
    """Import Open3D for TSDF integration and meshing."""
    import open3d as o3d
    return o3d


# ── Step 1: Collect image paths ─────────────────────────────────────

def collect_image_paths(image_dir: Path, max_images: int) -> list[str]:
    """Collect and sort image paths, with subsampling if needed."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    all_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if len(all_paths) == 0:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(all_paths)} images in {image_dir}")

    # Subsample uniformly if too many images
    if len(all_paths) > max_images:
        indices = np.linspace(0, len(all_paths) - 1, max_images, dtype=int)
        all_paths = [all_paths[i] for i in indices]
        print(f"Subsampled to {len(all_paths)} images (max_images={max_images})")

    return [str(p) for p in all_paths]


# ── Step 2: MASt3R Inference ────────────────────────────────────────

def run_mast3r_inference(image_paths, args, modules):
    """
    Run MASt3R pairwise inference and global alignment.

    Returns:
        scene: The globally aligned scene object with:
            - scene.get_pts3d()       → list of (H,W,3) pointmaps
            - scene.get_confidence()  → list of (H,W) confidence maps
            - scene.get_im_poses()    → (N,4,4) camera-to-world matrices
            - scene.get_focals()      → (N,) focal lengths
            - scene.imgs              → list of (H,W,3) RGB images
    """
    torch = modules["torch"]
    AsymmetricMASt3R = modules["AsymmetricMASt3R"]
    inference_fn = modules["inference"]
    load_images = modules["load_images"]
    make_pairs = modules["make_pairs"]
    global_aligner = modules["global_aligner"]
    GlobalAlignerMode = modules["GlobalAlignerMode"]

    # Load model
    print(f"\nLoading MASt3R model from {args.checkpoint}...")
    model = AsymmetricMASt3R.from_pretrained(str(args.checkpoint))
    model = model.to(args.device).eval()
    print("Model loaded.")

    # Load images
    print(f"Loading {len(image_paths)} images at size={args.image_size}...")
    images = load_images(image_paths, size=args.image_size)
    print(f"Images loaded. Shape: {images[0]['img'].shape}")

    # Make pairs
    print(f"Building scene graph (type={args.scene_graph})...")
    pairs = make_pairs(
        images,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=True,
    )
    print(f"Created {len(pairs)} image pairs.")

    # Run pairwise inference
    print("Running MASt3R pairwise inference...")
    output = inference_fn(pairs, model, args.device, batch_size=1)
    print("Pairwise inference complete.")

    # Global alignment
    print(f"Running global alignment (niter={args.niter}, lr={args.lr})...")
    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(images) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=args.device, mode=mode)
    loss = scene.compute_global_alignment(
        init="mst",
        niter=args.niter,
        schedule="cosine",
        lr=args.lr,
    )
    print(f"Global alignment complete. Final loss: {loss:.4f}")

    return scene


# ── Step 3: Extract aligned data ────────────────────────────────────

def extract_scene_data(scene, confidence_threshold: float):
    """
    Extract aligned pointmaps, colors, confidence, poses, and focals.

    Returns dict with:
        pts3d_list:     list of (H,W,3) numpy arrays (3D points)
        colors_list:    list of (H,W,3) numpy arrays (RGB, 0-1)
        confidence_list: list of (H,W) numpy arrays
        poses:          (N,4,4) numpy array (cam-to-world)
        focals:         (N,) numpy array
        masks:          list of (H,W) boolean arrays (confidence > thr)
    """
    import torch

    with torch.no_grad():
        pts3d_list = [p.cpu().numpy() for p in scene.get_pts3d()]

        # Extract confidence maps: PointCloudOptimizer uses get_conf()
        if hasattr(scene, "get_conf"):
            confidence_list = [c.cpu().numpy() for c in scene.get_conf()]
        elif hasattr(scene, "im_conf"):
            confidence_list = [c.cpu().numpy() for c in scene.im_conf]
        elif hasattr(scene, "get_confidence"):
            confidence_list = [c.cpu().numpy() for c in scene.get_confidence()]
        else:
            confidence_list = [np.ones(p.shape[:2], dtype=np.float32) for p in pts3d_list]

        poses_raw = scene.get_im_poses()
        if isinstance(poses_raw, (list, tuple)):
            poses = np.stack([p.cpu().numpy() for p in poses_raw])
        else:
            poses = poses_raw.cpu().numpy()     # (N, 4, 4)

        focals_raw = scene.get_focals()
        if isinstance(focals_raw, (list, tuple)):
            focals = np.array([float(f.cpu().numpy().squeeze()) for f in focals_raw])
        else:
            focals = focals_raw.cpu().numpy().flatten()   # (N,)

        # Extract optimized depthmaps if available
        if hasattr(scene, "get_depthmaps"):
            depthmaps = [d.cpu().numpy() for d in scene.get_depthmaps()]
        else:
            depthmaps = None

        # Extract principal points if available
        if hasattr(scene, "get_principal_points"):
            pp_raw = scene.get_principal_points()
            if isinstance(pp_raw, (list, tuple)):
                pps = [p.cpu().numpy().squeeze() for p in pp_raw]
            else:
                pps = pp_raw.cpu().numpy()
        else:
            pps = None

    # Images are already numpy (H,W,3) in [0,1]
    colors_list = list(scene.imgs)

    # Build confidence masks
    masks = []
    for conf, pts in zip(confidence_list, pts3d_list):
        mask = (conf > confidence_threshold) & np.all(np.isfinite(pts), axis=-1)
        masks.append(mask)

    total_pts = sum(m.sum() for m in masks)
    print(f"Extracted {len(pts3d_list)} views, {total_pts:,} points above confidence threshold.")

    return {
        "pts3d_list": pts3d_list,
        "colors_list": colors_list,
        "confidence_list": confidence_list,
        "poses": poses,
        "focals": focals,
        "masks": masks,
        "depthmaps": depthmaps,
        "pps": pps,
    }


# ── Step 4: Export raw point cloud ──────────────────────────────────

def export_point_cloud(scene_data, output_dir: Path):
    """Merge all views into a single colored point cloud and save as PLY."""
    o3d = import_open3d()

    all_pts = []
    all_colors = []

    for pts, colors, mask in zip(
        scene_data["pts3d_list"],
        scene_data["colors_list"],
        scene_data["masks"],
    ):
        H, W, _ = pts.shape
        pts_flat = pts.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        mask_flat = mask.reshape(-1)

        # Filter by confidence mask and finite values
        valid = mask_flat & np.all(np.isfinite(pts_flat), axis=1)
        all_pts.append(pts_flat[valid])
        all_colors.append(colors_flat[valid])

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Clip colors to [0, 1]
    all_colors = np.clip(all_colors, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Remove statistical outliers
    print("Removing statistical outliers...")
    pcd, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"Point cloud: {len(pcd.points):,} points after outlier removal.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pcd_path = output_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"Point cloud saved to: {pcd_path}")

    return pcd


# ── Step 5: TSDF Integration → Watertight Mesh ─────────────────────

def tsdf_integrate(scene_data, voxel_length: float, sdf_trunc_mult: float = 5):
    """
    Integrate per-view depth maps into a TSDF volume and extract mesh.

    The TSDF approach produces watertight meshes by construction.

    Auto-scales voxel_length and depth_trunc based on the actual scene
    geometry so that depths are not inadvertently discarded.
    """
    o3d = import_open3d()

    # ── First pass: compute global depth statistics to set parameters ──
    n_views = len(scene_data["pts3d_list"])
    all_valid_depths = []

    view_depths = []  # Store computed depths + metadata per view for reuse
    for i in range(n_views):
        pts3d = scene_data["pts3d_list"][i]    # (H, W, 3)
        mask = scene_data["masks"][i]          # (H, W)
        pose = np.squeeze(scene_data["poses"][i])  # (4, 4) cam2world

        H, W, _ = pts3d.shape
        extrinsic = np.linalg.inv(pose)

        pts_flat = pts3d.reshape(-1, 3)
        ones = np.ones((pts_flat.shape[0], 1))
        pts_homo = np.hstack([pts_flat, ones])

        pts_cam = (extrinsic @ pts_homo.T).T[:, :3]
        depth = pts_cam[:, 2].reshape(H, W)

        # Zero out invalid
        depth[~mask] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0

        valid = depth[depth > 0]
        if len(valid) > 0:
            all_valid_depths.append(valid)

        view_depths.append({
            "depth": depth,
            "extrinsic": extrinsic,
            "H": H, "W": W,
        })

    if len(all_valid_depths) == 0:
        print("WARNING: No valid depths found across any view — TSDF will be empty.")
        mesh = o3d.geometry.TriangleMesh()
        return mesh

    all_depths_concat = np.concatenate(all_valid_depths)
    depth_min = float(np.min(all_depths_concat))
    depth_max = float(np.max(all_depths_concat))
    depth_median = float(np.median(all_depths_concat))
    depth_p99 = float(np.percentile(all_depths_concat, 99))

    print(f"\n  Depth statistics across {n_views} views:")
    print(f"    min={depth_min:.4f}  median={depth_median:.4f}  "
          f"max={depth_max:.4f}  p99={depth_p99:.4f}")

    # ── Auto-scale voxel_length to scene extent ──
    # Use the 99th-percentile depth as the "scene diameter" reference.
    # Target ~500-800 voxels along the longest axis for a reasonable mesh.
    auto_voxel = depth_p99 / 600.0
    # Use the larger of user-specified and auto-computed voxel length
    effective_voxel = max(voxel_length, auto_voxel)
    sdf_trunc = effective_voxel * sdf_trunc_mult

    # depth_trunc should encompass virtually all valid depths
    depth_trunc = depth_p99 * 1.5

    print(f"\nTSDF Integration:")
    print(f"  Requested voxel_length = {voxel_length}")
    print(f"  Auto-computed voxel    = {auto_voxel:.6f}")
    print(f"  Effective voxel_length = {effective_voxel:.6f}")
    print(f"  sdf_trunc              = {sdf_trunc:.6f}")
    print(f"  depth_trunc            = {depth_trunc:.4f}")

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=effective_voxel,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # ── Second pass: integrate each view ──
    for i in range(n_views):
        colors = scene_data["colors_list"][i]  # (H, W, 3)
        focal = float(np.squeeze(scene_data["focals"][i]))

        vd = view_depths[i]
        H, W = vd["H"], vd["W"]
        depth = vd["depth"]
        extrinsic = vd["extrinsic"]

        cx, cy = W / 2.0, H / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, focal, focal, cx, cy)

        color_img = np.ascontiguousarray((np.clip(colors, 0, 1) * 255).astype(np.uint8))
        depth_img = np.ascontiguousarray(depth.astype(np.float32))
        extrinsic_c = np.ascontiguousarray(extrinsic.astype(np.float64))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        tsdf_volume.integrate(rgbd, intrinsic, extrinsic_c)

        valid_px = int(np.sum(depth > 0))
        print(f"  Integrated view {i + 1}/{n_views}  "
              f"(valid pixels: {valid_px:,})")

    # Extract mesh
    print("Extracting mesh from TSDF volume (Marching Cubes)...")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    print(f"TSDF mesh: {n_verts:,} vertices, {n_faces:,} faces")

    if n_verts == 0:
        print("WARNING: TSDF produced an empty mesh. This can happen when:")
        print("  - The depth maps are noisy or sparse")
        print("  - The voxel resolution is too fine for the scene scale")
        print("  Falling back to Poisson reconstruction.")

    return mesh


# ── Step 6: Alternative mesh via Poisson reconstruction ─────────────

def poisson_reconstruct(pcd, depth: int = 10):
    """
    Fallback: Poisson surface reconstruction from the point cloud.
    Also tends to produce watertight meshes.
    """
    o3d = import_open3d()

    print(f"\nPoisson reconstruction (depth={depth})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density vertices (removes extrapolation artifacts)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    print(f"Poisson mesh: {n_verts:,} vertices, {n_faces:,} faces")

    return mesh


# ── Step 7: Compute volume and save ─────────────────────────────────

def analyze_and_save(mesh, pcd, output_dir: Path, method_name: str):
    """Compute mesh properties and save to disk."""
    o3d = import_open3d()
    import trimesh

    # Guard: skip saving / analysis if mesh is empty
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    if n_verts == 0 or n_faces == 0:
        print(f"\nWARNING: {method_name} mesh is empty "
              f"({n_verts} vertices, {n_faces} faces). Skipping save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / f"mesh_{method_name}.ply"
    obj_path = output_dir / f"mesh_{method_name}.obj"

    # Save via Open3D
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"\nMesh saved to: {mesh_path}")
    print(f"Mesh saved to: {obj_path}")

    # Also verify with trimesh
    tmesh = trimesh.load(str(mesh_path))
    if isinstance(tmesh, trimesh.Scene):
        if tmesh.geometry:
            tmesh = trimesh.util.concatenate(tuple(tmesh.geometry.values()))
        else:
            print("WARNING: trimesh loaded an empty scene.")
            return

    print(f"\n{'=' * 50}")
    print(f"  Mesh Analysis ({method_name})")
    print(f"{'=' * 50}")
    print(f"  Vertices:    {len(tmesh.vertices):,}")
    print(f"  Faces:       {len(tmesh.faces):,}")
    print(f"  Watertight:  {tmesh.is_watertight}")

    if tmesh.is_watertight:
        volume = tmesh.volume
        print(f"  Volume:      {volume:.6f} cubic units")
    else:
        print(f"  Volume:      N/A (mesh is not watertight)")
        # Try Open3D volume anyway
        try:
            o3d_volume = mesh.get_volume()
            print(f"  Volume (Open3D, may be approximate): {o3d_volume:.6f}")
        except Exception:
            pass

    bounds = tmesh.bounds
    extents = tmesh.extents
    print(f"  Bounds min:  {bounds[0].tolist()}")
    print(f"  Bounds max:  {bounds[1].tolist()}")
    print(f"  Extents:     {extents.tolist()}")
    print(f"{'=' * 50}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "output"

    print("=" * 60)
    print("  MASt3R 3D Reconstruction Pipeline")
    print("=" * 60)
    print(f"  Image directory:  {args.image_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device:           {args.device}")
    print(f"  Max images:       {args.max_images}")
    print(f"  Image size:       {args.image_size}")
    print(f"  Voxel length:     {args.voxel_length}")
    print(f"  Confidence thr:   {args.confidence_threshold}")
    print("=" * 60)

    # Import MASt3R modules
    print("\nImporting MASt3R modules...")
    modules = import_mast3r()
    print("Modules loaded.")

    # Step 1: Collect images
    image_paths = collect_image_paths(args.image_dir, args.max_images)

    # Step 2: Run MASt3R inference + global alignment
    scene = run_mast3r_inference(image_paths, args, modules)

    # Step 3: Extract aligned data
    scene_data = extract_scene_data(scene, args.confidence_threshold)

    # Step 4: Export raw point cloud
    pcd = export_point_cloud(scene_data, args.output_dir)

    if args.skip_tsdf:
        print("\n--skip_tsdf specified. Skipping TSDF meshing.")
        print("Done! Point cloud saved to output/point_cloud.ply")
        return

    # Step 5: TSDF integration → watertight mesh
    tsdf_mesh = tsdf_integrate(
        scene_data,
        voxel_length=args.voxel_length,
        sdf_trunc_mult=DEFAULT_SDF_TRUNC_MULT,
    )

    tsdf_ok = len(tsdf_mesh.vertices) > 0
    if tsdf_ok:
        analyze_and_save(tsdf_mesh, pcd, args.output_dir, "tsdf")
    else:
        print("\nTSDF mesh was empty — skipping TSDF analysis.")

    # Step 6: Poisson reconstruction (always run as comparison / fallback)
    poisson_mesh = poisson_reconstruct(pcd)
    analyze_and_save(poisson_mesh, pcd, args.output_dir, "poisson")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    print(f"  Output files in: {args.output_dir}")
    print(f"    - point_cloud.ply    (raw point cloud)")
    print(f"    - mesh_tsdf.ply      (TSDF watertight mesh)")
    print(f"    - mesh_tsdf.obj      (TSDF mesh as OBJ)")
    print(f"    - mesh_poisson.ply   (Poisson mesh for comparison)")
    print(f"    - mesh_poisson.obj   (Poisson mesh as OBJ)")


if __name__ == "__main__":
    main()
