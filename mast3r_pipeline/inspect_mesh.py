"""
Mesh Inspection Utility
=======================

Load and inspect mesh files produced by the MASt3R pipeline.
Reports watertightness, vertex/face counts, volume, and bounding box.

Usage:
    python inspect_mesh.py                          # inspect all meshes in output/
    python inspect_mesh.py --mesh output/mesh_tsdf.ply
    python inspect_mesh.py --mesh output/mesh_poisson.ply --mesh output/mesh_tsdf.ply
"""

import argparse
from pathlib import Path

import trimesh
import numpy as np


def load_mesh(path: Path):
    """Load a mesh file, handling Scene containers."""
    if not path.exists():
        return None, f"{path.name} not found"

    loaded = trimesh.load(path)

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return None, f"{path.name} loaded as an empty scene"
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, f"{path.name} has zero vertices or faces"

    return mesh, None


def describe_mesh(label: str, path: Path):
    """Print detailed mesh diagnostics."""
    mesh, error = load_mesh(path)

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  File: {path}")
    print(f"{'─' * 50}")

    if error is not None:
        print(f"  Status: ❌ {error}")
        return None

    print(f"  Status:     ✅ loaded")
    print(f"  Vertices:   {len(mesh.vertices):,}")
    print(f"  Faces:      {len(mesh.faces):,}")
    print(f"  Watertight: {'✅ YES' if mesh.is_watertight else '❌ NO'}")

    if mesh.is_watertight:
        print(f"  Volume:     {mesh.volume:.6f} cubic units")
    else:
        print(f"  Volume:     N/A (not watertight)")

    bounds = mesh.bounds
    extents = mesh.extents
    print(f"  Bounds min: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}, {bounds[0][2]:.4f}]")
    print(f"  Bounds max: [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}, {bounds[1][2]:.4f}]")
    print(f"  Extents:    [{extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}]")

    # Additional diagnostics
    if hasattr(mesh, 'euler_number'):
        print(f"  Euler #:    {mesh.euler_number}")
    if hasattr(mesh, 'is_volume') and mesh.is_volume:
        print(f"  Is volume:  ✅ YES")
    else:
        print(f"  Is volume:  ❌ NO (non-manifold or has holes)")

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Inspect mesh files from MASt3R pipeline")
    parser.add_argument(
        "--mesh",
        type=Path,
        nargs="*",
        default=None,
        help="Paths to mesh files to inspect. If omitted, scans output/ for .ply files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory to scan for meshes (default: ./output/).",
    )
    args = parser.parse_args()

    # Collect mesh paths
    if args.mesh:
        mesh_paths = args.mesh
    else:
        if not args.output_dir.exists():
            print(f"Output directory not found: {args.output_dir}")
            print("Run reconstruct_mast3r.py first, or specify --mesh paths.")
            return

        mesh_paths = sorted(args.output_dir.glob("mesh_*.ply"))
        if not mesh_paths:
            # Also check for any .ply that isn't point_cloud.ply
            mesh_paths = sorted(
                p for p in args.output_dir.glob("*.ply")
                if "point_cloud" not in p.name
            )

        if not mesh_paths:
            print(f"No mesh files found in {args.output_dir}")
            return

    print("=" * 50)
    print("  Mesh Inspection Report")
    print("=" * 50)

    meshes = {}
    for path in mesh_paths:
        label = path.stem.replace("mesh_", "").replace("_", " ").title()
        mesh = describe_mesh(label, path)
        if mesh is not None:
            meshes[path.stem] = mesh

    # Summary / recommendation
    if meshes:
        print(f"\n{'=' * 50}")
        print("  Recommendation")
        print(f"{'=' * 50}")

        watertight = {k: m for k, m in meshes.items() if m.is_watertight}
        if watertight:
            # Pick the one with more faces (usually better quality)
            best_key = max(watertight, key=lambda k: len(watertight[k].faces))
            best = watertight[best_key]
            print(f"  Best watertight mesh: {best_key}")
            print(f"  Volume: {best.volume:.6f} cubic units")
        else:
            print("  ⚠ No watertight mesh found.")
            print("  Try adjusting --voxel_length or --confidence_threshold")
            print("  in reconstruct_mast3r.py.")


if __name__ == "__main__":
    main()
