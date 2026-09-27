#!/usr/bin/env python3
"""
Watertightness & Mesh Quality Checker
=====================================
Checks meshes for watertightness, vertex/face counts, and volume.
Works with both MASt3R output meshes (TSDF, Poisson) and COLMAP meshes.

Usage:
    python watertightness.py
    python watertightness.py --mesh mast3r_pipeline/output/mesh_tsdf.ply
    python watertightness.py --mesh dense_model/meshed-poisson.ply
"""

import argparse
import sys
from pathlib import Path

try:
    import trimesh
except ImportError as e:
    print(f"Error: {e}")
    print("\nRequired package 'trimesh' is not found in the current Python environment.")
    print("Please activate the mast3r conda environment:")
    print("    conda activate mast3r\n")
    sys.exit(1)


def load_mesh(path: Path):
    if not path.exists():
        return None, f"'{path}' not found"

    try:
        loaded = trimesh.load(path)
    except Exception as exc:
        return None, f"Failed to load {path.name}: {exc}"

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
    mesh, error = load_mesh(path)
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  File: {path}")
    print(f"{'─' * 50}")

    if error is not None:
        print(f"  Status:     ❌ {error}")
        return None

    print(f"  Status:     ✅ loaded")
    print(f"  Vertices:   {len(mesh.vertices):,}")
    print(f"  Faces:      {len(mesh.faces):,}")
    print(f"  Watertight: {'✅ YES' if mesh.is_watertight else '❌ NO'}")

    if mesh.is_watertight:
        print(f"  Volume:     {mesh.volume:.6f} cubic units")
    else:
        print(f"  Volume:     N/A (not watertight)")

    return mesh


def find_default_meshes(script_dir: Path) -> list[tuple[str, Path]]:
    """Look for candidate meshes in standard locations."""
    candidates = [
        # MASt3R outputs
        ("MASt3R TSDF Mesh", script_dir / "mast3r_pipeline" / "output" / "mesh_tsdf.ply"),
        ("MASt3R TSDF Mesh", script_dir / "output" / "mesh_tsdf.ply"),
        ("MASt3R Poisson Mesh", script_dir / "mast3r_pipeline" / "output" / "mesh_poisson.ply"),
        ("MASt3R Poisson Mesh", script_dir / "output" / "mesh_poisson.ply"),
        # COLMAP outputs
        ("COLMAP Delaunay Mesh", script_dir / "dense_model" / "meshed-delaunay.ply"),
        ("COLMAP Poisson Mesh", script_dir / "dense_model" / "meshed-poisson.ply"),
    ]

    seen = set()
    found = []
    for label, path in candidates:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            found.append((label, path))

    return found


def main():
    parser = argparse.ArgumentParser(description="Check mesh watertightness and volume")
    parser.add_argument(
        "--mesh",
        type=Path,
        nargs="*",
        default=None,
        help="Paths to specific mesh files to inspect (.ply / .obj).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.mesh:
        targets = [(f"Mesh ({p.stem})", p) for p in args.mesh]
    else:
        targets = find_default_meshes(script_dir)

    if not targets:
        print("No meshes found in default directories:")
        print("  - mast3r_pipeline/output/mesh_tsdf.ply")
        print("  - mast3r_pipeline/output/mesh_poisson.ply")
        print("  - dense_model/meshed-*.ply")
        print("\nPlease run reconstruct_mast3r.py first or specify: python watertightness.py --mesh <path>")
        return

    print("=" * 50)
    print("  Mesh Watertightness & Volume Report")
    print("=" * 50)

    evaluated = {}
    for label, path in targets:
        mesh = describe_mesh(label, path)
        if mesh is not None:
            evaluated[label] = mesh

    print(f"\n{'=' * 50}")
    print("  Summary & Recommendation")
    print(f"{'=' * 50}")

    watertight = {k: m for k, m in evaluated.items() if m.is_watertight}
    if watertight:
        best_name = next((k for k in watertight if "tsdf" in k.lower()), None)
        if not best_name:
            best_name = max(watertight, key=lambda k: len(watertight[k].faces))
        best_mesh = watertight[best_name]
        print(f"  Recommended volume mesh: ✅ {best_name}")
        print(f"  Watertight Volume:       {best_mesh.volume:.6f} cubic units")
    elif evaluated:
        print("  ⚠ None of the evaluated meshes are watertight.")
        print("  If using MASt3R, TSDF meshing is recommended for watertightness.")
    else:
        print("  No meshes could be loaded.")


if __name__ == "__main__":
    main()
