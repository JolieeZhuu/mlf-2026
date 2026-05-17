from pathlib import Path

import trimesh


def load_mesh(path: Path):
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
    mesh, error = load_mesh(path)
    print(label)

    if error is not None:
        print("  status:", error)
        return None

    print("  status: ok")
    print("  watertight:", mesh.is_watertight)
    print("  vertices:", len(mesh.vertices))
    print("  faces:", len(mesh.faces))
    print("  volume:", mesh.volume)
    return mesh


poisson = describe_mesh("Poisson mesh", Path("dense_model/meshed-poisson.ply"))
delaunay = describe_mesh("Delaunay mesh", Path("dense_model/meshed-delaunay.ply"))

print()
if delaunay is not None and delaunay.is_watertight:
    print("Recommended volume mesh: Delaunay")
elif poisson is not None and poisson.is_watertight:
    print("Recommended volume mesh: Poisson")
elif delaunay is not None:
    print("Neither mesh is watertight. Delaunay exists, but verify and repair it before trusting volume.")
elif poisson is not None:
    print("Neither mesh is watertight. Poisson exists, but verify and repair it before trusting volume.")
else:
    print("No usable mesh was found. Check the dense reconstruction and meshing steps first.")
