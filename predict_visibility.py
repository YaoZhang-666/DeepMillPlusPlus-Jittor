"""Predict a 102 x N DeepM accessibility matrix directly from an OBJ mesh."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / "DeepM" / "projects"
if str(PROJECTS) not in sys.path:
    sys.path.insert(0, str(PROJECTS))

from jittor_compat import torch  # noqa: E402
import ocnn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Jittor DeepM network on an OBJ and write its 102xN prediction."
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "DeepM" / "pretrained" / "deepm_00100.model.pth")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output matrix; 1 means inaccessible (DeepM convention).")
    parser.add_argument("--probabilities", type=Path,
                        help="Optional 102xN floating-point probability matrix.")
    parser.add_argument("--tool-params", type=float, nargs=4,
                        default=(1.5, 20.0, 15.0, 30.0),
                        metavar=("R_SMALL", "R_LARGE", "HEIGHT_1", "HEIGHT_2"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"No valid vertices found in {mesh_path}")
    if normals.shape != vertices.shape:
        raise ValueError(f"Could not compute one normal per vertex for {mesh_path}")

    # Reproduce tools/seg_deepmill_cutter.py and the test transform: center by
    # the point mean, scale the largest absolute coordinate to 0.8, and orient
    # unoriented normals into the positive xyz octant.
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    max_scale = float(np.max(np.abs(centered)))
    if max_scale <= 0.0:
        raise ValueError("The mesh has zero spatial extent")
    normalized = centered / max_scale * 0.8
    normals = np.abs(normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1.0e-12)
    return normalized.astype(np.float32), normals.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    started = time.perf_counter()
    xyz, normals = load_mesh(args.mesh)
    points = ocnn.octree.Points(
        torch.from_numpy(xyz),
        torch.from_numpy(normals),
    )

    sample_octree = ocnn.octree.Octree(depth=5, full_depth=2)
    sample_octree.build_octree(points)
    # The cutter-conditioning decoder reads ``batch_nnum``.  Follow the
    # training collator even for a batch containing just one mesh.
    octree = ocnn.octree.merge_octrees([sample_octree])
    octree.construct_all_neigh()
    points = ocnn.octree.merge_points([points])
    data = octree.get_input_feature("NDP", nempty=False)
    query_points = torch.cat([points.points, points.batch_id], dim=1)

    model = ocnn.models.UNet(7, 102, interp="linear", nempty=False)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if isinstance(checkpoint, dict) and "model_dict" in checkpoint:
        checkpoint = checkpoint["model_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    tool_params = torch.tensor([args.tool_params], dtype=torch.float32)
    logits = model(data, octree, octree.depth, query_points, tool_params)
    probabilities = torch.sigmoid(logits).numpy().astype(np.float32).T
    prediction = (probabilities > args.threshold).astype(np.int8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, prediction, fmt="%d")
    if args.probabilities is not None:
        args.probabilities.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.probabilities, probabilities, fmt="%.6f")

    elapsed = time.perf_counter() - started
    print(f"mesh vertices: {len(xyz)}")
    print(f"tool parameters: {list(args.tool_params)}")
    print(f"prediction shape: {prediction.shape} (1 = inaccessible)")
    print(f"inaccessible entries: {int(prediction.sum())}/{prediction.size}")
    print(f"vertices inaccessible from all directions: {int(prediction.all(axis=0).sum())}")
    print(f"saved: {args.output}")
    if args.probabilities is not None:
        print(f"saved probabilities: {args.probabilities}")
    print(f"network inference total: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
