"""Convert a DeepM direction prediction into DeepMill++ visibility input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeepM's 102×N direction labels to visibility_matrix.txt."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="DeepM visual/direction/<model>.txt file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination visibility_matrix.txt")
    parser.add_argument("--mesh", type=Path,
                        help="Optional OBJ mesh used to validate the N columns")
    parser.add_argument("--directions", type=int, default=102,
                        help="Expected number of sampled directions (default: 102)")
    parser.add_argument("--input-is-reachable", action="store_true",
                        help="Use this only when input 1 already means reachable; by default 1 means inaccessible.")
    return parser.parse_args()


def load_binary_matrix(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"DeepM output not found: {path}")
    matrix = np.loadtxt(path, dtype=np.int8)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} direction rows, got {matrix.shape[0]} rows."
        )
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("The DeepM direction matrix must contain only binary 0/1 values.")
    return matrix


def validate_vertex_count(mesh_path: Path, columns: int) -> None:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"Unable to read mesh vertices from: {mesh_path}")
    if vertices.shape[0] != columns:
        raise ValueError(
            f"Matrix has {columns} columns but mesh has {vertices.shape[0]} vertices. "
            "They must use the same vertex order."
        )


def main() -> None:
    args = parse_args()
    deepm_matrix = load_binary_matrix(args.input, args.directions)
    if args.mesh is not None:
        validate_vertex_count(args.mesh, deepm_matrix.shape[1])

    # DeepM's direction_visual() writes 1 for an inaccessible prediction.
    # deepmill2.py uses 1 to mean that the vertex is allowed for this view.
    visibility = deepm_matrix if args.input_is_reachable else 1 - deepm_matrix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, visibility, fmt="%d")
    print(
        f"Saved {args.output}: shape={visibility.shape}, "
        f"allowed entries={int(visibility.sum())}/{visibility.size}"
    )


if __name__ == "__main__":
    main()
