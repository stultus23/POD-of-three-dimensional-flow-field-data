"""Generate synthetic 3D DAT snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--n", type=int, default=5, help="Number of snapshots")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(0, 1, args.nx)
    ys = np.linspace(0, 1, args.ny)
    zs = np.linspace(0, 1, args.nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)

    for i in range(args.n):
        u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.1 * i
        v = np.cos(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
        w = np.sin(2 * np.pi * Z) * np.cos(2 * np.pi * X)
        data = np.stack([u.reshape(-1), v.reshape(-1), w.reshape(-1)], axis=1)
        mat = np.hstack([coords, data])
        np.savetxt(out / f"snapshot_{i:03d}.dat", mat, fmt="%.6f")


if __name__ == "__main__":
    main()
