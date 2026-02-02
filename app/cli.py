"""CLI entry for PODFlow."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.io.dat_parser import parse_dat_file
from app.io.export import export_results
from app.pod.pod_compute import compute_pod
from app.utils.models import ExportSettings, ParseSettings, PodSettings
from app.utils.validators import validate_snapshots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PODFlow CLI")
    parser.add_argument("--input", required=True, nargs="+", help="Input DAT files")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--delimiter", default=" ", help="Delimiter (default: space)")
    parser.add_argument("--header", type=int, default=0, help="Header rows to skip")
    parser.add_argument("--no-coords", action="store_true", help="Data has no coordinates")
    parser.add_argument("--modes", type=int, default=0, help="Number of modes (0=auto)")
    parser.add_argument("--method", choices=["svd", "randomized"], default="svd")
    parser.add_argument("--no-mean", action="store_true", help="Disable mean subtraction")
    parser.add_argument("--normalize", action="store_true", help="Normalize snapshots")
    return parser.parse_args()


def run_cli() -> int:
    args = _parse_args()
    parse = ParseSettings(
        delimiter=args.delimiter,
        header_rows=args.header,
        has_coords=not args.no_coords,
    )
    pod = PodSettings(
        mean_subtract=not args.no_mean,
        normalize=args.normalize,
        n_modes=None if args.modes == 0 else args.modes,
        method=args.method,
    )

    snaps = [parse_dat_file(p, parse) for p in args.input]
    ok, msg = validate_snapshots(snaps)
    if not ok:
        raise SystemExit(msg)

    result = compute_pod(snaps, pod)
    export = ExportSettings()
    export_results(result, export, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
