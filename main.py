#!/usr/bin/env python3
"""Command-line entry point for grid generation and adaptive search."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adaptive_search import SearchSettings, run_search
from src.physics_model import generate_grid, inclusive_grid, plot_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physics-guided Gaussian-beam laser-welding simulator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser(
        "grid",
        help="Generate a full Cartesian grid, CSV and two 3D scatter plots.",
    )
    grid.add_argument("--power-start", type=float, default=100.0)
    grid.add_argument("--power-stop", type=float, default=6000.0)
    grid.add_argument("--power-step", type=float, default=500.0)
    grid.add_argument("--spot-start", type=float, default=50.0)
    grid.add_argument("--spot-stop", type=float, default=300.0)
    grid.add_argument("--spot-step", type=float, default=50.0)
    grid.add_argument("--speed-start", type=float, default=10.0)
    grid.add_argument("--speed-stop", type=float, default=1000.0)
    grid.add_argument("--speed-step", type=float, default=100.0)
    grid.add_argument("--marker-size", type=float, default=24.0)
    grid.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/grid"),
    )

    search = subparsers.add_parser(
        "search",
        help="Run coarse-to-fine multi-branch Gaussian-process search.",
    )
    search.add_argument("--depth-min", type=float, default=2.5)
    search.add_argument(
        "--spatter-max",
        type=int,
        default=5,
        help="Exclusive upper bound: 5 means spatter level < 5.",
    )
    search.add_argument("--coarse-points", type=int, default=3)
    search.add_argument("--prediction-points", type=int, default=21)
    search.add_argument("--max-steps", type=int, default=8)
    search.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/search"),
    )
    return parser


def run_grid(args: argparse.Namespace) -> None:
    powers = inclusive_grid(args.power_start, args.power_stop, args.power_step)
    spots = inclusive_grid(args.spot_start, args.spot_stop, args.spot_step)
    speeds = inclusive_grid(args.speed_start, args.speed_stop, args.speed_step)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "laser_welding_grid.csv"
    figure_path = args.output_dir / "laser_welding_3d_preview.png"

    frame = generate_grid(powers, spots, speeds)
    frame.to_csv(csv_path, index=False, float_format="%.8g")
    plot_grid(frame, figure_path, marker_size=args.marker_size)

    print(f"Generated grid points: {len(frame):,}")
    print(
        "Penetration depth: "
        f"{frame['penetration_depth_mm'].min():.4f} - "
        f"{frame['penetration_depth_mm'].max():.4f} mm"
    )
    print(
        "Spatter level: "
        f"{frame['spatter_level_0_9'].min()} - "
        f"{frame['spatter_level_0_9'].max()}"
    )
    print(csv_path.resolve())
    print(figure_path.resolve())


def run_adaptive_search(args: argparse.Namespace) -> None:
    if args.coarse_points < 3:
        raise ValueError("--coarse-points must be at least 3.")
    if args.prediction_points < 9:
        raise ValueError("--prediction-points must be at least 9.")
    if not 1 <= args.spatter_max <= 10:
        raise ValueError("--spatter-max must be from 1 through 10.")

    settings = SearchSettings(
        depth_min_mm=args.depth_min,
        spatter_max_exclusive=args.spatter_max,
        coarse_points_per_axis=args.coarse_points,
        prediction_points_per_axis=args.prediction_points,
        max_steps=args.max_steps,
    )
    summary = run_search(args.output_dir, settings)

    print(f"Status: {summary['status']}")
    print(f"Steps: {summary['completed_steps']}")
    print(f"Experiments: {summary['total_experiments']}")
    for box in summary["final_condition_boxes"]:
        print(
            f"{box['branch_id']}: "
            f"P={box['laser_power_kw']} kW, "
            f"d={box['spot_diameter_um']} um, "
            f"v={box['scan_speed_mm_s']} mm/s"
        )
    print((args.output_dir / "laser_welding_adaptive_search.gif").resolve())


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "grid":
        run_grid(args)
    else:
        run_adaptive_search(args)


if __name__ == "__main__":
    main()
