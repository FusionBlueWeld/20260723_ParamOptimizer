#!/usr/bin/env python3
"""Physics-guided synthetic grid for Gaussian-beam laser welding.

Assumptions
-----------
* Continuous-wave Gaussian beam.
* ``spot_um`` is the 1/e^2 intensity diameter at the work surface.
* Bead-on-plate welding of a thick mild-steel plate.
* Focus position, shielding gas, surface condition, beam quality and
  incidence angle are fixed.

The model is intended to generate physically structured synthetic data.  Its
absolute values must be calibrated against experiments before engineering use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from .plot_theme import (
    BACKGROUND,
    NEON_GREEN,
    style_3d_axis,
    style_colorbar,
    style_figure,
)


@dataclass(frozen=True)
class ModelParameters:
    """Material constants and tunable model coefficients."""

    # Representative mild-steel properties.
    density_kg_m3: float = 7850.0
    thermal_diffusivity_m2_s: float = 6.0e-6
    heat_capacity_j_kg_k: float = 750.0
    initial_temperature_k: float = 293.15
    melting_temperature_k: float = 1800.0
    latent_heat_fusion_j_kg: float = 2.70e5
    surface_tension_n_m: float = 1.5

    # Absorption and regime-transition parameters.
    absorptivity_conduction: float = 0.35
    absorptivity_keyhole: float = 0.72
    normalized_melting_threshold: float = 1.0
    normalized_keyhole_threshold: float = 6.0
    keyhole_peak_intensity_threshold_mw_cm2: float = 1.0
    melting_transition_width: float = 0.24
    keyhole_enthalpy_width: float = 0.32
    keyhole_intensity_width: float = 0.35

    # Penetration-depth coefficients.
    conduction_depth_ratio: float = 0.70
    conduction_rise_rate: float = 0.55
    drilling_coefficient_m3_j: float = 3.0e-11
    recoil_coefficient_s_m: float = 6.0e-6
    small_spot_floor_um: float = 140.0
    multireflection_factor: float = 1.15
    smooth_min_order: float = 4.0

    # Spatter-propensity coefficients.
    spatter_intensity_threshold_mw_cm2: float = 10.0
    spatter_intensity_width: float = 0.50
    collapse_width: float = 0.55
    front_tilt_threshold: float = 0.50
    front_tilt_width: float = 0.45
    # Continuous propensity assigned to level 9. Recalibrate this after
    # defining a physical inspection metric such as collected mass per length.
    spatter_level_9_reference: float = 1.80

    @property
    def melting_enthalpy_j_m3(self) -> float:
        sensible = self.heat_capacity_j_kg_k * (
            self.melting_temperature_k - self.initial_temperature_k
        )
        return self.density_kg_m3 * (
            sensible + self.latent_heat_fusion_j_kg
        )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""

    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return an increasing grid that always includes ``stop``."""

    if step <= 0:
        raise ValueError("Grid step must be positive.")
    if stop < start:
        raise ValueError("Grid stop must be greater than or equal to start.")

    count = int(np.floor((stop - start) / step + 1.0e-12))
    values = start + step * np.arange(count + 1, dtype=float)
    if values[-1] < stop - max(1.0, abs(stop)) * 1.0e-12:
        values = np.append(values, stop)
    else:
        values[-1] = stop
    return values


def evaluate_model(
    power_w: np.ndarray,
    spot_um: np.ndarray,
    speed_mm_s: np.ndarray,
    params: ModelParameters | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate penetration depth and spatter class on broadcastable arrays."""

    p = params or ModelParameters()

    power_w = np.asarray(power_w, dtype=float)
    spot_m = np.asarray(spot_um, dtype=float) * 1.0e-6
    speed_m_s = np.asarray(speed_mm_s, dtype=float) * 1.0e-3
    radius_m = spot_m / 2.0

    if np.any(power_w <= 0) or np.any(spot_m <= 0) or np.any(speed_m_s <= 0):
        raise ValueError("Power, spot diameter and scan speed must be positive.")

    # For I(r)=I_peak*exp(-2*r^2/w^2), d=2w is the 1/e^2 diameter.
    peak_intensity_w_m2 = 8.0 * power_w / (np.pi * spot_m**2)
    peak_intensity_mw_cm2 = peak_intensity_w_m2 / 1.0e10

    # Fixed-point update: keyhole formation increases absorptivity.
    eta = np.full(np.broadcast(power_w, spot_m, speed_m_s).shape,
                  p.absorptivity_conduction, dtype=float)
    keyhole_gate = np.zeros_like(eta)
    normalized_enthalpy = np.zeros_like(eta)

    for _ in range(8):
        normalized_enthalpy = (
            eta * power_w
            / (
                np.pi
                * p.melting_enthalpy_j_m3
                * np.sqrt(
                    p.thermal_diffusivity_m2_s
                    * speed_m_s
                    * radius_m**3
                )
            )
        )
        enthalpy_gate = sigmoid(
            (
                np.log(np.maximum(normalized_enthalpy, 1.0e-15))
                - np.log(p.normalized_keyhole_threshold)
            )
            / p.keyhole_enthalpy_width
        )
        intensity_gate = sigmoid(
            (
                np.log(
                    np.maximum(
                        eta * peak_intensity_mw_cm2,
                        1.0e-15,
                    )
                )
                - np.log(p.keyhole_peak_intensity_threshold_mw_cm2)
            )
            / p.keyhole_intensity_width
        )
        keyhole_gate = enthalpy_gate * intensity_gate
        eta = (
            p.absorptivity_conduction
            + (p.absorptivity_keyhole - p.absorptivity_conduction)
            * keyhole_gate
        )

    melting_gate = sigmoid(
        (
            np.log(np.maximum(normalized_enthalpy, 1.0e-15))
            - np.log(p.normalized_melting_threshold)
        )
        / p.melting_transition_width
    )

    # Conduction-mode depth: rises from zero and saturates below d-scale.
    excess_heat = np.maximum(
        normalized_enthalpy / p.normalized_melting_threshold - 1.0,
        0.0,
    )
    conduction_depth_m = (
        p.conduction_depth_ratio
        * spot_m
        * (1.0 - np.exp(-p.conduction_rise_rate * excess_heat))
    )

    # Keyhole depth: Fabbro-type P/(d*v) scaling with small-spot and
    # low-speed saturation corrections.
    effective_spot_m = np.sqrt(
        spot_m**2 + (p.small_spot_floor_um * 1.0e-6) ** 2
    )
    keyhole_depth_raw_m = (
        p.multireflection_factor
        * 4.0
        * p.drilling_coefficient_m3_j
        * p.absorptivity_keyhole
        * power_w
        / (np.pi * speed_m_s * effective_spot_m)
    )
    keyhole_depth_limit_m = (
        2.0
        * p.recoil_coefficient_s_m
        * p.absorptivity_keyhole
        * power_w
        / (np.pi * p.surface_tension_n_m)
    )
    n = p.smooth_min_order
    keyhole_depth_m = (
        keyhole_depth_raw_m ** (-n) + keyhole_depth_limit_m ** (-n)
    ) ** (-1.0 / n)

    penetration_depth_mm = (
        melting_gate
        * (
            (1.0 - keyhole_gate) * conduction_depth_m
            + keyhole_gate * keyhole_depth_m
        )
        * 1.0e3
    )

    # Spatter model: transition, low-speed collapse, recoil and high-speed
    # front-wall/humping contributions.
    transition_instability = 4.0 * keyhole_gate * (1.0 - keyhole_gate)

    capillary_stability = (
        p.recoil_coefficient_s_m * spot_m * speed_m_s
        / (
            2.0
            * p.surface_tension_n_m
            * p.drilling_coefficient_m3_j
        )
    )
    low_speed_collapse = keyhole_gate * sigmoid(
        -np.log(np.maximum(capillary_stability, 1.0e-15))
        / p.collapse_width
    )

    absorbed_peak_intensity_mw_cm2 = eta * peak_intensity_mw_cm2
    recoil_ejection = keyhole_gate * sigmoid(
        (
            np.log(np.maximum(absorbed_peak_intensity_mw_cm2, 1.0e-15))
            - np.log(p.spatter_intensity_threshold_mw_cm2)
        )
        / p.spatter_intensity_width
    )

    front_tilt_index = speed_m_s / np.maximum(
        p.drilling_coefficient_m3_j
        * p.absorptivity_keyhole
        * peak_intensity_w_m2,
        1.0e-15,
    )
    high_speed_humping = keyhole_gate * sigmoid(
        (
            np.log(np.maximum(front_tilt_index, 1.0e-15))
            - np.log(p.front_tilt_threshold)
        )
        / p.front_tilt_width
    )

    spatter_propensity = melting_gate * (
        1.5 * transition_instability
        + 1.0 * low_speed_collapse
        + 0.8 * recoil_ejection
        + 0.8 * high_speed_humping
    )
    spatter_level = np.rint(
        9.0
        * np.clip(
            spatter_propensity / p.spatter_level_9_reference,
            0.0,
            1.0,
        )
    ).astype(int)
    spatter_level = np.clip(spatter_level, 0, 9)

    regime_code = np.select(
        [
            melting_gate < 0.10,
            keyhole_gate < 0.20,
            keyhole_gate < 0.80,
        ],
        [0, 1, 2],
        default=3,
    )

    return {
        "penetration_depth_mm": penetration_depth_mm,
        "spatter_level_0_9": spatter_level,
        "spatter_propensity": spatter_propensity,
        "normalized_enthalpy": normalized_enthalpy,
        "peak_intensity_mw_cm2": peak_intensity_mw_cm2,
        "absorbed_peak_intensity_mw_cm2": absorbed_peak_intensity_mw_cm2,
        "line_energy_j_mm": power_w / np.asarray(speed_mm_s, dtype=float),
        "absorptivity": eta,
        "melting_gate": melting_gate,
        "keyhole_gate": keyhole_gate,
        "capillary_stability": capillary_stability,
        "front_tilt_index": front_tilt_index,
        "regime_code": regime_code,
    }


def generate_grid(
    power_values_w: np.ndarray,
    spot_values_um: np.ndarray,
    speed_values_mm_s: np.ndarray,
    params: ModelParameters | None = None,
) -> pd.DataFrame:
    """Create a full Cartesian grid and evaluate the model."""

    p_grid, d_grid, v_grid = np.meshgrid(
        power_values_w,
        spot_values_um,
        speed_values_mm_s,
        indexing="ij",
    )
    outputs = evaluate_model(p_grid, d_grid, v_grid, params)

    frame = pd.DataFrame(
        {
            "laser_power_w": p_grid.ravel(),
            "spot_diameter_um": d_grid.ravel(),
            "scan_speed_mm_s": v_grid.ravel(),
        }
    )
    for name, values in outputs.items():
        frame[name] = np.asarray(values).ravel()

    regime_names = np.array(
        ["no_melt", "conduction", "transition", "keyhole"],
        dtype=object,
    )
    frame["regime"] = regime_names[frame["regime_code"].astype(int)]
    return frame


def plot_grid(
    frame: pd.DataFrame,
    output_path: Path,
    marker_size: float = 24.0,
) -> None:
    """Plot penetration and spatter as side-by-side 3D scatter plots."""

    fig = plt.figure(figsize=(17.0, 7.2), constrained_layout=True)
    style_figure(fig)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    x = frame["laser_power_w"].to_numpy() / 1000.0
    y = frame["spot_diameter_um"].to_numpy()
    z = frame["scan_speed_mm_s"].to_numpy()

    depth = frame["penetration_depth_mm"].to_numpy()
    depth_scatter = axes[0].scatter(
        x,
        y,
        z,
        c=depth,
        cmap="jet",
        norm=Normalize(vmin=0.0, vmax=max(float(depth.max()), 1.0e-9)),
        s=marker_size,
        alpha=0.88,
        linewidths=0.0,
        depthshade=False,
    )
    axes[0].set_title("Maximum penetration depth")
    depth_cbar = fig.colorbar(
        depth_scatter,
        ax=axes[0],
        shrink=0.70,
        pad=0.08,
    )
    depth_cbar.set_label("Depth [mm]")
    style_colorbar(depth_cbar)

    spatter = frame["spatter_level_0_9"].to_numpy()
    spatter_scatter = axes[1].scatter(
        x,
        y,
        z,
        c=spatter,
        cmap="jet",
        norm=Normalize(vmin=0.0, vmax=9.0),
        s=marker_size,
        alpha=0.88,
        linewidths=0.0,
        depthshade=False,
    )
    axes[1].set_title("Spatter level")
    spatter_cbar = fig.colorbar(
        spatter_scatter,
        ax=axes[1],
        shrink=0.70,
        pad=0.08,
        ticks=np.arange(10),
    )
    spatter_cbar.set_label("Ordinal level [0: none, 9: heavy]")
    style_colorbar(spatter_cbar)

    for ax in axes:
        style_3d_axis(ax)
        ax.set_xlabel("Laser power [kW]", labelpad=10)
        ax.set_ylabel("1/e² spot diameter [µm]", labelpad=10)
        ax.set_zlabel("Scan speed [mm/s]", labelpad=10)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.view_init(elev=23.0, azim=-56.0)
        ax.grid(True)
        ax.set_box_aspect((1.15, 0.9, 1.0))

    fig.suptitle(
        "Physics-guided Gaussian-beam laser-welding parameter grid",
        fontsize=15,
        color="white",
    )
    fig.savefig(
        output_path,
        dpi=190,
        bbox_inches="tight",
        facecolor=BACKGROUND,
        edgecolor=NEON_GREEN,
    )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a physics-guided laser-welding grid, CSV and 3D plots."
        )
    )
    parser.add_argument("--power-start", type=float, default=100.0)
    parser.add_argument("--power-stop", type=float, default=6000.0)
    parser.add_argument("--power-step", type=float, default=500.0)
    parser.add_argument("--spot-start", type=float, default=50.0)
    parser.add_argument("--spot-stop", type=float, default=300.0)
    parser.add_argument("--spot-step", type=float, default=50.0)
    parser.add_argument("--speed-start", type=float, default=10.0)
    parser.add_argument("--speed-stop", type=float, default=1000.0)
    parser.add_argument("--speed-step", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for laser_welding_grid.csv and PNG preview.",
    )
    parser.add_argument("--marker-size", type=float, default=24.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    powers = inclusive_grid(
        args.power_start,
        args.power_stop,
        args.power_step,
    )
    spots = inclusive_grid(args.spot_start, args.spot_stop, args.spot_step)
    speeds = inclusive_grid(
        args.speed_start,
        args.speed_stop,
        args.speed_step,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "laser_welding_grid.csv"
    figure_path = args.output_dir / "laser_welding_3d_preview.png"

    frame = generate_grid(powers, spots, speeds)
    frame.to_csv(csv_path, index=False, float_format="%.8g")
    plot_grid(frame, figure_path, marker_size=args.marker_size)

    print(f"Generated {len(frame):,} grid points")
    print(f"Power values: {len(powers)}")
    print(f"Spot values: {len(spots)}")
    print(f"Speed values: {len(speeds)}")
    print(
        "Penetration range: "
        f"{frame['penetration_depth_mm'].min():.4f} to "
        f"{frame['penetration_depth_mm'].max():.4f} mm"
    )
    print(
        "Spatter levels: "
        f"{frame['spatter_level_0_9'].min()} to "
        f"{frame['spatter_level_0_9'].max()}"
    )
    print(csv_path.resolve())
    print(figure_path.resolve())


if __name__ == "__main__":
    main()
