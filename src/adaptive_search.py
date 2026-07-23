#!/usr/bin/env python3
"""Adaptive coarse-to-fine condition-range search for laser welding.

Each queried grid point is treated as a costly physical experiment.  Two
independent Gaussian-process regressors estimate penetration depth and spatter
level from the accumulated experiments.  The probability of satisfying both
user constraints defines a translucent pink feasible region.

At every step the program creates four transition frames: the current black
experimental grid, the pink Gaussian-process probability region, the recut
pink branch grids, and those grids promoted to the next black search ranges.
Disconnected feasible components become independent branches.  All transition
frames are assembled into an animated GIF.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import label
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from .physics_model import evaluate_model


PINK = "#ff69b4"
DEPTH_NORM = Normalize(vmin=0.0, vmax=12.0)
SPATTER_NORM = Normalize(vmin=0.0, vmax=9.0)
BRANCH_COLORS = (
    "#d81b60",
    "#1e88e5",
    "#ffc107",
    "#004d40",
    "#8e24aa",
)


@dataclass(frozen=True)
class SearchBox:
    """Axis-aligned search range."""

    power_min_w: float
    power_max_w: float
    spot_min_um: float
    spot_max_um: float
    speed_min_mm_s: float
    speed_max_mm_s: float
    branch_id: str = "B1"
    parent_id: str | None = None

    def widths(self) -> np.ndarray:
        return np.array(
            [
                self.power_max_w - self.power_min_w,
                self.spot_max_um - self.spot_min_um,
                self.speed_max_mm_s - self.speed_min_mm_s,
            ],
            dtype=float,
        )

    def as_bounds(self) -> np.ndarray:
        return np.array(
            [
                [self.power_min_w, self.power_max_w],
                [self.spot_min_um, self.spot_max_um],
                [self.speed_min_mm_s, self.speed_max_mm_s],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class SearchSettings:
    """Algorithm settings and stopping criteria."""

    depth_min_mm: float = 2.0
    spatter_max_exclusive: int = 6
    coarse_points_per_axis: int = 3
    fallback_points_per_axis: int = 3
    prediction_points_per_axis: int = 21
    feasible_probability: float = 0.55
    confidence_z: float = 2.0
    max_branches: int = 3
    min_component_points: int = 5
    max_steps: int = 8
    min_steps_before_convergence: int = 3
    min_steps_before_no_feasible: int = 2
    power_resolution_w: float = 250.0
    spot_resolution_um: float = 15.0
    speed_resolution_mm_s: float = 35.0
    depth_std_tolerance_mm: float = 0.35
    spatter_std_tolerance: float = 0.85
    pink_alpha: float = 0.10  # 90% transparent.
    random_seed: int = 7
    boundary_shift_fraction: float = 0.18
    uncertainty_expand_fraction: float = 0.15
    stagnation_volume_ratio: float = 0.78
    translation_fraction: float = 0.45
    global_reexplore_interval: int = 3
    global_probe_width_fraction: float = 0.28
    max_backtracks: int = 2
    max_global_reexplorations: int = 1
    max_boundary_shift_steps: int = 2
    max_uncertainty_reexpansions: int = 1
    max_probability_peak_shifts: int = 1
    box_merge_overlap_ratio: float = 0.30
    box_merge_containment_ratio: float = 0.80
    box_merge_max_inflation: float = 1.50
    box_merge_max_gap_cells: float = 1.0

    @property
    def spatter_gp_boundary(self) -> float:
        # Integer score < N is represented by the continuous boundary N-0.5.
        return float(self.spatter_max_exclusive) - 0.5


@dataclass
class BoxAnalysis:
    """Dense GP prediction and component results inside one branch box."""

    box: SearchBox
    power_axis: np.ndarray
    spot_axis: np.ndarray
    speed_axis: np.ndarray
    points: np.ndarray
    mean_depth: np.ndarray
    std_depth: np.ndarray
    mean_spatter: np.ndarray
    std_spatter: np.ndarray
    joint_probability: np.ndarray
    pink_mask: np.ndarray
    optimistic_mask: np.ndarray
    next_boxes: list[SearchBox]
    resolved: bool
    max_probability: float
    optimistic_count: int
    pink_count: int


@dataclass
class SearchUpdate:
    """Rule-selected update of the active search boxes."""

    boxes: list[SearchBox]
    action: str
    reason: str
    backtracked: bool = False
    global_reexploration: bool = False


def normalize_inputs(points: np.ndarray, domain: SearchBox) -> np.ndarray:
    bounds = domain.as_bounds()
    return (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def point_key(point: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(round(float(value), 8) for value in point)
    return values  # type: ignore[return-value]


def grid_fractions(points_per_axis: int, phase: int) -> np.ndarray:
    """Return a regular but phase-shifted 1D lattice in [0, 1].

    Phase zero includes the range boundaries. Later phases shift the lattice,
    so an unchanged branch still receives new experiments instead of repeating
    the same points.
    """

    if phase == 0:
        return np.linspace(0.0, 1.0, points_per_axis)
    base = (np.arange(points_per_axis, dtype=float) + 0.5) / points_per_axis
    shift = (((phase - 1) * 0.38196601125) % 1.0) / points_per_axis
    return np.sort((base + shift) % 1.0)


def box_grid(
    box: SearchBox,
    points_per_axis: int,
    phase: int,
) -> np.ndarray:
    fractions = grid_fractions(points_per_axis, phase)
    power = (
        box.power_min_w
        + fractions * (box.power_max_w - box.power_min_w)
    )
    spot = box.spot_min_um + fractions * (
        box.spot_max_um - box.spot_min_um
    )
    speed = box.speed_min_mm_s + fractions * (
        box.speed_max_mm_s - box.speed_min_mm_s
    )
    p_grid, d_grid, v_grid = np.meshgrid(
        power,
        spot,
        speed,
        indexing="ij",
    )
    return np.column_stack(
        [p_grid.ravel(), d_grid.ravel(), v_grid.ravel()]
    )


def run_experiments(
    boxes: list[SearchBox],
    step: int,
    points_per_axis: int,
    phase: int,
    experiment_cache: dict[tuple[float, float, float], dict[str, object]],
) -> tuple[np.ndarray, int]:
    """Evaluate only grid points that have not been measured previously."""

    pending: dict[tuple[float, float, float], dict[str, object]] = {}
    requested_points: list[np.ndarray] = []

    for box in boxes:
        points = box_grid(box, points_per_axis, phase)
        requested_points.append(points)
        for point in points:
            key = point_key(point)
            if key not in experiment_cache and key not in pending:
                pending[key] = {
                    "point": point.copy(),
                    "branch_id": box.branch_id,
                }
            elif key in pending:
                previous = str(pending[key]["branch_id"])
                if box.branch_id not in previous.split("|"):
                    pending[key]["branch_id"] = (
                        previous + "|" + box.branch_id
                    )

    if pending:
        new_points = np.vstack(
            [np.asarray(item["point"], dtype=float) for item in pending.values()]
        )
        outputs = evaluate_model(
            new_points[:, 0],
            new_points[:, 1],
            new_points[:, 2],
        )
        for index, (key, metadata) in enumerate(pending.items()):
            experiment_cache[key] = {
                "laser_power_w": float(new_points[index, 0]),
                "spot_diameter_um": float(new_points[index, 1]),
                "scan_speed_mm_s": float(new_points[index, 2]),
                "penetration_depth_mm": float(
                    outputs["penetration_depth_mm"][index]
                ),
                "spatter_level_0_9": int(
                    outputs["spatter_level_0_9"][index]
                ),
                "spatter_propensity": float(
                    outputs["spatter_propensity"][index]
                ),
                "normalized_enthalpy": float(
                    outputs["normalized_enthalpy"][index]
                ),
                "keyhole_gate": float(outputs["keyhole_gate"][index]),
                "first_sampled_step": int(step),
                "source_branch": str(metadata["branch_id"]),
            }

    all_requested = (
        np.vstack(requested_points)
        if requested_points
        else np.empty((0, 3), dtype=float)
    )
    return all_requested, len(pending)


def experiment_frame(
    experiment_cache: dict[tuple[float, float, float], dict[str, object]],
) -> pd.DataFrame:
    frame = pd.DataFrame(experiment_cache.values())
    if frame.empty:
        return frame
    return frame.sort_values(
        ["first_sampled_step", "laser_power_w", "spot_diameter_um",
         "scan_speed_mm_s"]
    ).reset_index(drop=True)


def fit_gaussian_processes(
    frame: pd.DataFrame,
    domain: SearchBox,
    random_seed: int,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Fit separate depth and spatter Gaussian-process regressors."""

    raw_points = frame[
        ["laser_power_w", "spot_diameter_um", "scan_speed_mm_s"]
    ].to_numpy(dtype=float)
    x = normalize_inputs(raw_points, domain)

    kernel = (
        ConstantKernel(1.0, (0.1, 20.0))
        * Matern(
            length_scale=np.array([0.30, 0.30, 0.30]),
            length_scale_bounds=(0.03, 3.0),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.015, noise_level_bounds=(1.0e-6, 0.3))
    )

    models = []
    for target in ("penetration_depth_mm", "spatter_level_0_9"):
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, frame[target].to_numpy(dtype=float))
        models.append(model)

    return models[0], models[1]


def make_component_boxes(
    mask: np.ndarray,
    probability: np.ndarray,
    box: SearchBox,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: SearchSettings,
) -> list[SearchBox]:
    """Convert disconnected 3D mask components into padded branch boxes."""

    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labels, count = label(mask, structure=structure)

    components: list[tuple[int, float, int]] = []
    for component_id in range(1, count + 1):
        component_mask = labels == component_id
        size = int(component_mask.sum())
        peak = float(probability[component_mask].max())
        if size >= settings.min_component_points or peak >= 0.80:
            components.append((size, peak, component_id))
    components.sort(reverse=True)

    selected: list[SearchBox] = []
    parent_bounds = box.as_bounds()
    for branch_index, (_, _, component_id) in enumerate(
        components[: settings.max_branches]
    ):
        indices = np.argwhere(labels == component_id)
        lower_indices = np.maximum(indices.min(axis=0) - 1, 0)
        upper_indices = np.minimum(
            indices.max(axis=0) + 1,
            np.array(mask.shape) - 1,
        )

        lower = np.array(
            [axes[axis][lower_indices[axis]] for axis in range(3)],
            dtype=float,
        )
        upper = np.array(
            [axes[axis][upper_indices[axis]] for axis in range(3)],
            dtype=float,
        )

        # A small safety margin protects against prematurely discarding the
        # probability boundary. It is clipped to the current parent box.
        padding = 0.02 * box.widths()
        lower = np.maximum(lower - padding, parent_bounds[:, 0])
        upper = np.minimum(upper + padding, parent_bounds[:, 1])

        child_id = (
            box.branch_id
            if len(components) == 1
            else f"{box.branch_id}.{branch_index + 1}"
        )
        selected.append(
            SearchBox(
                power_min_w=float(lower[0]),
                power_max_w=float(upper[0]),
                spot_min_um=float(lower[1]),
                spot_max_um=float(upper[1]),
                speed_min_mm_s=float(lower[2]),
                speed_max_mm_s=float(upper[2]),
                branch_id=child_id,
                parent_id=box.branch_id,
            )
        )
    return selected


def analyze_box(
    box: SearchBox,
    depth_gp: GaussianProcessRegressor,
    spatter_gp: GaussianProcessRegressor,
    domain: SearchBox,
    settings: SearchSettings,
) -> BoxAnalysis:
    n = settings.prediction_points_per_axis
    power_axis = np.linspace(box.power_min_w, box.power_max_w, n)
    spot_axis = np.linspace(box.spot_min_um, box.spot_max_um, n)
    speed_axis = np.linspace(box.speed_min_mm_s, box.speed_max_mm_s, n)
    p_grid, d_grid, v_grid = np.meshgrid(
        power_axis,
        spot_axis,
        speed_axis,
        indexing="ij",
    )
    points = np.column_stack(
        [p_grid.ravel(), d_grid.ravel(), v_grid.ravel()]
    )
    x = normalize_inputs(points, domain)

    mean_depth, std_depth = depth_gp.predict(x, return_std=True)
    mean_spatter, std_spatter = spatter_gp.predict(x, return_std=True)
    std_depth = np.maximum(std_depth, 0.03)
    std_spatter = np.maximum(std_spatter, 0.15)

    depth_probability = norm.cdf(
        (mean_depth - settings.depth_min_mm) / std_depth
    )
    spatter_probability = norm.cdf(
        (settings.spatter_gp_boundary - mean_spatter) / std_spatter
    )
    joint_probability = depth_probability * spatter_probability

    shape = (n, n, n)
    mean_depth_3d = mean_depth.reshape(shape)
    std_depth_3d = std_depth.reshape(shape)
    mean_spatter_3d = mean_spatter.reshape(shape)
    std_spatter_3d = std_spatter.reshape(shape)
    probability_3d = joint_probability.reshape(shape)

    optimistic_mask = (
        (
            mean_depth_3d
            + settings.confidence_z * std_depth_3d
            >= settings.depth_min_mm
        )
        & (
            mean_spatter_3d
            - settings.confidence_z * std_spatter_3d
            < settings.spatter_gp_boundary
        )
    )
    pink_mask = (
        probability_3d >= settings.feasible_probability
    ) & optimistic_mask

    selection_mask = pink_mask.copy()
    if not selection_mask.any() and optimistic_mask.any():
        # If no point reaches the standard probability cutoff, keep only the
        # most promising uncertain core instead of discarding the branch.
        optimistic_probabilities = probability_3d[optimistic_mask]
        fallback_cutoff = max(
            float(np.quantile(optimistic_probabilities, 0.93)),
            0.70 * float(optimistic_probabilities.max()),
        )
        selection_mask = optimistic_mask & (
            probability_3d >= fallback_cutoff
        )

    next_boxes = make_component_boxes(
        selection_mask,
        probability_3d,
        box,
        (power_axis, spot_axis, speed_axis),
        settings,
    )

    cell_size = box.widths() / float(n - 1)
    input_resolved = bool(
        cell_size[0] <= settings.power_resolution_w
        and cell_size[1] <= settings.spot_resolution_um
        and cell_size[2] <= settings.speed_resolution_mm_s
    )

    boundary_mask = (
        (probability_3d >= 0.25)
        & (probability_3d <= 0.75)
        & optimistic_mask
    )
    if boundary_mask.any():
        depth_uncertainty = float(
            np.quantile(std_depth_3d[boundary_mask], 0.90)
        )
        spatter_uncertainty = float(
            np.quantile(std_spatter_3d[boundary_mask], 0.90)
        )
        output_resolved = bool(
            depth_uncertainty <= settings.depth_std_tolerance_mm
            and spatter_uncertainty <= settings.spatter_std_tolerance
        )
    else:
        output_resolved = bool(pink_mask.any())

    return BoxAnalysis(
        box=box,
        power_axis=power_axis,
        spot_axis=spot_axis,
        speed_axis=speed_axis,
        points=points,
        mean_depth=mean_depth_3d,
        std_depth=std_depth_3d,
        mean_spatter=mean_spatter_3d,
        std_spatter=std_spatter_3d,
        joint_probability=probability_3d,
        pink_mask=pink_mask,
        optimistic_mask=optimistic_mask,
        next_boxes=next_boxes,
        resolved=input_resolved and output_resolved and bool(next_boxes),
        max_probability=float(probability_3d.max()),
        optimistic_count=int(optimistic_mask.sum()),
        pink_count=int(pink_mask.sum()),
    )


def box_from_bounds(
    bounds: np.ndarray,
    branch_id: str,
    parent_id: str | None,
) -> SearchBox:
    """Build a search box from a ``(3, 2)`` bounds array."""

    return SearchBox(
        power_min_w=float(bounds[0, 0]),
        power_max_w=float(bounds[0, 1]),
        spot_min_um=float(bounds[1, 0]),
        spot_max_um=float(bounds[1, 1]),
        speed_min_mm_s=float(bounds[2, 0]),
        speed_max_mm_s=float(bounds[2, 1]),
        branch_id=branch_id,
        parent_id=parent_id,
    )


def clip_shifted_bounds(
    center: np.ndarray,
    widths: np.ndarray,
    domain: SearchBox,
) -> np.ndarray:
    """Clip a fixed-size box to the domain while preserving its width."""

    domain_bounds = domain.as_bounds()
    widths = np.minimum(widths, domain.widths())
    lower = center - widths / 2.0
    upper = center + widths / 2.0
    for axis in range(3):
        if lower[axis] < domain_bounds[axis, 0]:
            upper[axis] += domain_bounds[axis, 0] - lower[axis]
            lower[axis] = domain_bounds[axis, 0]
        if upper[axis] > domain_bounds[axis, 1]:
            lower[axis] -= upper[axis] - domain_bounds[axis, 1]
            upper[axis] = domain_bounds[axis, 1]
        lower[axis] = max(lower[axis], domain_bounds[axis, 0])
        upper[axis] = min(upper[axis], domain_bounds[axis, 1])
    return np.column_stack([lower, upper])


def shift_box_at_parent_boundary(
    child: SearchBox,
    parent: SearchBox,
    domain: SearchBox,
    settings: SearchSettings,
) -> tuple[SearchBox, bool]:
    """Shift and expand a child that touches a parent-search boundary."""

    child_bounds = child.as_bounds()
    parent_bounds = parent.as_bounds()
    domain_bounds = domain.as_bounds()
    cell = parent.widths() / float(settings.prediction_points_per_axis - 1)
    lower = child_bounds[:, 0].copy()
    upper = child_bounds[:, 1].copy()
    touched = False

    for axis in range(3):
        low_touch = lower[axis] <= parent_bounds[axis, 0] + 1.05 * cell[axis]
        high_touch = upper[axis] >= parent_bounds[axis, 1] - 1.05 * cell[axis]
        delta = settings.boundary_shift_fraction * parent.widths()[axis]
        can_shift_low = parent_bounds[axis, 0] > domain_bounds[axis, 0]
        can_shift_high = parent_bounds[axis, 1] < domain_bounds[axis, 1]
        if low_touch and high_touch:
            if can_shift_low:
                lower[axis] -= 0.5 * delta
                touched = True
            if can_shift_high:
                upper[axis] += 0.5 * delta
                touched = True
            continue
        if low_touch and can_shift_low:
            lower[axis] -= delta
            upper[axis] -= 0.35 * delta
            touched = True
        if high_touch and can_shift_high:
            lower[axis] += 0.35 * delta
            upper[axis] += delta
            touched = True

    bounds = np.column_stack(
        [
            np.maximum(lower, domain_bounds[:, 0]),
            np.minimum(upper, domain_bounds[:, 1]),
        ]
    )
    return (
        box_from_bounds(bounds, child.branch_id, child.parent_id),
        touched,
    )


def expand_box_for_uncertainty(
    child: SearchBox,
    parent: SearchBox,
    domain: SearchBox,
    settings: SearchSettings,
) -> SearchBox:
    """Symmetrically re-expand a branch while GP uncertainty is high."""

    bounds = child.as_bounds()
    center = bounds.mean(axis=1)
    widths = (
        child.widths()
        + settings.uncertainty_expand_fraction * parent.widths()
    )
    expanded = clip_shifted_bounds(center, widths, domain)
    return box_from_bounds(expanded, child.branch_id, child.parent_id)


def translate_box_toward_probability_peak(
    child: SearchBox,
    analysis: BoxAnalysis,
    domain: SearchBox,
    settings: SearchSettings,
) -> SearchBox:
    """Move a stagnant branch toward its highest joint-probability point."""

    peak_index = int(np.argmax(analysis.joint_probability))
    peak = analysis.points[peak_index]
    bounds = child.as_bounds()
    center = bounds.mean(axis=1)
    shifted_center = center + settings.translation_fraction * (peak - center)
    shifted = clip_shifted_bounds(shifted_center, child.widths(), domain)
    return box_from_bounds(shifted, child.branch_id, child.parent_id)


def analysis_has_high_uncertainty(
    analysis: BoxAnalysis,
    settings: SearchSettings,
) -> bool:
    """Check GP uncertainty in the current optimistic candidate region."""

    mask = analysis.optimistic_mask
    if not mask.any():
        return False
    depth_q = float(np.quantile(analysis.std_depth[mask], 0.90))
    spatter_q = float(np.quantile(analysis.std_spatter[mask], 0.90))
    return bool(
        depth_q > settings.depth_std_tolerance_mm
        or spatter_q > settings.spatter_std_tolerance
    )


def make_global_probe_box(
    global_analysis: BoxAnalysis,
    existing_boxes: list[SearchBox],
    domain: SearchBox,
    step: int,
    settings: SearchSettings,
) -> SearchBox:
    """Create a domain-wide confirmation box outside retained branches."""

    probability = global_analysis.joint_probability.ravel()
    depth_uncertainty = global_analysis.std_depth.ravel()
    spatter_uncertainty = global_analysis.std_spatter.ravel()
    uncertainty = (
        depth_uncertainty / max(settings.depth_std_tolerance_mm, 1.0e-9)
        + spatter_uncertainty / max(settings.spatter_std_tolerance, 1.0e-9)
    )
    uncertainty /= max(float(uncertainty.max()), 1.0e-9)
    score = probability + 0.20 * uncertainty

    points = global_analysis.points
    outside = np.ones(len(points), dtype=bool)
    for box in existing_boxes:
        bounds = box.as_bounds()
        outside &= ~np.all(
            (points >= bounds[:, 0]) & (points <= bounds[:, 1]),
            axis=1,
        )
    if outside.any():
        score = np.where(outside, score, -np.inf)

    center = points[int(np.argmax(score))]
    widths = settings.global_probe_width_fraction * domain.widths()
    bounds = clip_shifted_bounds(center, widths, domain)
    return box_from_bounds(bounds, f"G{step}", "GLOBAL")


def choose_rule_based_update(
    analyses: list[BoxAnalysis],
    domain: SearchBox,
    settings: SearchSettings,
    step: int,
    ancestor_boxes: list[SearchBox] | None,
    backtracks_used: int,
    global_reexplorations_used: int,
    boundary_shifts_used: int,
    uncertainty_reexpansions_used: int,
    probability_peak_shifts_used: int,
    global_analysis: BoxAnalysis | None,
    require_global_audit: bool,
) -> SearchUpdate:
    """Select shrink, shift, expansion, translation or recovery by rules."""

    candidates = [
        child for analysis in analyses for child in analysis.next_boxes
    ]
    any_optimistic = any(
        analysis.optimistic_count > 0 for analysis in analyses
    )

    if not any_optimistic:
        if (
            ancestor_boxes
            and backtracks_used < settings.max_backtracks
        ):
            return SearchUpdate(
                boxes=ancestor_boxes,
                action="backtrack",
                reason="No optimistic points remained; restored parent scope.",
                backtracked=True,
            )
        if (
            global_analysis is not None
            and global_reexplorations_used
            < settings.max_global_reexplorations
        ):
            probe = make_global_probe_box(
                global_analysis,
                [],
                domain,
                step,
                settings,
            )
            return SearchUpdate(
                boxes=[probe],
                action="global_reexploration",
                reason="No local candidate remained; opened a global probe.",
                global_reexploration=True,
            )
        return SearchUpdate(
            boxes=[],
            action="no_feasible_region",
            reason="No optimistic points remained after recovery rules.",
        )

    periodic_due = (
        step % settings.global_reexplore_interval == 0
        and global_reexplorations_used
        < settings.max_global_reexplorations
    )
    if (
        (periodic_due or require_global_audit)
        and global_analysis is not None
        and candidates
    ):
        probe = make_global_probe_box(
            global_analysis,
            candidates,
            domain,
            step,
            settings,
        )
        retained = candidates[: max(0, settings.max_branches - 1)]
        return SearchUpdate(
            boxes=retained + [probe],
            action="periodic_global_reexploration",
            reason="Added a global probe before accepting local convergence.",
            global_reexploration=True,
        )

    shifted: list[SearchBox] = []
    boundary_touched = False
    for analysis in analyses:
        for child in analysis.next_boxes:
            updated, touched = shift_box_at_parent_boundary(
                child,
                analysis.box,
                domain,
                settings,
            )
            shifted.append(updated)
            boundary_touched |= touched
    if (
        boundary_touched
        and boundary_shifts_used < settings.max_boundary_shift_steps
    ):
        return SearchUpdate(
            boxes=shifted[: settings.max_branches],
            action="boundary_shift_expand",
            reason="Candidate touched a parent boundary; shifted outward.",
        )

    if (
        uncertainty_reexpansions_used
        < settings.max_uncertainty_reexpansions
        and any(
            analysis_has_high_uncertainty(analysis, settings)
            for analysis in analyses
        )
    ):
        expanded = [
            expand_box_for_uncertainty(
                child,
                analysis.box,
                domain,
                settings,
            )
            for analysis in analyses
            for child in analysis.next_boxes
        ]
        return SearchUpdate(
            boxes=expanded[: settings.max_branches],
            action="uncertainty_reexpand",
            reason="Boundary uncertainty exceeded tolerance.",
        )

    stagnant = any(
        box_volume(child)
        / max(box_volume(analysis.box), 1.0e-12)
        >= settings.stagnation_volume_ratio
        for analysis in analyses
        for child in analysis.next_boxes
    )
    if (
        stagnant
        and probability_peak_shifts_used
        < settings.max_probability_peak_shifts
    ):
        translated = [
            translate_box_toward_probability_peak(
                child,
                analysis,
                domain,
                settings,
            )
            for analysis in analyses
            for child in analysis.next_boxes
        ]
        return SearchUpdate(
            boxes=translated[: settings.max_branches],
            action="probability_peak_shift",
            reason="Range reduction stalled; shifted toward probability peak.",
        )

    return SearchUpdate(
        boxes=candidates[: settings.max_branches],
        action="local_shrink",
        reason="Candidate is internal and sufficiently resolved.",
    )


def joint_feasible_probability_at_point(
    point: np.ndarray,
    depth_gp: GaussianProcessRegressor,
    spatter_gp: GaussianProcessRegressor,
    domain: SearchBox,
    settings: SearchSettings,
) -> float:
    """Evaluate the two-constraint joint probability at one bridge point."""

    x = normalize_inputs(np.asarray(point, dtype=float)[None, :], domain)
    depth_mean, depth_std = depth_gp.predict(x, return_std=True)
    spatter_mean, spatter_std = spatter_gp.predict(x, return_std=True)
    depth_sigma = max(float(depth_std[0]), 0.03)
    spatter_sigma = max(float(spatter_std[0]), 0.15)
    depth_probability = norm.cdf(
        (float(depth_mean[0]) - settings.depth_min_mm) / depth_sigma
    )
    spatter_probability = norm.cdf(
        (
            settings.spatter_gp_boundary - float(spatter_mean[0])
        )
        / spatter_sigma
    )
    return float(depth_probability * spatter_probability)


def box_pair_merge_metrics(
    first: SearchBox,
    second: SearchBox,
    settings: SearchSettings,
) -> dict[str, object]:
    """Calculate overlap, envelope inflation and nearest bridge point."""

    first_bounds = first.as_bounds()
    second_bounds = second.as_bounds()
    intersection_lower = np.maximum(
        first_bounds[:, 0],
        second_bounds[:, 0],
    )
    intersection_upper = np.minimum(
        first_bounds[:, 1],
        second_bounds[:, 1],
    )
    intersection_widths = np.maximum(
        intersection_upper - intersection_lower,
        0.0,
    )
    intersection_volume = float(np.prod(intersection_widths))
    first_volume = box_volume(first)
    second_volume = box_volume(second)
    smaller_volume = max(min(first_volume, second_volume), 1.0e-12)
    overlap_ratio = intersection_volume / smaller_volume

    envelope_lower = np.minimum(
        first_bounds[:, 0],
        second_bounds[:, 0],
    )
    envelope_upper = np.maximum(
        first_bounds[:, 1],
        second_bounds[:, 1],
    )
    envelope_bounds = np.column_stack([envelope_lower, envelope_upper])
    envelope_volume = float(np.prod(envelope_upper - envelope_lower))
    union_volume = max(
        first_volume + second_volume - intersection_volume,
        1.0e-12,
    )
    inflation = envelope_volume / union_volume

    gap = np.maximum(
        np.maximum(
            first_bounds[:, 0] - second_bounds[:, 1],
            second_bounds[:, 0] - first_bounds[:, 1],
        ),
        0.0,
    )
    grid_gap = np.maximum(
        np.minimum(first.widths(), second.widths())
        / max(settings.coarse_points_per_axis - 1, 1),
        np.array(
            [
                settings.power_resolution_w,
                settings.spot_resolution_um,
                settings.speed_resolution_mm_s,
            ],
            dtype=float,
        ),
    )
    near_grid_neighbor = bool(
        np.all(gap <= settings.box_merge_max_gap_cells * grid_gap)
    )

    bridge = np.empty(3, dtype=float)
    for axis in range(3):
        if intersection_lower[axis] <= intersection_upper[axis]:
            bridge[axis] = (
                intersection_lower[axis] + intersection_upper[axis]
            ) / 2.0
        elif first_bounds[axis, 1] < second_bounds[axis, 0]:
            bridge[axis] = (
                first_bounds[axis, 1] + second_bounds[axis, 0]
            ) / 2.0
        else:
            bridge[axis] = (
                second_bounds[axis, 1] + first_bounds[axis, 0]
            ) / 2.0

    return {
        "overlap_ratio": float(overlap_ratio),
        "envelope_inflation": float(inflation),
        "near_grid_neighbor": near_grid_neighbor,
        "bridge_point": bridge,
        "envelope_bounds": envelope_bounds,
        "intersection_volume": intersection_volume,
    }


def merge_overlapping_boxes(
    boxes: list[SearchBox],
    depth_gp: GaussianProcessRegressor,
    spatter_gp: GaussianProcessRegressor,
    domain: SearchBox,
    settings: SearchSettings,
    step: int,
) -> tuple[list[SearchBox], list[dict[str, object]]]:
    """Iteratively merge redundant overlapping or connected search boxes."""

    merged_boxes = list(boxes)
    origins: dict[str, list[str]] = {
        box.branch_id: [box.branch_id] for box in merged_boxes
    }
    events: list[dict[str, object]] = []
    merge_index = 1

    while True:
        selected_pair: tuple[int, int, dict[str, object], float] | None = None
        for first_index in range(len(merged_boxes)):
            for second_index in range(first_index + 1, len(merged_boxes)):
                first = merged_boxes[first_index]
                second = merged_boxes[second_index]
                metrics = box_pair_merge_metrics(first, second, settings)
                overlap_ratio = float(metrics["overlap_ratio"])
                inflation = float(metrics["envelope_inflation"])
                containment = (
                    overlap_ratio >= settings.box_merge_containment_ratio
                )
                substantial_overlap = (
                    overlap_ratio >= settings.box_merge_overlap_ratio
                )
                nearby = bool(metrics["near_grid_neighbor"])
                if (
                    not containment
                    and not substantial_overlap
                    and not nearby
                ):
                    continue
                if inflation > settings.box_merge_max_inflation:
                    continue

                bridge_probability = joint_feasible_probability_at_point(
                    np.asarray(metrics["bridge_point"], dtype=float),
                    depth_gp,
                    spatter_gp,
                    domain,
                    settings,
                )
                probability_connected = (
                    bridge_probability >= settings.feasible_probability
                )
                if not containment and not probability_connected:
                    continue

                selected_pair = (
                    first_index,
                    second_index,
                    metrics,
                    bridge_probability,
                )
                break
            if selected_pair is not None:
                break

        if selected_pair is None:
            break

        first_index, second_index, metrics, bridge_probability = selected_pair
        first = merged_boxes[first_index]
        second = merged_boxes[second_index]
        merged_id = f"M{step}.{merge_index}"
        merged_from = (
            origins.get(first.branch_id, [first.branch_id])
            + origins.get(second.branch_id, [second.branch_id])
        )
        merged_box = box_from_bounds(
            np.asarray(metrics["envelope_bounds"], dtype=float),
            merged_id,
            f"{first.branch_id}|{second.branch_id}",
        )
        events.append(
            {
                "merged_box": merged_id,
                "merged_from": merged_from,
                "overlap_ratio": float(metrics["overlap_ratio"]),
                "envelope_inflation": float(
                    metrics["envelope_inflation"]
                ),
                "near_grid_neighbor": bool(
                    metrics["near_grid_neighbor"]
                ),
                "bridge_probability": bridge_probability,
                "result_box": box_to_dict(merged_box),
            }
        )

        merged_boxes.pop(second_index)
        merged_boxes.pop(first_index)
        merged_boxes.append(merged_box)
        origins[merged_id] = merged_from
        merge_index += 1

    return merged_boxes, events


def draw_box(
    ax: plt.Axes,
    box: SearchBox,
    color: str,
    linewidth: float = 1.8,
    alpha: float = 1.0,
) -> None:
    x0, x1 = box.power_min_w / 1000.0, box.power_max_w / 1000.0
    y0, y1 = box.spot_min_um, box.spot_max_um
    z0, z1 = box.speed_min_mm_s, box.speed_max_mm_s
    vertices = {
        (0, 0, 0): (x0, y0, z0),
        (1, 0, 0): (x1, y0, z0),
        (0, 1, 0): (x0, y1, z0),
        (1, 1, 0): (x1, y1, z0),
        (0, 0, 1): (x0, y0, z1),
        (1, 0, 1): (x1, y0, z1),
        (0, 1, 1): (x0, y1, z1),
        (1, 1, 1): (x1, y1, z1),
    }
    for key, start in vertices.items():
        for axis in range(3):
            if key[axis] == 0:
                neighbor = list(key)
                neighbor[axis] = 1
                end = vertices[tuple(neighbor)]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )


def draw_wire_grid(
    ax: plt.Axes,
    box: SearchBox,
    points_per_axis: int,
    phase: int,
    color: str = "black",
    linewidth: float = 0.65,
    alpha: float = 0.48,
) -> None:
    fractions = grid_fractions(points_per_axis, phase)
    xs = (
        box.power_min_w
        + fractions * (box.power_max_w - box.power_min_w)
    ) / 1000.0
    ys = box.spot_min_um + fractions * (
        box.spot_max_um - box.spot_min_um
    )
    zs = box.speed_min_mm_s + fractions * (
        box.speed_max_mm_s - box.speed_min_mm_s
    )
    for y in ys:
        for z in zs:
            ax.plot(xs, np.full_like(xs, y), np.full_like(xs, z),
                    color=color, linewidth=linewidth, alpha=alpha)
    for x in xs:
        for z in zs:
            ax.plot(np.full_like(ys, x), ys, np.full_like(ys, z),
                    color=color, linewidth=linewidth, alpha=alpha)
    for x in xs:
        for y in ys:
            ax.plot(np.full_like(zs, x), np.full_like(zs, y), zs,
                    color=color, linewidth=linewidth, alpha=alpha)
    draw_box(
        ax,
        box,
        color=color,
        linewidth=max(1.5, 2.0 * linewidth),
        alpha=min(1.0, alpha + 0.30),
    )


def plot_pink_regions(
    ax: plt.Axes,
    analyses: list[BoxAnalysis],
    settings: SearchSettings,
    rng: np.random.Generator,
    max_points: int = 4500,
) -> None:
    all_points: list[np.ndarray] = []
    for analysis in analyses:
        mask = analysis.pink_mask.ravel()
        if mask.any():
            all_points.append(analysis.points[mask])
    if not all_points:
        return
    points = np.vstack(all_points)
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
    ax.scatter(
        points[:, 0] / 1000.0,
        points[:, 1],
        points[:, 2],
        color=PINK,
        s=22,
        alpha=settings.pink_alpha,
        linewidths=0,
        depthshade=False,
    )


def measured_constraint_margin(
    frame: pd.DataFrame,
    settings: SearchSettings,
) -> np.ndarray:
    """Combine the two measured constraints into one signed color quantity.

    The minimum normalized margin is positive only when both constraints pass.
    """

    depth_margin = (
        frame["penetration_depth_mm"].to_numpy(dtype=float)
        - settings.depth_min_mm
    ) / max(settings.depth_min_mm, 0.25)
    spatter_margin = (
        settings.spatter_gp_boundary
        - frame["spatter_level_0_9"].to_numpy(dtype=float)
    ) / max(settings.spatter_gp_boundary, 1.0)
    return np.clip(np.minimum(depth_margin, spatter_margin), -1.0, 1.0)


def rows_inside_boxes(
    frame: pd.DataFrame,
    boxes: list[SearchBox],
) -> pd.DataFrame:
    if frame.empty or not boxes:
        return frame.iloc[0:0].copy()
    keep = np.zeros(len(frame), dtype=bool)
    power = frame["laser_power_w"].to_numpy(dtype=float)
    spot = frame["spot_diameter_um"].to_numpy(dtype=float)
    speed = frame["scan_speed_mm_s"].to_numpy(dtype=float)
    for box in boxes:
        keep |= (
            (power >= box.power_min_w - 1.0e-7)
            & (power <= box.power_max_w + 1.0e-7)
            & (spot >= box.spot_min_um - 1.0e-7)
            & (spot <= box.spot_max_um + 1.0e-7)
            & (speed >= box.speed_min_mm_s - 1.0e-7)
            & (speed <= box.speed_max_mm_s + 1.0e-7)
        )
    return frame.loc[keep].copy()


def rows_at_requested_points(
    frame: pd.DataFrame,
    requested_points: np.ndarray,
) -> pd.DataFrame:
    if frame.empty or len(requested_points) == 0:
        return frame.iloc[0:0].copy()
    requested_keys = {point_key(point) for point in requested_points}
    keep = [
        point_key((row.laser_power_w, row.spot_diameter_um,
                   row.scan_speed_mm_s)) in requested_keys
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[np.asarray(keep, dtype=bool)].copy()


def plot_measured_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    output_column: str,
    norm: Normalize,
    marker_size: float = 40.0,
) -> None:
    if frame.empty:
        return
    ax.scatter(
        frame["laser_power_w"] / 1000.0,
        frame["spot_diameter_um"],
        frame["scan_speed_mm_s"],
        c=frame[output_column].to_numpy(dtype=float),
        cmap="jet",
        norm=norm,
        s=marker_size,
        edgecolor="black",
        linewidth=0.28,
        alpha=0.96,
        depthshade=False,
    )


def axis_limits(boxes: list[SearchBox]) -> tuple[tuple[float, float], ...]:
    if not boxes:
        return ((0.1, 6.0), (50.0, 300.0), (10.0, 1000.0))
    power_min = min(box.power_min_w for box in boxes) / 1000.0
    power_max = max(box.power_max_w for box in boxes) / 1000.0
    spot_min = min(box.spot_min_um for box in boxes)
    spot_max = max(box.spot_max_um for box in boxes)
    speed_min = min(box.speed_min_mm_s for box in boxes)
    speed_max = max(box.speed_max_mm_s for box in boxes)

    def padded(low: float, high: float) -> tuple[float, float]:
        width = max(high - low, 1.0e-9)
        return low - 0.04 * width, high + 0.04 * width

    return padded(power_min, power_max), padded(spot_min, spot_max), padded(
        speed_min, speed_max
    )


def style_axis(
    ax: plt.Axes,
    boxes: list[SearchBox],
    title: str,
    step: int,
    frame: pd.DataFrame,
    settings: SearchSettings,
) -> None:
    limits = axis_limits(boxes)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_xlabel("Laser power [kW]", labelpad=8)
    ax.set_ylabel("1/e² spot diameter [µm]", labelpad=8)
    ax.set_zlabel("Scan speed [mm/s]", labelpad=8)
    ax.set_title(title, fontsize=12, pad=12)
    ax.view_init(elev=24.0, azim=-57.0)
    ax.set_box_aspect((1.15, 0.90, 1.0))
    ax.grid(True, alpha=0.18)
    query = (
        f"Query: depth ≥ {settings.depth_min_mm:g} mm, "
        f"spatter < {settings.spatter_max_exclusive}"
    )
    ax.text2D(
        0.02,
        0.97,
        f"{query}\nStep {step} | cumulative experiments: {len(frame)}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
    )


def draw_stage(
    ax: plt.Axes,
    stage: int,
    step: int,
    active_boxes: list[SearchBox],
    next_boxes: list[SearchBox],
    analyses: list[BoxAnalysis],
    frame: pd.DataFrame,
    requested_points: np.ndarray,
    grid_points_per_axis: int,
    grid_phase: int,
    settings: SearchSettings,
    rng: np.random.Generator,
) -> None:
    if stage == 1:
        for box in active_boxes:
            draw_wire_grid(ax, box, grid_points_per_axis, grid_phase)
        if len(requested_points):
            ax.scatter(
                requested_points[:, 0] / 1000.0,
                requested_points[:, 1],
                requested_points[:, 2],
                color="black",
                s=22,
                alpha=0.82,
                depthshade=False,
            )
        title = "1. Coarse experimental grid"
        display_boxes = active_boxes
    elif stage == 2:
        for box in active_boxes:
            draw_box(ax, box, color="black", linewidth=1.0, alpha=0.45)
        plot_pink_regions(ax, analyses, settings, rng)
        measured_feasible = (
            (frame["penetration_depth_mm"] >= settings.depth_min_mm)
            & (
                frame["spatter_level_0_9"]
                < settings.spatter_max_exclusive
            )
        )
        infeasible = frame.loc[~measured_feasible]
        feasible = frame.loc[measured_feasible]
        ax.scatter(
            infeasible["laser_power_w"] / 1000.0,
            infeasible["spot_diameter_um"],
            infeasible["scan_speed_mm_s"],
            color="#424242",
            s=11,
            alpha=0.40,
            depthshade=False,
        )
        if not feasible.empty:
            ax.scatter(
                feasible["laser_power_w"] / 1000.0,
                feasible["spot_diameter_um"],
                feasible["scan_speed_mm_s"],
                color="#00a651",
                edgecolor="white",
                linewidth=0.25,
                s=28,
                alpha=0.95,
                depthshade=False,
            )
        title = "2. GP feasible region (90% transparent pink)"
        display_boxes = active_boxes
    else:
        plot_pink_regions(ax, analyses, settings, rng)
        for box in active_boxes:
            draw_box(ax, box, color="#616161", linewidth=0.8, alpha=0.35)
        for index, box in enumerate(next_boxes):
            color = BRANCH_COLORS[index % len(BRANCH_COLORS)]
            draw_box(ax, box, color=color, linewidth=2.7, alpha=0.95)
            ax.text(
                box.power_min_w / 1000.0,
                box.spot_min_um,
                box.speed_max_mm_s,
                box.branch_id,
                color=color,
                fontsize=9,
                weight="bold",
            )
        title = "3. Selected branch boxes for next step"
        display_boxes = active_boxes + next_boxes

    style_axis(
        ax,
        display_boxes or active_boxes,
        title,
        step,
        frame,
        settings,
    )


def render_step_figures(
    output_dir: Path,
    step: int,
    active_boxes: list[SearchBox],
    next_boxes: list[SearchBox],
    analyses: list[BoxAnalysis],
    frame: pd.DataFrame,
    requested_points: np.ndarray,
    grid_points_per_axis: int,
    grid_phase: int,
    settings: SearchSettings,
) -> tuple[list[Path], Path]:
    """Save three individual stage plots and one combined step summary."""

    rng = np.random.default_rng(settings.random_seed + step)
    individual_paths: list[Path] = []

    for stage in (1, 2, 3):
        fig = plt.figure(figsize=(7.4, 6.3), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        draw_stage(
            ax,
            stage,
            step,
            active_boxes,
            next_boxes,
            analyses,
            frame,
            requested_points,
            grid_points_per_axis,
            grid_phase,
            settings,
            rng,
        )
        path = output_dir / f"step_{step:02d}_{stage}.png"
        fig.savefig(path, dpi=145, facecolor="white")
        plt.close(fig)
        individual_paths.append(path)

    summary = plt.figure(figsize=(19.2, 6.2), constrained_layout=True)
    for stage in (1, 2, 3):
        ax = summary.add_subplot(1, 3, stage, projection="3d")
        draw_stage(
            ax,
            stage,
            step,
            active_boxes,
            next_boxes,
            analyses,
            frame,
            requested_points,
            grid_points_per_axis,
            grid_phase,
            settings,
            rng,
        )
    summary.suptitle(
        "Adaptive coarse-to-fine laser-welding condition search",
        fontsize=15,
    )
    summary_path = output_dir / f"step_{step:02d}_summary.png"
    summary.savefig(
        summary_path,
        dpi=135,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(summary)
    return individual_paths, summary_path


def box_volume(box: SearchBox) -> float:
    return float(np.prod(np.maximum(box.widths(), 0.0)))


def retained_volume_percent(
    boxes: list[SearchBox],
    domain: SearchBox,
) -> float:
    if not boxes:
        return 0.0
    # Branch components are normally disjoint. The sum is intentionally used
    # here because it communicates total experimental search volume.
    return 100.0 * sum(box_volume(box) for box in boxes) / box_volume(domain)


def style_transition_axis(
    ax: plt.Axes,
    domain: SearchBox,
) -> None:
    """Use one fixed coordinate system for every GIF frame."""

    limits = axis_limits([domain])
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_xlabel("Laser power [kW]", labelpad=10)
    ax.set_ylabel("1/e² spot diameter [µm]", labelpad=10)
    ax.set_zlabel("Scan speed [mm/s]", labelpad=10)
    ax.view_init(elev=24.0, azim=-57.0)
    ax.set_box_aspect((1.15, 0.90, 1.0))
    ax.grid(True, alpha=0.12)


def create_output_figure(
    title: str,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create the paired penetration-depth and spatter 3D charts."""

    fig = plt.figure(figsize=(17.0, 7.2), constrained_layout=True)
    axes = (
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    )
    fig.suptitle(title, fontsize=16, weight="bold")
    return fig, axes


def add_output_colorbars(
    fig: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes],
) -> None:
    """Add one output scale to each chart without a separate legend."""

    specs = (
        (DEPTH_NORM, "Penetration depth [mm]", None),
        (SPATTER_NORM, "Spatter level [0–9]", np.arange(10)),
    )
    for ax, (norm, label, ticks) in zip(axes, specs):
        colorbar_map = plt.cm.ScalarMappable(norm=norm, cmap="jet")
        colorbar_map.set_array([])
        colorbar = fig.colorbar(
            colorbar_map,
            ax=ax,
            shrink=0.70,
            pad=0.07,
            ticks=ticks,
        )
        colorbar.set_label(label, fontsize=9)


def save_figure_atomic(
    fig: plt.Figure,
    path: Path,
    dpi: int = 145,
    max_attempts: int = 3,
) -> None:
    """Save a verified PNG and atomically publish it at ``path``."""

    temporary = path.with_name(f"{path.stem}.tmp.png")
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            fig.savefig(
                temporary,
                dpi=dpi,
                facecolor="white",
                format="png",
            )
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return
        except (OSError, SyntaxError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
    raise OSError(f"Could not write a complete PNG: {path}") from last_error


def render_initial_grid_frame(
    output_dir: Path,
    domain: SearchBox,
    grid_points_per_axis: int,
    settings: SearchSettings,
) -> tuple[Path, int]:
    """Render the initial black search grid before any experiments."""

    fig, axes = create_output_figure("Step 0  |  Action 2/2")
    for ax in axes:
        draw_wire_grid(
            ax,
            domain,
            grid_points_per_axis,
            phase=0,
            color="black",
            linewidth=0.72,
            alpha=0.52,
        )
        style_transition_axis(ax, domain)
    add_output_colorbars(fig, axes)
    path = output_dir / "animation_step_00_initial_grid.png"
    save_figure_atomic(fig, path)
    plt.close(fig)
    return path, 1400


def render_empty_space_frame(
    output_dir: Path,
    domain: SearchBox,
    settings: SearchSettings,
) -> tuple[Path, int]:
    """Render the empty parameter space before the initial grid appears."""

    fig, axes = create_output_figure("Step 0  |  Action 1/2")
    for ax in axes:
        style_transition_axis(ax, domain)
    add_output_colorbars(fig, axes)
    path = output_dir / "animation_start_empty_space.png"
    save_figure_atomic(fig, path)
    plt.close(fig)
    return path, 1200


def render_transition_frames(
    output_dir: Path,
    step: int,
    domain: SearchBox,
    active_boxes: list[SearchBox],
    next_boxes: list[SearchBox],
    analyses: list[BoxAnalysis],
    frame: pd.DataFrame,
    requested_points: np.ndarray,
    grid_points_per_axis: int,
    grid_phase: int,
    settings: SearchSettings,
    terminal_status: str | None,
) -> tuple[list[Path], list[int]]:
    """Render four paired-output frames for one search step."""

    paths: list[Path] = []
    durations = [1200, 1650, 1250, 950]
    output_specs = (
        ("penetration_depth_mm", DEPTH_NORM),
        ("spatter_level_0_9", SPATTER_NORM),
    )
    current_rows = rows_at_requested_points(frame, requested_points)

    for action_number in range(1, 5):
        fig, axes = create_output_figure(
            f"Step {step}  |  Action {action_number}/4"
        )
        for ax, (output_column, norm) in zip(axes, output_specs):
            if action_number == 1:
                for box in active_boxes:
                    draw_wire_grid(
                        ax,
                        box,
                        grid_points_per_axis,
                        grid_phase,
                        color="black",
                        linewidth=0.72,
                        alpha=0.52,
                    )
                plot_measured_points(
                    ax,
                    current_rows,
                    output_column,
                    norm,
                    marker_size=48.0,
                )

            elif action_number == 2:
                for box in active_boxes:
                    draw_wire_grid(
                        ax,
                        box,
                        grid_points_per_axis,
                        grid_phase,
                        color="black",
                        linewidth=0.60,
                        alpha=0.34,
                    )
                plot_pink_regions(
                    ax,
                    analyses,
                    settings,
                    np.random.default_rng(
                        settings.random_seed + 100 * step
                    ),
                    max_points=6500,
                )
                plot_measured_points(
                    ax,
                    current_rows,
                    output_column,
                    norm,
                    marker_size=39.0,
                )

            elif action_number == 3:
                # Keep the joint GP probability region visible while the
                # measured points are cleared and both output grids are recut.
                plot_pink_regions(
                    ax,
                    analyses,
                    settings,
                    np.random.default_rng(
                        settings.random_seed + 100 * step
                    ),
                    max_points=6500,
                )
                for box in next_boxes:
                    draw_wire_grid(
                        ax,
                        box,
                        grid_points_per_axis,
                        grid_phase + 1,
                        color=PINK,
                        linewidth=1.20,
                        alpha=0.82,
                    )
                    ax.text(
                        box.power_min_w / 1000.0,
                        box.spot_min_um,
                        box.speed_max_mm_s,
                        box.branch_id,
                        color="#c2185b",
                        fontsize=10,
                        weight="bold",
                    )
                if not next_boxes:
                    ax.text2D(
                        0.50,
                        0.50,
                        "No promising region remains",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=16,
                        color="#c62828",
                        weight="bold",
                    )

            else:
                # The pink distribution is cleared when both recut grids
                # become the next active black grids.
                for box in next_boxes:
                    draw_wire_grid(
                        ax,
                        box,
                        grid_points_per_axis,
                        grid_phase + 1,
                        color="black",
                        linewidth=0.78,
                        alpha=0.56,
                    )

            style_transition_axis(ax, domain)

        add_output_colorbars(fig, axes)
        path = (
            output_dir
            / f"animation_step_{step:02d}_action_{action_number:02d}.png"
        )
        save_figure_atomic(fig, path)
        plt.close(fig)
        paths.append(path)

    if terminal_status in {"converged", "no_feasible_region"}:
        durations[-1] = 1900
    return paths, durations


def create_gif(
    summary_paths: list[Path],
    output_path: Path,
    duration_ms: int | list[int] = 1500,
) -> None:
    images: list[Image.Image] = []
    for path in summary_paths:
        image = Image.open(path).convert("RGB")
        images.append(image)
    if not images:
        raise ValueError("No summary images were generated.")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for image in images:
        image.close()


def box_to_dict(box: SearchBox) -> dict[str, object]:
    result = asdict(box)
    result["laser_power_kw"] = [
        box.power_min_w / 1000.0,
        box.power_max_w / 1000.0,
    ]
    result["spot_diameter_um"] = [box.spot_min_um, box.spot_max_um]
    result["scan_speed_mm_s"] = [
        box.speed_min_mm_s,
        box.speed_max_mm_s,
    ]
    return result


def run_search(
    output_dir: Path,
    settings: SearchSettings,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    domain = SearchBox(
        power_min_w=100.0,
        power_max_w=6000.0,
        spot_min_um=50.0,
        spot_max_um=300.0,
        speed_min_mm_s=10.0,
        speed_max_mm_s=1000.0,
        branch_id="B1",
    )
    active_boxes = [domain]
    experiment_cache: dict[
        tuple[float, float, float],
        dict[str, object],
    ] = {}
    empty_path, empty_duration = render_empty_space_frame(
        output_dir,
        domain,
        settings,
    )
    initial_path, initial_duration = render_initial_grid_frame(
        output_dir,
        domain,
        settings.coarse_points_per_axis,
        settings,
    )
    animation_paths: list[Path] = [empty_path, initial_path]
    animation_durations: list[int] = [empty_duration, initial_duration]
    step_history: list[dict[str, object]] = []
    status = "max_steps_inconclusive"
    final_boxes: list[SearchBox] = []
    force_dense_grid = False
    search_stack: list[list[SearchBox]] = []
    backtracks_used = 0
    global_reexplorations_used = 0
    boundary_shifts_used = 0
    uncertainty_reexpansions_used = 0
    probability_peak_shifts_used = 0

    for step in range(1, settings.max_steps + 1):
        grid_points = (
            settings.fallback_points_per_axis
            if force_dense_grid
            else settings.coarse_points_per_axis
        )
        force_dense_grid = False

        requested_points, new_experiments = run_experiments(
            active_boxes,
            step,
            grid_points,
            step - 1,
            experiment_cache,
        )
        frame = experiment_frame(experiment_cache)
        depth_gp, spatter_gp = fit_gaussian_processes(
            frame,
            domain,
            settings.random_seed,
        )
        analyses = [
            analyze_box(
                box,
                depth_gp,
                spatter_gp,
                domain,
                settings,
            )
            for box in active_boxes
        ]

        any_optimistic = any(
            analysis.optimistic_count > 0 for analysis in analyses
        )
        all_resolved = bool(analyses) and all(
            analysis.resolved for analysis in analyses
        )

        require_global_audit = bool(
            all_resolved
            and step >= settings.min_steps_before_convergence
            and global_reexplorations_used == 0
        )
        periodic_due = bool(
            step % settings.global_reexplore_interval == 0
            and global_reexplorations_used
            < settings.max_global_reexplorations
        )
        recovery_needs_global = bool(
            not any_optimistic
            and (
                not search_stack
                or backtracks_used >= settings.max_backtracks
            )
            and global_reexplorations_used
            < settings.max_global_reexplorations
        )
        global_analysis = (
            analyze_box(
                domain,
                depth_gp,
                spatter_gp,
                domain,
                settings,
            )
            if (
                require_global_audit
                or periodic_due
                or recovery_needs_global
            )
            else None
        )
        update = choose_rule_based_update(
            analyses=analyses,
            domain=domain,
            settings=settings,
            step=step,
            ancestor_boxes=search_stack[-1] if search_stack else None,
            backtracks_used=backtracks_used,
            global_reexplorations_used=global_reexplorations_used,
            boundary_shifts_used=boundary_shifts_used,
            uncertainty_reexpansions_used=uncertainty_reexpansions_used,
            probability_peak_shifts_used=probability_peak_shifts_used,
            global_analysis=global_analysis,
            require_global_audit=require_global_audit,
        )
        next_boxes, merge_events = merge_overlapping_boxes(
            update.boxes,
            depth_gp,
            spatter_gp,
            domain,
            settings,
            step,
        )
        update.boxes = next_boxes

        if update.backtracked:
            backtracks_used += 1
        if update.global_reexploration:
            global_reexplorations_used += 1
        if update.action == "boundary_shift_expand":
            boundary_shifts_used += 1
        elif update.action == "uncertainty_reexpand":
            uncertainty_reexpansions_used += 1
        elif update.action == "probability_peak_shift":
            probability_peak_shifts_used += 1

        if (
            all_resolved
            and step >= settings.min_steps_before_convergence
            and next_boxes
            and update.action == "local_shrink"
            and global_reexplorations_used > 0
            and not merge_events
        ):
            status = "converged"
            final_boxes = next_boxes
        elif update.action == "no_feasible_region":
            if step >= settings.min_steps_before_no_feasible:
                status = "no_feasible_region"
                final_boxes = []
            else:
                update = SearchUpdate(
                    boxes=active_boxes,
                    action="dense_confirmation",
                    reason="Confirming candidate absence with a denser grid.",
                )
                next_boxes = active_boxes
                force_dense_grid = True

        terminal_status = (
            status
            if status in {"converged", "no_feasible_region"}
            else None
        )
        transition_paths, transition_durations = render_transition_frames(
            output_dir,
            step,
            domain,
            active_boxes,
            next_boxes,
            analyses,
            frame,
            requested_points,
            grid_points,
            step - 1,
            settings,
            terminal_status,
        )
        animation_paths.extend(transition_paths)
        animation_durations.extend(transition_durations)

        step_history.append(
            {
                "step": step,
                "active_branches": len(active_boxes),
                "next_branches": len(next_boxes),
                "new_experiments": new_experiments,
                "cumulative_experiments": len(frame),
                "max_feasible_probability": max(
                    analysis.max_probability for analysis in analyses
                ),
                "pink_prediction_points": sum(
                    analysis.pink_count for analysis in analyses
                ),
                "optimistic_prediction_points": sum(
                    analysis.optimistic_count for analysis in analyses
                ),
                "all_boundaries_resolved": all_resolved,
                "search_action": update.action,
                "search_action_reason": update.reason,
                "backtracks_used": backtracks_used,
                "global_reexplorations_used": global_reexplorations_used,
                "boundary_shifts_used": boundary_shifts_used,
                "uncertainty_reexpansions_used": (
                    uncertainty_reexpansions_used
                ),
                "probability_peak_shifts_used": (
                    probability_peak_shifts_used
                ),
                "box_merge_count": len(merge_events),
                "box_merge_events": merge_events,
                "active_boxes": [
                    box_to_dict(box) for box in active_boxes
                ],
                "next_boxes": [box_to_dict(box) for box in next_boxes],
            }
        )

        if status in {"converged", "no_feasible_region"}:
            break
        if not next_boxes:
            status = "no_feasible_region"
            final_boxes = []
            break
        if update.backtracked:
            if search_stack:
                search_stack.pop()
        else:
            search_stack.append(active_boxes)
        active_boxes = next_boxes
        final_boxes = next_boxes

    gif_path = output_dir / "laser_welding_adaptive_search.gif"
    create_gif(animation_paths, gif_path, duration_ms=animation_durations)

    experiments_path = output_dir / "adaptive_search_experiments.csv"
    final_frame = experiment_frame(experiment_cache)
    final_frame.to_csv(experiments_path, index=False, float_format="%.8g")

    summary = {
        "status": status,
        "query": {
            "penetration_depth_min_mm": settings.depth_min_mm,
            "spatter_level_max_exclusive": settings.spatter_max_exclusive,
        },
        "total_experiments": int(len(final_frame)),
        "completed_steps": int(len(step_history)),
        "final_condition_boxes": [
            box_to_dict(box) for box in final_boxes
        ],
        "settings": asdict(settings),
        "step_history": step_history,
        "gif_file": gif_path.name,
        "animation_frames": [path.name for path in animation_paths],
        "animation_frame_durations_ms": animation_durations,
        "experiments_file": experiments_path.name,
    }
    summary_path = output_dir / "adaptive_search_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Coarse-to-fine multi-branch Gaussian-process condition search."
        )
    )
    parser.add_argument("--depth-min", type=float, default=2.0)
    parser.add_argument("--spatter-max", type=int, default=6)
    parser.add_argument("--coarse-points", type=int, default=3)
    parser.add_argument("--prediction-points", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("adaptive_search_output"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    print("Final boxes:")
    for box in summary["final_condition_boxes"]:
        print(
            "  "
            f"{box['branch_id']}: "
            f"P={box['laser_power_kw']} kW, "
            f"d={box['spot_diameter_um']} µm, "
            f"v={box['scan_speed_mm_s']} mm/s"
        )
    print((args.output_dir / "laser_welding_adaptive_search.gif").resolve())


if __name__ == "__main__":
    main()
