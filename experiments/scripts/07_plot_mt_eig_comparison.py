#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot isotropic-only vs MT-marginalized EIG maps from eig_calc outputs."
    )
    parser.add_argument(
        "--isotropic",
        required=True,
        type=Path,
        help="Verbose eig_calc .npz output for isotropic-only run.",
    )
    parser.add_argument(
        "--marginalized",
        required=True,
        type=Path,
        help="Verbose eig_calc .npz output for MT-marginalized run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PNG path for the comparison figure.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path.",
    )
    parser.add_argument(
        "--depth-slice",
        type=float,
        default=10.0,
        help="Depth slice to visualize in km.",
    )
    parser.add_argument(
        "--magnitude-slice",
        type=float,
        default=5.0,
        help="Magnitude slice to visualize in Mw.",
    )
    parser.add_argument(
        "--depth-tol",
        type=float,
        default=5.0,
        help="Initial depth tolerance in km for selecting nearby training samples.",
    )
    parser.add_argument(
        "--mag-tol",
        type=float,
        default=0.5,
        help="Initial magnitude tolerance in Mw for selecting nearby training samples.",
    )
    parser.add_argument(
        "--min-slice-points",
        type=int,
        default=150,
        help="Minimum number of sampled events to include near the slice after adaptive widening.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=120,
        help="Grid resolution for the map in each horizontal dimension.",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=1200,
        help="Maximum number of slice points to use when fitting each GP surface.",
    )
    return parser.parse_args()


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def mean_ig_per_event(data) -> tuple[np.ndarray, np.ndarray]:
    theta_data = np.asarray(data["theta_data"], dtype=float)
    ig = np.asarray(data["ig"], dtype=float)
    n_events = int(theta_data.shape[0])
    if n_events == 0:
        raise ValueError("theta_data is empty.")
    if ig.size % n_events != 0:
        raise ValueError(
            f"ig size ({ig.size}) is not divisible by theta_data rows ({n_events})."
        )
    return theta_data, ig.reshape(n_events, -1).mean(axis=1)


def infer_lat_lon_bounds(data, theta_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "location_bounds" in data.files:
        try:
            location_bounds = data["location_bounds"]
            if getattr(location_bounds, "shape", None) == ():
                location_bounds = location_bounds.item()

            if hasattr(location_bounds, "sample_bounds"):
                sample_bounds = np.asarray(location_bounds.sample_bounds, dtype=float)
                if sample_bounds.shape == (2, 2):
                    return sample_bounds[0], sample_bounds[1]

            location_bounds = np.asarray(location_bounds, dtype=float)
            if location_bounds.shape == (2, 2):
                return location_bounds[0], location_bounds[1]
        except Exception:
            pass

    if "lat_range" in data.files:
        lon_key = "long_range" if "long_range" in data.files else "lon_range"
        if lon_key in data.files:
            return (
                np.asarray(data["lat_range"], dtype=float),
                np.asarray(data[lon_key], dtype=float),
            )

    lat = theta_data[:, 0]
    lon = theta_data[:, 1]
    lat_pad = max(0.05 * max(lat.max() - lat.min(), 1.0), 1e-6)
    lon_pad = max(0.05 * max(lon.max() - lon.min(), 1.0), 1e-6)
    return (
        np.array([lat.min() - lat_pad, lat.max() + lat_pad], dtype=float),
        np.array([lon.min() - lon_pad, lon.max() + lon_pad], dtype=float),
    )


def adaptive_slice_mask(
    theta_data: np.ndarray,
    depth_slice: float,
    magnitude_slice: float,
    depth_tol: float,
    mag_tol: float,
    min_points: int,
) -> tuple[np.ndarray, float, float]:
    depth_tol_curr = float(depth_tol)
    mag_tol_curr = float(mag_tol)
    max_depth_tol = max(depth_tol_curr, float(theta_data[:, 2].max() - theta_data[:, 2].min()))
    max_mag_tol = max(mag_tol_curr, float(theta_data[:, 3].max() - theta_data[:, 3].min()))

    for _ in range(12):
        mask = (
            (np.abs(theta_data[:, 2] - depth_slice) <= depth_tol_curr)
            & (np.abs(theta_data[:, 3] - magnitude_slice) <= mag_tol_curr)
        )
        if int(np.count_nonzero(mask)) >= int(min_points):
            return mask, depth_tol_curr, mag_tol_curr

        if depth_tol_curr >= max_depth_tol and mag_tol_curr >= max_mag_tol:
            break

        depth_tol_curr = min(max_depth_tol, depth_tol_curr * 1.5)
        mag_tol_curr = min(max_mag_tol, mag_tol_curr * 1.5)

    return mask, depth_tol_curr, mag_tol_curr


def maybe_subsample_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_train_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[0] <= int(max_train_points):
        return x_train, y_train

    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(x_train.shape[0], size=int(max_train_points), replace=False))
    return x_train[keep], y_train[keep]


def fit_surface(
    theta_data: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_selected = int(np.count_nonzero(mask))
    if n_selected < 8:
        raise ValueError(
            f"Need at least 8 slice points to fit a map; only found {n_selected}."
        )

    x_train = theta_data[mask][:, :2]
    y_train = response[mask]

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[0.25, 0.25], length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    model.fit(x_train, y_train)

    lat = np.linspace(float(lat_range[0]), float(lat_range[1]), int(grid_size))
    lon = np.linspace(float(lon_range[0]), float(lon_range[1]), int(grid_size))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_xy = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    pred = model.predict(grid_xy).reshape(lat_grid.shape)
    return lat_grid, lon_grid, pred


def format_kernel(model: GaussianProcessRegressor) -> str:
    return str(model.kernel_) if hasattr(model, "kernel_") else str(model.kernel)


def fit_surface_with_model(
    theta_data: np.ndarray,
    response: np.ndarray,
    mask: np.ndarray,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    grid_size: int,
    max_train_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GaussianProcessRegressor]:
    n_selected = int(np.count_nonzero(mask))
    if n_selected < 8:
        raise ValueError(
            f"Need at least 8 slice points to fit a map; only found {n_selected}."
        )

    x_train = theta_data[mask][:, :2]
    y_train = response[mask]
    x_train, y_train = maybe_subsample_training(x_train, y_train, max_train_points)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[0.25, 0.25], length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    model.fit(x_train, y_train)

    lat = np.linspace(float(lat_range[0]), float(lat_range[1]), int(grid_size))
    lon = np.linspace(float(lon_range[0]), float(lon_range[1]), int(grid_size))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_xy = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    pred = model.predict(grid_xy).reshape(lat_grid.shape)
    return lat_grid, lon_grid, pred, model


def write_summary_csv(
    output_path: Path,
    *,
    run_id: str,
    git_sha: str,
    timestamp: str,
    params_json: str,
    depth_slice: float,
    magnitude_slice: float,
    depth_tol_used: float,
    mag_tol_used: float,
    n_slice_points: int,
    eig_iso_mean: float,
    eig_marg_mean: float,
    delta_mean: float,
    delta_min: float,
    delta_max: float,
    kernel_iso: str,
    kernel_marg: str,
) -> None:
    row = {
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "params": params_json,
        "depth_slice_km": depth_slice,
        "magnitude_slice_mw": magnitude_slice,
        "depth_tol_used_km": depth_tol_used,
        "mag_tol_used_mw": mag_tol_used,
        "n_slice_points": n_slice_points,
        "eig_iso_mean": eig_iso_mean,
        "eig_marg_mean": eig_marg_mean,
        "delta_mean": delta_mean,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "kernel_iso": kernel_iso,
        "kernel_marg": kernel_marg,
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    iso_path = args.isotropic if args.isotropic.is_absolute() else (REPO_ROOT / args.isotropic)
    marg_path = (
        args.marginalized
        if args.marginalized.is_absolute()
        else (REPO_ROOT / args.marginalized)
    )
    output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    summary_csv = None
    if args.summary_csv is not None:
        summary_csv = (
            args.summary_csv
            if args.summary_csv.is_absolute()
            else (REPO_ROOT / args.summary_csv)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)

    iso_data = load_npz(iso_path)
    marg_data = load_npz(marg_path)

    theta_iso, ig_iso = mean_ig_per_event(iso_data)
    theta_marg, ig_marg = mean_ig_per_event(marg_data)

    mag_min = float(np.min(theta_iso[:, 3]))
    mag_max = float(np.max(theta_iso[:, 3]))
    depth_min = float(np.min(theta_iso[:, 2]))
    depth_max = float(np.max(theta_iso[:, 2]))

    if args.magnitude_slice < (mag_min - 1e-9) or args.magnitude_slice > (mag_max + 1e-9):
        raise ValueError(
            "Requested magnitude slice is outside the sampled event range for this run. "
            f"Requested Mw={args.magnitude_slice:.3f}, available range=[{mag_min:.3f}, {mag_max:.3f}]."
        )
    if args.depth_slice < (depth_min - 1e-9) or args.depth_slice > (depth_max + 1e-9):
        raise ValueError(
            "Requested depth slice is outside the sampled event range for this run. "
            f"Requested depth={args.depth_slice:.3f} km, available range=[{depth_min:.3f}, {depth_max:.3f}] km."
        )

    if theta_iso.shape != theta_marg.shape or not np.allclose(theta_iso, theta_marg):
        raise ValueError(
            "theta_data mismatch between isotropic and marginalized outputs; "
            "rerun both jobs from the same inputs before comparing."
        )

    sensors_iso = np.asarray(iso_data["sensors"], dtype=float)
    sensors_marg = np.asarray(marg_data["sensors"], dtype=float)
    if sensors_iso.shape != sensors_marg.shape or not np.allclose(sensors_iso, sensors_marg):
        raise ValueError(
            "Sensor network mismatch between isotropic and marginalized outputs."
        )

    lat_range, lon_range = infer_lat_lon_bounds(iso_data, theta_iso)
    mask, depth_tol_used, mag_tol_used = adaptive_slice_mask(
        theta_iso,
        depth_slice=args.depth_slice,
        magnitude_slice=args.magnitude_slice,
        depth_tol=args.depth_tol,
        mag_tol=args.mag_tol,
        min_points=args.min_slice_points,
    )
    n_slice_points = int(np.count_nonzero(mask))
    if n_slice_points == 0:
        raise ValueError(
            "No event samples were found near the requested slice. "
            f"Requested depth={args.depth_slice:.3f} km, Mw={args.magnitude_slice:.3f}; "
            f"available depth range=[{depth_min:.3f}, {depth_max:.3f}] km, "
            f"available Mw range=[{mag_min:.3f}, {mag_max:.3f}]."
        )
    print(
        "[slice]",
        f"selected_points={n_slice_points}",
        f"depth_tol_used={depth_tol_used:.3f}",
        f"mag_tol_used={mag_tol_used:.3f}",
        f"max_train_points={args.max_train_points}",
    )

    lat_grid, lon_grid, eig_iso_grid, iso_model = fit_surface_with_model(
        theta_iso,
        ig_iso,
        mask,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=args.grid_size,
        max_train_points=args.max_train_points,
    )
    _, _, eig_marg_grid, marg_model = fit_surface_with_model(
        theta_marg,
        ig_marg,
        mask,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=args.grid_size,
        max_train_points=args.max_train_points,
    )
    diff_grid = eig_marg_grid - eig_iso_grid

    shared_vmin = float(min(np.min(eig_iso_grid), np.min(eig_marg_grid)))
    shared_vmax = float(max(np.max(eig_iso_grid), np.max(eig_marg_grid)))
    diff_abs = float(np.max(np.abs(diff_grid)))
    if diff_abs <= 0.0:
        diff_abs = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), constrained_layout=True)

    pcm0 = axes[0].pcolormesh(
        lon_grid,
        lat_grid,
        eig_iso_grid,
        shading="auto",
        cmap="viridis",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[0].scatter(
        sensors_iso[:, 1],
        sensors_iso[:, 0],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
    )
    axes[0].set_title("Isotropic Source Only")
    axes[0].set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")

    pcm1 = axes[1].pcolormesh(
        lon_grid,
        lat_grid,
        eig_marg_grid,
        shading="auto",
        cmap="viridis",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[1].scatter(
        sensors_iso[:, 1],
        sensors_iso[:, 0],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
    )
    axes[1].set_title("Source-Shape Uncertainty Included")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")

    pcm2 = axes[2].pcolormesh(
        lon_grid,
        lat_grid,
        diff_grid,
        shading="auto",
        cmap="coolwarm",
        vmin=-diff_abs,
        vmax=diff_abs,
    )
    axes[2].scatter(
        sensors_iso[:, 1],
        sensors_iso[:, 0],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
    )
    axes[2].set_title("Change in Information Gain")
    axes[2].set_xlabel("Longitude (deg)")
    axes[2].set_ylabel("Latitude (deg)")

    cbar_shared = fig.colorbar(pcm1, ax=axes[:2], shrink=0.93)
    cbar_shared.set_label("Expected Information Gain")
    cbar_diff = fig.colorbar(pcm2, ax=axes[2], shrink=0.93)
    cbar_diff.set_label("MT marginalized - isotropic")

    fig.suptitle(
        "Moment-Tensor Comparison at Fixed Event Slice\n"
        f"Depth = {args.depth_slice:.1f} km, Mw = {args.magnitude_slice:.2f}, "
        f"slice points = {n_slice_points}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    if summary_csv is not None:
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = f"mt_eig_comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        params_json = json.dumps(
            {
                "isotropic_npz": str(iso_path),
                "marginalized_npz": str(marg_path),
                "depth_slice_km": args.depth_slice,
                "magnitude_slice_mw": args.magnitude_slice,
                "depth_tol_initial_km": args.depth_tol,
                "mag_tol_initial_mw": args.mag_tol,
                "depth_tol_used_km": depth_tol_used,
                "mag_tol_used_mw": mag_tol_used,
                "min_slice_points": args.min_slice_points,
                "grid_size": args.grid_size,
            },
            sort_keys=True,
        )
        write_summary_csv(
            summary_csv,
            run_id=run_id,
            git_sha=get_git_sha(REPO_ROOT),
            timestamp=timestamp,
            params_json=params_json,
            depth_slice=args.depth_slice,
            magnitude_slice=args.magnitude_slice,
            depth_tol_used=depth_tol_used,
            mag_tol_used=mag_tol_used,
            n_slice_points=n_slice_points,
            eig_iso_mean=float(np.mean(eig_iso_grid)),
            eig_marg_mean=float(np.mean(eig_marg_grid)),
            delta_mean=float(np.mean(diff_grid)),
            delta_min=float(np.min(diff_grid)),
            delta_max=float(np.max(diff_grid)),
            kernel_iso=format_kernel(iso_model),
            kernel_marg=format_kernel(marg_model),
        )

    print(f"Wrote figure: {output_path}")
    if summary_csv is not None:
        print(f"Wrote summary CSV: {summary_csv}")
    print(
        "Slice summary:",
        f"depth={args.depth_slice:.2f} km,",
        f"mw={args.magnitude_slice:.2f},",
        f"n_points={n_slice_points},",
        f"depth_tol_used={depth_tol_used:.3f},",
        f"mag_tol_used={mag_tol_used:.3f}",
    )


if __name__ == "__main__":
    main()
