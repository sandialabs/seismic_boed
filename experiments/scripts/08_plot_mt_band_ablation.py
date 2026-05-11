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
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        description=(
            "Plot low/high magnitude with-MT vs without-MT information surfaces "
            "using azimuth-averaged radial fits within fixed depth/magnitude slices."
        )
    )
    parser.add_argument("--low-with-mt", required=True, type=Path)
    parser.add_argument("--low-without-mt", required=True, type=Path)
    parser.add_argument("--high-with-mt", required=True, type=Path)
    parser.add_argument("--high-without-mt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--max-train-points", type=int, default=1200)
    parser.add_argument("--depth-slice", type=float, default=10.0)
    parser.add_argument("--depth-tol", type=float, default=2.5)
    parser.add_argument("--low-magnitude-slice", type=float, default=1.5)
    parser.add_argument("--low-mag-tol", type=float, default=0.2)
    parser.add_argument("--high-magnitude-slice", type=float, default=4.75)
    parser.add_argument("--high-mag-tol", type=float, default=0.1)
    parser.add_argument("--min-slice-points", type=int, default=150)
    parser.add_argument("--n-profile-bins", type=int, default=10)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


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

    lat = theta_data[:, 0]
    lon = theta_data[:, 1]
    lat_pad = max(0.05 * max(lat.max() - lat.min(), 1.0), 1e-6)
    lon_pad = max(0.05 * max(lon.max() - lon.min(), 1.0), 1e-6)
    return (
        np.array([lat.min() - lat_pad, lat.max() + lat_pad], dtype=float),
        np.array([lon.min() - lon_pad, lon.max() + lon_pad], dtype=float),
    )


def validate_matched_runs(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    sensors_a: np.ndarray,
    sensors_b: np.ndarray,
    label: str,
) -> None:
    if theta_a.shape != theta_b.shape or not np.allclose(theta_a, theta_b):
        raise ValueError(f"{label}: theta_data mismatch between with-MT and without-MT runs.")
    if sensors_a.shape != sensors_b.shape or not np.allclose(sensors_a, sensors_b):
        raise ValueError(f"{label}: sensor mismatch between with-MT and without-MT runs.")


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

    mask = np.zeros(theta_data.shape[0], dtype=bool)
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
    x_train: np.ndarray, y_train: np.ndarray, max_train_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[0] <= int(max_train_points):
        return x_train, y_train

    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(x_train.shape[0], size=int(max_train_points), replace=False))
    return x_train[keep], y_train[keep]


def latlon_offsets_km(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(center_lat))
    lat_km = (lat_arr - float(center_lat)) * km_per_deg_lat
    lon_km = (lon_arr - float(center_lon)) * km_per_deg_lon
    return lat_km, lon_km


def compute_radius_km(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    lat_km, lon_km = latlon_offsets_km(lat, lon, center_lat, center_lon)
    return np.sqrt(lat_km**2 + lon_km**2)


def fit_radial_model(
    radius_km: np.ndarray,
    response: np.ndarray,
    max_train_points: int,
) -> tuple[Pipeline, dict]:
    radius_km = np.asarray(radius_km, dtype=float).reshape(-1, 1)
    response = np.asarray(response, dtype=float)
    if radius_km.shape[0] < 8:
        raise ValueError(
            f"Need at least 8 slice points to fit a radial model; only found {radius_km.shape[0]}."
        )

    x_train, y_train = maybe_subsample_training(radius_km, response, max_train_points)
    x_fit, x_test, y_fit, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    kernel = Matern(
        length_scale=np.ones(1, dtype=float),
        length_scale_bounds=(1e-2, 1e3),
        nu=1.5,
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("gp", gp),
        ]
    )
    model.fit(x_fit, y_fit)

    y_pred_test = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred_test)) if y_test.size > 1 else np.nan,
        "mse": float(mean_squared_error(y_test, y_pred_test)) if y_test.size > 0 else np.nan,
        "n_train": int(x_fit.shape[0]),
        "n_test": int(x_test.shape[0]),
    }
    return model, metrics


def radial_surface_from_model(
    model: Pipeline,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    center_lat: float,
    center_lon: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(float(lat_range[0]), float(lat_range[1]), int(grid_size))
    lon = np.linspace(float(lon_range[0]), float(lon_range[1]), int(grid_size))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_radius = compute_radius_km(
        lat_grid.ravel(),
        lon_grid.ravel(),
        center_lat=center_lat,
        center_lon=center_lon,
    ).reshape(-1, 1)
    pred_grid = model.predict(grid_radius).reshape(lat_grid.shape)

    radius_curve_km = np.linspace(0.0, float(np.max(grid_radius)), 240)
    radial_curve = model.predict(radius_curve_km.reshape(-1, 1))
    return lat_grid, lon_grid, pred_grid, radius_curve_km, radial_curve


def summarize_radial_profile(
    radius_km: np.ndarray,
    with_values: np.ndarray,
    without_values: np.ndarray,
    n_bins: int,
) -> dict:
    radius_km = np.asarray(radius_km, dtype=float)
    with_values = np.asarray(with_values, dtype=float)
    without_values = np.asarray(without_values, dtype=float)

    quantiles = np.linspace(0.0, 1.0, int(max(n_bins, 2)) + 1)
    edges = np.unique(np.quantile(radius_km, quantiles))
    if edges.size < 3:
        edges = np.linspace(float(radius_km.min()), float(radius_km.max()) + 1e-9, 3)

    radius_mid = []
    count = []
    with_mean = []
    with_sem = []
    without_mean = []
    without_sem = []

    for idx in range(edges.size - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == edges.size - 2:
            keep = (radius_km >= lo) & (radius_km <= hi)
        else:
            keep = (radius_km >= lo) & (radius_km < hi)
        if not np.any(keep):
            continue

        n_keep = int(np.count_nonzero(keep))
        radius_mid.append(float(np.median(radius_km[keep])))
        count.append(n_keep)
        with_mean.append(float(np.mean(with_values[keep])))
        without_mean.append(float(np.mean(without_values[keep])))

        if n_keep > 1:
            with_sem.append(float(np.std(with_values[keep], ddof=1) / np.sqrt(n_keep)))
            without_sem.append(float(np.std(without_values[keep], ddof=1) / np.sqrt(n_keep)))
        else:
            with_sem.append(0.0)
            without_sem.append(0.0)

    with_mean_arr = np.asarray(with_mean, dtype=float)
    without_mean_arr = np.asarray(without_mean, dtype=float)
    return {
        "radius_km": np.asarray(radius_mid, dtype=float),
        "count": np.asarray(count, dtype=int),
        "with_mean": with_mean_arr,
        "with_sem": np.asarray(with_sem, dtype=float),
        "without_mean": without_mean_arr,
        "without_sem": np.asarray(without_sem, dtype=float),
        "delta_mean": with_mean_arr - without_mean_arr,
    }


def format_kernel(model: Pipeline) -> str:
    gp = model.named_steps["gp"]
    return str(gp.kernel_) if hasattr(gp, "kernel_") else str(gp.kernel)


def build_band_slice_data(
    *,
    label: str,
    title: str,
    with_mt_path: Path,
    without_mt_path: Path,
    depth_slice: float,
    magnitude_slice: float,
    depth_tol: float,
    mag_tol: float,
    min_slice_points: int,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    grid_size: int,
    max_train_points: int,
    n_profile_bins: int,
) -> dict:
    with_mt = load_npz(with_mt_path)
    without_mt = load_npz(without_mt_path)

    theta_with, ig_with = mean_ig_per_event(with_mt)
    theta_without, ig_without = mean_ig_per_event(without_mt)
    sensors_with = np.asarray(with_mt["sensors"], dtype=float)
    sensors_without = np.asarray(without_mt["sensors"], dtype=float)
    validate_matched_runs(theta_with, theta_without, sensors_with, sensors_without, label)

    mag_min = float(np.min(theta_with[:, 3]))
    mag_max = float(np.max(theta_with[:, 3]))
    depth_min = float(np.min(theta_with[:, 2]))
    depth_max = float(np.max(theta_with[:, 2]))
    if magnitude_slice < (mag_min - 1e-9) or magnitude_slice > (mag_max + 1e-9):
        raise ValueError(
            f"{label}: requested magnitude slice {magnitude_slice:.3f} is outside "
            f"available range [{mag_min:.3f}, {mag_max:.3f}]."
        )
    if depth_slice < (depth_min - 1e-9) or depth_slice > (depth_max + 1e-9):
        raise ValueError(
            f"{label}: requested depth slice {depth_slice:.3f} is outside "
            f"available range [{depth_min:.3f}, {depth_max:.3f}] km."
        )

    mask, depth_tol_used, mag_tol_used = adaptive_slice_mask(
        theta_with,
        depth_slice=depth_slice,
        magnitude_slice=magnitude_slice,
        depth_tol=depth_tol,
        mag_tol=mag_tol,
        min_points=min_slice_points,
    )
    n_slice_points = int(np.count_nonzero(mask))
    if n_slice_points == 0:
        raise ValueError(
            f"{label}: no event samples found near requested slice depth={depth_slice:.3f} km, "
            f"Mw={magnitude_slice:.3f}."
        )

    center_lat = float(np.mean(sensors_with[:, 0]))
    center_lon = float(np.mean(sensors_with[:, 1]))
    radius_km = compute_radius_km(
        theta_with[mask, 0],
        theta_with[mask, 1],
        center_lat=center_lat,
        center_lon=center_lon,
    )

    with_model, with_metrics = fit_radial_model(
        radius_km=radius_km,
        response=ig_with[mask],
        max_train_points=max_train_points,
    )
    without_model, without_metrics = fit_radial_model(
        radius_km=radius_km,
        response=ig_without[mask],
        max_train_points=max_train_points,
    )

    (
        lat_grid,
        lon_grid,
        with_grid,
        radius_curve_km,
        with_curve,
    ) = radial_surface_from_model(
        with_model,
        lat_range=lat_range,
        lon_range=lon_range,
        center_lat=center_lat,
        center_lon=center_lon,
        grid_size=grid_size,
    )
    _, _, without_grid, _, without_curve = radial_surface_from_model(
        without_model,
        lat_range=lat_range,
        lon_range=lon_range,
        center_lat=center_lat,
        center_lon=center_lon,
        grid_size=grid_size,
    )

    profile = summarize_radial_profile(
        radius_km=radius_km,
        with_values=ig_with[mask],
        without_values=ig_without[mask],
        n_bins=n_profile_bins,
    )

    return {
        "label": label,
        "title": title,
        "sensors": sensors_with,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "with_grid": with_grid,
        "without_grid": without_grid,
        "with_model": with_model,
        "without_model": without_model,
        "with_metrics": with_metrics,
        "without_metrics": without_metrics,
        "with_eig": float(with_mt["eig"]),
        "without_eig": float(without_mt["eig"]),
        "with_seig": float(with_mt["seig"]),
        "without_seig": float(without_mt["seig"]),
        "with_miness": float(with_mt["miness"]),
        "without_miness": float(without_mt["miness"]),
        "mag_min": mag_min,
        "mag_max": mag_max,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_slice": float(depth_slice),
        "magnitude_slice": float(magnitude_slice),
        "depth_tol_used": float(depth_tol_used),
        "mag_tol_used": float(mag_tol_used),
        "n_slice_points": n_slice_points,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "radius_curve_km": radius_curve_km,
        "with_curve": with_curve,
        "without_curve": without_curve,
        "profile": profile,
        "raw_with": np.asarray(ig_with[mask], dtype=float),
        "raw_without": np.asarray(ig_without[mask], dtype=float),
        "slice_with_mean": float(np.mean(ig_with[mask])),
        "slice_without_mean": float(np.mean(ig_without[mask])),
    }


def write_summary_csv(
    output_path: Path,
    *,
    run_id: str,
    git_sha: str,
    timestamp: str,
    params_json: str,
    band_rows: list[dict],
) -> None:
    rows = []
    for row in band_rows:
        rows.append(
            {
                "run_id": run_id,
                "git_sha": git_sha,
                "timestamp": timestamp,
                "params": params_json,
                **row,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    low_with = resolve_path(args.low_with_mt)
    low_without = resolve_path(args.low_without_mt)
    high_with = resolve_path(args.high_with_mt)
    high_without = resolve_path(args.high_without_mt)
    output_path = resolve_path(args.output)
    summary_csv = None if args.summary_csv is None else resolve_path(args.summary_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)

    theta_low = mean_ig_per_event(load_npz(low_with))[0]
    theta_high = mean_ig_per_event(load_npz(high_with))[0]
    low_lat_range, low_lon_range = infer_lat_lon_bounds(load_npz(low_with), theta_low)
    high_lat_range, high_lon_range = infer_lat_lon_bounds(load_npz(high_with), theta_high)
    lat_range = np.array(
        [min(low_lat_range[0], high_lat_range[0]), max(low_lat_range[1], high_lat_range[1])],
        dtype=float,
    )
    lon_range = np.array(
        [min(low_lon_range[0], high_lon_range[0]), max(low_lon_range[1], high_lon_range[1])],
        dtype=float,
    )

    band_data = [
        build_band_slice_data(
            label="low",
            title="Low Magnitude Band (Mw 0.5-2.0)",
            with_mt_path=low_with,
            without_mt_path=low_without,
            depth_slice=args.depth_slice,
            magnitude_slice=args.low_magnitude_slice,
            depth_tol=args.depth_tol,
            mag_tol=args.low_mag_tol,
            min_slice_points=args.min_slice_points,
            lat_range=lat_range,
            lon_range=lon_range,
            grid_size=args.grid_size,
            max_train_points=args.max_train_points,
            n_profile_bins=args.n_profile_bins,
        ),
        build_band_slice_data(
            label="high",
            title="High Magnitude Band (Mw 4.5-5.0)",
            with_mt_path=high_with,
            without_mt_path=high_without,
            depth_slice=args.depth_slice,
            magnitude_slice=args.high_magnitude_slice,
            depth_tol=args.depth_tol,
            mag_tol=args.high_mag_tol,
            min_slice_points=args.min_slice_points,
            lat_range=lat_range,
            lon_range=lon_range,
            grid_size=args.grid_size,
            max_train_points=args.max_train_points,
            n_profile_bins=args.n_profile_bins,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.5), constrained_layout=True)

    for row_idx, band in enumerate(band_data):
        with_grid = band["with_grid"]
        without_grid = band["without_grid"]
        lat_grid = band["lat_grid"]
        lon_grid = band["lon_grid"]
        sensors = band["sensors"]
        profile = band["profile"]

        shared_vmin = float(min(np.min(with_grid), np.min(without_grid)))
        shared_vmax = float(max(np.max(with_grid), np.max(without_grid)))
        run_delta = band["with_eig"] - band["without_eig"]
        run_delta_pct = 100.0 * run_delta / max(abs(band["without_eig"]), 1e-12)
        slice_delta = band["slice_with_mean"] - band["slice_without_mean"]

        ax0, ax1, ax2 = axes[row_idx]
        subtitle = (
            f"Slice: depth={band['depth_slice']:.1f} km, Mw~{band['magnitude_slice']:.2f}, "
            f"n={band['n_slice_points']}"
        )

        pcm0 = ax0.pcolormesh(
            lon_grid,
            lat_grid,
            without_grid,
            shading="auto",
            cmap="viridis",
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        ax0.scatter(
            sensors[:, 1],
            sensors[:, 0],
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
        )
        ax0.scatter(
            band["center_lon"],
            band["center_lat"],
            s=40,
            c="black",
            marker="+",
            linewidths=1.2,
        )
        ax0.set_title(
            f"{band['title']}\nWithout MT (azimuth-averaged)\n"
            f"{subtitle}, run EIG={band['without_eig']:.3f}",
            fontsize=11,
        )
        ax0.set_xlabel("Longitude (deg)")
        ax0.set_ylabel("Latitude (deg)")

        pcm1 = ax1.pcolormesh(
            lon_grid,
            lat_grid,
            with_grid,
            shading="auto",
            cmap="viridis",
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        ax1.scatter(
            sensors[:, 1],
            sensors[:, 0],
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
        )
        ax1.scatter(
            band["center_lon"],
            band["center_lat"],
            s=40,
            c="black",
            marker="+",
            linewidths=1.2,
        )
        ax1.set_title(
            f"{band['title']}\nWith MT (azimuth-averaged)\n"
            f"{subtitle}, run EIG={band['with_eig']:.3f}",
            fontsize=11,
        )
        ax1.set_xlabel("Longitude (deg)")
        ax1.set_ylabel("Latitude (deg)")

        ax2.scatter(
            band["radius_km"],
            band["raw_without"],
            s=10,
            alpha=0.10,
            color="#355C7D",
            label="Without MT points",
        )
        ax2.scatter(
            band["radius_km"],
            band["raw_with"],
            s=10,
            alpha=0.10,
            color="#C06C84",
            label="With MT points",
        )
        ax2.errorbar(
            profile["radius_km"],
            profile["without_mean"],
            yerr=profile["without_sem"],
            fmt="o",
            ms=4,
            lw=1,
            capsize=2,
            color="#355C7D",
            label="Without MT bins",
        )
        ax2.errorbar(
            profile["radius_km"],
            profile["with_mean"],
            yerr=profile["with_sem"],
            fmt="o",
            ms=4,
            lw=1,
            capsize=2,
            color="#C06C84",
            label="With MT bins",
        )
        ax2.plot(
            band["radius_curve_km"],
            band["without_curve"],
            color="#355C7D",
            lw=2,
            label="Without MT fit",
        )
        ax2.plot(
            band["radius_curve_km"],
            band["with_curve"],
            color="#C06C84",
            lw=2,
            label="With MT fit",
        )
        ax2.set_title(
            f"{band['title']}\nObserved Slice EIG vs Distance\n"
            f"slice mean: {band['slice_without_mean']:.3f} -> {band['slice_with_mean']:.3f}",
            fontsize=11,
        )
        ax2.set_xlabel("Distance From Array Center (km)")
        ax2.set_ylabel("Observed Slice EIG")
        ax2.grid(alpha=0.2, linewidth=0.6)
        ax2.legend(frameon=False, fontsize=8, loc="best")
        ax2.text(
            0.02,
            0.98,
            f"run delta: {run_delta:+.3f} ({run_delta_pct:+.1f}%)\n"
            f"slice delta: {slice_delta:+.3f}",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )

        cbar_row = fig.colorbar(pcm1, ax=[ax0, ax1], shrink=0.93)
        cbar_row.set_label("Expected Information Gain")

    fig.suptitle(
        "Moment-Tensor Ablation by Magnitude Regime\n"
        "Circularized surfaces from actual slice values",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.008,
        (
            "Left and middle panels are azimuth-averaged schematics derived from the "
            "actual selected slice values. Right panels show the underlying distance-binned data."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    if summary_csv is not None:
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = f"mt_band_ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        params_json = json.dumps(
            {
                "low_with_mt": str(low_with),
                "low_without_mt": str(low_without),
                "high_with_mt": str(high_with),
                "high_without_mt": str(high_without),
                "grid_size": args.grid_size,
                "max_train_points": args.max_train_points,
                "depth_slice": args.depth_slice,
                "depth_tol": args.depth_tol,
                "low_magnitude_slice": args.low_magnitude_slice,
                "low_mag_tol": args.low_mag_tol,
                "high_magnitude_slice": args.high_magnitude_slice,
                "high_mag_tol": args.high_mag_tol,
                "min_slice_points": args.min_slice_points,
                "n_profile_bins": args.n_profile_bins,
            },
            sort_keys=True,
        )
        band_rows = []
        for band in band_data:
            delta_mean = band["profile"]["delta_mean"]
            peak_idx = int(np.argmax(delta_mean)) if delta_mean.size else 0
            peak_radius = (
                float(band["profile"]["radius_km"][peak_idx]) if delta_mean.size else np.nan
            )
            band_rows.append(
                {
                    "band": band["label"],
                    "band_title": band["title"],
                    "with_eig": band["with_eig"],
                    "without_eig": band["without_eig"],
                    "delta_eig": band["with_eig"] - band["without_eig"],
                    "delta_eig_pct": 100.0
                    * (band["with_eig"] - band["without_eig"])
                    / max(abs(band["without_eig"]), 1e-12),
                    "with_seig": band["with_seig"],
                    "without_seig": band["without_seig"],
                    "with_miness": band["with_miness"],
                    "without_miness": band["without_miness"],
                    "mag_min": band["mag_min"],
                    "mag_max": band["mag_max"],
                    "depth_min": band["depth_min"],
                    "depth_max": band["depth_max"],
                    "depth_slice": band["depth_slice"],
                    "magnitude_slice": band["magnitude_slice"],
                    "depth_tol_used": band["depth_tol_used"],
                    "mag_tol_used": band["mag_tol_used"],
                    "n_slice_points": band["n_slice_points"],
                    "center_lat": band["center_lat"],
                    "center_lon": band["center_lon"],
                    "kernel_with": format_kernel(band["with_model"]),
                    "kernel_without": format_kernel(band["without_model"]),
                    "with_r2": band["with_metrics"]["r2"],
                    "with_mse": band["with_metrics"]["mse"],
                    "with_n_train": band["with_metrics"]["n_train"],
                    "with_n_test": band["with_metrics"]["n_test"],
                    "without_r2": band["without_metrics"]["r2"],
                    "without_mse": band["without_metrics"]["mse"],
                    "without_n_train": band["without_metrics"]["n_train"],
                    "without_n_test": band["without_metrics"]["n_test"],
                    "slice_with_mean": band["slice_with_mean"],
                    "slice_without_mean": band["slice_without_mean"],
                    "slice_delta_mean": band["slice_with_mean"] - band["slice_without_mean"],
                    "profile_delta_peak": float(np.max(delta_mean)) if delta_mean.size else np.nan,
                    "profile_delta_trough": float(np.min(delta_mean)) if delta_mean.size else np.nan,
                    "profile_delta_peak_radius_km": peak_radius,
                }
            )
        write_summary_csv(
            summary_csv,
            run_id=run_id,
            git_sha=get_git_sha(REPO_ROOT),
            timestamp=timestamp,
            params_json=params_json,
            band_rows=band_rows,
        )

    print(f"Wrote figure: {output_path}")
    if summary_csv is not None:
        print(f"Wrote summary CSV: {summary_csv}")
    for band in band_data:
        print(
            f"[{band['label']}] with_eig={band['with_eig']:.4f} "
            f"without_eig={band['without_eig']:.4f} "
            f"delta={band['with_eig'] - band['without_eig']:.4f} "
            f"slice_depth={band['depth_slice']:.2f} "
            f"slice_mw={band['magnitude_slice']:.2f} "
            f"n={band['n_slice_points']} "
            f"slice_delta={band['slice_with_mean'] - band['slice_without_mean']:.4f} "
            f"with_r2={band['with_metrics']['r2']:.3f} "
            f"without_r2={band['without_metrics']['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
