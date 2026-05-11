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
        description=(
            "Plot low/high magnitude with-MT vs without-MT information surfaces "
            "using fixed depth/magnitude slices within each band."
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


def format_kernel(model: GaussianProcessRegressor) -> str:
    return str(model.kernel_) if hasattr(model, "kernel_") else str(model.kernel)


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

    lat_grid, lon_grid, with_grid, with_model = fit_surface_with_model(
        theta_with,
        ig_with,
        mask,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=grid_size,
        max_train_points=max_train_points,
    )
    _, _, without_grid, without_model = fit_surface_with_model(
        theta_without,
        ig_without,
        mask,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=grid_size,
        max_train_points=max_train_points,
    )

    delta_grid = with_grid - without_grid

    return {
        "label": label,
        "title": title,
        "sensors": sensors_with,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "with_grid": with_grid,
        "without_grid": without_grid,
        "delta_grid": delta_grid,
        "with_model": with_model,
        "without_model": without_model,
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
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), constrained_layout=True)

    for row_idx, band in enumerate(band_data):
        with_grid = band["with_grid"]
        without_grid = band["without_grid"]
        delta_grid = band["delta_grid"]
        lat_grid = band["lat_grid"]
        lon_grid = band["lon_grid"]
        sensors = band["sensors"]

        shared_vmin = float(min(np.min(with_grid), np.min(without_grid)))
        shared_vmax = float(max(np.max(with_grid), np.max(without_grid)))
        delta_abs = float(np.max(np.abs(delta_grid)))
        if delta_abs <= 0.0:
            delta_abs = 1e-12

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
        ax0.set_title(f"{band['title']}\nWithout MT\n{subtitle}", fontsize=11)
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
        ax1.set_title(f"{band['title']}\nWith MT\n{subtitle}", fontsize=11)
        ax1.set_xlabel("Longitude (deg)")
        ax1.set_ylabel("Latitude (deg)")

        pcm2 = ax2.pcolormesh(
            lon_grid,
            lat_grid,
            delta_grid,
            shading="auto",
            cmap="coolwarm",
            vmin=-delta_abs,
            vmax=delta_abs,
        )
        ax2.scatter(
            sensors[:, 1],
            sensors[:, 0],
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
        )
        ax2.set_title(
            f"{band['title']}\nWith MT - Without MT\n{subtitle}",
            fontsize=11,
        )
        ax2.set_xlabel("Longitude (deg)")
        ax2.set_ylabel("Latitude (deg)")

        cbar_row = fig.colorbar(pcm1, ax=[ax0, ax1], shrink=0.93)
        cbar_row.set_label("Expected Information Gain")
        cbar_delta = fig.colorbar(pcm2, ax=ax2, shrink=0.93)
        cbar_delta.set_label("With MT - Without MT")

    fig.suptitle(
        "Moment-Tensor Ablation by Magnitude Regime\nFixed-Slice Lat/Lon Surfaces",
        fontsize=16,
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
            },
            sort_keys=True,
        )
        band_rows = []
        for band in band_data:
            band_rows.append(
                {
                    "band": band["label"],
                    "band_title": band["title"],
                    "with_eig": band["with_eig"],
                    "without_eig": band["without_eig"],
                    "delta_eig": band["with_eig"] - band["without_eig"],
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
                    "kernel_with": format_kernel(band["with_model"]),
                    "kernel_without": format_kernel(band["without_model"]),
                    "delta_grid_mean": float(np.mean(band["delta_grid"])),
                    "delta_grid_min": float(np.min(band["delta_grid"])),
                    "delta_grid_max": float(np.max(band["delta_grid"])),
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
            f"n={band['n_slice_points']}"
        )


if __name__ == "__main__":
    main()
