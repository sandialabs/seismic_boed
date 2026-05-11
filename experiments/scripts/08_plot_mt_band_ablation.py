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


BANDS = [
    ("low", "Low Magnitude (Mw 0.5-2.0)"),
    ("high", "High Magnitude (Mw 4.5-5.0)"),
]


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
            "and their deltas."
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


def maybe_subsample_training(
    x_train: np.ndarray, y_train: np.ndarray, max_train_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[0] <= int(max_train_points):
        return x_train, y_train

    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(x_train.shape[0], size=int(max_train_points), replace=False))
    return x_train[keep], y_train[keep]


def fit_latlon_surface(
    theta_data: np.ndarray,
    response: np.ndarray,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    grid_size: int,
    max_train_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GaussianProcessRegressor]:
    x_train = theta_data[:, :2]
    y_train = response
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


def build_band_data(
    label: str,
    title: str,
    with_mt_path: Path,
    without_mt_path: Path,
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

    lat_grid, lon_grid, with_grid, with_model = fit_latlon_surface(
        theta_with,
        ig_with,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=grid_size,
        max_train_points=max_train_points,
    )
    _, _, without_grid, without_model = fit_latlon_surface(
        theta_without,
        ig_without,
        lat_range=lat_range,
        lon_range=lon_range,
        grid_size=grid_size,
        max_train_points=max_train_points,
    )

    delta_grid = with_grid - without_grid

    return {
        "label": label,
        "title": title,
        "theta_data": theta_with,
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
        "mag_min": float(np.min(theta_with[:, 3])),
        "mag_max": float(np.max(theta_with[:, 3])),
        "depth_min": float(np.min(theta_with[:, 2])),
        "depth_max": float(np.max(theta_with[:, 2])),
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
    theta_all = np.vstack([theta_low, theta_high])
    lat = theta_all[:, 0]
    lon = theta_all[:, 1]
    lat_pad = max(0.03 * max(lat.max() - lat.min(), 1.0), 1e-6)
    lon_pad = max(0.03 * max(lon.max() - lon.min(), 1.0), 1e-6)
    lat_range = np.array([lat.min() - lat_pad, lat.max() + lat_pad], dtype=float)
    lon_range = np.array([lon.min() - lon_pad, lon.max() + lon_pad], dtype=float)

    band_data = [
        build_band_data(
            "low",
            "Low Magnitude (Mw 0.5-2.0)",
            low_with,
            low_without,
            lat_range=lat_range,
            lon_range=lon_range,
            grid_size=args.grid_size,
            max_train_points=args.max_train_points,
        ),
        build_band_data(
            "high",
            "High Magnitude (Mw 4.5-5.0)",
            high_with,
            high_without,
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
        ax0.set_title(
            f"{band['title']}\nWithout MT, EIG={band['without_eig']:.3f}",
            fontsize=12,
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
        ax1.set_title(
            f"{band['title']}\nWith MT, EIG={band['with_eig']:.3f}",
            fontsize=12,
        )
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
            f"{band['title']}\nWith MT - Without MT = {band['with_eig'] - band['without_eig']:.3f}",
            fontsize=12,
        )
        ax2.set_xlabel("Longitude (deg)")
        ax2.set_ylabel("Latitude (deg)")

        cbar_row = fig.colorbar(pcm1, ax=[ax0, ax1], shrink=0.93)
        cbar_row.set_label("Expected Information Gain")
        cbar_delta = fig.colorbar(pcm2, ax=ax2, shrink=0.93)
        cbar_delta.set_label("With MT - Without MT")

    fig.suptitle(
        "Moment-Tensor Ablation by Magnitude Regime",
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
            f"delta={band['with_eig'] - band['without_eig']:.4f}"
        )


if __name__ == "__main__":
    main()
