#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility plotter that follows the old eig_vis.py behavior."
    )
    parser.add_argument("data_file", type=Path)
    parser.add_argument("control_file", type=Path)
    parser.add_argument("--bounds-file", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path("eig_plots"))
    parser.add_argument("--stepsize", type=int, default=100)
    return parser.parse_args()


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path)


def select_training_samples(
    samples: np.ndarray,
    targets: np.ndarray,
    depth_slice: float,
    mag_slice: float,
    depth_tol: float,
    mag_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_low = depth_slice - depth_tol
    depth_high = depth_slice + depth_tol
    mag_low = mag_slice - mag_tol
    mag_high = mag_slice + mag_tol

    mask = (
        (samples[:, 2] <= depth_high)
        & (samples[:, 2] >= depth_low)
        & (samples[:, 3] <= mag_high)
        & (samples[:, 3] >= mag_low)
    )

    training_inputs = samples[mask]
    training_targets = targets[mask]
    return training_inputs, training_targets


def infer_lat_long_range(data, bounds_file: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if "lat_range" in data.files and "long_range" in data.files:
        return np.asarray(data["lat_range"], dtype=float), np.asarray(data["long_range"], dtype=float)

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

    if bounds_file is not None:
        with open(bounds_file, "r", encoding="utf-8") as f:
            domain = json.load(f)
        return np.asarray(domain["lat_range"], dtype=float), np.asarray(domain["lon_range"], dtype=float)

    raise ValueError(
        "Could not infer lat/lon plotting ranges from the npz. Pass --bounds-file."
    )


def load_control(control_file: Path) -> tuple[float, float, float, float]:
    with open(control_file, "r", encoding="utf-8") as f:
        depth_step, mag_step = np.fromstring(f.readline(), dtype=float, sep=",")
        depth_tol, mag_tol = np.fromstring(f.readline(), dtype=float, sep=",")
    return float(depth_step), float(mag_step), float(depth_tol), float(mag_tol)


def plot_surface(
    data,
    t0: float,
    *,
    output_path: Path,
    stepsize: int,
    depth_step: float,
    mag_step: float,
    depth_tol: float,
    mag_tol: float,
    bounds_file: Path | None,
) -> None:
    print(f"Configuring data for plots: {time.time() - t0}")
    target = np.asarray(data["ig"], dtype=float)
    inputs = np.asarray(data["theta_data"], dtype=float)

    lat_range, long_range = infer_lat_long_range(data, bounds_file)
    depth_range = np.asarray(data["depth_range"], dtype=float)
    mag_range = np.asarray(data["mag_range"], dtype=float)

    target = target.reshape(len(inputs), -1).mean(axis=1)

    x = np.linspace(lat_range[0], lat_range[1], stepsize)
    y = np.linspace(long_range[0], long_range[1], stepsize)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T

    domain = np.zeros((stepsize**2, 4))
    domain[:, :2] = xy

    depth_slices = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
    mag_slices = np.arange(mag_range[0], mag_range[1] + mag_step, mag_step)

    now = datetime.now()
    timestamp = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    save_dir = output_path / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    total_plots = len(depth_slices) * len(mag_slices)
    curr_plot = 1

    sensors = np.asarray(data["sensors"], dtype=float)

    for depth_slice in depth_slices:
        for mag_slice in mag_slices:
            print(f"Generating plot {curr_plot} of {total_plots}: {time.time() - t0}")
            domain[:, 2] = depth_slice
            domain[:, 3] = mag_slice

            training_inputs, training_targets = select_training_samples(
                inputs,
                target,
                depth_slice,
                mag_slice,
                depth_tol,
                mag_tol,
            )

            if training_inputs.shape[0] == 0:
                print(
                    f"No samples found within {depth_tol} of depth slice and within "
                    f"{mag_tol} of mag slice, skipping this plot"
                )
                curr_plot += 1
                continue

            print(
                f"Training GP model with {len(training_targets)} samples: {time.time() - t0}"
            )
            model = GPR()
            model.fit(training_inputs, training_targets)

            preds = model.predict(domain)

            plt.figure(figsize=(8.5, 6.5))
            plt.pcolormesh(
                xv,
                yv,
                preds.reshape((stepsize, stepsize)),
                shading="auto",
                cmap="viridis",
            )
            plt.colorbar()
            plt.scatter(
                sensors[:, 0],
                sensors[:, 1],
                marker="o",
                facecolors="none",
                edgecolors="red",
                label="Sensor location",
            )
            plt.xlabel("Latitude")
            plt.ylabel("Longitude")
            plt.title(
                f"Expected Information Gain for events with depth = {depth_slice}, mag = {mag_slice}"
            )
            plt.legend()

            plotname = f"depth-{np.round(depth_slice,3)}_mag-{np.round(mag_slice,3)}.pdf"
            plt.savefig(save_dir / plotname, dpi=300)
            plt.close()
            curr_plot += 1

    print(f"Wrote plots to {save_dir}")


def main() -> None:
    args = parse_args()
    data_file = resolve_path(args.data_file)
    control_file = resolve_path(args.control_file)
    bounds_file = resolve_path(args.bounds_file)
    output_path = resolve_path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading data: {t0}")

    depth_step, mag_step, depth_tol, mag_tol = load_control(control_file)
    data = np.load(data_file, allow_pickle=True)

    plot_surface(
        data,
        t0,
        output_path=output_path,
        stepsize=args.stepsize,
        depth_step=depth_step,
        mag_step=mag_step,
        depth_tol=depth_tol,
        mag_tol=mag_tol,
        bounds_file=bounds_file,
    )


if __name__ == "__main__":
    main()
