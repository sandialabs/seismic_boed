#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize eig_calc outputs with the original eig_vis slice-fitting workflow."
        )
    )
    parser.add_argument("data_file", type=Path, help="NPZ output from eig_calc.py")
    parser.add_argument(
        "control_file",
        type=Path,
        help="Two-line control file: depth_step,mag_step then depth_tol,mag_tol",
    )
    parser.add_argument(
        "--bounds-file",
        type=Path,
        default=None,
        help="Optional JSON bounds file when the NPZ does not contain plain lat/lon ranges.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("eig_plots"),
        help="Directory where the timestamped plot folder will be created.",
    )
    parser.add_argument("--stepsize", type=int, default=100)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--range-from-data-file",
        type=Path,
        action="append",
        default=[],
        help="Additional NPZ file(s) to include when deriving a shared color scale.",
    )
    parser.add_argument("--range-pad-frac", type=float, default=0.02)
    return parser.parse_args()


def select_training_samples(
    samples,
    targets,
    depth_slice,
    mag_slice,
    depth_tol,
    mag_tol,
):
    depth_low = depth_slice - depth_tol
    depth_high = depth_slice + depth_tol

    mag_low = mag_slice - mag_tol
    mag_high = mag_slice + mag_tol

    mask = [
        ((samples[:, 2] <= depth_high) & (samples[:, 2] >= depth_low))
        & ((samples[:, 3] <= mag_high) & (samples[:, 3] >= mag_low))
    ]

    training_inputs = samples[tuple(mask)]
    training_targets = targets[tuple(mask)]

    return training_inputs, training_targets


def load_control(control_file: Path) -> tuple[float, float, float, float]:
    with open(control_file, "r", encoding="utf-8") as f:
        depth_step, mag_step = np.fromstring(f.readline(), dtype=float, sep=",")
        depth_tol, mag_tol = np.fromstring(f.readline(), dtype=float, sep=",")
    return float(depth_step), float(mag_step), float(depth_tol), float(mag_tol)


def infer_lat_long_range(data, bounds_file: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if "lat_range" in data.files:
        lat_range = np.asarray(data["lat_range"], dtype=float)
    else:
        lat_range = None

    if "long_range" in data.files:
        long_range = np.asarray(data["long_range"], dtype=float)
    elif "lon_range" in data.files:
        long_range = np.asarray(data["lon_range"], dtype=float)
    else:
        long_range = None

    if lat_range is not None and long_range is not None:
        return lat_range, long_range

    if "location_bounds" in data.files:
        try:
            bounds = data["location_bounds"]
            if getattr(bounds, "shape", None) == ():
                bounds = bounds.item()

            if hasattr(bounds, "sample_bounds"):
                sample_bounds = np.asarray(bounds.sample_bounds, dtype=float)
                if sample_bounds.shape == (2, 2):
                    return sample_bounds[0], sample_bounds[1]

            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape == (2, 2):
                return bounds[0], bounds[1]
        except Exception:
            pass

    if bounds_file is not None:
        with open(bounds_file, "r", encoding="utf-8") as f:
            domain = json.load(f)
        return np.asarray(domain["lat_range"], dtype=float), np.asarray(
            domain["lon_range"], dtype=float
        )

    raise ValueError(
        "Could not infer lat/long plotting ranges from the NPZ. Pass --bounds-file."
    )


def infer_color_range(
    data_files: list[Path],
    *,
    explicit_vmin: float | None,
    explicit_vmax: float | None,
    range_pad_frac: float,
) -> tuple[float | None, float | None]:
    if explicit_vmin is not None or explicit_vmax is not None:
        return explicit_vmin, explicit_vmax

    mins = []
    maxs = []
    for data_file in data_files:
        with np.load(data_file, allow_pickle=True) as data:
            inputs = np.asarray(data["theta_data"], dtype=float)
            target = np.asarray(data["ig"], dtype=float).reshape(len(inputs), -1).mean(axis=1)
            mins.append(float(np.min(target)))
            maxs.append(float(np.max(target)))

    if not mins:
        return None, None

    vmin = min(mins)
    vmax = max(maxs)
    span = vmax - vmin
    pad = max(float(range_pad_frac) * span, 1e-9)
    return vmin - pad, vmax + pad


def plot_surface(
    data,
    t0,
    *,
    depth_step,
    mag_step,
    depth_tol,
    mag_tol,
    stepsize,
    output_path,
    bounds_file,
    color_vmin,
    color_vmax,
):
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

    for depth_slice in depth_slices:
        for mag_slice in mag_slices:
            print(f"Generating plot {curr_plot} of {total_plots}: {time.time() - t0}")
            domain[:, 2] = depth_slice
            domain[:, 3] = mag_slice

            training_inputs, training_targets = select_training_samples(
                inputs, target, depth_slice, mag_slice, depth_tol, mag_tol
            )

            if training_inputs.shape[0] == 0:
                print(
                    f"No samples found within {depth_tol} of depth slice and within "
                    f"{mag_tol} of mag slice, skipping this plot"
                )
                curr_plot += 1
                continue

            print(f"Training GP model with {len(training_targets)} samples: {time.time() - t0}")
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
                vmin=color_vmin,
                vmax=color_vmax,
            )
            plt.colorbar(label="Expected Information Gain")

            plt.scatter(
                data["sensors"][:, 0],
                data["sensors"][:, 1],
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
            plt.savefig(save_dir / plotname, dpi=300, bbox_inches="tight")
            plt.close()
            curr_plot += 1

    print(f"Wrote plots to {save_dir}")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(f"Loading data: {t0}")

    depth_step, mag_step, depth_tol, mag_tol = load_control(args.control_file)
    with np.load(args.data_file, allow_pickle=True) as data:
        color_vmin, color_vmax = infer_color_range(
            [args.data_file, *args.range_from_data_file],
            explicit_vmin=args.vmin,
            explicit_vmax=args.vmax,
            range_pad_frac=args.range_pad_frac,
        )
        if color_vmin is not None and color_vmax is not None:
            print(f"Using color range vmin={color_vmin:.4f}, vmax={color_vmax:.4f}")

        plot_surface(
            data,
            t0,
            depth_step=depth_step,
            mag_step=mag_step,
            depth_tol=depth_tol,
            mag_tol=mag_tol,
            stepsize=args.stepsize,
            output_path=args.output_path,
            bounds_file=args.bounds_file,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
        )


if __name__ == "__main__":
    main()
