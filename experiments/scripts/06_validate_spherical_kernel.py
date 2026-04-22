#!/usr/bin/env python3

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geo_kernels import SphericalRBF, great_circle_distance_deg
from spatial_domain import SpatialDomain, latlon_to_unit, unit_to_latlon


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def destination_point(center_latlon, azimuth_deg, distance_deg):
    center_vec = latlon_to_unit(np.asarray(center_latlon, dtype=float).reshape(1, 2))[0]
    ref_vec = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref_vec, center_vec)) > 0.99:
        ref_vec = np.array([0.0, 1.0, 0.0])

    basis_1 = np.cross(ref_vec, center_vec)
    basis_1 = basis_1 / np.linalg.norm(basis_1)
    basis_2 = np.cross(center_vec, basis_1)

    azimuth = np.deg2rad(azimuth_deg)
    distance = np.deg2rad(distance_deg)
    tangent = np.cos(azimuth) * basis_1 + np.sin(azimuth) * basis_2
    point_vec = np.cos(distance) * center_vec + np.sin(distance) * tangent
    return unit_to_latlon(point_vec)[0]


def true_geodesic_field(points_latlon, cap_center):
    cap_center = np.asarray(cap_center, dtype=float).reshape(1, 2)
    center_a = destination_point(cap_center[0], azimuth_deg=40.0, distance_deg=5.0)
    center_b = destination_point(cap_center[0], azimuth_deg=215.0, distance_deg=8.0)

    dist_a = great_circle_distance_deg(points_latlon, center_a.reshape(1, 2)).ravel()
    dist_b = great_circle_distance_deg(points_latlon, center_b.reshape(1, 2)).ravel()
    dist_cap = great_circle_distance_deg(points_latlon, cap_center).ravel()

    bump_a = 1.15 * np.exp(-0.5 * (dist_a / 3.5) ** 2)
    bump_b = -0.80 * np.exp(-0.5 * (dist_b / 5.0) ** 2)
    ring = 0.15 * np.cos(np.deg2rad(3.0 * dist_cap))
    return bump_a + bump_b + ring


def build_models(length_scale_deg):
    spherical_kernel = (
        1.0
        * SphericalRBF(length_scale=length_scale_deg, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    )
    planar_kernel = (
        1.0
        * RBF(length_scale=[length_scale_deg, length_scale_deg], length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    )

    common_kwargs = {
        "alpha": 1e-10,
        "normalize_y": True,
        "optimizer": None,
    }
    return (
        GaussianProcessRegressor(kernel=spherical_kernel, **common_kwargs),
        GaussianProcessRegressor(kernel=planar_kernel, **common_kwargs),
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def corrcoef(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 1.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def isotropy_diagnostic(reference_point, length_scale_deg, max_distance_deg=12.0, npts=80):
    distances = np.linspace(0.0, max_distance_deg, npts)
    north_points = np.vstack(
        [destination_point(reference_point, azimuth_deg=0.0, distance_deg=d) for d in distances]
    )
    east_points = np.vstack(
        [destination_point(reference_point, azimuth_deg=90.0, distance_deg=d) for d in distances]
    )

    spherical_kernel = SphericalRBF(length_scale=length_scale_deg, length_scale_bounds="fixed")
    planar_kernel = RBF(
        length_scale=[length_scale_deg, length_scale_deg], length_scale_bounds="fixed"
    )
    ref = np.asarray(reference_point, dtype=float).reshape(1, 2)
    spherical_north = spherical_kernel(ref, north_points).ravel()
    spherical_east = spherical_kernel(ref, east_points).ravel()
    planar_north = planar_kernel(ref, north_points).ravel()
    planar_east = planar_kernel(ref, east_points).ravel()

    return {
        "distance_deg": distances,
        "spherical_north": spherical_north,
        "spherical_east": spherical_east,
        "planar_north": planar_north,
        "planar_east": planar_east,
        "planar_max_direction_gap": float(np.max(np.abs(planar_north - planar_east))),
        "spherical_max_direction_gap": float(np.max(np.abs(spherical_north - spherical_east))),
    }


def write_isotropy_figure(output_path, diagnostic):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        diagnostic["distance_deg"],
        diagnostic["spherical_north"],
        label="Spherical kernel",
        color="#1f6f8b",
        linewidth=2.4,
    )
    ax.plot(
        diagnostic["distance_deg"],
        diagnostic["planar_north"],
        label="Planar kernel: north-south",
        color="#bc5b38",
        linewidth=2.0,
    )
    ax.plot(
        diagnostic["distance_deg"],
        diagnostic["planar_east"],
        label="Planar kernel: east-west",
        color="#d8a31a",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_xlabel("Angular separation from reference point (degrees)")
    ax.set_ylabel("Kernel covariance")
    ax.set_title("Kernel Isotropy at High Latitude")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_learning_curve_figure(output_path, learning_rows):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    rows = list(learning_rows)
    n_train = [row["n_train"] for row in rows if row["model"] == "spherical_geodesic"]
    spherical_rmse = [row["rmse"] for row in rows if row["model"] == "spherical_geodesic"]
    planar_rmse = [row["rmse"] for row in rows if row["model"] == "legacy_planar"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        n_train,
        spherical_rmse,
        marker="o",
        color="#1f6f8b",
        linewidth=2.4,
        label="Spherical geodesic kernel",
    )
    ax.plot(
        n_train,
        planar_rmse,
        marker="s",
        color="#bc5b38",
        linewidth=2.2,
        linestyle="--",
        label="Legacy planar lat/lon kernel",
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Holdout RMSE (synthetic target units)")
    ax.set_title("64-Sample Spherical-Kernel Learning Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_residual_map_figure(
    output_path,
    holdout_points,
    y_true,
    planar_pred,
    spherical_pred,
    cap_center,
    cap_radius_deg,
):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    residual_planar = planar_pred - y_true
    residual_spherical = spherical_pred - y_true
    residual_lim = float(
        np.max(np.abs(np.concatenate([residual_planar, residual_spherical])))
    )
    value_lim = float(np.max(np.abs(y_true)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharex=True, sharey=True)
    panels = [
        ("True synthetic field", y_true, value_lim, "viridis"),
        ("Planar GP residual", residual_planar, residual_lim, "coolwarm"),
        ("Spherical GP residual", residual_spherical, residual_lim, "coolwarm"),
    ]
    for ax, (title, values, clim, cmap) in zip(axes, panels):
        scatter = ax.scatter(
            holdout_points[:, 1],
            holdout_points[:, 0],
            c=values,
            s=18,
            cmap=cmap,
            vmin=-clim,
            vmax=clim,
            edgecolors="none",
        )
        ax.scatter(
            [cap_center[1]],
            [cap_center[0]],
            color="black",
            s=30,
            marker="x",
            linewidths=1.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude (degrees)")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, shrink=0.82)
    axes[0].set_ylabel("Latitude (degrees)")
    fig.suptitle(
        f"High-Latitude Spherical-Cap Validation (radius = {cap_radius_deg:.1f} deg)", y=0.99
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    output_dir = REPO_ROOT / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = "spherical_kernel_validation_v1"
    sha = git_sha()
    cap_center = np.array([72.0, -150.0])
    cap_radius_deg = 15.0
    fixed_length_scale_deg = 6.0
    train_sizes = [8, 16, 32, 64]
    holdout_count = 2048

    domain = SpatialDomain.from_dict(
        {
            "geometry_mode": "spherical",
            "domain_type": "spherical_cap",
            "center_lat": float(cap_center[0]),
            "center_lon": float(cap_center[1]),
            "radius_deg": cap_radius_deg,
            "depth_range": [0.0, 1.0],
            "mag_range": [1.0, 2.0],
        }
    )

    train_points_full = domain.sample_points(max(train_sizes), 0)
    holdout_points = domain.sample_points(holdout_count, 5000)
    y_holdout = true_geodesic_field(holdout_points, cap_center)
    learning_rows = []
    last_planar_pred = None
    last_spherical_pred = None

    params = {
        "cap_center_lat": float(cap_center[0]),
        "cap_center_lon": float(cap_center[1]),
        "cap_radius_deg": cap_radius_deg,
        "fixed_length_scale_deg": fixed_length_scale_deg,
        "train_sizes": train_sizes,
        "holdout_count": holdout_count,
    }
    params_json = json.dumps(params, sort_keys=True)

    for n_train in train_sizes:
        train_points = train_points_full[:n_train]
        y_train = true_geodesic_field(train_points, cap_center)
        spherical_gp, planar_gp = build_models(fixed_length_scale_deg)

        spherical_gp.fit(train_points, y_train)
        planar_gp.fit(train_points, y_train)

        spherical_pred = spherical_gp.predict(holdout_points)
        planar_pred = planar_gp.predict(holdout_points)

        if n_train == max(train_sizes):
            last_spherical_pred = spherical_pred
            last_planar_pred = planar_pred

        for model_name, pred in [
            ("spherical_geodesic", spherical_pred),
            ("legacy_planar", planar_pred),
        ]:
            learning_rows.append(
                {
                    "run_id": run_id,
                    "git_sha": sha,
                    "timestamp": run_timestamp,
                    "params": params_json,
                    "model": model_name,
                    "n_train": n_train,
                    "rmse": rmse(y_holdout, pred),
                    "mae": mae(y_holdout, pred),
                    "corr": corrcoef(y_holdout, pred),
                    "max_abs_error": float(np.max(np.abs(pred - y_holdout))),
                }
            )

    isotropy = isotropy_diagnostic(cap_center, fixed_length_scale_deg)
    row_64_spherical = next(
        row
        for row in learning_rows
        if row["model"] == "spherical_geodesic" and row["n_train"] == 64
    )
    row_64_planar = next(
        row
        for row in learning_rows
        if row["model"] == "legacy_planar" and row["n_train"] == 64
    )
    rmse_improvement_pct = 100.0 * (
        row_64_planar["rmse"] - row_64_spherical["rmse"]
    ) / row_64_planar["rmse"]

    csv_path = output_dir / "spherical_kernel_validation_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run_id",
                "git_sha",
                "timestamp",
                "params",
                "model",
                "n_train",
                "rmse",
                "mae",
                "corr",
                "max_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(learning_rows)

    fig_isotropy_path = output_dir / "fig_spherical_kernel_isotropy.png"
    fig_learning_path = output_dir / "fig_spherical_kernel_learning_curve.png"
    fig_residual_path = output_dir / "fig_spherical_kernel_residual_maps.png"
    figure_paths = [
        write_isotropy_figure(fig_isotropy_path, isotropy),
        write_learning_curve_figure(fig_learning_path, learning_rows),
        write_residual_map_figure(
            fig_residual_path,
            holdout_points,
            y_holdout,
            last_planar_pred,
            last_spherical_pred,
            cap_center,
            cap_radius_deg,
        ),
    ]
    generated_figure_paths = [path for path in figure_paths if path is not None]

    report_path = output_dir / "spherical_kernel_validation_report.md"
    report_lines = [
        "# Spherical Kernel Validation",
        "",
        "## Setup",
        f"- Validation domain: spherical cap centered at `({cap_center[0]:.1f}, {cap_center[1]:.1f})` with radius `{cap_radius_deg:.1f} deg`.",
        f"- Synthetic training sizes: `{train_sizes}`.",
        f"- Holdout points: `{holdout_count}`.",
        f"- Fixed kernel length scale for both models: `{fixed_length_scale_deg:.1f} deg`.",
        "- Comparison isolates geometry only: both GPs use the same nominal length scale and noise level, but one measures distance geodesically and the other in raw lat/lon coordinates.",
        "",
        "## Key Result",
        f"- At `64` training samples, planar holdout RMSE is `{row_64_planar['rmse']:.4f}`.",
        f"- At `64` training samples, spherical holdout RMSE is `{row_64_spherical['rmse']:.4f}`.",
        f"- Relative RMSE reduction from the spherical kernel is `{rmse_improvement_pct:.1f}%`.",
        f"- At high latitude, the planar kernel's max north-vs-east covariance mismatch is `{isotropy['planar_max_direction_gap']:.4f}`.",
        f"- The spherical kernel's corresponding directional mismatch is `{isotropy['spherical_max_direction_gap']:.4e}`.",
        "",
        "## Interpretation",
        "- The spherical kernel stays isotropic with respect to angular separation, so points the same great-circle distance apart receive the same covariance.",
        "- The planar lat/lon kernel distorts east-west versus north-south distance at high latitude, which shows up as larger residual structure in the synthetic cap experiment.",
        "- This validation is synthetic rather than end-to-end EIG, but it isolates exactly the geometry change introduced in the new BO kernel path.",
        "",
        "## Figures",
    ]
    if generated_figure_paths:
        report_lines.extend([f"- `{path.name}`" for path in generated_figure_paths])
    else:
        report_lines.append(
            "- Figure generation skipped because `matplotlib` is not installed in this environment."
        )
    report_lines.extend(
        [
            "",
            "## Artifact",
            f"- Metrics CSV: `{csv_path.name}`",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(report_path)
    print(csv_path)
    for figure_path in generated_figure_paths:
        print(figure_path)


if __name__ == "__main__":
    main()
