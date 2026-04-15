#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_domain import SpatialDomain, sobol_matrix


def sample_rectangular_global(nsamp):
    samples = sobol_matrix(2, nsamp, 0)
    lat = samples[:, 0] * 180.0 - 90.0
    lon = samples[:, 1] * 360.0 - 180.0
    return np.column_stack((lat, lon))


def latitude_metrics(points_latlon):
    lat = points_latlon[:, 0]
    sin_lat = np.sin(np.deg2rad(lat))
    polar_fraction = np.mean(np.abs(lat) >= 60.0)
    equatorial_fraction = np.mean(np.abs(lat) <= 30.0)

    lat_hist, lat_edges = np.histogram(lat, bins=9, range=(-90.0, 90.0))
    sin_hist, sin_edges = np.histogram(sin_lat, bins=9, range=(-1.0, 1.0))

    return {
        "polar_fraction_abs_lat_ge_60": polar_fraction,
        "equatorial_fraction_abs_lat_le_30": equatorial_fraction,
        "lat_bin_max_over_min": lat_hist.max() / lat_hist.min(),
        "sin_lat_bin_max_over_min": sin_hist.max() / sin_hist.min(),
        "lat_hist": lat_hist.tolist(),
        "lat_edges": lat_edges.tolist(),
        "sin_hist": sin_hist.tolist(),
        "sin_edges": sin_edges.tolist(),
    }


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def write_latitude_histogram(output_path, old_points, new_points):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(-90.0, 90.0, 19)
    ax.hist(
        old_points[:, 0],
        bins=bins,
        alpha=0.55,
        label="Legacy rectangular global",
        color="#bc5b38",
        edgecolor="white",
    )
    ax.hist(
        new_points[:, 0],
        bins=bins,
        alpha=0.55,
        label="Spherical surface global",
        color="#1f6f8b",
        edgecolor="white",
    )
    ax.axvline(-60.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(60.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(-30.0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(30.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Sample count")
    ax.set_title("Global Sampling Latitude Histogram")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_sin_lat_histogram(output_path, old_points, new_points):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(-1.0, 1.0, 19)
    ax.hist(
        np.sin(np.deg2rad(old_points[:, 0])),
        bins=bins,
        alpha=0.55,
        label="Legacy rectangular global",
        color="#bc5b38",
        edgecolor="white",
    )
    ax.hist(
        np.sin(np.deg2rad(new_points[:, 0])),
        bins=bins,
        alpha=0.55,
        label="Spherical surface global",
        color="#1f6f8b",
        edgecolor="white",
    )
    ax.set_xlabel("sin(latitude)")
    ax.set_ylabel("Sample count")
    ax.set_title("Uniform-on-Sphere Diagnostic via sin(latitude)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_global_scatter_panel(output_path, old_points, new_points, max_points=1200):
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    n_old = min(max_points, old_points.shape[0])
    n_new = min(max_points, new_points.shape[0])
    old_plot = old_points[:n_old]
    new_plot = new_points[:n_new]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True, sharey=True)
    panels = [
        (axes[0], old_plot, "Legacy rectangular global"),
        (axes[1], new_plot, "Spherical surface global"),
    ]
    for ax, points, title in panels:
        ax.scatter(
            points[:, 1],
            points[:, 0],
            s=8,
            alpha=0.6,
            color="#1f6f8b" if "Spherical" in title else "#bc5b38",
            edgecolors="none",
        )
        ax.axhline(-60.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(60.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(-30.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(30.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Latitude (degrees)")
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-90.0, 90.0)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Longitude (degrees)")
    fig.suptitle("Global Sampling in Lon/Lat Coordinates", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    output_dir = REPO_ROOT / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    nsamp = 4096
    old_points = sample_rectangular_global(nsamp)
    globe_domain = SpatialDomain.from_dict(
        {
            "geometry_mode": "spherical",
            "domain_type": "globe",
            "depth_range": [0.0, 1.0],
            "mag_range": [1.0, 2.0],
        }
    )
    new_points = globe_domain.sample_points(nsamp, 0)

    old_metrics = latitude_metrics(old_points)
    new_metrics = latitude_metrics(new_points)

    fig_latitude_hist_path = output_dir / "fig_global_sampling_latitude_hist.png"
    fig_sin_lat_hist_path = output_dir / "fig_global_sampling_sin_lat_hist.png"
    fig_scatter_panel_path = output_dir / "fig_global_sampling_lonlat_scatter.png"
    figure_paths = [
        write_latitude_histogram(fig_latitude_hist_path, old_points, new_points),
        write_sin_lat_histogram(fig_sin_lat_hist_path, old_points, new_points),
        write_global_scatter_panel(fig_scatter_panel_path, old_points, new_points),
    ]
    generated_figure_paths = [path for path in figure_paths if path is not None]

    csv_path = output_dir / "global_geometry_sampling_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sampler", "metric", "value"])
        for sampler, metrics in [
            ("legacy_rectangular_global", old_metrics),
            ("spherical_surface_global", new_metrics),
        ]:
            for key in [
                "polar_fraction_abs_lat_ge_60",
                "equatorial_fraction_abs_lat_le_30",
                "lat_bin_max_over_min",
                "sin_lat_bin_max_over_min",
            ]:
                writer.writerow([sampler, key, metrics[key]])

    report_path = output_dir / "global_geometry_sampling_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Global Geometry Sampling Validation",
                "",
                f"Sample count: `{nsamp}`",
                "",
                "## Result",
                f"- Legacy rectangular global sampling puts `{old_metrics['polar_fraction_abs_lat_ge_60']:.3f}` of points above `|lat| >= 60°`.",
                f"- Spherical surface sampling puts `{new_metrics['polar_fraction_abs_lat_ge_60']:.3f}` of points above `|lat| >= 60°`.",
                f"- Legacy equatorial share (`|lat| <= 30°`) is `{old_metrics['equatorial_fraction_abs_lat_le_30']:.3f}`.",
                f"- Spherical equatorial share (`|lat| <= 30°`) is `{new_metrics['equatorial_fraction_abs_lat_le_30']:.3f}`.",
                "",
                "## Interpretation",
                "- The old global rectangle transform is uniform in latitude, so it over-samples polar regions relative to surface area.",
                "- The spherical transform is close to uniform in `sin(latitude)`, which is the correct signature for uniform sampling on the sphere.",
                "",
                "## Histogram Diagnostics",
                f"- Legacy latitude-bin spread: `{old_metrics['lat_bin_max_over_min']:.3f}` max/min.",
                f"- New latitude-bin spread: `{new_metrics['lat_bin_max_over_min']:.3f}` max/min.",
                f"- Legacy `sin(lat)`-bin spread: `{old_metrics['sin_lat_bin_max_over_min']:.3f}` max/min.",
                f"- New `sin(lat)`-bin spread: `{new_metrics['sin_lat_bin_max_over_min']:.3f}` max/min.",
                "",
                "## Figures",
            ]
            + [
                f"- `{path.name}`"
                for path in generated_figure_paths
            ]
            + [
                ""
                if generated_figure_paths
                else "- Figure generation skipped because `matplotlib` is not installed in this environment.",
                "",
                "The new sampler is the expected latitude-density correction for teleseismic/global geometry runs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(report_path)
    print(csv_path)
    for figure_path in generated_figure_paths:
        print(figure_path)


if __name__ == "__main__":
    main()
