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
                "The new sampler is the expected latitude-density correction for teleseismic/global geometry runs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
