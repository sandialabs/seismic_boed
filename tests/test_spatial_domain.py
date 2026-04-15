import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spatial_domain import SpatialDomain, angular_distance_deg
from uniform_prior import eval_importance, eval_theta_prior, generate_theta_data
from utils import read_bounds, read_spatial_domain


class SpatialDomainTests(unittest.TestCase):
    def test_globe_sampling_approximates_uniform_surface_area(self):
        domain = SpatialDomain.from_dict(
            {
                "geometry_mode": "spherical",
                "domain_type": "globe",
                "depth_range": [0.0, 10.0],
                "mag_range": [1.0, 2.0],
            }
        )

        points = domain.sample_points(2048, 0)
        self.assertTrue(np.all(domain.contains(points)))

        lon_hist, _ = np.histogram(points[:, 1], bins=8, range=(-180.0, 180.0))
        sin_lat = np.sin(np.deg2rad(points[:, 0]))
        sin_hist, _ = np.histogram(sin_lat, bins=8, range=(-1.0, 1.0))

        self.assertLess(lon_hist.max() / lon_hist.min(), 1.6)
        self.assertLess(sin_hist.max() / sin_hist.min(), 1.6)

    def test_spherical_cap_sampling_respects_radius_and_area_law(self):
        radius_deg = 25.0
        domain = SpatialDomain.from_dict(
            {
                "geometry_mode": "spherical",
                "domain_type": "spherical_cap",
                "center_lat": 12.0,
                "center_lon": -35.0,
                "radius_deg": radius_deg,
                "depth_range": [0.0, 10.0],
                "mag_range": [1.0, 2.0],
            }
        )

        points = domain.sample_points(2048, 0)
        distances = angular_distance_deg(points, [[12.0, -35.0]])
        self.assertTrue(np.all(distances <= radius_deg + 1e-8))

        cos_distance = np.cos(np.deg2rad(distances))
        lower = math.cos(math.radians(radius_deg))
        hist, _ = np.histogram(cos_distance, bins=6, range=(lower, 1.0))
        self.assertLess(hist.max() / hist.min(), 1.8)

    def test_spherical_polygon_rejection_and_validation(self):
        polygon = [[-20.0, -10.0], [-20.0, 10.0], [15.0, 8.0], [12.0, -12.0]]
        domain = SpatialDomain.from_dict(
            {
                "geometry_mode": "spherical",
                "domain_type": "polygon",
                "coordinates_1": polygon,
                "depth_range": [0.0, 10.0],
                "mag_range": [1.0, 2.0],
            }
        )

        points = domain.sample_points(256, 0)
        self.assertTrue(np.all(domain.contains(points)))
        self.assertFalse(domain.contains([[40.0, 40.0]])[0])
        self.assertGreater(domain.area_measure, 0.0)

        with self.assertRaises(ValueError):
            SpatialDomain.from_dict(
                {
                    "geometry_mode": "spherical",
                    "domain_type": "polygon",
                    "coordinates_1": [[170.0, -10.0], [170.0, 10.0], [-170.0, 10.0], [-170.0, -10.0]],
                    "depth_range": [0.0, 10.0],
                    "mag_range": [1.0, 2.0],
                }
            )

    def test_planar_regression_matches_legacy_bounds_behavior(self):
        repo_root = Path(__file__).resolve().parents[1]
        bounds_file = repo_root / "ta_array_domain.json"

        legacy_bounds, depth_range, mag_range = read_bounds(str(bounds_file), sensor_bounds=False)
        spatial_domain = read_spatial_domain(str(bounds_file), sensor_bounds=False)

        theta_legacy = generate_theta_data(legacy_bounds, depth_range, mag_range, 32, 0)
        theta_domain = generate_theta_data(spatial_domain, depth_range, mag_range, 32, 0)

        np.testing.assert_allclose(theta_legacy, theta_domain)
        np.testing.assert_allclose(
            eval_theta_prior(theta_legacy, legacy_bounds, depth_range, mag_range),
            eval_theta_prior(theta_domain, spatial_domain, depth_range, mag_range),
        )
        np.testing.assert_allclose(
            eval_importance(theta_legacy, legacy_bounds, depth_range, mag_range),
            eval_importance(theta_domain, spatial_domain, depth_range, mag_range),
        )

    def test_spherical_bounds_round_trip_through_utils(self):
        bounds = {
            "geometry_mode": "spherical",
            "domain_type": "spherical_cap",
            "center_lat": 30.0,
            "center_lon": 70.0,
            "radius_deg": 18.0,
            "depth_range": [5.0, 20.0],
            "mag_range": [3.0, 6.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bounds_path = Path(tmpdir) / "cap_bounds.json"
            bounds_path.write_text(json.dumps(bounds), encoding="utf-8")

            domain = read_spatial_domain(str(bounds_path), sensor_bounds=False)
            legacy_bounds, depth_range, mag_range = read_bounds(
                str(bounds_path), sensor_bounds=False
            )

            self.assertEqual(domain.geometry_mode, "spherical")
            self.assertEqual(domain.domain_type, "spherical_cap")
            self.assertEqual(depth_range, bounds["depth_range"])
            self.assertEqual(mag_range, bounds["mag_range"])
            self.assertEqual(np.asarray(legacy_bounds).shape[1], 2)


if __name__ == "__main__":
    unittest.main()
