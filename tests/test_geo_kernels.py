import math
import unittest

import numpy as np

from spatial_domain import SpatialDomain

try:
    from geo_kernels import SphericalRBF, build_base_spatial_kernel, great_circle_distance_deg

    HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised when sklearn is unavailable
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN, "sklearn is required for kernel tests")
class GeoKernelTests(unittest.TestCase):
    def test_great_circle_distance_wraps_longitude_correctly(self):
        points_a = np.array([[0.0, 170.0], [0.0, 0.0]])
        points_b = np.array([[0.0, -170.0], [10.0, 0.0]])

        dists = great_circle_distance_deg(points_a, points_b)

        self.assertAlmostEqual(dists[0, 0], 20.0, places=6)
        self.assertAlmostEqual(dists[1, 1], 10.0, places=6)

    def test_spherical_rbf_uses_angular_distance(self):
        kernel = SphericalRBF(length_scale=20.0, length_scale_bounds="fixed")
        points = np.array(
            [
                [0.0, 0.0],
                [0.0, 20.0],
                [0.0, 170.0],
                [0.0, -170.0],
            ]
        )

        K = kernel(points)

        self.assertAlmostEqual(K[0, 0], 1.0, places=12)
        self.assertAlmostEqual(K[2, 3], math.exp(-0.5), places=10)
        self.assertLess(K[0, 2], 1e-12)
        self.assertGreater(K[0, 1], K[0, 2])

    def test_spherical_rbf_gradient_shape_matches_sklearn_expectation(self):
        kernel = SphericalRBF(length_scale=15.0)
        points = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 15.0]])

        K, grad = kernel(points, eval_gradient=True)

        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(grad.shape, (3, 3, 1))
        np.testing.assert_allclose(np.diag(grad[:, :, 0]), 0.0)

    def test_spherical_rbf_kernel_matrix_is_psd(self):
        domain = SpatialDomain.from_dict(
            {
                "geometry_mode": "spherical",
                "domain_type": "spherical_cap",
                "center_lat": 72.0,
                "center_lon": -150.0,
                "radius_deg": 15.0,
            },
            sensor_bounds=True,
        )
        points = domain.sample_points(24, 0)
        kernel = SphericalRBF(length_scale=8.0, length_scale_bounds="fixed")

        K = kernel(points)

        np.testing.assert_allclose(K, K.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(K)
        self.assertGreater(eigvals.min(), -1e-10)

    def test_kernel_builder_switches_between_planar_and_spherical(self):
        spherical_domain = SpatialDomain.from_dict(
            {
                "geometry_mode": "spherical",
                "domain_type": "spherical_cap",
                "center_lat": 20.0,
                "center_lon": -150.0,
                "radius_deg": 10.0,
            },
            sensor_bounds=True,
        )
        planar_domain = SpatialDomain.from_dict(
            {
                "lat_range": [34.0, 36.0],
                "lon_range": [-118.0, -116.0],
            },
            sensor_bounds=True,
        )

        spherical_kernel = build_base_spatial_kernel(spherical_domain)
        planar_kernel = build_base_spatial_kernel(planar_domain)

        self.assertIsInstance(spherical_kernel, SphericalRBF)
        self.assertEqual(planar_kernel.__class__.__name__, "RBF")


if __name__ == "__main__":
    unittest.main()
