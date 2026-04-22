import numpy as np
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    NormalizedKernelMixin,
    RBF,
    StationaryKernelMixin,
    WhiteKernel,
)

from spatial_domain import ensure_2d_points, latlon_to_unit


def great_circle_distance_deg(X, Y=None):
    X = ensure_2d_points(X)
    Y = X if Y is None else ensure_2d_points(Y)

    x_unit = latlon_to_unit(X)
    y_unit = x_unit if Y is X else latlon_to_unit(Y)
    cos_angles = np.clip(x_unit @ y_unit.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angles))


class SphericalRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """RBF kernel over latitude/longitude pairs using great-circle distance."""

    def __init__(self, length_scale=10.0, length_scale_bounds=(0.5, 180.0)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = ensure_2d_points(X)
        if X.shape[1] != 2:
            raise ValueError("SphericalRBF expects [lat, lon] points with shape (n, 2).")

        if Y is None:
            Y = X
            Y_is_none = True
        else:
            Y = ensure_2d_points(Y)
            Y_is_none = False
            if Y.shape[1] != 2:
                raise ValueError("SphericalRBF expects [lat, lon] points with shape (n, 2).")

        if eval_gradient and not Y_is_none:
            raise ValueError("Kernel gradient can only be evaluated when Y is None.")

        length_scale = float(np.squeeze(np.asarray(self.length_scale, dtype=float)))
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")

        dists = great_circle_distance_deg(X, Y)
        scaled = dists / length_scale
        K = np.exp(-0.5 * scaled**2)

        if not eval_gradient:
            return K

        if self.hyperparameter_length_scale.fixed:
            return K, np.empty((X.shape[0], X.shape[0], 0))

        gradient = (K * scaled**2)[:, :, np.newaxis]
        return K, gradient


def _default_spherical_length_scale(spatial_domain):
    if spatial_domain.domain_type == "globe":
        return 20.0

    if spatial_domain.domain_type == "spherical_cap":
        return max(1.0, min(30.0, 0.5 * spatial_domain.radius_deg))

    lat_span = spatial_domain.sample_bounds[0, 1] - spatial_domain.sample_bounds[0, 0]
    lon_span = spatial_domain.sample_bounds[1, 1] - spatial_domain.sample_bounds[1, 0]
    return max(1.0, min(30.0, 0.25 * max(lat_span, lon_span)))


def build_base_spatial_kernel(spatial_domain):
    if spatial_domain.geometry_mode == "spherical":
        length_scale = _default_spherical_length_scale(spatial_domain)
        lower = max(0.5, length_scale / 10.0)
        upper = min(180.0, max(5.0, length_scale * 10.0))
        if lower >= upper:
            lower = max(0.5, upper / 10.0)
        return SphericalRBF(
            length_scale=length_scale,
            length_scale_bounds=(lower, upper),
        )

    return RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.2, 1.0))


def build_spatial_kernel(spatial_domain):
    return (
        1.0 * build_base_spatial_kernel(spatial_domain)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 5e-1))
    )
