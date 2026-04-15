import numpy as np

from spatial_domain import SpatialDomain, polygon_contains_points, sobol_matrix


SPATIAL_DOMAIN_AWARE = True


def _ensure_domain(bounds):
    if isinstance(bounds, SpatialDomain):
        return bounds
    return SpatialDomain.from_legacy_bounds(bounds)


def check_valid(bounds, points):
    if isinstance(bounds, SpatialDomain):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        latlon_points = np.column_stack((points[:, 1], points[:, 0]))
        return bounds.contains(latlon_points)

    masks = []
    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)
    if len(bounds.shape) == 2:
        bounds = bounds.reshape((1, *bounds.shape))

    for polygon in bounds:
        valid_pts_idx = polygon_contains_points(polygon, points)
        masks.append(valid_pts_idx)
    point_is_valid = np.any(masks, axis=0)

    return point_is_valid


def compute_sample_bounds(input_bounds):
    domain = _ensure_domain(input_bounds)
    lat_range = domain.sample_bounds[0]
    lon_range = domain.sample_bounds[1]
    return np.array([lon_range, lat_range], dtype=float)


def calc_area(bounds, nsamp=1000, skip=0):
    del nsamp, skip
    domain = _ensure_domain(bounds)
    return domain.area_measure


def _sample_location_and_tails(bounds, depth_range, mag_range, nsamp, skip):
    domain = _ensure_domain(bounds)
    dim_num = 4
    count = 0
    collected = np.empty((0, dim_num))

    while collected.shape[0] < nsamp:
        curr_len = nsamp - collected.shape[0]
        sbvals = sobol_matrix(dim_num, curr_len, skip + count)
        candidate_latlon = domain.candidate_points_from_unit(sbvals[:, :2])
        valid_idx = domain.contains(candidate_latlon)

        if np.any(valid_idx):
            valid_latlon = candidate_latlon[valid_idx]
            valid_samples = sbvals[valid_idx]

            max_mag = 1 - 10 ** (-mag_range[1])
            min_mag = 1 - 10 ** (-mag_range[0])

            theta_batch = np.zeros((valid_latlon.shape[0], dim_num))
            theta_batch[:, 0] = valid_latlon[:, 0]
            theta_batch[:, 1] = valid_latlon[:, 1]
            theta_batch[:, 2] = (
                valid_samples[:, 2] * (depth_range[1] - depth_range[0]) + depth_range[0]
            )
            theta_batch[:, 3] = valid_samples[:, 3] * (max_mag - min_mag) + min_mag
            theta_batch[:, 3] = -np.log(1 - theta_batch[:, 3]) / np.log(10)
            collected = np.vstack((collected, theta_batch))

        count += curr_len

    return collected[:nsamp]


def generate_theta_data(bounds, depth_range, mag_range, nsamp, skip):
    """
    Rejection sample events using a location distribution determined by the
    spatial domain, a uniform depth, and the legacy exponential magnitude prior.
    """
    return _sample_location_and_tails(bounds, depth_range, mag_range, nsamp, skip)


def sample_theta_space(bounds, depth_range, mag_range, nsamp, skip):
    """
    Draw event-hypothesis samples from the same importance distribution used for
    data-generation events.
    """
    return _sample_location_and_tails(bounds, depth_range, mag_range, nsamp, skip)


def _evaluate_joint_density(thetas, bounds, depth_range, mag_range):
    domain = _ensure_domain(bounds)
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim == 1:
        thetas = thetas.reshape((1, -1))

    location_prob = domain.location_density(thetas[:, :2])
    depth_prob = 1 / np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10) / 10 ** thetas[:, 3]) / (
        (1 - 10 ** (-mag_range[1])) - (1 - 10 ** (-mag_range[0]))
    )
    return location_prob * depth_prob * mag_prob


def eval_theta_prior(thetas, bounds, depth_range, mag_range):
    return _evaluate_joint_density(thetas, bounds, depth_range, mag_range)


def eval_importance(thetas, bounds, depth_range, mag_range):
    return _evaluate_joint_density(thetas, bounds, depth_range, mag_range)
