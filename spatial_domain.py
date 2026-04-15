import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import sobol_seq as sq
except ImportError:  # pragma: no cover - exercised in environments without sobol_seq
    sq = None


_HALTON_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]


_GLOBE_BOUNDS_LONLAT = np.array(
    [
        [-180.0, -90.0],
        [-180.0, 90.0],
        [180.0, 90.0],
        [180.0, -90.0],
    ]
)
_POLE_TEST_POINTS = np.array(
    [
        [0.0, 89.999],
        [90.0, 89.999],
        [-90.0, 89.999],
        [179.999, 89.999],
        [0.0, -89.999],
        [90.0, -89.999],
        [-90.0, -89.999],
        [179.999, -89.999],
    ]
)


def normalize_longitudes(longitudes):
    values = np.asarray(longitudes, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def sobol_matrix(dim_num, nsamp, skip):
    if sq is not None:
        sbvals = np.full((nsamp, dim_num), np.nan)
        for j in range(nsamp):
            sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1 + skip + j)
        return sbvals

    if dim_num > len(_HALTON_PRIMES):
        raise ValueError(
            f"Pure-Python low-discrepancy fallback only supports up to {len(_HALTON_PRIMES)} dimensions."
        )

    def van_der_corput(index, base):
        value = 0.0
        denom = 1.0
        while index > 0:
            index, remainder = divmod(index, base)
            denom *= base
            value += remainder / denom
        return value

    sbvals = np.zeros((nsamp, dim_num), dtype=float)
    for dim in range(dim_num):
        base = _HALTON_PRIMES[dim]
        for row in range(nsamp):
            sbvals[row, dim] = van_der_corput(skip + row + 1, base)
    return sbvals


def ensure_2d_points(points):
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2) as [lat, lon].")
    return values


def latlon_to_unit(points_latlon):
    points_latlon = ensure_2d_points(points_latlon)
    lat_rad = np.deg2rad(points_latlon[:, 0])
    lon_rad = np.deg2rad(points_latlon[:, 1])
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        )
    )


def unit_to_latlon(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / np.clip(norms, 1e-15, None)
    lat = np.rad2deg(np.arcsin(np.clip(unit_vectors[:, 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0]))
    lon = normalize_longitudes(lon)
    return np.column_stack((lat, lon))


def angular_distance_deg(points_latlon, center_latlon):
    points_latlon = ensure_2d_points(points_latlon)
    center_latlon = ensure_2d_points(center_latlon)
    center_vec = latlon_to_unit(center_latlon)[0]
    point_vecs = latlon_to_unit(points_latlon)
    cos_angles = np.clip(point_vecs @ center_vec, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angles))


def is_axis_aligned_rectangle(polygon_lonlat):
    polygon = np.asarray(polygon_lonlat, dtype=float)
    if polygon.shape != (4, 2):
        return False
    lon_vals = np.unique(np.round(polygon[:, 0], decimals=12))
    lat_vals = np.unique(np.round(polygon[:, 1], decimals=12))
    if lon_vals.size != 2 or lat_vals.size != 2:
        return False
    expected = {
        (lon_vals[0], lat_vals[0]),
        (lon_vals[0], lat_vals[1]),
        (lon_vals[1], lat_vals[0]),
        (lon_vals[1], lat_vals[1]),
    }
    actual = {(round(lon, 12), round(lat, 12)) for lon, lat in polygon}
    return expected == actual


def planar_polygon_area(polygon_lonlat):
    polygon = np.asarray(polygon_lonlat, dtype=float)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_contains_points(polygon_lonlat, points_lonlat):
    polygon = np.asarray(polygon_lonlat, dtype=float)
    points = np.asarray(points_lonlat, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    x = points[:, 0]
    y = points[:, 1]
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)

    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        yi = poly_y[i]
        yj = poly_y[j]
        xi = poly_x[i]
        xj = poly_x[j]
        crosses = (yi > y) != (yj > y)
        denom = (yj - yi) + 1e-15
        x_intersect = (xj - xi) * (y - yi) / denom + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def spherical_triangle_area(vector_a, vector_b, vector_c):
    def central_angle(vec1, vec2):
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    side_a = central_angle(vector_b, vector_c)
    side_b = central_angle(vector_c, vector_a)
    side_c = central_angle(vector_a, vector_b)
    semiperimeter = 0.5 * (side_a + side_b + side_c)

    tan_terms = np.array(
        [
            np.tan(0.5 * semiperimeter),
            np.tan(0.5 * (semiperimeter - side_a)),
            np.tan(0.5 * (semiperimeter - side_b)),
            np.tan(0.5 * (semiperimeter - side_c)),
        ]
    )
    tan_terms = np.clip(tan_terms, 0.0, None)
    return 4.0 * np.arctan(np.sqrt(np.prod(tan_terms)))


def spherical_polygon_area_sr(polygon_lonlat):
    polygon = np.asarray(polygon_lonlat, dtype=float)
    if polygon.shape[0] < 3:
        raise ValueError("Spherical polygons need at least 3 vertices.")

    unit_vectors = latlon_to_unit(np.column_stack((polygon[:, 1], polygon[:, 0])))
    origin = unit_vectors[0]
    area = 0.0
    for idx in range(1, unit_vectors.shape[0] - 1):
        area += spherical_triangle_area(origin, unit_vectors[idx], unit_vectors[idx + 1])
    return area


@dataclass
class SpatialDomain:
    geometry_mode: str
    domain_type: str
    polygons_lonlat: tuple = field(default_factory=tuple)
    center_lat: float = None
    center_lon: float = None
    radius_deg: float = None
    depth_range: tuple = None
    mag_range: tuple = None
    source_path: str = None
    sample_bounds: np.ndarray = field(init=False)
    area_measure: float = field(init=False)
    _paths: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self.geometry_mode = str(self.geometry_mode).lower()
        self.domain_type = str(self.domain_type).lower()
        self.polygons_lonlat = tuple(
            np.asarray(polygon, dtype=float).copy() for polygon in self.polygons_lonlat
        )
        self.center_lon = (
            None if self.center_lon is None else float(normalize_longitudes(self.center_lon))
        )
        self.center_lat = None if self.center_lat is None else float(self.center_lat)
        self.radius_deg = None if self.radius_deg is None else float(self.radius_deg)
        self._paths = tuple()
        self.sample_bounds = self._compute_sample_bounds()
        self.area_measure = self._compute_area_measure()

    @classmethod
    def from_file(cls, bounds_file, sensor_bounds=False):
        bounds_path = Path(bounds_file).expanduser().resolve()
        with open(bounds_path, "r") as f:
            bounds = json.load(f)
        domain = cls.from_dict(bounds, sensor_bounds=sensor_bounds)
        domain.source_path = str(bounds_path)
        return domain

    @classmethod
    def from_dict(cls, bounds, sensor_bounds=False):
        bounds = dict(bounds)
        geometry_mode = str(bounds.get("geometry_mode", "planar")).lower()

        allowable_keys = {
            "geometry_mode",
            "domain_type",
            "lat_range",
            "lon_range",
            "depth_range",
            "mag_range",
            "center_lat",
            "center_lon",
            "radius_deg",
        }
        coords_keys = sorted(
            [key for key in bounds.keys() if key.startswith("coordinates_")],
            key=lambda key: int(key.split("_")[-1]),
        )
        allowable_keys.update(coords_keys)
        for key in bounds.keys():
            if key not in allowable_keys:
                raise ValueError(
                    f"Key {key} in bounds file is not supported. Please remove."
                )

        depth_range = None if sensor_bounds else tuple(bounds["depth_range"])
        mag_range = None if sensor_bounds else tuple(bounds["mag_range"])

        if geometry_mode == "planar":
            has_lat = "lat_range" in bounds
            has_lon = "lon_range" in bounds
            has_coords = len(coords_keys) > 0

            if has_lat and not has_lon:
                raise ValueError(
                    "Bounds file contains 'lat_range' but not 'lon_range'. Both must be specified."
                )
            if has_lon and not has_lat:
                raise ValueError(
                    "Bounds file contains 'lon_range' but not 'lat_range'. Both must be specified."
                )
            if has_lat and has_coords:
                raise ValueError(
                    "Bounds file contains both range and polygon bounds. Use one representation."
                )

            if has_lat:
                lat_range = bounds["lat_range"]
                lon_range = bounds["lon_range"]
                polygons = (
                    np.array(
                        [
                            [lon_range[0], lat_range[0]],
                            [lon_range[0], lat_range[1]],
                            [lon_range[1], lat_range[1]],
                            [lon_range[1], lat_range[0]],
                        ]
                    ),
                )
            elif has_coords:
                polygons = tuple(np.asarray(bounds[key], dtype=float) for key in coords_keys)
            else:
                raise ValueError("Planar bounds require either ranges or coordinates.")

            return cls(
                geometry_mode="planar",
                domain_type="polygon",
                polygons_lonlat=polygons,
                depth_range=depth_range,
                mag_range=mag_range,
            )

        if geometry_mode != "spherical":
            raise ValueError(
                f"geometry_mode must be 'planar' or 'spherical', not '{geometry_mode}'."
            )

        domain_type = str(bounds.get("domain_type", "")).lower()
        if domain_type not in {"globe", "polygon", "spherical_cap"}:
            raise ValueError(
                "Spherical bounds require domain_type of 'globe', 'polygon', or 'spherical_cap'."
            )

        if domain_type == "globe":
            return cls(
                geometry_mode="spherical",
                domain_type="globe",
                depth_range=depth_range,
                mag_range=mag_range,
            )

        if domain_type == "polygon":
            if len(coords_keys) == 0:
                raise ValueError("Spherical polygon bounds require coordinates_* entries.")
            polygons = tuple(np.asarray(bounds[key], dtype=float) for key in coords_keys)
            domain = cls(
                geometry_mode="spherical",
                domain_type="polygon",
                polygons_lonlat=polygons,
                depth_range=depth_range,
                mag_range=mag_range,
            )
            domain._validate_simple_spherical_polygons()
            return domain

        for required_key in ["center_lat", "center_lon", "radius_deg"]:
            if required_key not in bounds:
                raise ValueError(
                    f"Spherical cap bounds require '{required_key}' in the bounds file."
                )
        return cls(
            geometry_mode="spherical",
            domain_type="spherical_cap",
            center_lat=bounds["center_lat"],
            center_lon=bounds["center_lon"],
            radius_deg=bounds["radius_deg"],
            depth_range=depth_range,
            mag_range=mag_range,
        )

    @classmethod
    def from_legacy_bounds(cls, bounds, depth_range=None, mag_range=None):
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds, dtype=float)
        if len(bounds.shape) == 2:
            bounds = bounds.reshape((1, *bounds.shape))
        polygons = tuple(np.asarray(bounds[idx], dtype=float) for idx in range(len(bounds)))
        return cls(
            geometry_mode="planar",
            domain_type="polygon",
            polygons_lonlat=polygons,
            depth_range=depth_range,
            mag_range=mag_range,
        )

    def to_legacy_bounds(self):
        if self.geometry_mode == "spherical" and self.domain_type == "globe":
            return _GLOBE_BOUNDS_LONLAT.copy()
        if self.geometry_mode == "spherical" and self.domain_type == "spherical_cap":
            boundary = self.cap_boundary_polygon(npts=361)
            return boundary
        if len(self.polygons_lonlat) == 1:
            return self.polygons_lonlat[0].copy()
        return [polygon.copy() for polygon in self.polygons_lonlat]

    def contains(self, points_latlon):
        points_latlon = ensure_2d_points(points_latlon)
        lat = points_latlon[:, 0]
        lon = normalize_longitudes(points_latlon[:, 1])
        finite_mask = np.isfinite(lat) & np.isfinite(lon)
        valid = np.zeros(points_latlon.shape[0], dtype=bool)

        if self.geometry_mode == "spherical" and self.domain_type == "globe":
            valid[finite_mask] = np.abs(lat[finite_mask]) <= 90.0 + 1e-12
            return valid

        if self.geometry_mode == "spherical" and self.domain_type == "spherical_cap":
            distances = angular_distance_deg(
                np.column_stack((lat[finite_mask], lon[finite_mask])),
                np.array([[self.center_lat, self.center_lon]]),
            )
            valid[finite_mask] = distances <= self.radius_deg + 1e-9
            return valid

        if len(self.polygons_lonlat) == 0:
            return valid

        lonlat_points = np.column_stack((lon, lat))
        masks = [
            polygon_contains_points(polygon, lonlat_points)
            for polygon in self.polygons_lonlat
        ]
        if masks:
            valid[finite_mask] = np.any(np.vstack(masks)[:, finite_mask], axis=0)
        return valid

    def candidate_points_from_unit(self, unit_xy):
        unit_xy = ensure_2d_points(unit_xy)
        if self.geometry_mode == "spherical" and self.domain_type == "globe":
            lon = unit_xy[:, 0] * 360.0 - 180.0
            lat = np.rad2deg(np.arcsin(np.clip(2.0 * unit_xy[:, 1] - 1.0, -1.0, 1.0)))
            return np.column_stack((lat, normalize_longitudes(lon)))

        if self.geometry_mode == "spherical" and self.domain_type == "spherical_cap":
            azimuth = 2.0 * np.pi * unit_xy[:, 0]
            radius_rad = np.deg2rad(self.radius_deg)
            cos_distance = 1.0 - unit_xy[:, 1] * (1.0 - np.cos(radius_rad))
            sin_distance = np.sqrt(np.clip(1.0 - cos_distance**2, 0.0, 1.0))

            center_vec = latlon_to_unit([[self.center_lat, self.center_lon]])[0]
            ref_vec = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(ref_vec, center_vec)) > 0.99:
                ref_vec = np.array([0.0, 1.0, 0.0])
            basis_1 = np.cross(ref_vec, center_vec)
            basis_1 = basis_1 / np.linalg.norm(basis_1)
            basis_2 = np.cross(center_vec, basis_1)

            tangent = (
                np.cos(azimuth)[:, None] * basis_1[None, :]
                + np.sin(azimuth)[:, None] * basis_2[None, :]
            )
            points = cos_distance[:, None] * center_vec[None, :] + sin_distance[:, None] * tangent
            return unit_to_latlon(points)

        lon_range = self.sample_bounds[1]
        lat_range = self.sample_bounds[0]
        lon = unit_xy[:, 0] * (lon_range[1] - lon_range[0]) + lon_range[0]
        lat = unit_xy[:, 1] * (lat_range[1] - lat_range[0]) + lat_range[0]
        return np.column_stack((lat, normalize_longitudes(lon)))

    def sample_points(self, nsamp, skip=0):
        dim_num = 2
        count = 0
        collected = np.empty((0, 2))

        while collected.shape[0] < nsamp:
            curr_len = nsamp - collected.shape[0]
            samples = sobol_matrix(dim_num, curr_len, skip + count)
            candidates = self.candidate_points_from_unit(samples)
            valid = self.contains(candidates)
            collected = np.vstack((collected, candidates[valid]))
            count += curr_len

        return collected[:nsamp]

    def location_density(self, points_latlon):
        points_latlon = ensure_2d_points(points_latlon)
        mask = self.contains(points_latlon)
        density = np.zeros(points_latlon.shape[0], dtype=float)
        if self.area_measure <= 0:
            return density
        density[mask] = 1.0 / self.area_measure
        return density

    def cap_boundary_polygon(self, npts=181):
        if self.domain_type != "spherical_cap":
            raise ValueError("Cap boundary polygon is only defined for spherical_cap domains.")
        unit_xy = np.column_stack((np.linspace(0.0, 1.0, npts, endpoint=False), np.ones(npts)))
        boundary_latlon = self.candidate_points_from_unit(unit_xy)
        return np.column_stack((boundary_latlon[:, 1], boundary_latlon[:, 0]))

    def _validate_simple_spherical_polygons(self):
        for polygon in self.polygons_lonlat:
            longitudes = polygon[:, 0]
            if np.ptp(longitudes) > 180.0 + 1e-9:
                raise ValueError(
                    "Spherical polygon domains that cross the dateline are not supported in v1."
                )
            if np.any(polygon_contains_points(polygon, _POLE_TEST_POINTS)):
                raise ValueError(
                    "Spherical polygon domains that enclose a pole are not supported in v1."
                )

    def _compute_sample_bounds(self):
        if self.geometry_mode == "spherical" and self.domain_type == "globe":
            return np.array([[-90.0, 90.0], [-180.0, 180.0]])

        if self.geometry_mode == "spherical" and self.domain_type == "spherical_cap":
            boundary_polygon = self.cap_boundary_polygon(npts=721)
            lon = boundary_polygon[:, 0]
            lat = boundary_polygon[:, 1]
        else:
            if len(self.polygons_lonlat) == 0:
                raise ValueError("Polygon domains require at least one polygon.")
            all_points = np.vstack(self.polygons_lonlat)
            lon = all_points[:, 0]
            lat = all_points[:, 1]

        return np.array([[lat.min(), lat.max()], [lon.min(), lon.max()]])

    def _compute_area_measure(self):
        if self.geometry_mode == "spherical":
            if self.domain_type == "globe":
                return 4.0 * np.pi
            if self.domain_type == "spherical_cap":
                radius_rad = np.deg2rad(self.radius_deg)
                return 2.0 * np.pi * (1.0 - np.cos(radius_rad))
            return sum(spherical_polygon_area_sr(polygon) for polygon in self.polygons_lonlat)

        if len(self.polygons_lonlat) == 1 and is_axis_aligned_rectangle(self.polygons_lonlat[0]):
            polygon = self.polygons_lonlat[0]
            lon_range = [polygon[:, 0].min(), polygon[:, 0].max()]
            lat_range = [polygon[:, 1].min(), polygon[:, 1].max()]
            return np.abs(lon_range[1] - lon_range[0]) * np.abs(lat_range[1] - lat_range[0])

        sample_count = 8192
        samples = sobol_matrix(2, sample_count, 0)
        candidates = self.candidate_points_from_unit(samples)
        valid = self.contains(candidates)
        lat_bounds = self.sample_bounds[0]
        lon_bounds = self.sample_bounds[1]
        box_area = np.abs(lat_bounds[1] - lat_bounds[0]) * np.abs(lon_bounds[1] - lon_bounds[0])
        return box_area * (np.count_nonzero(valid) / sample_count)
