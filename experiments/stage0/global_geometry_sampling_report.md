# Global Geometry Sampling Validation

Sample count: `4096`

## Result
- Legacy rectangular global sampling puts `0.333` of points above `|lat| >= 60°`.
- Spherical surface sampling puts `0.134` of points above `|lat| >= 60°`.
- Legacy equatorial share (`|lat| <= 30°`) is `0.333`.
- Spherical equatorial share (`|lat| <= 30°`) is `0.500`.

## Interpretation
- The old global rectangle transform is uniform in latitude, so it over-samples polar regions relative to surface area.
- The spherical transform is close to uniform in `sin(latitude)`, which is the correct signature for uniform sampling on the sphere.

## Histogram Diagnostics
- Legacy latitude-bin spread: `1.002` max/min.
- New latitude-bin spread: `5.893` max/min.
- Legacy `sin(lat)`-bin spread: `3.048` max/min.
- New `sin(lat)`-bin spread: `1.002` max/min.

The new sampler is the expected latitude-density correction for teleseismic/global geometry runs.
