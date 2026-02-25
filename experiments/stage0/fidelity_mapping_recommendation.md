# Fidelity Mapping Recommendation

- run_id: `stage0_fidelity_mapping_sensitivity`
- git_sha: `ab8fe30515d88a09385183e1f0e3b08978c2300c`
- timestamp_utc: `2026-02-25T22:30:44.491977+00:00`

## Recommended Strategy

- recommendation: `clamped_linear`
- selected_formula: `gaussian_variance = clip(1 + 2 * ((fidelity - raw_min) / (raw_max - raw_min)), 1, 3)`
- configured_raw_range: `raw_min=0.0, raw_max=0.2`

## Why

- `direct` keeps raw sensor value and usually fails Gaussian-variance gate.
- `fixed_nominal` is gate-safe but discards sensor-to-sensor fidelity differences.
- `clamped_linear` stays in training support and preserves relative fidelity ranking.

- power_prediction_note: `torch unavailable in this environment; finite prediction fractions are 0 and gate metrics are used for comparison`

## Concrete Formula Used

- `gaussian_variance = clip(1 + 2 * ((fidelity - 0.0) / 0.2), 1, 3)`

Sensitivity CSV: `experiments/stage0/fidelity_mapping_sensitivity.csv`
