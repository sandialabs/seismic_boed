# Runtime Gating Smoke Report

- run_id: `stage0_runtime_gating_smoke`
- git_sha: `ab8fe30515d88a09385183e1f0e3b08978c2300c`
- timestamp_utc: `2026-02-25T22:30:02.000147+00:00`
- fidelity_mapping: `clamped_linear`

## Results

- Full-domain rejects: 3111 / 3150 (98.76%)
- In-domain rejects: 0 / 3150 (0.00%)
- Full-domain non-zero rejects: PASS
- In-domain near-zero rejects (<2%): PASS

## Gate Failure Counters (full-domain)

- failed_lat_local: 0
- failed_lon_local: 681
- failed_depth_m: 2025
- failed_gaussian_variance: 0
- failed_mt_norm: 2997

## Overall: PASS

Metrics CSV: `experiments/stage0/runtime_gating_smoke_metrics.csv`
