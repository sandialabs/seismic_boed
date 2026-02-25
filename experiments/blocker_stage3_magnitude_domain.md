# Blocker: Stage 3 Magnitude Posterior Domain Mismatch

- run_id: `stage0_5_mt_magnitude_domain_check`
- git_sha: `ab8fe30515d88a09385183e1f0e3b08978c2300c`
- timestamp_utc: `2026-02-25T21:27:47.513727+00:00`

## Blocking Condition

- Scenario A (fixed Mw=5.0) passed while Scenario B (sampled magnitude range) failed.
- Magnitude inference is blocked until retraining with magnitude-varying MT data.

## Evidence

- A overall_pass: True
- B overall_pass: False
- B mt_diag_any_absz_gt10_frac: 45.6667%
- B non_mt_any_absz_gt10_frac: 0.0000%

## Proposed Fix

- Retrain the ML power model using training data that spans the intended magnitude range and corresponding MT scaling.
- Re-run Stage 0.5 and require Scenario B to pass before Stage 3 posterior inference.
