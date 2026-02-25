# ML Feature Alignment Report

- run_id: `stage0_ml_feature_alignment`
- git_sha: `ab8fe30515d88a09385183e1f0e3b08978c2300c`
- timestamp_utc: `2026-02-25T21:17:12.361453+00:00`
- params: `n_events=250;seed=7;sensors=all;mapping=source_relative_lat_lon+depth_meters`

## Dataset Summary

- Events sampled: 250
- Sensors used: 9
- Runtime feature rows: 2250
- Training feature rows: 200000

## Z-Norm Comparison

- Runtime z-norm median: 4.9081
- Runtime z-norm p99: 17658392.6507
- Training z-norm median: 2.9677
- Training z-norm p99: 5.4648

## AGENTS Pass Criteria

- Criterion 1 (runtime median < 6): PASS
- Criterion 2 (runtime p99 < 8): FAIL
- Criterion 3 (no feature with |z|>10 for >1% rows): FAIL
- Features failing Criterion 3: m_rr, m_tt, m_pp

## Overall: FAIL

Metrics CSV: `experiments/stage0/ml_feature_alignment_metrics.csv`
