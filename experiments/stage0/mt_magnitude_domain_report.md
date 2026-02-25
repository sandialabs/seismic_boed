# Stage 0.5 MT-Magnitude Domain Check

- run_id: `stage0_5_mt_magnitude_domain_check`
- git_sha: `ab8fe30515d88a09385183e1f0e3b08978c2300c`
- timestamp_utc: `2026-02-25T21:27:47.513727+00:00`
- params: `n_events=600;seed=11;sensors=all`

## Scenario Results

- A fixed Mw=5.0: median z-norm=5.4764, p99=5.9367, overall_pass=True
- B sampled Mw from domain mag_range=[0.5, 9.5]: median z-norm=5.4632, p99=16332630.9576, overall_pass=False

## Per-Feature |z|>10 Exceedance (B sampled Mw)

- m_tt (idx 6): 45.6667%
- m_pp (idx 7): 45.6667%
- m_rr (idx 5): 45.0000%
- Lat (idx 0): 0.0000%
- Lon (idx 1): 0.0000%
- Depth (idx 2): 0.0000%
- Distance_to_source_km (idx 3): 0.0000%
- Gaussian_variance (idx 4): 0.0000%
- m_rt (idx 8): 0.0000%
- m_rp (idx 9): 0.0000%
- m_tp (idx 10): 0.0000%

## MT Diagonal OOD Attribution

- MT diagonal features tested: m_rr, m_tt, m_pp
- Any MT diagonal |z|>10 row fraction (B): 45.6667%
- Any non-MT feature |z|>10 row fraction (B): 0.0000%
- MT diagonal alone cause OOD under (B): True

## Conclusion

**Model valid only near Mw~5**

Metrics CSV: `experiments/stage0/mt_magnitude_domain_metrics.csv`
