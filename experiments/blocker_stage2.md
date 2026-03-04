# Blocker: Stage 2 Isotropy Audit

- run_id: `stage2_isotropy_audit_20260304T211645Z`
- git_sha: `452ec303b59d6394f9f336a1774d8405d9ccda29`
- timestamp_utc: `2026-03-04T21:16:45.003583+00:00`
- params: `{"hist_bins": 80, "input_csv": "/Users/jpcalla/Desktop/seismic_oed/maike_code/mixeddata (3).csv", "max_rows": null, "output_dir": "/Users/jpcalla/Desktop/seismic_oed/experiments/stage2", "power_floor": 1e-30, "progress_every": 5000, "seed": 7, "source_for_prediction": {"lat": 0.0, "lon": 0.0, "seismic_type": 0}}`
- input_csv: `/Users/jpcalla/Desktop/seismic_oed/maike_code/mixeddata (3).csv`

## Exact Traceback

```text
Traceback (most recent call last):
  File "/Users/jpcalla/Desktop/seismic_oed/experiments/scripts/02_isotropy_audit.py", line 400, in main
    power_model = ml_utils.get_power_model()
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jpcalla/Desktop/seismic_oed/ml_utils.py", line 264, in get_power_model
    raise RuntimeError(
RuntimeError: torch is not installed; cannot initialize seismic power model.
```

## Proposed Fix

- Install `torch` in the Python environment used to run this script.
- Run with an environment that has `torch`, `numpy`, `pandas`, `matplotlib`, `joblib`, and `obspy` available.
- On HPC, activate the project env and re-run: `python3 experiments/scripts/02_isotropy_audit.py`.
