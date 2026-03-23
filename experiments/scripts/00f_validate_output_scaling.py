#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ml_utils


def get_git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_sensors(inputs_path: Path) -> np.ndarray:
    return np.loadtxt(inputs_path, delimiter=",", skiprows=5)


def sample_indomain_events(sensors: np.ndarray, n_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat_center = float(np.mean(sensors[:, 0]))
    lon_center = float(np.mean(sensors[:, 1]))
    lat = rng.uniform(lat_center - 0.2, lat_center + 0.2, n_events)
    lon = rng.uniform(lon_center - 0.2, lon_center + 0.2, n_events)
    depth = rng.uniform(5.0, 20.0, n_events)
    mag = np.full(n_events, ml_utils.REFERENCE_MW, dtype=float)
    return np.column_stack([lat, lon, depth, mag])


def main() -> None:
    os.environ.setdefault("SEISMIC_OED_FIDELITY_MAP", "clamped_linear")
    output_dir = REPO_ROOT / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = load_sensors(REPO_ROOT / "inputs.dat")
    power_model = ml_utils.get_power_model()
    gaussian_variance = ml_utils.map_sensor_fidelity_to_gaussian_variance(
        sensors[:, 2], strategy=os.environ["SEISMIC_OED_FIDELITY_MAP"]
    )

    mw_targets = np.array([3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    events = sample_indomain_events(sensors, n_events=25, seed=31415)

    run_id = "stage0_output_scaling_validation"
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)

    rows = []
    all_abs_errors = []

    for event_idx, theta_ref in enumerate(events):
        ref_pred = power_model.predict_log_power(
            theta_ref, sensors, gaussian_variance=gaussian_variance
        )

        for mw_target in mw_targets:
            theta_target = theta_ref.copy()
            theta_target[3] = float(mw_target)

            pred = power_model.predict_log_power(
                theta_target, sensors, gaussian_variance=gaussian_variance
            )

            expected_shift = ml_utils.log_power_magnitude_shift(
                mw_target, mw_ref=ml_utils.REFERENCE_MW
            )
            delta = pred - ref_pred
            abs_error = np.abs(delta - expected_shift)
            all_abs_errors.extend(abs_error.tolist())

            rows.append(
                {
                    "run_id": run_id,
                    "git_sha": git_sha,
                    "timestamp": timestamp,
                    "params": (
                        f"event_idx={event_idx};mw_target={mw_target:.2f};"
                        f"mw_ref={ml_utils.REFERENCE_MW:.2f};"
                        f"fidelity_map={os.environ['SEISMIC_OED_FIDELITY_MAP']}"
                    ),
                    "event_idx": int(event_idx),
                    "mw_target": float(mw_target),
                    "mw_ref": float(ml_utils.REFERENCE_MW),
                    "expected_shift": float(expected_shift),
                    "mean_delta": float(np.mean(delta)),
                    "std_delta": float(np.std(delta, ddof=0)),
                    "max_abs_error": float(np.max(abs_error)),
                    "mean_abs_error": float(np.mean(abs_error)),
                    "num_sensors": int(len(delta)),
                }
            )

    metrics_df = pd.DataFrame(rows)
    out_csv = output_dir / "output_scaling_validation_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    all_abs_errors = np.asarray(all_abs_errors, dtype=float)
    max_abs_error_overall = float(np.max(all_abs_errors))
    mean_abs_error_overall = float(np.mean(all_abs_errors))
    pass_threshold = 1e-9
    overall_pass = max_abs_error_overall <= pass_threshold

    report = [
        "# Output Scaling Validation Report",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        f"- fidelity_mapping: `{os.environ['SEISMIC_OED_FIDELITY_MAP']}`",
        f"- reference_mw: `{ml_utils.REFERENCE_MW:.2f}`",
        f"- sampled_events: `{len(events)}`",
        f"- mw_targets: `{mw_targets.tolist()}`",
        "",
        "## Validation Rule",
        "",
        "- Expected shift in natural-log power:",
        "  `delta_log_power = 3 * ln(10) * (Mw_target - Mw_ref)`",
        "- Compare direct model predictions at each `Mw_target` against the fixed-`Mw_ref` prediction plus this analytic shift.",
        "",
        "## Results",
        "",
        f"- overall_mean_abs_error: `{mean_abs_error_overall:.3e}`",
        f"- overall_max_abs_error: `{max_abs_error_overall:.3e}`",
        f"- pass_threshold: `{pass_threshold:.1e}`",
        f"- overall_pass: `{'PASS' if overall_pass else 'FAIL'}`",
        "",
        f"Metrics CSV: `experiments/stage0/{out_csv.name}`",
    ]

    out_md = output_dir / "output_scaling_validation_report.md"
    out_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Overall pass: {overall_pass}")
    print(f"Overall max abs error: {max_abs_error_overall:.3e}")


if __name__ == "__main__":
    main()
