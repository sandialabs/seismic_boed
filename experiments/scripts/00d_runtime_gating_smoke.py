#!/usr/bin/env python3
import json
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

import like_models


def get_git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def load_sensors(inputs_path: Path) -> np.ndarray:
    return np.loadtxt(inputs_path, delimiter=",", skiprows=5)


def load_bounds(bounds_path: Path) -> dict:
    with bounds_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_full_domain_events(bounds: dict, n_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(bounds["lat_range"][0], bounds["lat_range"][1], n_events)
    lon = rng.uniform(bounds["lon_range"][0], bounds["lon_range"][1], n_events)
    depth = rng.uniform(bounds["depth_range"][0], bounds["depth_range"][1], n_events)
    mag = rng.uniform(bounds["mag_range"][0], bounds["mag_range"][1], n_events)
    return np.column_stack([lat, lon, depth, mag])


def sample_indomain_events(sensors: np.ndarray, n_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat_center = float(np.mean(sensors[:, 0]))
    lon_center = float(np.mean(sensors[:, 1]))
    lat = rng.uniform(lat_center - 0.2, lat_center + 0.2, n_events)
    lon = rng.uniform(lon_center - 0.2, lon_center + 0.2, n_events)
    depth = rng.uniform(5.0, 20.0, n_events)
    mag = rng.uniform(4.95, 5.05, n_events)
    return np.column_stack([lat, lon, depth, mag])


def run_cohort(events: np.ndarray, sensors: np.ndarray) -> dict:
    like_models.reset_power_gate_diagnostics()
    enabled_total = 0
    checked_total = 0
    finite_pred_total = 0

    for theta in events:
        power = like_models.compute_power(theta, sensors, stype="seismic")
        enabled = power[:, 3].astype(bool)
        enabled_total += int(np.count_nonzero(enabled))
        checked_total += int(len(enabled))
        finite_pred_total += int(np.count_nonzero(np.isfinite(power[:, 0])))

    d = like_models.get_power_gate_diagnostics()
    d["enabled_count"] = int(enabled_total)
    d["checked_count"] = int(checked_total)
    d["finite_pred_count"] = int(finite_pred_total)
    d["reject_fraction"] = float(d["power_disabled_count"] / max(d["total_checked"], 1))
    return d


def main() -> None:
    os.environ["SEISMIC_OED_FIDELITY_MAP"] = "clamped_linear"
    output_dir = REPO_ROOT / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = load_sensors(REPO_ROOT / "inputs.dat")
    bounds = load_bounds(REPO_ROOT / "ta_array_domain.json")
    full_events = sample_full_domain_events(bounds, n_events=350, seed=21)
    indomain_events = sample_indomain_events(sensors, n_events=350, seed=22)

    run_id = "stage0_runtime_gating_smoke"
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)

    full_stats = run_cohort(full_events, sensors)
    in_stats = run_cohort(indomain_events, sensors)

    rows = []
    for cohort, stats, params in [
        ("full_domain", full_stats, "events=350;source=ta_array_domain"),
        ("synthetic_in_domain", in_stats, "events=350;depth_km=5..20;mw=4.95..5.05"),
    ]:
        row = {
            "run_id": run_id,
            "git_sha": git_sha,
            "timestamp": timestamp,
            "params": params,
            "cohort": cohort,
        }
        row.update(stats)
        rows.append(row)

    out_csv = output_dir / "runtime_gating_smoke_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    full_nonzero_reject = full_stats["power_disabled_count"] > 0
    in_near_zero_reject = in_stats["reject_fraction"] < 0.02
    overall_pass = full_nonzero_reject and in_near_zero_reject

    report = [
        "# Runtime Gating Smoke Report",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        "- fidelity_mapping: `clamped_linear`",
        "",
        "## Results",
        "",
        (
            f"- Full-domain rejects: {full_stats['power_disabled_count']} / "
            f"{full_stats['total_checked']} ({full_stats['reject_fraction']:.2%})"
        ),
        (
            f"- In-domain rejects: {in_stats['power_disabled_count']} / "
            f"{in_stats['total_checked']} ({in_stats['reject_fraction']:.2%})"
        ),
        f"- Full-domain non-zero rejects: {'PASS' if full_nonzero_reject else 'FAIL'}",
        f"- In-domain near-zero rejects (<2%): {'PASS' if in_near_zero_reject else 'FAIL'}",
        "",
        "## Gate Failure Counters (full-domain)",
        "",
        f"- failed_lat_local: {full_stats['failed_lat_local']}",
        f"- failed_lon_local: {full_stats['failed_lon_local']}",
        f"- failed_depth_m: {full_stats['failed_depth_m']}",
        f"- failed_gaussian_variance: {full_stats['failed_gaussian_variance']}",
        f"- failed_mt_norm: {full_stats['failed_mt_norm']}",
        "",
        f"## Overall: {'PASS' if overall_pass else 'FAIL'}",
        "",
        f"Metrics CSV: `experiments/stage0/{out_csv.name}`",
    ]

    out_md = output_dir / "runtime_gating_smoke_report.md"
    out_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Overall pass: {overall_pass}")


if __name__ == "__main__":
    main()
