#!/usr/bin/env python3
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from obspy import geodetics

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def magnitude_to_moment_tensor_isotropic(mag: float) -> np.ndarray:
    m0 = 10 ** (1.5 * mag + 9.1)
    return np.array([m0, m0, m0, 0.0, 0.0, 0.0], dtype=float)


FEATURE_NAMES = [
    "Lat",
    "Lon",
    "Depth",
    "Distance_to_source_km",
    "Gaussian_variance",
    "m_rr",
    "m_tt",
    "m_pp",
    "m_rt",
    "m_rp",
    "m_tp",
]


def get_git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def load_sensors(inputs_path: Path) -> np.ndarray:
    # First 5 lines are config/file pointers, then sensor rows: lat, lon, variance, type, subtype
    return np.loadtxt(inputs_path, delimiter=",", skiprows=5)


def load_bounds(bounds_path: Path) -> dict:
    with bounds_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_events(bounds: dict, n_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(bounds["lat_range"][0], bounds["lat_range"][1], n_events)
    lon = rng.uniform(bounds["lon_range"][0], bounds["lon_range"][1], n_events)
    depth = rng.uniform(bounds["depth_range"][0], bounds["depth_range"][1], n_events)
    mag = rng.uniform(bounds["mag_range"][0], bounds["mag_range"][1], n_events)
    return np.column_stack([lat, lon, depth, mag])


def build_runtime_features(events: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    sens_lat = sensors[:, 0]
    sens_lon = sensors[:, 1]
    sens_var = sensors[:, 2]
    all_features = []

    for src_lat, src_lon, src_depth_km, src_mag in events:
        dists_deg = [
            geodetics.locations2degrees(src_lat, src_lon, s_lat, s_lon)
            for s_lat, s_lon in zip(sens_lat, sens_lon)
        ]
        dists_km = geodetics.degrees2kilometers(np.array(dists_deg))
        mt = magnitude_to_moment_tensor_isotropic(src_mag)

        n_sensors = len(sensors)
        features = np.zeros((n_sensors, 11), dtype=float)
        features[:, 0] = sens_lat - src_lat
        features[:, 1] = sens_lon - src_lon
        features[:, 2] = src_depth_km * 1000.0
        features[:, 3] = dists_km
        features[:, 4] = sens_var
        features[:, 5:11] = np.tile(mt, (n_sensors, 1))
        all_features.append(features)

    return np.vstack(all_features)


def load_training_features(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=FEATURE_NAMES)
    return df[FEATURE_NAMES].to_numpy(dtype=float)


def main() -> None:
    repo_root = REPO_ROOT
    output_dir = repo_root / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = repo_root / "x_scaler_38_1000000.pkl"
    inputs_path = repo_root / "inputs.dat"
    bounds_path = repo_root / "ta_array_domain.json"
    train_csv_path = repo_root / "maike_code" / "mixeddata (3).csv"

    n_events = 250
    seed = 7
    run_id = "stage0_ml_feature_alignment"
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(repo_root)
    params = (
        f"n_events={n_events};seed={seed};sensors=all;"
        "mapping=source_relative_lat_lon+depth_meters"
    )

    scaler = joblib.load(scaler_path)
    sensors = load_sensors(inputs_path)
    bounds = load_bounds(bounds_path)
    events = sample_events(bounds, n_events=n_events, seed=seed)

    runtime_features = build_runtime_features(events, sensors)
    train_features = load_training_features(train_csv_path)

    runtime_z = scaler.transform(runtime_features)
    train_z = scaler.transform(train_features)

    runtime_znorm = np.linalg.norm(runtime_z, axis=1)
    train_znorm = np.linalg.norm(train_z, axis=1)

    per_feature_rows = []
    runtime_abs = np.abs(runtime_z)
    for i, name in enumerate(FEATURE_NAMES):
        per_feature_rows.append(
            {
                "run_id": run_id,
                "git_sha": git_sha,
                "timestamp": timestamp,
                "params": params,
                "feature_index": i,
                "feature_name": name,
                "runtime_z_min": float(np.min(runtime_z[:, i])),
                "runtime_z_max": float(np.max(runtime_z[:, i])),
                "train_z_min": float(np.min(train_z[:, i])),
                "train_z_max": float(np.max(train_z[:, i])),
                "runtime_absz_gt10_frac": float(np.mean(runtime_abs[:, i] > 10.0)),
            }
        )

    summary_row = {
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "params": params,
        "feature_index": -1,
        "feature_name": "ALL",
        "runtime_z_min": float(np.min(runtime_z)),
        "runtime_z_max": float(np.max(runtime_z)),
        "train_z_min": float(np.min(train_z)),
        "train_z_max": float(np.max(train_z)),
        "runtime_absz_gt10_frac": float(np.mean(runtime_abs > 10.0)),
        "runtime_znorm_median": float(np.median(runtime_znorm)),
        "runtime_znorm_p99": float(np.quantile(runtime_znorm, 0.99)),
        "train_znorm_median": float(np.median(train_znorm)),
        "train_znorm_p99": float(np.quantile(train_znorm, 0.99)),
    }
    per_feature_rows.append(summary_row)

    metrics_df = pd.DataFrame(per_feature_rows)
    metrics_path = output_dir / "ml_feature_alignment_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    runtime_median = float(np.median(runtime_znorm))
    runtime_p99 = float(np.quantile(runtime_znorm, 0.99))
    feature_extreme_fails = [
        r["feature_name"] for r in per_feature_rows[:-1] if r["runtime_absz_gt10_frac"] > 0.01
    ]

    pass_criterion_1 = runtime_median < 6.0
    pass_criterion_2 = runtime_p99 < 8.0
    pass_criterion_3 = len(feature_extreme_fails) == 0
    overall_pass = pass_criterion_1 and pass_criterion_2 and pass_criterion_3

    report_lines = [
        "# ML Feature Alignment Report",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        f"- params: `{params}`",
        "",
        "## Dataset Summary",
        "",
        f"- Events sampled: {len(events)}",
        f"- Sensors used: {len(sensors)}",
        f"- Runtime feature rows: {len(runtime_features)}",
        f"- Training feature rows: {len(train_features)}",
        "",
        "## Z-Norm Comparison",
        "",
        f"- Runtime z-norm median: {runtime_median:.4f}",
        f"- Runtime z-norm p99: {runtime_p99:.4f}",
        f"- Training z-norm median: {np.median(train_znorm):.4f}",
        f"- Training z-norm p99: {np.quantile(train_znorm, 0.99):.4f}",
        "",
        "## AGENTS Pass Criteria",
        "",
        f"- Criterion 1 (runtime median < 6): {'PASS' if pass_criterion_1 else 'FAIL'}",
        f"- Criterion 2 (runtime p99 < 8): {'PASS' if pass_criterion_2 else 'FAIL'}",
        (
            "- Criterion 3 (no feature with |z|>10 for >1% rows): "
            + ("PASS" if pass_criterion_3 else "FAIL")
        ),
        (
            "- Features failing Criterion 3: "
            + (", ".join(feature_extreme_fails) if feature_extreme_fails else "none")
        ),
        "",
        f"## Overall: {'PASS' if overall_pass else 'FAIL'}",
        "",
        f"Metrics CSV: `{metrics_path.relative_to(repo_root)}`",
    ]

    report_path = output_dir / "ml_feature_alignment_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {report_path}")
    print(f"Runtime z-norm median: {runtime_median:.6f}")
    print(f"Runtime z-norm p99: {runtime_p99:.6f}")
    print(f"Overall pass: {overall_pass}")


if __name__ == "__main__":
    main()
