#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from obspy import geodetics

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import like_models
import ml_utils

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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def load_sensors(inputs_path: Path) -> np.ndarray:
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


def build_features(theta: np.ndarray, sensors: np.ndarray, mapped_gv: np.ndarray) -> np.ndarray:
    src_lat, src_lon, src_depth_km, src_mag = theta
    sens_lat = sensors[:, 0]
    sens_lon = sensors[:, 1]
    dists_deg = [
        geodetics.locations2degrees(src_lat, src_lon, slat, slon)
        for slat, slon in zip(sens_lat, sens_lon)
    ]
    dists_km = geodetics.degrees2kilometers(np.array(dists_deg))

    mt = ml_utils.magnitude_to_moment_tensor_isotropic(src_mag)
    n_sensors = sensors.shape[0]
    features = np.zeros((n_sensors, 11), dtype=float)
    features[:, 0] = sens_lat - src_lat
    features[:, 1] = sens_lon - src_lon
    features[:, 2] = src_depth_km * 1000.0
    features[:, 3] = dists_km
    features[:, 4] = mapped_gv
    features[:, 5:11] = np.tile(mt, (n_sensors, 1))
    return features


def evaluate_strategy(strategy: str, events: np.ndarray, sensors: np.ndarray, scaler) -> dict:
    os.environ["SEISMIC_OED_FIDELITY_MAP"] = strategy
    like_models.reset_power_gate_diagnostics()

    all_features = []
    finite_preds = 0
    total_preds = 0
    enabled_count = 0
    for theta in events:
        mapped = ml_utils.map_sensor_fidelity_to_gaussian_variance(sensors[:, 2], strategy=strategy)
        all_features.append(build_features(theta, sensors, mapped))
        out = like_models.compute_power(theta, sensors, stype="seismic")
        enabled = out[:, 3].astype(bool)
        enabled_count += int(np.count_nonzero(enabled))
        finite_preds += int(np.count_nonzero(np.isfinite(out[:, 0])))
        total_preds += int(out.shape[0])

    feats = np.vstack(all_features)
    z = scaler.transform(feats)
    znorm = np.linalg.norm(z, axis=1)
    gv_col = z[:, 4]

    d = like_models.get_power_gate_diagnostics()
    return {
        "strategy": strategy,
        "z_gv_min": float(np.min(gv_col)),
        "z_gv_max": float(np.max(gv_col)),
        "z_gv_abs_gt10_frac": float(np.mean(np.abs(gv_col) > 10.0)),
        "znorm_median": float(np.median(znorm)),
        "znorm_p99": float(np.quantile(znorm, 0.99)),
        "enabled_fraction": float(enabled_count / max(total_preds, 1)),
        "finite_pred_fraction": float(finite_preds / max(total_preds, 1)),
        "reject_fraction": float(d["power_disabled_count"] / max(d["total_checked"], 1)),
        "failed_gaussian_variance": int(d["failed_gaussian_variance"]),
    }


def recommend(rows: List[Dict]) -> str:
    by_name = {r["strategy"]: r for r in rows}
    clamped = by_name["clamped_linear"]
    direct = by_name["direct"]
    fixed = by_name["fixed_nominal"]
    if (
        clamped["enabled_fraction"] >= fixed["enabled_fraction"]
        and clamped["reject_fraction"] <= fixed["reject_fraction"] + 1e-12
        and direct["enabled_fraction"] < 0.1
    ):
        return "clamped_linear"
    return "fixed_nominal"


def main() -> None:
    out_dir = REPO_ROOT / "experiments" / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)

    sensors = load_sensors(REPO_ROOT / "inputs.dat")
    bounds = load_bounds(REPO_ROOT / "ta_array_domain.json")
    scaler = joblib.load(REPO_ROOT / "x_scaler_38_1000000.pkl")
    events = sample_events(bounds, n_events=220, seed=31)

    run_id = "stage0_fidelity_mapping_sensitivity"
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)
    params = "events=220;cohort=full_domain;mapping=direct|clamped_linear|fixed_nominal"

    rows = []
    for strategy in ["direct", "clamped_linear", "fixed_nominal"]:
        r = evaluate_strategy(strategy, events, sensors, scaler)
        r.update({"run_id": run_id, "git_sha": git_sha, "timestamp": timestamp, "params": params})
        rows.append(r)

    rec = recommend(rows)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "fidelity_mapping_sensitivity.csv"
    df.to_csv(out_csv, index=False)

    formula = "gaussian_variance = clip(1 + 2 * ((fidelity - 0.0) / 0.2), 1, 3)"
    rationale = [
        "- `direct` keeps raw sensor value and usually fails Gaussian-variance gate.",
        "- `fixed_nominal` is gate-safe but discards sensor-to-sensor fidelity differences.",
        "- `clamped_linear` stays in training support and preserves relative fidelity ranking.",
    ]
    report = [
        "# Fidelity Mapping Recommendation",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        "",
        "## Recommended Strategy",
        "",
        f"- recommendation: `{rec}`",
        "- selected_formula: `gaussian_variance = clip(1 + 2 * ((fidelity - raw_min) / (raw_max - raw_min)), 1, 3)`",
        "- configured_raw_range: `raw_min=0.0, raw_max=0.2`",
        "",
        "## Why",
        "",
        *rationale,
        "",
        (
            "- power_prediction_note: "
            + (
                "`torch available; finite prediction fractions reflect live ML inference`"
                if ml_utils.TORCH_AVAILABLE
                else "`torch unavailable in this environment; finite prediction fractions are 0 and gate metrics are used for comparison`"
            )
        ),
        "",
        "## Concrete Formula Used",
        "",
        f"- `{formula}`",
        "",
        f"Sensitivity CSV: `experiments/stage0/{out_csv.name}`",
    ]
    out_md = out_dir / "fidelity_mapping_recommendation.md"
    out_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Recommended mapping: {rec}")


if __name__ == "__main__":
    main()
