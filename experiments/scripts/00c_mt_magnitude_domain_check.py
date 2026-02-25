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
MT_DIAG_FEATURES = ["m_rr", "m_tt", "m_pp"]
MT_DIAG_INDICES = [5, 6, 7]


def magnitude_to_moment_tensor_isotropic(mag: float) -> np.ndarray:
    m0 = 10 ** (1.5 * mag + 9.1)
    return np.array([m0, m0, m0, 0.0, 0.0, 0.0], dtype=float)


def get_git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def load_sensors(inputs_path: Path) -> np.ndarray:
    return np.loadtxt(inputs_path, delimiter=",", skiprows=5)


def load_bounds(bounds_path: Path) -> dict:
    with bounds_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_events(bounds: dict, n_events: int, seed: int, mode: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(bounds["lat_range"][0], bounds["lat_range"][1], n_events)
    lon = rng.uniform(bounds["lon_range"][0], bounds["lon_range"][1], n_events)
    depth = rng.uniform(bounds["depth_range"][0], bounds["depth_range"][1], n_events)

    if mode == "A_fixed_mw5":
        mag = np.full(n_events, 5.0, dtype=float)
    elif mode == "B_sampled_mag_range":
        mag = rng.uniform(bounds["mag_range"][0], bounds["mag_range"][1], n_events)
    else:
        raise ValueError(f"Unknown mode: {mode}")

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


def scenario_stats(scaler, features: np.ndarray, scenario: str) -> dict:
    z = scaler.transform(features)
    abs_z = np.abs(z)
    znorm = np.linalg.norm(z, axis=1)

    feature_rows = []
    for i, fname in enumerate(FEATURE_NAMES):
        feature_rows.append(
            {
                "scenario": scenario,
                "feature_index": i,
                "feature_name": fname,
                "z_min": float(np.min(z[:, i])),
                "z_max": float(np.max(z[:, i])),
                "absz_gt10_frac": float(np.mean(abs_z[:, i] > 10.0)),
            }
        )

    summary = {
        "scenario": scenario,
        "rows": int(features.shape[0]),
        "znorm_median": float(np.median(znorm)),
        "znorm_p95": float(np.quantile(znorm, 0.95)),
        "znorm_p99": float(np.quantile(znorm, 0.99)),
        "all_absz_gt10_frac": float(np.mean(abs_z > 10.0)),
        "mt_diag_any_absz_gt10_frac": float(
            np.mean(np.any(abs_z[:, MT_DIAG_INDICES] > 10.0, axis=1))
        ),
        "non_mt_any_absz_gt10_frac": float(
            np.mean(np.any(abs_z[:, :5] > 10.0, axis=1) | np.any(abs_z[:, 8:] > 10.0, axis=1))
        ),
    }

    pass_criterion_1 = summary["znorm_median"] < 6.0
    pass_criterion_2 = summary["znorm_p99"] < 8.0
    pass_criterion_3 = all(row["absz_gt10_frac"] <= 0.01 for row in feature_rows)
    summary["pass_criterion_1_median_lt_6"] = bool(pass_criterion_1)
    summary["pass_criterion_2_p99_lt_8"] = bool(pass_criterion_2)
    summary["pass_criterion_3_no_feat_gt1pct_absz10"] = bool(pass_criterion_3)
    summary["overall_pass"] = bool(pass_criterion_1 and pass_criterion_2 and pass_criterion_3)

    return {"z": z, "feature_rows": feature_rows, "summary": summary}


def main() -> None:
    output_dir = REPO_ROOT / "experiments" / "stage0"
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = REPO_ROOT / "x_scaler_38_1000000.pkl"
    inputs_path = REPO_ROOT / "inputs.dat"
    bounds_path = REPO_ROOT / "ta_array_domain.json"

    n_events = 600
    seed = 11
    run_id = "stage0_5_mt_magnitude_domain_check"
    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)

    scaler = joblib.load(scaler_path)
    sensors = load_sensors(inputs_path)
    bounds = load_bounds(bounds_path)

    events_a = sample_events(bounds, n_events=n_events, seed=seed, mode="A_fixed_mw5")
    events_b = sample_events(bounds, n_events=n_events, seed=seed, mode="B_sampled_mag_range")

    feats_a = build_runtime_features(events_a, sensors)
    feats_b = build_runtime_features(events_b, sensors)

    stats_a = scenario_stats(scaler, feats_a, "A_fixed_mw5")
    stats_b = scenario_stats(scaler, feats_b, "B_sampled_mag_range")

    rows = []
    for feature_row in stats_a["feature_rows"] + stats_b["feature_rows"]:
        rows.append(
            {
                "run_id": run_id,
                "git_sha": git_sha,
                "timestamp": timestamp,
                "params": f"n_events={n_events};seed={seed};sensors=all",
                **feature_row,
            }
        )

    for summary in [stats_a["summary"], stats_b["summary"]]:
        rows.append(
            {
                "run_id": run_id,
                "git_sha": git_sha,
                "timestamp": timestamp,
                "params": f"n_events={n_events};seed={seed};sensors=all",
                "scenario": summary["scenario"],
                "feature_index": -1,
                "feature_name": "ALL",
                "z_min": np.nan,
                "z_max": np.nan,
                "absz_gt10_frac": summary["all_absz_gt10_frac"],
                "rows": summary["rows"],
                "znorm_median": summary["znorm_median"],
                "znorm_p95": summary["znorm_p95"],
                "znorm_p99": summary["znorm_p99"],
                "mt_diag_any_absz_gt10_frac": summary["mt_diag_any_absz_gt10_frac"],
                "non_mt_any_absz_gt10_frac": summary["non_mt_any_absz_gt10_frac"],
                "pass_criterion_1_median_lt_6": summary["pass_criterion_1_median_lt_6"],
                "pass_criterion_2_p99_lt_8": summary["pass_criterion_2_p99_lt_8"],
                "pass_criterion_3_no_feat_gt1pct_absz10": summary[
                    "pass_criterion_3_no_feat_gt1pct_absz10"
                ],
                "overall_pass": summary["overall_pass"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "mt_magnitude_domain_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Is OOD in B attributable to MT diagonal features alone?
    b_feat = pd.DataFrame(stats_b["feature_rows"])
    mt_diag_exceed = b_feat[b_feat["feature_name"].isin(MT_DIAG_FEATURES)]["absz_gt10_frac"]
    non_mt_exceed = b_feat[~b_feat["feature_name"].isin(MT_DIAG_FEATURES)]["absz_gt10_frac"]
    mt_diag_only_cause = bool((mt_diag_exceed > 0.01).any() and (non_mt_exceed <= 0.01).all())

    a_pass = bool(stats_a["summary"]["overall_pass"])
    b_pass = bool(stats_b["summary"]["overall_pass"])

    if a_pass and not b_pass:
        conclusion = "Model valid only near Mw~5"
    else:
        conclusion = "Model valid across range"

    report_lines = [
        "# Stage 0.5 MT-Magnitude Domain Check",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        f"- params: `n_events={n_events};seed={seed};sensors=all`",
        "",
        "## Scenario Results",
        "",
        f"- A fixed Mw=5.0: median z-norm={stats_a['summary']['znorm_median']:.4f}, p99={stats_a['summary']['znorm_p99']:.4f}, overall_pass={a_pass}",
        f"- B sampled Mw from domain mag_range={bounds['mag_range']}: median z-norm={stats_b['summary']['znorm_median']:.4f}, p99={stats_b['summary']['znorm_p99']:.4f}, overall_pass={b_pass}",
        "",
        "## Per-Feature |z|>10 Exceedance (B sampled Mw)",
        "",
    ]

    b_sorted = b_feat.sort_values("absz_gt10_frac", ascending=False)
    for _, row in b_sorted.iterrows():
        report_lines.append(
            f"- {row['feature_name']} (idx {int(row['feature_index'])}): {row['absz_gt10_frac']:.4%}"
        )

    report_lines += [
        "",
        "## MT Diagonal OOD Attribution",
        "",
        f"- MT diagonal features tested: {', '.join(MT_DIAG_FEATURES)}",
        f"- Any MT diagonal |z|>10 row fraction (B): {stats_b['summary']['mt_diag_any_absz_gt10_frac']:.4%}",
        f"- Any non-MT feature |z|>10 row fraction (B): {stats_b['summary']['non_mt_any_absz_gt10_frac']:.4%}",
        f"- MT diagonal alone cause OOD under (B): {mt_diag_only_cause}",
        "",
        "## Conclusion",
        "",
        f"**{conclusion}**",
        "",
        f"Metrics CSV: `experiments/stage0/{metrics_path.name}`",
    ]

    report_path = output_dir / "mt_magnitude_domain_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if a_pass and not b_pass:
        blocker_path = REPO_ROOT / "experiments" / "blocker_stage3_magnitude_domain.md"
        blocker_lines = [
            "# Blocker: Stage 3 Magnitude Posterior Domain Mismatch",
            "",
            f"- run_id: `{run_id}`",
            f"- git_sha: `{git_sha}`",
            f"- timestamp_utc: `{timestamp}`",
            "",
            "## Blocking Condition",
            "",
            "- Scenario A (fixed Mw=5.0) passed while Scenario B (sampled magnitude range) failed.",
            "- Magnitude inference is blocked until retraining with magnitude-varying MT data.",
            "",
            "## Evidence",
            "",
            f"- A overall_pass: {a_pass}",
            f"- B overall_pass: {b_pass}",
            f"- B mt_diag_any_absz_gt10_frac: {stats_b['summary']['mt_diag_any_absz_gt10_frac']:.4%}",
            f"- B non_mt_any_absz_gt10_frac: {stats_b['summary']['non_mt_any_absz_gt10_frac']:.4%}",
            "",
            "## Proposed Fix",
            "",
            "- Retrain the ML power model using training data that spans the intended magnitude range and corresponding MT scaling.",
            "- Re-run Stage 0.5 and require Scenario B to pass before Stage 3 posterior inference.",
        ]
        blocker_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")
        print(f"Wrote: {blocker_path}")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {report_path}")
    print(f"Conclusion: {conclusion}")
    print(f"A overall_pass: {a_pass}")
    print(f"B overall_pass: {b_pass}")


if __name__ == "__main__":
    main()
