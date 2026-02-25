#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires pandas for CSV/summary outputs. Install pandas before running."
    ) from exc

try:
    from obspy import geodetics
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires obspy for geodetic distance calculations. Install obspy before running."
    ) from exc

try:
    from scipy import stats
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires scipy for Pearson/Spearman metrics. Install scipy before running."
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires matplotlib to write required PNG outputs. Install matplotlib before running."
    ) from exc

REPO_ROOT = Path(os.environ.get("SEISMIC_OED_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires torch for ML power inference. Install torch in this environment."
    ) from exc

try:
    import mpi4py  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "Stage 1 requires mpi4py in the environment. Install mpi4py before running."
    ) from exc

import like_models
import ml_utils
import utils


def get_git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: SNR crosswalk (old model vs power-derived).")
    parser.add_argument("--inputs", default="inputs.dat", help="Path to inputs.dat")
    parser.add_argument("--seed", type=int, default=20260225, help="Reproducible seed / Sobol skip")
    parser.add_argument(
        "--n-events",
        type=int,
        default=2500,
        help="Number of sampled events. Total pairs = n_events * n_sensors (must be >= 20k).",
    )
    parser.add_argument(
        "--noise-floors",
        default="1e-8,3e-8,1e-7",
        help="Comma-separated linear noise floors (power units).",
    )
    return parser.parse_args()


def compute_correlations(old_vals: np.ndarray, new_vals: np.ndarray) -> tuple[float, float]:
    if old_vals.size < 2:
        return np.nan, np.nan
    pearson = stats.pearsonr(old_vals, new_vals).statistic
    spearman = stats.spearmanr(old_vals, new_vals).statistic
    return float(pearson), float(spearman)


def compute_metric_row(
    df: pd.DataFrame, cohort: str, extrapolation: bool, noise_floor: float, metadata: dict
) -> dict:
    work = df.copy()
    work["new_log_snr"] = work["pred_log_power"].to_numpy() - np.log(noise_floor)

    valid = np.isfinite(work["old_log_snr"].to_numpy()) & np.isfinite(work["new_log_snr"].to_numpy())
    work = work.loc[valid].copy()

    if work.empty:
        out = {
            "cohort": cohort,
            "extrapolation": extrapolation,
            "noise_floor": float(noise_floor),
            "n_pairs_total": int(len(df)),
            "n_pairs_used": 0,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse_z": np.nan,
            "frac_abs_resid_gt_1sigma": np.nan,
            "frac_abs_resid_gt_2sigma": np.nan,
            "resid_sigma": np.nan,
            "distance_km_mean": np.nan,
        }
        out.update(metadata)
        return out

    old_vals = work["old_log_snr"].to_numpy()
    new_vals = work["new_log_snr"].to_numpy()

    old_std = float(np.std(old_vals, ddof=0))
    new_std = float(np.std(new_vals, ddof=0))
    old_z = (old_vals - float(np.mean(old_vals))) / (old_std if old_std > 0 else 1.0)
    new_z = (new_vals - float(np.mean(new_vals))) / (new_std if new_std > 0 else 1.0)
    residual = new_z - old_z
    resid_sigma = float(np.std(residual, ddof=0))

    pearson_r, spearman_r = compute_correlations(old_vals, new_vals)
    rmse_z = float(np.sqrt(np.mean((new_z - old_z) ** 2)))
    one_sigma_thresh = resid_sigma
    two_sigma_thresh = 2.0 * resid_sigma

    out = {
        "cohort": cohort,
        "extrapolation": extrapolation,
        "noise_floor": float(noise_floor),
        "n_pairs_total": int(len(df)),
        "n_pairs_used": int(len(work)),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "rmse_z": rmse_z,
        "frac_abs_resid_gt_1sigma": float(np.mean(np.abs(residual) > one_sigma_thresh)),
        "frac_abs_resid_gt_2sigma": float(np.mean(np.abs(residual) > two_sigma_thresh)),
        "resid_sigma": resid_sigma,
        "distance_km_mean": float(work["distance_km"].mean()),
    }
    out.update(metadata)
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if not ml_utils.TORCH_AVAILABLE:
        raise RuntimeError(
            "ml_utils reports TORCH_AVAILABLE=False; cannot run power-derived SNR crosswalk."
        )

    if "SEISMIC_OED_FIDELITY_MAP" not in os.environ:
        os.environ["SEISMIC_OED_FIDELITY_MAP"] = "clamped_linear"
    fidelity_strategy = os.environ["SEISMIC_OED_FIDELITY_MAP"]

    out_dir = REPO_ROOT / "experiments" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = (REPO_ROOT / args.inputs).resolve()
    nlpts_data, nlpts_space, ndata, bounds_fname, sampling_fname, sensors = utils.read_input_file(str(inputs_path))
    bounds_path = (REPO_ROOT / bounds_fname).resolve()
    latlon_bounds, depth_range, mag_range = utils.read_bounds(str(bounds_path), sensor_bounds=False)

    sampling_mod_name = Path(sampling_fname).stem
    sampling_mod = importlib.import_module(sampling_mod_name)
    events = sampling_mod.generate_theta_data(
        bounds=latlon_bounds,
        depth_range=depth_range,
        mag_range=mag_range,
        nsamp=args.n_events,
        skip=args.seed,
    )
    n_events = int(events.shape[0])
    n_sensors = int(sensors.shape[0])
    total_pairs = n_events * n_sensors

    if total_pairs < 20000:
        raise RuntimeError(
            f"Need at least 20k event-sensor pairs; got {total_pairs}. "
            "Increase --n-events."
        )

    noise_floors = [float(x.strip()) for x in args.noise_floors.split(",") if x.strip()]
    if len(noise_floors) < 3:
        raise RuntimeError("Provide at least 3 noise floor values via --noise-floors.")

    power_model = ml_utils.get_power_model()
    gaussian_variance = ml_utils.map_sensor_fidelity_to_gaussian_variance(
        sensors[:, 2], strategy=fidelity_strategy
    )

    pair_rows = []
    gate_totals = {
        "failed_lat_local": 0,
        "failed_lon_local": 0,
        "failed_depth_m": 0,
        "failed_gaussian_variance": 0,
        "failed_mt_norm": 0,
        "domain_valid_count": 0,
    }

    for event_idx, theta in enumerate(events):
        enabled_mask, gate_masks, gate_meta = ml_utils.evaluate_power_domain_gates(
            theta=theta,
            sensors=sensors,
            gaussian_variance=gaussian_variance,
        )
        gate_totals["failed_lat_local"] += int(np.count_nonzero(~gate_masks["lat_local"]))
        gate_totals["failed_lon_local"] += int(np.count_nonzero(~gate_masks["lon_local"]))
        gate_totals["failed_depth_m"] += int(np.count_nonzero(~gate_masks["depth_m"]))
        gate_totals["failed_gaussian_variance"] += int(np.count_nonzero(~gate_masks["gaussian_variance"]))
        gate_totals["failed_mt_norm"] += int(np.count_nonzero(~gate_masks["mt_norm"]))
        gate_totals["domain_valid_count"] += int(np.count_nonzero(enabled_mask))

        src_lat, src_lon, _, src_mag = theta
        dist_deg = np.array(
            [geodetics.locations2degrees(src_lat, src_lon, rlat, rlon) for rlat, rlon in sensors[:, :2]]
        )
        dist_km = geodetics.degrees2kilometers(dist_deg)

        old_log_snr = like_models.seismic_snr_cal(dist_km, src_mag, sensors[:, 2])
        gated_power = like_models.compute_power(theta, sensors, stype="seismic")
        pred_log_power_gated = gated_power[:, 0]
        pred_log_power_full = power_model.predict_log_power(
            theta=theta,
            sensors=sensors,
            gaussian_variance=gaussian_variance,
        )

        for sensor_idx in range(n_sensors):
            pair_rows.append(
                {
                    "event_idx": event_idx,
                    "sensor_idx": sensor_idx,
                    "distance_km": float(dist_km[sensor_idx]),
                    "magnitude": float(src_mag),
                    "old_log_snr": float(old_log_snr[sensor_idx]),
                    "pred_log_power_gated": float(pred_log_power_gated[sensor_idx]),
                    "pred_log_power_full": float(pred_log_power_full[sensor_idx]),
                    "is_domain_valid": bool(enabled_mask[sensor_idx]),
                    "mt_norm_ratio": float(gate_meta["mt_norm_ratio"]),
                }
            )

    pairs_df = pd.DataFrame(pair_rows)

    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)
    run_id = f"stage1_snr_crosswalk_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    params = {
        "seed": args.seed,
        "n_events": n_events,
        "n_sensors": n_sensors,
        "total_pairs": total_pairs,
        "noise_floors": noise_floors,
        "fidelity_map": fidelity_strategy,
        "inputs_file": str(inputs_path.relative_to(REPO_ROOT)),
        "bounds_file": str(bounds_path.relative_to(REPO_ROOT)),
        "sampling_module": sampling_mod_name,
        "nlpts_data_header": int(nlpts_data),
        "nlpts_space_header": int(nlpts_space),
        "ndata_header": int(ndata),
    }
    params_json = json.dumps(params, sort_keys=True)
    metadata = {
        "run_id": run_id,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "params_json": params_json,
    }

    records = []
    cohort_a_df = pairs_df.loc[pairs_df["is_domain_valid"]].copy()
    cohort_a_df["pred_log_power"] = cohort_a_df["pred_log_power_gated"]

    cohort_b_df = pairs_df.copy()
    cohort_b_df["pred_log_power"] = cohort_b_df["pred_log_power_full"]

    for nf in noise_floors:
        records.append(
            compute_metric_row(
                df=cohort_a_df,
                cohort="A_domain_valid",
                extrapolation=False,
                noise_floor=nf,
                metadata=metadata,
            )
        )
        records.append(
            compute_metric_row(
                df=cohort_b_df,
                cohort="B_full_domain_extrapolation",
                extrapolation=True,
                noise_floor=nf,
                metadata=metadata,
            )
        )

    metrics_df = pd.DataFrame(records)
    metrics_csv = out_dir / "snr_crosswalk.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    gate_total = float(total_pairs)
    domain_valid_count = int(gate_totals["domain_valid_count"])
    breakdown_rows = [
        {
            **metadata,
            "cohort": "A_domain_valid",
            "extrapolation": False,
            "total_pairs": total_pairs,
            "pair_count": domain_valid_count,
            "pair_fraction": domain_valid_count / gate_total,
            "failed_lat_local": gate_totals["failed_lat_local"],
            "failed_lon_local": gate_totals["failed_lon_local"],
            "failed_depth_m": gate_totals["failed_depth_m"],
            "failed_gaussian_variance": gate_totals["failed_gaussian_variance"],
            "failed_mt_norm": gate_totals["failed_mt_norm"],
        },
        {
            **metadata,
            "cohort": "B_full_domain_extrapolation",
            "extrapolation": True,
            "total_pairs": total_pairs,
            "pair_count": total_pairs,
            "pair_fraction": 1.0,
            "failed_lat_local": gate_totals["failed_lat_local"],
            "failed_lon_local": gate_totals["failed_lon_local"],
            "failed_depth_m": gate_totals["failed_depth_m"],
            "failed_gaussian_variance": gate_totals["failed_gaussian_variance"],
            "failed_mt_norm": gate_totals["failed_mt_norm"],
        },
    ]
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_csv = out_dir / "domain_validity_breakdown.csv"
    breakdown_df.to_csv(breakdown_csv, index=False)

    primary_nf = noise_floors[0]
    plot_a = cohort_a_df.copy()
    plot_a = plot_a[np.isfinite(plot_a["old_log_snr"]) & np.isfinite(plot_a["pred_log_power"])]
    plot_a["new_log_snr"] = plot_a["pred_log_power"] - np.log(primary_nf)
    plot_a["cohort"] = "A_domain_valid"
    plot_b = cohort_b_df.copy()
    plot_b = plot_b[np.isfinite(plot_b["old_log_snr"]) & np.isfinite(plot_b["pred_log_power"])]
    plot_b["new_log_snr"] = plot_b["pred_log_power"] - np.log(primary_nf)
    plot_b["cohort"] = "B_full_domain_extrapolation"
    plot_df = pd.concat([plot_a, plot_b], axis=0, ignore_index=True)

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, label in zip(axes, ["A_domain_valid", "B_full_domain_extrapolation"]):
        d = plot_df.loc[plot_df["cohort"] == label]
        ax.scatter(d["old_log_snr"], d["new_log_snr"], s=6, alpha=0.2, edgecolors="none")
        minv = float(np.nanmin([d["old_log_snr"].min(), d["new_log_snr"].min()]))
        maxv = float(np.nanmax([d["old_log_snr"].max(), d["new_log_snr"].max()]))
        ax.plot([minv, maxv], [minv, maxv], linestyle="--", linewidth=1.0, color="black")
        ax.set_title(label)
        ax.set_xlabel("Old log SNR (natural log units)")
        ax.set_ylabel("New log SNR (natural log units)")
    fig1.suptitle(f"Old vs New SNR (noise floor={primary_nf:.2e} power units)")
    scatter_path = out_dir / "fig_scatter_old_vs_new.png"
    fig1.savefig(scatter_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for label, color in [("A_domain_valid", "tab:blue"), ("B_full_domain_extrapolation", "tab:orange")]:
        d = plot_df.loc[plot_df["cohort"] == label].copy()
        old_vals = d["old_log_snr"].to_numpy()
        new_vals = d["new_log_snr"].to_numpy()
        old_z = (old_vals - old_vals.mean()) / (old_vals.std(ddof=0) if old_vals.std(ddof=0) > 0 else 1.0)
        new_z = (new_vals - new_vals.mean()) / (new_vals.std(ddof=0) if new_vals.std(ddof=0) > 0 else 1.0)
        d["residual_z"] = new_z - old_z

        bins = np.linspace(float(d["distance_km"].min()), float(d["distance_km"].max()), 16)
        d["dist_bin"] = pd.cut(d["distance_km"], bins=bins, include_lowest=True)
        grouped = d.groupby("dist_bin", observed=True)
        x = grouped["distance_km"].mean().to_numpy()
        y = grouped["residual_z"].mean().to_numpy()
        ylo = grouped["residual_z"].quantile(0.10).to_numpy()
        yhi = grouped["residual_z"].quantile(0.90).to_numpy()

        ax2.plot(x, y, label=label, color=color, linewidth=2.0)
        ax2.fill_between(x, ylo, yhi, color=color, alpha=0.18)

    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Source-sensor distance (km)")
    ax2.set_ylabel("Residual (new_z - old_z), z-score units")
    ax2.set_title(f"Residual by Distance (noise floor={primary_nf:.2e} power units)")
    ax2.legend()
    resid_path = out_dir / "fig_residual_by_distance.png"
    fig2.savefig(resid_path, dpi=200)
    plt.close(fig2)

    summary_lines = [
        "# Stage 1 SNR Crosswalk Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        f"- total_event_sensor_pairs: `{total_pairs}`",
        f"- events: `{n_events}`",
        f"- sensors: `{n_sensors}`",
        f"- fidelity_mapping: `{fidelity_strategy}`",
        f"- primary_noise_floor: `{primary_nf:.3e}` (power units)",
        "",
        "## Cohorts",
        "",
        "- `A_domain_valid`: ML-domain-valid subset (gated).",
        "- `B_full_domain_extrapolation`: full OED domain using ungated ML power prediction (explicit extrapolation).",
        "",
        "## Metrics (all noise floors)",
        "",
        metrics_df[
            [
                "cohort",
                "noise_floor",
                "n_pairs_total",
                "n_pairs_used",
                "pearson_r",
                "spearman_r",
                "rmse_z",
                "frac_abs_resid_gt_1sigma",
                "frac_abs_resid_gt_2sigma",
            ]
        ].to_markdown(index=False),
        "",
        "## Domain Validity",
        "",
        breakdown_df[
            [
                "cohort",
                "pair_count",
                "pair_fraction",
                "failed_lat_local",
                "failed_lon_local",
                "failed_depth_m",
                "failed_gaussian_variance",
                "failed_mt_norm",
            ]
        ].to_markdown(index=False),
        "",
        "## Output Files",
        "",
        "- `experiments/stage1/snr_crosswalk.csv`",
        "- `experiments/stage1/fig_scatter_old_vs_new.png`",
        "- `experiments/stage1/fig_residual_by_distance.png`",
        "- `experiments/stage1/summary.md`",
        "- `experiments/stage1/domain_validity_breakdown.csv`",
    ]
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    required_files = [
        metrics_csv,
        scatter_path,
        resid_path,
        summary_path,
        breakdown_csv,
    ]
    file_checks = [(p, p.exists() and p.stat().st_size > 0) for p in required_files]

    has_both_cohorts = set(metrics_df["cohort"]) == {
        "A_domain_valid",
        "B_full_domain_extrapolation",
    }

    print("\nCompact metric table:")
    print(
        metrics_df[
            [
                "cohort",
                "noise_floor",
                "n_pairs_used",
                "pearson_r",
                "spearman_r",
                "rmse_z",
                "frac_abs_resid_gt_1sigma",
                "frac_abs_resid_gt_2sigma",
            ]
        ].to_string(index=False)
    )

    print("\nAcceptance checks:")
    print(f"1) >=20k event-sensor pairs: {'PASS' if total_pairs >= 20000 else 'FAIL'} ({total_pairs})")
    print(f"2) both cohorts present: {'PASS' if has_both_cohorts else 'FAIL'} ({sorted(set(metrics_df['cohort']))})")
    all_files_ok = all(ok for _, ok in file_checks)
    print(f"3) 5 required files exist and non-empty: {'PASS' if all_files_ok else 'FAIL'}")
    for path_obj, ok in file_checks:
        print(f"   - {path_obj.relative_to(REPO_ROOT)}: {'OK' if ok else 'MISSING/EMPTY'}")
    print("4) compact metric table printed above: PASS")


if __name__ == "__main__":
    main()
