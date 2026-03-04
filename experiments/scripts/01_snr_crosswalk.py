#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
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

def resolve_repo_root() -> Path:
    candidates = []
    env_root = os.environ.get("SEISMIC_OED_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.cwd().resolve())

    # Include all parents of this script as fallback search locations.
    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / "like_models.py").exists() and (c / "ml_utils.py").exists():
            return c

    # Last fallback to original default behavior.
    return Path(__file__).resolve().parents[2]


REPO_ROOT = resolve_repo_root()
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

try:
    import like_models
    import ml_utils
    import utils
except ModuleNotFoundError as exc:
    raise RuntimeError(
        f"Failed importing project modules from repo root '{REPO_ROOT}'. "
        f"sys.path[0:5]={sys.path[:5]}. "
        "Set SEISMIC_OED_ROOT to your repository root and rerun."
    ) from exc


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates during event evaluation and output generation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress print interval in events per cohort when --verbose is enabled.",
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


def build_monotonic_bins(values: np.ndarray, n_edges: int = 16) -> np.ndarray | None:
    finite_vals = np.asarray(values, dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        return None

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if np.isclose(vmin, vmax):
        # Expand a tiny symmetric interval so cut() has strictly increasing edges.
        span = max(1e-6, abs(vmin) * 1e-6, 1.0)
        bins = np.linspace(vmin - span, vmax + span, n_edges)
    else:
        bins = np.linspace(vmin, vmax, n_edges)

    bins = np.unique(bins)
    if bins.size < 2:
        return None
    return bins


def sample_mw5_indomain_events(sensors: np.ndarray, n_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat_center = float(np.mean(sensors[:, 0]))
    lon_center = float(np.mean(sensors[:, 1]))
    lat = rng.uniform(lat_center - 0.2, lat_center + 0.2, n_events)
    lon = rng.uniform(lon_center - 0.2, lon_center + 0.2, n_events)
    depth = rng.uniform(5.0, 20.0, n_events)  # km
    mag = rng.uniform(4.9, 5.1, n_events)
    return np.column_stack([lat, lon, depth, mag])


def evaluate_pairs_for_events(
    events: np.ndarray,
    sensors: np.ndarray,
    gaussian_variance: np.ndarray,
    power_model,
    mode: str,
    cohort_label: str,
    verbose: bool = False,
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict]:
    pair_rows = []
    gate_totals = {
        "failed_lat_local": 0,
        "failed_lon_local": 0,
        "failed_depth_m": 0,
        "failed_gaussian_variance": 0,
        "failed_mt_norm": 0,
        "domain_valid_count": 0,
    }

    n_events = int(events.shape[0])
    start_t = time.time()
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

        if mode == "gated":
            pred_log_power = like_models.compute_power(theta, sensors, stype="seismic")[:, 0]
        elif mode == "full":
            pred_log_power = power_model.predict_log_power(
                theta=theta,
                sensors=sensors,
                gaussian_variance=gaussian_variance,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for sensor_idx in range(sensors.shape[0]):
            pair_rows.append(
                {
                    "event_idx": event_idx,
                    "sensor_idx": sensor_idx,
                    "distance_km": float(dist_km[sensor_idx]),
                    "magnitude": float(src_mag),
                    "old_log_snr": float(old_log_snr[sensor_idx]),
                    "pred_log_power": float(pred_log_power[sensor_idx]),
                    "is_domain_valid": bool(enabled_mask[sensor_idx]),
                    "mt_norm_ratio": float(gate_meta["mt_norm_ratio"]),
                }
            )

        if verbose and (((event_idx + 1) % max(progress_every, 1) == 0) or (event_idx + 1 == n_events)):
            elapsed = time.time() - start_t
            rate = (event_idx + 1) / max(elapsed, 1e-9)
            pct = 100.0 * (event_idx + 1) / max(n_events, 1)
            print(
                f"[progress] cohort={cohort_label} mode={mode} "
                f"events={event_idx + 1}/{n_events} ({pct:.1f}%) "
                f"rate={rate:.2f} evt/s elapsed={elapsed:.1f}s"
            )

    if verbose:
        elapsed = time.time() - start_t
        print(
            f"[done] cohort={cohort_label} mode={mode} "
            f"pairs={len(pair_rows)} domain_valid={gate_totals['domain_valid_count']} "
            f"elapsed={elapsed:.1f}s"
        )

    return pd.DataFrame(pair_rows), gate_totals


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    t_main = time.time()

    if not ml_utils.TORCH_AVAILABLE:
        raise RuntimeError(
            "ml_utils reports TORCH_AVAILABLE=False; cannot run power-derived SNR crosswalk."
        )

    if "SEISMIC_OED_FIDELITY_MAP" not in os.environ:
        os.environ["SEISMIC_OED_FIDELITY_MAP"] = "clamped_linear"
    fidelity_strategy = os.environ["SEISMIC_OED_FIDELITY_MAP"]

    if args.verbose:
        print("[start] Stage 1 SNR crosswalk")
        print(f"[config] seed={args.seed} n_events_per_cohort={args.n_events} fidelity_map={fidelity_strategy}")

    out_dir = REPO_ROOT / "experiments" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = (REPO_ROOT / args.inputs).resolve()
    nlpts_data, nlpts_space, ndata, bounds_fname, sampling_fname, sensors = utils.read_input_file(str(inputs_path))
    bounds_path = (REPO_ROOT / bounds_fname).resolve()
    latlon_bounds, depth_range, mag_range = utils.read_bounds(str(bounds_path), sensor_bounds=False)

    sampling_mod_name = Path(sampling_fname).stem
    sampling_mod = importlib.import_module(sampling_mod_name)
    events_b = sampling_mod.generate_theta_data(
        bounds=latlon_bounds,
        depth_range=depth_range,
        mag_range=mag_range,
        nsamp=args.n_events,
        skip=args.seed,
    )
    events_a = sample_mw5_indomain_events(sensors=sensors, n_events=args.n_events, seed=args.seed + 1)

    n_events_a = int(events_a.shape[0])
    n_events_b = int(events_b.shape[0])
    n_sensors = int(sensors.shape[0])
    total_pairs_a = n_events_a * n_sensors
    total_pairs_b = n_events_b * n_sensors
    total_pairs_evaluated = total_pairs_a + total_pairs_b

    if total_pairs_evaluated < 20000:
        raise RuntimeError(
            f"Need at least 20k event-sensor pairs; got {total_pairs_evaluated}. "
            "Increase --n-events."
        )

    noise_floors = [float(x.strip()) for x in args.noise_floors.split(",") if x.strip()]
    if len(noise_floors) < 3:
        raise RuntimeError("Provide at least 3 noise floor values via --noise-floors.")
    if args.verbose:
        print(f"[config] noise_floors={noise_floors}")

    power_model = ml_utils.get_power_model()
    gaussian_variance = ml_utils.map_sensor_fidelity_to_gaussian_variance(
        sensors[:, 2], strategy=fidelity_strategy
    )

    pairs_a, gates_a = evaluate_pairs_for_events(
        events=events_a,
        sensors=sensors,
        gaussian_variance=gaussian_variance,
        power_model=power_model,
        mode="gated",
        cohort_label="A_domain_valid",
        verbose=args.verbose,
        progress_every=args.progress_every,
    )
    pairs_b, gates_b = evaluate_pairs_for_events(
        events=events_b,
        sensors=sensors,
        gaussian_variance=gaussian_variance,
        power_model=power_model,
        mode="full",
        cohort_label="B_full_domain_extrapolation",
        verbose=args.verbose,
        progress_every=args.progress_every,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(REPO_ROOT)
    run_id = f"stage1_snr_crosswalk_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    params = {
        "seed": args.seed,
        "n_events_cohort_a": n_events_a,
        "n_events_cohort_b": n_events_b,
        "n_sensors": n_sensors,
        "total_pairs_cohort_a": total_pairs_a,
        "total_pairs_cohort_b": total_pairs_b,
        "total_pairs_evaluated": total_pairs_evaluated,
        "cohort_a_sampling": {
            "latlon_center": "sensor_mean",
            "lat_halfwidth_deg": 0.2,
            "lon_halfwidth_deg": 0.2,
            "depth_km_range": [5.0, 20.0],
            "mag_range": [4.9, 5.1],
        },
        "cohort_b_sampling": "uniform_prior.generate_theta_data from inputs bounds",
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
    cohort_a_df = pairs_a.loc[pairs_a["is_domain_valid"]].copy()
    cohort_b_df = pairs_b.copy()

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
    if args.verbose:
        print(f"[write] {metrics_csv}")

    breakdown_rows = [
        {
            **metadata,
            "cohort": "A_domain_valid",
            "extrapolation": False,
            "total_pairs": total_pairs_a,
            "pair_count": int(cohort_a_df.shape[0]),
            "pair_fraction": float(cohort_a_df.shape[0]) / max(total_pairs_a, 1),
            "failed_lat_local": gates_a["failed_lat_local"],
            "failed_lon_local": gates_a["failed_lon_local"],
            "failed_depth_m": gates_a["failed_depth_m"],
            "failed_gaussian_variance": gates_a["failed_gaussian_variance"],
            "failed_mt_norm": gates_a["failed_mt_norm"],
            "domain_valid_count": gates_a["domain_valid_count"],
        },
        {
            **metadata,
            "cohort": "B_full_domain_extrapolation",
            "extrapolation": True,
            "total_pairs": total_pairs_b,
            "pair_count": total_pairs_b,
            "pair_fraction": 1.0,
            "failed_lat_local": gates_b["failed_lat_local"],
            "failed_lon_local": gates_b["failed_lon_local"],
            "failed_depth_m": gates_b["failed_depth_m"],
            "failed_gaussian_variance": gates_b["failed_gaussian_variance"],
            "failed_mt_norm": gates_b["failed_mt_norm"],
            "domain_valid_count": gates_b["domain_valid_count"],
        },
    ]
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_csv = out_dir / "domain_validity_breakdown.csv"
    breakdown_df.to_csv(breakdown_csv, index=False)
    if args.verbose:
        print(f"[write] {breakdown_csv}")

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
        if d.empty:
            ax.text(0.5, 0.5, "No finite points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            ax.set_xlabel("Old log SNR (natural log units)")
            ax.set_ylabel("New log SNR (natural log units)")
            continue
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
    if args.verbose:
        print(f"[write] {scatter_path}")

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for label, color in [("A_domain_valid", "tab:blue"), ("B_full_domain_extrapolation", "tab:orange")]:
        d = plot_df.loc[plot_df["cohort"] == label].copy()
        if d.empty:
            continue
        old_vals = d["old_log_snr"].to_numpy()
        new_vals = d["new_log_snr"].to_numpy()
        old_z = (old_vals - old_vals.mean()) / (old_vals.std(ddof=0) if old_vals.std(ddof=0) > 0 else 1.0)
        new_z = (new_vals - new_vals.mean()) / (new_vals.std(ddof=0) if new_vals.std(ddof=0) > 0 else 1.0)
        d["residual_z"] = new_z - old_z

        bins = build_monotonic_bins(d["distance_km"].to_numpy(), n_edges=16)
        if bins is None:
            continue

        d["dist_bin"] = pd.cut(
            d["distance_km"],
            bins=bins,
            include_lowest=True,
            duplicates="drop",
        )
        grouped = d.groupby("dist_bin", observed=True)
        x = grouped["distance_km"].mean().to_numpy()
        y = grouped["residual_z"].mean().to_numpy()
        ylo = grouped["residual_z"].quantile(0.10).to_numpy()
        yhi = grouped["residual_z"].quantile(0.90).to_numpy()

        if x.size == 0:
            continue

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
    if args.verbose:
        print(f"[write] {resid_path}")

    summary_lines = [
        "# Stage 1 SNR Crosswalk Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- git_sha: `{git_sha}`",
        f"- timestamp_utc: `{timestamp}`",
        f"- total_event_sensor_pairs: `{total_pairs_evaluated}`",
        f"- events_cohort_a: `{n_events_a}`",
        f"- events_cohort_b: `{n_events_b}`",
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
    if args.verbose:
        print(f"[write] {summary_path}")

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
    print(
        f"1) >=20k event-sensor pairs: "
        f"{'PASS' if total_pairs_evaluated >= 20000 else 'FAIL'} ({total_pairs_evaluated})"
    )
    print(f"2) both cohorts present: {'PASS' if has_both_cohorts else 'FAIL'} ({sorted(set(metrics_df['cohort']))})")
    all_files_ok = all(ok for _, ok in file_checks)
    print(f"3) 5 required files exist and non-empty: {'PASS' if all_files_ok else 'FAIL'}")
    for path_obj, ok in file_checks:
        print(f"   - {path_obj.relative_to(REPO_ROOT)}: {'OK' if ok else 'MISSING/EMPTY'}")
    print("4) compact metric table printed above: PASS")
    if args.verbose:
        print(f"[done] total_elapsed={time.time() - t_main:.1f}s")


if __name__ == "__main__":
    main()
