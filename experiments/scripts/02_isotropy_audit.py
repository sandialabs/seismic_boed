#!/usr/bin/env python3
"""Stage 2: Isotropic Slice Audit on maike_code/mixeddata (3).csv."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "maike_code" / "mixeddata (3).csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "stage2"
BLOCKER_PATH = REPO_ROOT / "experiments" / "blocker_stage2.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def observation_variance_power(
    log_power,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    log_power_ref: float = -15.5,
    decay_rate: float = 0.5,
):
    """Mirror like_models.observation_variance_power."""
    import numpy as np

    sigma = sigma_min + (sigma_max - sigma_min) / (
        1.0 + np.exp(decay_rate * (log_power - log_power_ref))
    )
    return sigma**2


@dataclass
class RunMeta:
    run_id: str
    git_sha: str
    timestamp: str
    params: str


def write_blocker(
    *,
    meta: RunMeta,
    input_csv: Path,
    traceback_text: str,
    fix_lines: Iterable[str],
) -> None:
    BLOCKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Blocker: Stage 2 Isotropy Audit",
        "",
        f"- run_id: `{meta.run_id}`",
        f"- git_sha: `{meta.git_sha}`",
        f"- timestamp_utc: `{meta.timestamp}`",
        f"- params: `{meta.params}`",
        f"- input_csv: `{input_csv}`",
        "",
        "## Exact Traceback",
        "",
        "```text",
        traceback_text.rstrip(),
        "```",
        "",
        "## Proposed Fix",
        "",
    ]
    lines.extend([f"- {line}" for line in fix_lines])
    BLOCKER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote blocker: {BLOCKER_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 isotropy audit.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV path (default: maike_code/mixeddata (3).csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for Stage 2 artifacts.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional deterministic subsample size for faster runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="RNG seed for deterministic row subsampling when --max_rows is used.",
    )
    parser.add_argument(
        "--power_floor",
        type=float,
        default=1e-30,
        help="Clamp floor used in target log(Integ_power)=log(max(Integ_power, power_floor)).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5000,
        help="Prediction progress print cadence in rows.",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=80,
        help="Number of bins for isotropy histogram.",
    )
    return parser.parse_args()


def compute_isotropy_and_moment_stats(df):
    import numpy as np

    m_rr = df["m_rr"].to_numpy(dtype=float)
    m_tt = df["m_tt"].to_numpy(dtype=float)
    m_pp = df["m_pp"].to_numpy(dtype=float)
    m_rt = df["m_rt"].to_numpy(dtype=float)
    m_rp = df["m_rp"].to_numpy(dtype=float)
    m_tp = df["m_tp"].to_numpy(dtype=float)

    # Build full symmetric 3x3 tensor from Voigt components.
    tensor = np.zeros((len(df), 3, 3), dtype=float)
    tensor[:, 0, 0] = m_rr
    tensor[:, 1, 1] = m_tt
    tensor[:, 2, 2] = m_pp
    tensor[:, 0, 1] = m_rt
    tensor[:, 1, 0] = m_rt
    tensor[:, 0, 2] = m_rp
    tensor[:, 2, 0] = m_rp
    tensor[:, 1, 2] = m_tp
    tensor[:, 2, 1] = m_tp

    trace = np.trace(tensor, axis1=1, axis2=2)
    fro_norm = np.linalg.norm(tensor, ord="fro", axis=(1, 2))

    denom = np.sqrt(3.0) * fro_norm
    iso_score = np.divide(
        np.abs(trace),
        denom,
        out=np.zeros_like(trace, dtype=float),
        where=denom > 0.0,
    )

    m0_frob_equiv = fro_norm / np.sqrt(3.0)
    m0_trace = np.abs(trace) / 3.0

    m0_frob_safe = np.clip(m0_frob_equiv, 1e-30, None)
    m0_trace_safe = np.clip(m0_trace, 1e-30, None)
    mw_frob_equiv = (np.log10(m0_frob_safe) - 9.1) / 1.5
    mw_trace = (np.log10(m0_trace_safe) - 9.1) / 1.5

    return iso_score, m0_frob_equiv, mw_frob_equiv, m0_trace, mw_trace


def assign_subset_labels(iso_score) -> Tuple:
    import numpy as np

    ge_095 = iso_score >= 0.95
    ge_090 = iso_score >= 0.90
    le_020 = iso_score <= 0.20

    subset_label = np.full(len(iso_score), "other", dtype=object)
    subset_label[le_020] = "strongly_anisotropic_le_0.20"
    subset_label[ge_090] = "near_isotropic_ge_0.90"
    subset_label[ge_095] = "near_isotropic_ge_0.95"

    return subset_label, ge_095, ge_090, le_020


def run_prediction_loop(df, power_model, progress_every: int):
    import numpy as np

    n_rows = len(df)
    pred_log_power = np.full(n_rows, np.nan, dtype=float)

    lat = df["Lat"].to_numpy(dtype=float)
    lon = df["Lon"].to_numpy(dtype=float)
    depth_m = df["Depth"].to_numpy(dtype=float)
    gaussian_variance = df["Gaussian_variance"].to_numpy(dtype=float)
    mw_frob_equiv = df["mw_frob_equiv"].to_numpy(dtype=float)

    for i in range(n_rows):
        theta = np.array([0.0, 0.0, depth_m[i] / 1000.0, mw_frob_equiv[i]], dtype=float)
        sensor = np.array([[lat[i], lon[i], gaussian_variance[i], 0.0, 0.0]], dtype=float)
        pred_log_power[i] = float(
            power_model.predict_log_power(
                theta,
                sensor,
                gaussian_variance=np.array([gaussian_variance[i]], dtype=float),
            )[0]
        )
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"Predicted {i + 1}/{n_rows} rows")

    return pred_log_power


def summarize_subset(df, subset_name: str, mask):
    import numpy as np

    sub = df.loc[mask]
    n_rows = int(len(sub))
    if n_rows == 0:
        return {
            "subset": subset_name,
            "n_rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_nll": np.nan,
            "median_nll": np.nan,
            "mean_abs_error": np.nan,
            "p90_abs_error": np.nan,
            "iso_min": np.nan,
            "iso_median": np.nan,
            "iso_max": np.nan,
            "m0_frob_mean": np.nan,
            "m0_frob_std": np.nan,
            "m0_frob_min": np.nan,
            "m0_frob_max": np.nan,
            "mw_frob_mean": np.nan,
            "mw_frob_std": np.nan,
            "mw_frob_min": np.nan,
            "mw_frob_max": np.nan,
            "m0_trace_mean": np.nan,
            "mw_trace_mean": np.nan,
            "power_clamped_frac": np.nan,
        }

    err = sub["error"].to_numpy(dtype=float)
    abs_err = sub["abs_error"].to_numpy(dtype=float)
    sq_err = sub["squared_error"].to_numpy(dtype=float)
    nll = sub["nll_log_power"].to_numpy(dtype=float)

    return {
        "subset": subset_name,
        "n_rows": n_rows,
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mean_nll": float(np.mean(nll)),
        "median_nll": float(np.median(nll)),
        "mean_abs_error": float(np.mean(abs_err)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "iso_min": float(np.min(sub["iso_score"])),
        "iso_median": float(np.median(sub["iso_score"])),
        "iso_max": float(np.max(sub["iso_score"])),
        "m0_frob_mean": float(np.mean(sub["m0_frob_equiv"])),
        "m0_frob_std": float(np.std(sub["m0_frob_equiv"])),
        "m0_frob_min": float(np.min(sub["m0_frob_equiv"])),
        "m0_frob_max": float(np.max(sub["m0_frob_equiv"])),
        "mw_frob_mean": float(np.mean(sub["mw_frob_equiv"])),
        "mw_frob_std": float(np.std(sub["mw_frob_equiv"])),
        "mw_frob_min": float(np.min(sub["mw_frob_equiv"])),
        "mw_frob_max": float(np.max(sub["mw_frob_equiv"])),
        "m0_trace_mean": float(np.mean(sub["m0_trace"])),
        "mw_trace_mean": float(np.mean(sub["mw_trace"])),
        "power_clamped_frac": float(np.mean(sub["power_clamped"])),
    }


def choose_fix(traceback_text: str) -> Tuple[str, ...]:
    text = traceback_text.lower()
    if "no module named 'torch'" in text or "torch is not installed" in text:
        return (
            "Install `torch` in the Python environment used to run this script.",
            "Run with an environment that has `torch`, `numpy`, `pandas`, `matplotlib`, `joblib`, and `obspy` available.",
            "On HPC, activate the project env and re-run: `python3 experiments/scripts/02_isotropy_audit.py`.",
        )
    if "no such file or directory" in text and "checkpoint_n1000000_seed38_epoch999.pt" in text:
        return (
            "Ensure ML artifacts are present at repo root: checkpoint and scaler files.",
            "Set `SEISMIC_OED_ROOT` to this repo if running from another working directory.",
            "Re-run after artifacts are accessible.",
        )
    return (
        "Resolve the exception in the traceback and rerun Stage 2.",
        "Verify the runtime has all dependencies and ML artifacts accessible from repo root.",
    )


def main() -> int:
    args = parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"stage2_isotropy_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    git_sha = get_git_sha(REPO_ROOT)

    input_csv = args.input_csv if args.input_csv.is_absolute() else (REPO_ROOT / args.input_csv)
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    )

    params_obj: Dict[str, object] = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "max_rows": args.max_rows,
        "seed": args.seed,
        "power_floor": args.power_floor,
        "progress_every": args.progress_every,
        "hist_bins": args.hist_bins,
        "source_for_prediction": {"lat": 0.0, "lon": 0.0, "seismic_type": 0},
    }
    params = json.dumps(params_obj, sort_keys=True)
    meta = RunMeta(run_id=run_id, git_sha=git_sha, timestamp=timestamp, params=params)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        import ml_utils
    except Exception:
        tb = traceback.format_exc()
        write_blocker(
            meta=meta,
            input_csv=input_csv,
            traceback_text=tb,
            fix_lines=choose_fix(tb),
        )
        return 1

    required_cols = [
        "Lat",
        "Lon",
        "Depth",
        "Distance_to_source_km",
        "Gaussian_variance",
        "Integ_power",
        "m_rr",
        "m_tt",
        "m_pp",
        "m_rt",
        "m_rp",
        "m_tp",
    ]

    try:
        df = pd.read_csv(input_csv, usecols=required_cols)
        total_rows = len(df)
        if args.max_rows is not None and args.max_rows > 0 and args.max_rows < total_rows:
            rng = np.random.default_rng(args.seed)
            pick = np.sort(rng.choice(total_rows, size=args.max_rows, replace=False))
            df = df.iloc[pick].copy()
            df.insert(0, "source_row_index", pick.astype(int))
            sampled = True
        else:
            df = df.copy()
            df.insert(0, "source_row_index", np.arange(total_rows, dtype=int))
            sampled = False

        iso_score, m0_frob, mw_frob, m0_trace, mw_trace = compute_isotropy_and_moment_stats(df)
        subset_label, ge_095, ge_090, le_020 = assign_subset_labels(iso_score)

        df["iso_score"] = iso_score
        df["subset_label"] = subset_label
        df["subset_ge_0p95"] = ge_095
        df["subset_ge_0p90"] = ge_090
        df["subset_le_0p20"] = le_020
        df["m0_frob_equiv"] = m0_frob
        df["mw_frob_equiv"] = mw_frob
        df["m0_trace"] = m0_trace
        df["mw_trace"] = mw_trace

        integ_power = df["Integ_power"].to_numpy(dtype=float)
        target_log_power = np.log(np.clip(integ_power, args.power_floor, None))
        df["power_clamped"] = integ_power <= args.power_floor
        df["target_log_power"] = target_log_power

        power_model = ml_utils.get_power_model()
        df["pred_log_power"] = run_prediction_loop(
            df=df, power_model=power_model, progress_every=args.progress_every
        )

        df["error"] = df["pred_log_power"] - df["target_log_power"]
        df["abs_error"] = np.abs(df["error"])
        df["squared_error"] = df["error"] ** 2

        gaussian_variance = np.clip(df["Gaussian_variance"].to_numpy(dtype=float), 1e-12, None)
        sigma_sq_total = (
            0.3**2 + gaussian_variance + observation_variance_power(df["pred_log_power"].to_numpy())
        )
        df["nll_log_power"] = 0.5 * np.log(2.0 * np.pi * sigma_sq_total) + (
            0.5 * (df["error"] ** 2) / sigma_sq_total
        )

        row_csv_path = output_dir / "isotropy_audit.csv"
        fig_path = output_dir / "fig_iso_score_hist.png"
        summary_csv_path = output_dir / "model_error_by_isotropy.csv"
        summary_md_path = output_dir / "summary.md"

        row_output = df.copy()
        row_output.insert(0, "params", params)
        row_output.insert(0, "timestamp", timestamp)
        row_output.insert(0, "git_sha", git_sha)
        row_output.insert(0, "run_id", run_id)
        row_output.to_csv(row_csv_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["iso_score"], bins=args.hist_bins, color="#1f77b4", edgecolor="black")
        ax.set_xlabel("Isotropy Score (unitless)")
        ax.set_ylabel("Row Count")
        ax.set_title("Isotropy Score Distribution")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

        subset_specs = [
            ("full_dataset", np.ones(len(df), dtype=bool)),
            ("near_isotropic_ge_0.95", df["subset_ge_0p95"].to_numpy(dtype=bool)),
            ("near_isotropic_ge_0.90", df["subset_ge_0p90"].to_numpy(dtype=bool)),
            ("strongly_anisotropic_le_0.20", df["subset_le_0p20"].to_numpy(dtype=bool)),
        ]

        summary_rows = [summarize_subset(df, name, mask) for name, mask in subset_specs]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.insert(0, "params", params)
        summary_df.insert(0, "timestamp", timestamp)
        summary_df.insert(0, "git_sha", git_sha)
        summary_df.insert(0, "run_id", run_id)
        summary_df.to_csv(summary_csv_path, index=False)

        count_full = int(len(df))
        count_095 = int(df["subset_ge_0p95"].sum())
        count_090 = int(df["subset_ge_0p90"].sum())
        count_020 = int(df["subset_le_0p20"].sum())
        clamped_frac = float(df["power_clamped"].mean())

        error_table_lines = [
            "| subset | n_rows | MAE | RMSE | Mean NLL |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, row in summary_df.iterrows():
            error_table_lines.append(
                "| "
                + f"{row['subset']} | {int(row['n_rows'])} | "
                + f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['mean_nll']:.6f} |"
            )

        summary_lines = [
            "# Stage 2 Isotropy Audit Summary",
            "",
            f"- run_id: `{run_id}`",
            f"- git_sha: `{git_sha}`",
            f"- timestamp_utc: `{timestamp}`",
            f"- params: `{params}`",
            "",
            "## Input and Sampling",
            "",
            f"- Input CSV: `{input_csv.relative_to(REPO_ROOT)}`",
            f"- Total rows in CSV: {total_rows}",
            f"- Rows analyzed: {len(df)}",
            f"- Deterministic subsampling applied: {sampled}",
            (
                f"- Subsample details: max_rows={args.max_rows}, seed={args.seed}"
                if sampled
                else "- Subsample details: none (full dataset used)"
            ),
            "",
            "## Stage 2 Subset Sizes",
            "",
            f"- Full dataset: {count_full}",
            f"- Near-isotropic (iso_score >= 0.95): {count_095}",
            f"- Near-isotropic (iso_score >= 0.90): {count_090}",
            f"- Strongly anisotropic (iso_score <= 0.20): {count_020}",
            "",
            "## Magnitude and Scalar-Moment Distribution Check",
            "",
            (
                "- Frobenius-equivalent scalar moment (M0) stats: "
                f"mean={df['m0_frob_equiv'].mean():.6e}, std={df['m0_frob_equiv'].std():.6e}, "
                f"min={df['m0_frob_equiv'].min():.6e}, max={df['m0_frob_equiv'].max():.6e}"
            ),
            (
                "- Frobenius-equivalent Mw stats: "
                f"mean={df['mw_frob_equiv'].mean():.6f}, std={df['mw_frob_equiv'].std():.6f}, "
                f"min={df['mw_frob_equiv'].min():.6f}, max={df['mw_frob_equiv'].max():.6f}"
            ),
            (
                "- Trace-derived scalar moment (|trace(M)|/3) stats: "
                f"mean={df['m0_trace'].mean():.6e}, std={df['m0_trace'].std():.6e}"
            ),
            (
                "- Trace-derived Mw stats: "
                f"mean={df['mw_trace'].mean():.6f}, std={df['mw_trace'].std():.6f}"
            ),
            "",
            "## Error Metrics by Subset",
            "",
            *error_table_lines,
            "",
            "## Target Clamp",
            "",
            (
                "- Target is computed as `log(max(Integ_power, power_floor))` "
                f"with `power_floor={args.power_floor}`."
            ),
            f"- Fraction of rows clamped: {clamped_frac:.6%}",
            "",
            "## Output Files",
            "",
            f"- `{row_csv_path.relative_to(REPO_ROOT)}`",
            f"- `{fig_path.relative_to(REPO_ROOT)}`",
            f"- `{summary_csv_path.relative_to(REPO_ROOT)}`",
            f"- `{summary_md_path.relative_to(REPO_ROOT)}`",
        ]

        summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        print(f"Wrote: {row_csv_path}")
        print(f"Wrote: {fig_path}")
        print(f"Wrote: {summary_csv_path}")
        print(f"Wrote: {summary_md_path}")
        print(f"Key counts -> full={count_full}, iso>=0.95={count_095}, iso>=0.90={count_090}, iso<=0.20={count_020}")
        return 0

    except Exception:
        tb = traceback.format_exc()
        write_blocker(
            meta=meta,
            input_csv=input_csv,
            traceback_text=tb,
            fix_lines=choose_fix(tb),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
