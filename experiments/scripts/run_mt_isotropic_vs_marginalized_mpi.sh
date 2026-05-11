#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <inputs.dat> <output_prefix>"
  echo "Example: $0 inputs.dat experiments/figures/mt_compare/ta_mt_compare"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_PREFIX="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_MCA_PSEC="${MPI_MCA_PSEC:-native}"
N_RANKS="${N_RANKS:-512}"
VERBOSE="${VERBOSE:-1}"

DEPTH_SLICE_KM="${DEPTH_SLICE_KM:-10.0}"
MAG_SLICE="${MAG_SLICE:-5.0}"
DEPTH_TOL_KM="${DEPTH_TOL_KM:-5.0}"
MAG_TOL="${MAG_TOL:-0.5}"
MIN_SLICE_POINTS="${MIN_SLICE_POINTS:-150}"
GRID_SIZE="${GRID_SIZE:-120}"

MT_PRIOR_SAMPLES="${SEISMIC_OED_MT_PRIOR_SAMPLES:-50}"
MT_PRIOR_DF="${SEISMIC_OED_MT_PRIOR_DF:-5.0}"
MT_PRIOR_DEV_SCALE="${SEISMIC_OED_MT_PRIOR_DEV_SCALE:-0.25}"
MT_PRIOR_MODE="${SEISMIC_OED_MT_PRIOR_MODE:-student_t_iso_centered}"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

OUT_ISO="${OUTPUT_PREFIX}_isotropic.npz"
OUT_MARG="${OUTPUT_PREFIX}_mt_marginalized.npz"
OUT_FIG="${OUTPUT_PREFIX}_eig_comparison.png"
OUT_CSV="${OUTPUT_PREFIX}_eig_comparison_summary.csv"

export SEISMIC_OED_ROOT="$REPO_ROOT"
export SEISMIC_OED_USE_MT_POWER=1

echo "Moment-tensor stakeholder figure run"
echo "  repo_root: $REPO_ROOT"
echo "  input_file: $INPUT_FILE"
echo "  output_prefix: $OUTPUT_PREFIX"
echo "  n_ranks: $N_RANKS"
echo "  verbose: $VERBOSE"
echo "  depth_slice_km: $DEPTH_SLICE_KM"
echo "  mag_slice: $MAG_SLICE"
echo "  mt_prior_samples: $MT_PRIOR_SAMPLES"
echo "  mt_prior_df: $MT_PRIOR_DF"
echo "  mt_prior_dev_scale: $MT_PRIOR_DEV_SCALE"
echo
echo "Expected MPI launch pattern:"
echo "  $MPIRUN_BIN -n $N_RANKS --mca psec $MPI_MCA_PSEC $PYTHON_BIN eig_calc.py ..."
echo "This matches the target 19-node allocation using 512 usable ranks."
echo

echo "[1/3] Running isotropic-only EIG surface -> $OUT_ISO"
SEISMIC_OED_MT_MARGINALIZE=0 \
"$MPIRUN_BIN" \
  -x SEISMIC_OED_ROOT \
  -x SEISMIC_OED_USE_MT_POWER \
  -x SEISMIC_OED_MT_MARGINALIZE \
  -n "$N_RANKS" \
  --mca psec "$MPI_MCA_PSEC" \
  "$PYTHON_BIN" eig_calc.py "$INPUT_FILE" "$OUT_ISO" "$VERBOSE"

echo "[2/3] Running MT-marginalized EIG surface -> $OUT_MARG"
SEISMIC_OED_MT_MARGINALIZE=1 \
SEISMIC_OED_MT_PRIOR_SAMPLES="$MT_PRIOR_SAMPLES" \
SEISMIC_OED_MT_PRIOR_DF="$MT_PRIOR_DF" \
SEISMIC_OED_MT_PRIOR_DEV_SCALE="$MT_PRIOR_DEV_SCALE" \
SEISMIC_OED_MT_PRIOR_MODE="$MT_PRIOR_MODE" \
"$MPIRUN_BIN" \
  -x SEISMIC_OED_ROOT \
  -x SEISMIC_OED_USE_MT_POWER \
  -x SEISMIC_OED_MT_MARGINALIZE \
  -x SEISMIC_OED_MT_PRIOR_SAMPLES \
  -x SEISMIC_OED_MT_PRIOR_DF \
  -x SEISMIC_OED_MT_PRIOR_DEV_SCALE \
  -x SEISMIC_OED_MT_PRIOR_MODE \
  -n "$N_RANKS" \
  --mca psec "$MPI_MCA_PSEC" \
  "$PYTHON_BIN" eig_calc.py "$INPUT_FILE" "$OUT_MARG" "$VERBOSE"

echo "[3/3] Rendering comparison figure -> $OUT_FIG"
"$PYTHON_BIN" experiments/scripts/07_plot_mt_eig_comparison.py \
  --isotropic "$OUT_ISO" \
  --marginalized "$OUT_MARG" \
  --output "$OUT_FIG" \
  --summary-csv "$OUT_CSV" \
  --depth-slice "$DEPTH_SLICE_KM" \
  --magnitude-slice "$MAG_SLICE" \
  --depth-tol "$DEPTH_TOL_KM" \
  --mag-tol "$MAG_TOL" \
  --min-slice-points "$MIN_SLICE_POINTS" \
  --grid-size "$GRID_SIZE"

echo
echo "Done."
echo "Outputs:"
echo "  $OUT_ISO"
echo "  $OUT_MARG"
echo "  $OUT_FIG"
echo "  $OUT_CSV"
