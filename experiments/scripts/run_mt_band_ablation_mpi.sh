#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <output_prefix>"
  echo "Example: $0 experiments/figures/mt_band_ablation/ta_array"
  exit 1
fi

OUTPUT_PREFIX="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_MCA_PSEC="${MPI_MCA_PSEC:-native}"
N_RANKS="${N_RANKS:-512}"
VERBOSE="${VERBOSE:-1}"
MAX_TRAIN_POINTS="${MAX_TRAIN_POINTS:-1200}"
GRID_SIZE="${GRID_SIZE:-120}"
DEPTH_SLICE_KM="${DEPTH_SLICE_KM:-10.0}"
DEPTH_TOL_KM="${DEPTH_TOL_KM:-2.5}"
LOW_MAG_SLICE="${LOW_MAG_SLICE:-1.5}"
LOW_MAG_TOL="${LOW_MAG_TOL:-0.2}"
HIGH_MAG_SLICE="${HIGH_MAG_SLICE:-4.75}"
HIGH_MAG_TOL="${HIGH_MAG_TOL:-0.1}"
MIN_SLICE_POINTS="${MIN_SLICE_POINTS:-150}"

LOW_INPUT="${LOW_INPUT:-experiments/inputs/inputs_mw_0p5_2p0.dat}"
HIGH_INPUT="${HIGH_INPUT:-experiments/inputs/inputs_mw_4p5_5p0.dat}"

LOW_PREFIX="${OUTPUT_PREFIX}_low"
HIGH_PREFIX="${OUTPUT_PREFIX}_high"

LOW_WITH_MT="${LOW_PREFIX}_with_mt.npz"
LOW_WITHOUT_MT="${LOW_PREFIX}_without_mt.npz"
HIGH_WITH_MT="${HIGH_PREFIX}_with_mt.npz"
HIGH_WITHOUT_MT="${HIGH_PREFIX}_without_mt.npz"
OUT_FIG="${OUTPUT_PREFIX}_mt_band_ablation.png"
OUT_CSV="${OUTPUT_PREFIX}_mt_band_ablation_summary.csv"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

echo "MT band-ablation run"
echo "  repo_root: $REPO_ROOT"
echo "  low_input: $LOW_INPUT"
echo "  high_input: $HIGH_INPUT"
echo "  output_prefix: $OUTPUT_PREFIX"
echo "  n_ranks: $N_RANKS"
echo "  verbose: $VERBOSE"
echo "  depth_slice_km: $DEPTH_SLICE_KM"
echo "  low_mag_slice: $LOW_MAG_SLICE"
echo "  high_mag_slice: $HIGH_MAG_SLICE"
echo

echo "[1/6] Low magnitude band: with MT vs without MT"
bash experiments/scripts/run_with_without_mt_mpi.sh "$LOW_INPUT" "$LOW_PREFIX" "$N_RANKS" "$VERBOSE"

echo "[2/6] High magnitude band: with MT vs without MT"
bash experiments/scripts/run_with_without_mt_mpi.sh "$HIGH_INPUT" "$HIGH_PREFIX" "$N_RANKS" "$VERBOSE"

echo "[3/6] Rendering low/high band comparison figure"
"$PYTHON_BIN" experiments/scripts/08_plot_mt_band_ablation.py \
  --low-with-mt "$LOW_WITH_MT" \
  --low-without-mt "$LOW_WITHOUT_MT" \
  --high-with-mt "$HIGH_WITH_MT" \
  --high-without-mt "$HIGH_WITHOUT_MT" \
  --output "$OUT_FIG" \
  --summary-csv "$OUT_CSV" \
  --grid-size "$GRID_SIZE" \
  --max-train-points "$MAX_TRAIN_POINTS" \
  --depth-slice "$DEPTH_SLICE_KM" \
  --depth-tol "$DEPTH_TOL_KM" \
  --low-magnitude-slice "$LOW_MAG_SLICE" \
  --low-mag-tol "$LOW_MAG_TOL" \
  --high-magnitude-slice "$HIGH_MAG_SLICE" \
  --high-mag-tol "$HIGH_MAG_TOL" \
  --min-slice-points "$MIN_SLICE_POINTS"

echo
echo "Done."
echo "Outputs:"
echo "  $LOW_WITH_MT"
echo "  $LOW_WITHOUT_MT"
echo "  $HIGH_WITH_MT"
echo "  $HIGH_WITHOUT_MT"
echo "  $OUT_FIG"
echo "  $OUT_CSV"
