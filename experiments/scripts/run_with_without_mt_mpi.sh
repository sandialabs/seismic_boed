#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <inputs.dat> <output_prefix> <n_ranks> <verbose>"
  echo "Example: $0 inputs.dat experiments/figures/legacy_ta/default_compare 512 1"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_PREFIX="$2"
N_RANKS="$3"
VERBOSE="$4"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_MCA_PSEC="${MPI_MCA_PSEC:-native}"

OUT_WITH_MT="${OUTPUT_PREFIX}_with_mt.npz"
OUT_WITHOUT_MT="${OUTPUT_PREFIX}_without_mt.npz"

echo "[1/2] Running with MT power enabled -> ${OUT_WITH_MT}"
SEISMIC_OED_USE_MT_POWER=1 \
"$MPIRUN_BIN" -x SEISMIC_OED_USE_MT_POWER -n "$N_RANKS" --mca psec "$MPI_MCA_PSEC" \
  "$PYTHON_BIN" eig_calc.py "$INPUT_FILE" "$OUT_WITH_MT" "$VERBOSE"

echo "[2/2] Running with MT power disabled -> ${OUT_WITHOUT_MT}"
SEISMIC_OED_USE_MT_POWER=0 \
"$MPIRUN_BIN" -x SEISMIC_OED_USE_MT_POWER -n "$N_RANKS" --mca psec "$MPI_MCA_PSEC" \
  "$PYTHON_BIN" eig_calc.py "$INPUT_FILE" "$OUT_WITHOUT_MT" "$VERBOSE"

echo "Done."
echo "Outputs:"
echo "  $OUT_WITH_MT"
echo "  $OUT_WITHOUT_MT"
