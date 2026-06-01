#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <inputs.dat> <output_npz> <n_ranks> <verbose>"
  echo "Example: $0 inputs.dat experiments/figures/legacy_ta/default_legacy.npz 512 1"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
N_RANKS="$3"
VERBOSE="$4"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_MCA_PSEC="${MPI_MCA_PSEC:-native}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Legacy no-MT run"
echo "  repo_root: $REPO_ROOT"
echo "  input_file: $INPUT_FILE"
echo "  output_file: $OUTPUT_FILE"
echo "  n_ranks: $N_RANKS"
echo "  verbose: $VERBOSE"
echo

SEISMIC_OED_USE_MT_POWER=0 \
"$MPIRUN_BIN" -x SEISMIC_OED_USE_MT_POWER -n "$N_RANKS" --mca psec "$MPI_MCA_PSEC" \
  "$PYTHON_BIN" eig_calc.py "$INPUT_FILE" "$OUTPUT_FILE" "$VERBOSE"
