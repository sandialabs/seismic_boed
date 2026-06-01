# Moment Tensor Handoff

This branch keeps the moment-tensor runtime code and a minimal set of inputs and helper scripts.

## Default run path

The repository default is now the known-good `inputs.dat` geometry.

MT-enabled run:

```bash
mpirun -n 512 --mca psec native python3 eig_calc.py inputs.dat output_mt.npz 1
```

Legacy no-MT run:

```bash
SEISMIC_OED_USE_MT_POWER=0 \
mpirun -x SEISMIC_OED_USE_MT_POWER -n 512 --mca psec native \
python3 eig_calc.py inputs.dat output_legacy.npz 1
```

Paired comparison helper:

```bash
bash experiments/scripts/run_with_without_mt_mpi.sh \
  inputs.dat \
  experiments/figures/legacy_ta/default_compare \
  512 \
  1
```

## Moment-tensor options

MT power is enabled by default. To include MT uncertainty marginalization:

```bash
SEISMIC_OED_MT_MARGINALIZE=1 \
SEISMIC_OED_MT_PRIOR_SAMPLES=50 \
mpirun -x SEISMIC_OED_MT_MARGINALIZE -x SEISMIC_OED_MT_PRIOR_SAMPLES -n 512 --mca psec native \
python3 eig_calc.py inputs.dat output_mt_marginalized.npz 1
```

## Plotting

Use the original plotting entry point:

```bash
python3 eig_vis.py output_mt.npz experiments/inputs/eig_vis_control_wide.txt --bounds-file ta_array_domain.json
```

To compare two runs on the same color scale:

```bash
python3 eig_vis.py output_legacy.npz experiments/inputs/eig_vis_control_wide.txt \
  --bounds-file ta_array_domain.json \
  --range-from-data-file output_mt.npz
```

```bash
python3 eig_vis.py output_mt.npz experiments/inputs/eig_vis_control_wide.txt \
  --bounds-file ta_array_domain.json \
  --range-from-data-file output_legacy.npz
```

## Kept comparison inputs

- `experiments/inputs/inputs_legacy_ta_mw_0p5_2p0.dat`
- `experiments/inputs/inputs_legacy_ta_mw_4p5_5p0.dat`

These use the same sensor geometry as `inputs.dat` with narrower magnitude windows for direct low/high comparisons.

## Kept helper scripts

- `experiments/scripts/run_legacy_no_mt_mpi.sh`
- `experiments/scripts/run_with_without_mt_mpi.sh`

The removed files were validation reports, exploratory plotting workflows, and one-off comparison scripts that are no longer part of the supported handoff path.
