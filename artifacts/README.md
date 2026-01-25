# Artifacts

This directory stores outputs and validation artifacts from the paper.

## Layout:

- `artifacts/data_figs/`: dataset plots produced by `src/scripts/analysis.py`
  and `src/scripts/data_plotting.py`.
- `artifacts/descriptor_calc/`: logs/timing outputs from descriptor generation runs
  (`src/descriptor_calc/run.sh`).
- `artifacts/models/`: pretrained weights (AP predictor, masked CVAE).
- `artifacts/validation/`: validation outputs grouped into:
  - `figs/`: figures created by `src/modeling/validation/validation_plotting.py`.
  - `results/`: tables from `src/modeling/validation/*` evaluation scripts.
  - `gen_peptides/`: generated peptides and filtered sets from `src/modeling/validation/generate.py`.
- `artifacts/md_sims/`: MD validation data from the paper (runs + outputs). Scripts live in
  `src/md_sims/cg/`.
- `artifacts/legacy_runs/`: older outputs moved out of working directories to keep the repository tidy;
  nothing is deleted.
