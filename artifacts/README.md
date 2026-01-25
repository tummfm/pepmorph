# Artifacts

This directory stores outputs and validation artifacts from the paper.

## Layout:

- `artifacts/data_figs/`: dataset plots produced by `src/scripts/analysis.py`
  and `src/scripts/data_plotting.py`.
- `artifacts/descriptor_calc/`: logs/timing outputs from descriptor generation runs.
- `artifacts/models/`: pretrained weights (AP predictor, masked CVAE).
- `artifacts/validation/`: current validation outputs (figs, results tables, generated peptides)
- `artifacts/md_sims/`: MD validation data from the paper (runs + outputs)
- `artifacts/legacy_runs/`: older outputs moved out of working directories to keep the repository tidy; nothing is deleted.
