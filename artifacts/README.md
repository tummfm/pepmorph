Artifacts

This directory stores outputs and validation artifacts from the paper. These files are
not required for model training or descriptor calculation, but are kept so validation
results can be reproduced without rerunning long jobs.

Layout:
- `artifacts/data_figs/`: dataset plots produced by `src/pepmorph/scripts/analysis.py`
  and `src/pepmorph/scripts/data_plotting.py`.
- `artifacts/descriptor_calc/`: logs/timing outputs from descriptor generation runs.
- `artifacts/models/`: pretrained weights (AP predictor, masked CVAE).
- `artifacts/validation/`: current validation outputs (figs, results tables, generated
  peptides). Validation scripts default to these paths.
- `artifacts/md_sims/`: MD validation data from the paper (runs + outputs). CG analysis
  scripts now live under `src/pepmorph/md_sims`, while runs live in
  `artifacts/md_sims/cg/robustness/runs/` and `artifacts/md_sims/cg/pepmorph-main/runs/`.
- `artifacts/legacy_runs/`: older outputs moved out of working directories to keep the
  repository tidy; nothing is deleted.
