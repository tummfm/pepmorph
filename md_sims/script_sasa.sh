#!/usr/bin/env bash
set -euo pipefail

# Loop over all subdirectories in the current directory
for dir in */; do
  dir=${dir%/}  # remove trailing slash
  echo "Processing '${dir}' …"

  pushd "$dir/prod" >/dev/null
    echo 1 | gmx sasa -f trajout.xtc -s ../box/peptide-cg.gro
  popd >/dev/null

  echo "Finished '${dir}'."
done
