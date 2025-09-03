#!/usr/bin/env bash
set -euo pipefail

# Loop over all subdirectories in the current directory
for dir in */; do
  dir=${dir%/}  # remove trailing slash
  echo "Processing '${dir}' …"

  # Check required files exist
  if [[ -f "$dir/prod/prod.xtc" && -f "$dir/prod/prod.tpr" ]]; then
    pushd "$dir/prod" >/dev/null

      # trjconv: needs two selections (center group, then output group)
      # Replace the two "1"s below if your group indices differ.
      gmx trjconv -f prod.xtc -s prod.tpr -pbc mol -center -o trajout.xtc <<EOF
1
1
EOF

    popd >/dev/null
  else
    echo "  Skipping '${dir}': missing prod/prod.xtc or prod/prod.tpr."
  fi

  echo "Finished '${dir}'."
done
