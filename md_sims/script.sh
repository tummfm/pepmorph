#!/usr/bin/env bash
set -euo pipefail

# Loop over all subdirectories in the current directory
for dir in */; do
  dir=${dir%/}  # remove trailing slash
  echo "Processing '${dir}' …"

  # Skip if prod/prod.xtc already exists
  if [[ -f "$dir/prod/prod.xtc" ]]; then
    echo "  Skipping '${dir}': prod/prod.xtc already exists."
    continue
  fi

  # Check that all required subdirectories exist
  if [[ -d "$dir/top" && -d "$dir/box" && -d "$dir/solvate" && -d "$dir/emin" && -d "$dir/prod" && -d "$dir/coords" ]]; then

    # 1) Run martinize2 in top/
    pushd "$dir/top" >/dev/null
      martinize2 \
        -f ../peptide.pdb \
        -o peptide-top.top \
        -x ../coords/peptide-cg.pdb \
        -p backbone \
        -ff martini3001
    popd >/dev/null

    # 2) Insert molecules in box/
    pushd "$dir/box" >/dev/null
      gmx insert-molecules \
        -box 15 15 15 \
        -radius 0.15 \
        -nmol 300 \
        -ci ../coords/peptide-cg.pdb \
        -o peptide-cg.gro
    popd >/dev/null

    # 3) Text replacements in top/
    #    a) replace 'molecule_0 1' by 'molecule_0 300'
    sed -i 's/molecule_0    1/molecule_0    300/g' "$dir/top/peptide-top.top"

    #    b) replace all Q5 → Q4 and TC5 → SC4 in molecule_0.itp
    sed -i 's/Q5/Q4/g; s/TC5/SC4/g' "$dir/top/molecule_0.itp"

    # 4) Solvation and ion‐adding in solvate/
    pushd "$dir/solvate" >/dev/null
      gmx solvate \
        -cp ../box/peptide-cg.gro \
        -cs martini_water.gro \
        -o solvated.gro \
        -p ../top/peptide-top.top

      gmx grompp \
        -f ../emin/em.mdp \
        -c solvated.gro \
        -p ../top/peptide-top.top \
        -o ions.tpr

      # Non-interactive selection of group “13” for genion
      echo 13 | gmx genion \
        -s ions.tpr \
        -o solvated_ions.gro \
        -p ../top/peptide-top.top \
        -pname NA \
        -nname CL \
        -neutral
    popd >/dev/null

    # 5) Energy minimization in emin/
    pushd "$dir/emin" >/dev/null
      gmx grompp \
        -f em.mdp \
        -c ../solvate/solvated_ions.gro \
        -p ../top/peptide-top.top \
        -o em.tpr

      gmx mdrun \
        -v \
        -deffnm em \
        -nb gpu \
        -gpu_id 0
    popd >/dev/null

    pushd "$dir/prod" >/dev/null
      gmx grompp \
        -f prod.mdp \
        -c ../emin/em.gro \
        -p ../top/peptide-top.top \
        -r ../solvate/solvated_ions.gro \
        -o prod.tpr
      
      gmx mdrun \
        -v \
        -deffnm prod \
        -nb gpu \
        -gpu_id 3
    popd >/dev/null

    echo "Finished '${dir}'."
  else
    echo "  Skipping '${dir}': missing one of (top/, box/, solvate/, emin/, prod/, coords/)" >&2
  fi
done
