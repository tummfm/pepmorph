#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./flatten_box_prod.sh /path/to/root --dry-run
#   ./flatten_box_prod.sh /path/to/root
#
# What it does:
#   root/.../folder_last/{box,prod}/...files -> root/.../folder_last/...files
#   then deletes empty box/prod directories

ROOT="${1:-.}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# If you ONLY want certain files, list them here (globs allowed).
# Leave as ("*") to move everything.
WANTED_PATTERNS=("*")
# Examples:
# WANTED_PATTERNS=("trajout.xtc" "sasa.xvg" "peptide-cg.gro" "*.log")

log() { printf '%s\n' "$*" >&2; }

matches_wanted() {
  local base="$1"
  for pat in "${WANTED_PATTERNS[@]}"; do
    if [[ "$base" == $pat ]]; then
      return 0
    fi
  done
  return 1
}

unique_target_path() {
  local parent="$1" label="$2" base="$3"
  local target="$parent/$base"

  if [[ ! -e "$target" ]]; then
    printf '%s\n' "$target"
    return 0
  fi

  # Collision: prefix with box_/prod_ and add numeric suffix if needed
  local pref="$parent/${label}_$base"
  if [[ ! -e "$pref" ]]; then
    printf '%s\n' "$pref"
    return 0
  fi

  local i=2
  while :; do
    local cand="$parent/${label}_${i}_$base"
    if [[ ! -e "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
    i=$((i+1))
  done
}

do_mv() {
  local src="$1" dst="$2"
  if (( DRY_RUN )); then
    log "[DRY] mv -v -- '$src' '$dst'"
  else
    mv -v -- "$src" "$dst"
  fi
}

do_rmdir_if_empty() {
  local d="$1"
  # remove if empty (and only if it exists)
  if [[ -d "$d" ]]; then
    if (( DRY_RUN )); then
      if [[ -z "$(find "$d" -mindepth 1 -print -quit 2>/dev/null || true)" ]]; then
        log "[DRY] rmdir -- '$d'"
      fi
    else
      rmdir --ignore-fail-on-non-empty -- "$d" 2>/dev/null || true
    fi
  fi
}

log "Root: $ROOT"
(( DRY_RUN )) && log "Running in DRY-RUN mode (no changes will be made)."

# Find every directory named box or prod under ROOT
while IFS= read -r -d '' dir; do
  label="$(basename "$dir")"          # box or prod
  parent="$(dirname "$dir")"          # folder_last

  # Move all matching files (recursively) from dir to parent (flattened)
  while IFS= read -r -d '' file; do
    base="$(basename "$file")"
    if ! matches_wanted "$base"; then
      continue
    fi

    target="$(unique_target_path "$parent" "$label" "$base")"
    do_mv "$file" "$target"
  done < <(find "$dir" -type f -print0)

  # Try to remove box/prod (will only succeed if empty)
  do_rmdir_if_empty "$dir"

done < <(find "$ROOT" -type d \( -name box -o -name prod \) -print0)

log "Done."
