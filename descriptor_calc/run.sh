#!/usr/bin/env bash
set -euo pipefail

OVERALL_LOG="${OVERALL_LOG:-overall_orchestrator.log}"
TIMINGS_CSV="${TIMINGS_CSV:-timings.csv}"

GEN_CMD=(python3 gen.py peptides.fst --sa-workers=8 --num-workers=8)
DATASET_CMD=(python3 generate_pep_dataset.py)

ts() { date +"%Y-%m-%d %H:%M:%S"; }

log() {
  local msg="$1"
  echo "[$(ts)] $msg" | tee -a "$OVERALL_LOG"
}

run_step() {
  local step_name="$1"
  local workdir="$2"
  local local_log="$3"
  shift 3

  if [[ ! -d "$workdir" ]]; then
    log "ERROR: Missing directory: $workdir (step: $step_name)"
    return 1
  fi

  log "START: $step_name (dir=$workdir)"
  local start_epoch end_epoch elapsed status

  start_epoch="$(date +%s)"

  (
    cd "$workdir"
    {
      echo "[$(ts)] CMD: $*"
      "$@"
    } >> "$local_log" 2>&1
  )

  status=$?
  end_epoch="$(date +%s)"
  elapsed=$(( end_epoch - start_epoch ))

  if [[ $status -eq 0 ]]; then
    log "DONE:  $step_name (elapsed=${elapsed}s)"
  else
    log "FAIL:  $step_name (exit=$status, elapsed=${elapsed}s). See: $workdir/$local_log"
  fi

  # Columns: timestamp, step, directory, elapsed_seconds, exit_code, local_log
  printf "%s,%s,%s,%s,%s,%s\n" \
    "$(ts)" "$step_name" "$workdir" "$elapsed" "$status" "$workdir/$local_log" \
    >> "$TIMINGS_CSV"

  return $status
}

init_outputs() {
  : > "$OVERALL_LOG"
  log "Orchestrator started in $(pwd)"

  if [[ ! -f "$TIMINGS_CSV" ]]; then
    printf "timestamp,step,directory,elapsed_seconds,exit_code,local_log\n" > "$TIMINGS_CSV"
  else
    if ! head -n 1 "$TIMINGS_CSV" | grep -q '^timestamp,step,'; then
      tmp="$(mktemp)"
      printf "timestamp,step,directory,elapsed_seconds,exit_code,local_log\n" > "$tmp"
      cat "$TIMINGS_CSV" >> "$tmp"
      mv "$tmp" "$TIMINGS_CSV"
    fi
  fi
}

init_outputs

run_step "gen_spheres" "spheres" "log.txt" "${GEN_CMD[@]}"
run_step "gen_fibers"  "fibers"  "log.txt" "${GEN_CMD[@]}"

run_step "dataset_spheres" "spheres/pepfold_pipeline" "generate_pep_dataset.log" "${DATASET_CMD[@]}"
run_step "dataset_fibers"  "fibers/pepfold_pipeline"  "generate_pep_dataset.log" "${DATASET_CMD[@]}"

log "All steps completed successfully."
log "Overall log: $(pwd)/$OVERALL_LOG"
log "Timing CSV:  $(pwd)/$TIMINGS_CSV"
