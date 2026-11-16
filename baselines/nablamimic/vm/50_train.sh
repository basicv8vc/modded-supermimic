#!/bin/bash
# Launch one training run detached from any local CLI connection:
#   tmux nm_<RUN_ID>   -> 50_pipeline.sh (train -> eval -> report -> final sync)
#   tmux sync_<RUN_ID> -> 90_sync.sh loop (periodic Drive sync every SYNC_INTERVAL s)
# The run_id is <RUN>_s<SEED>_<UTC timestamp>: unique per launch, so re-running never
# overwrites a previous experiment. Usage: RUN=ea SEED=0 BUDGET_MINUTES=8 bash 50_train.sh
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== train launch $(date) ==="

RUN="${RUN:-ea}"
SEED="${SEED:-0}"
BUDGET_MINUTES="${BUDGET_MINUTES:-8}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
RUNS="${RUNS:-$WORK/runs}"
LOGDIR="$WORK/logs"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"
DRIVE_RUNS="$DRIVE_ROOT/supermimic/nablamimic/runs"

mkdir -p "$LOGDIR" "$RUNS"
[ -x "$VENV/bin/python" ] || { echo "FATAL: venv missing, run 20_deps.sh first"; exit 1; }
[ -d "$WORK/data/lafan1_g1" ] || { echo "FATAL: data missing, run 30_data.sh first"; exit 1; }

# --- unique run id: name + seed + UTC timestamp (never overwrites) ----------
RUN_ID="$(bash /content/run_id.sh "$RUN" "$SEED")"
[ ! -e "$RUNS/$RUN_ID" ] || { echo "FATAL: $RUNS/$RUN_ID already exists"; exit 1; }
echo "$RUN_ID" > "$LOGDIR/active_run_id.txt"
echo "$RUN_ID $SEED $BUDGET_MINUTES $(date -u +%FT%TZ)" >> "$LOGDIR/run_history.txt"
echo "run_id=$RUN_ID seed=$SEED budget=${BUDGET_MINUTES}min"

# --- Drive run dir (unique per run; keeps old results untouched) -----------
DRIVE_RUN=""
if [ -d "$DRIVE_ROOT" ]; then
  if mkdir -p "$DRIVE_RUNS/$RUN_ID"; then
    DRIVE_RUN="$DRIVE_RUNS/$RUN_ID"
    echo "$DRIVE_RUN" > "$LOGDIR/${RUN_ID}.drive_dir.txt"
    echo "drive run dir: $DRIVE_RUN"
  fi
fi
if [ -z "$DRIVE_RUN" ]; then
  echo "WARN: Drive not mounted -> results live on the VM disk only (collect before down!)"
fi

# --- tmux: pipeline (train -> eval -> report) ------------------------------
tmux kill-session -t "nm_$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "nm_$RUN_ID" \
  "WORK='$WORK' REPO='$REPO' VENV='$VENV' RUNS='$RUNS' RUN_ID='$RUN_ID' SEED='$SEED' \
BUDGET_MINUTES='$BUDGET_MINUTES' DRIVE_ROOT='$DRIVE_ROOT' bash /content/50_pipeline.sh"

# --- tmux: periodic Drive sync until the pipeline session exits ------------
if [ -n "$DRIVE_RUN" ]; then
  tmux kill-session -t "sync_$RUN_ID" 2>/dev/null || true
  tmux new-session -d -s "sync_$RUN_ID" \
    "WORK='$WORK' RUN_ID='$RUN_ID' SRC='$RUNS/$RUN_ID' DST='$DRIVE_RUN' \
SYNC_INTERVAL='$SYNC_INTERVAL' bash /content/90_sync.sh loop"
fi

sleep 5
tmux ls || true
echo "poll: RUN_ID=$RUN_ID with 60_status.sh (or run.sh status)"
echo "=== train LAUNCHED $(date) ==="
