#!/bin/bash
# Official evaluation for one run: bash scripts/evaluate_run.sh RUN_DIR
# (eval_plan picks the checkpoints: every 250 updates of the first hour, then every 1000,
# plus the last one; MuJoCo Warp, 10 solver iterations, deterministic EMA actor, 40 LAFAN1
# clips played to their end, fall forfeits the rest of the clip). Already-evaluated
# checkpoints are skipped, so re-running resumes. Idempotent.
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== eval start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
RUNS="${RUNS:-$WORK/runs}"
LOGDIR="$WORK/logs"
PY="$VENV/bin/python"
RUN_ID="${RUN_ID:-$(cat "$LOGDIR/active_run_id.txt" 2>/dev/null)}"
[ -n "$RUN_ID" ] || { echo "FATAL: RUN_ID not set and no active_run_id.txt"; exit 1; }
RUN_DIR="$RUNS/$RUN_ID"

export LAFAN1_G1_DIR="$WORK/data/lafan1_g1"
export MOTION_CACHE_DIR="$WORK/data/cache"
export MENAGERIE_DIR="$REPO/third_party/mujoco_menagerie"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

[ -f "$RUN_DIR/log.jsonl" ] || { echo "FATAL: $RUN_DIR/log.jsonl missing (no training output)"; exit 1; }
ls "$RUN_DIR/ckpt" 2>/dev/null | head -5 || echo "WARN: no ckpt dir yet"
cd "$REPO"
PYTHON="$PY" bash scripts/evaluate_run.sh "$RUN_DIR" 2>&1 | tee -a "$LOGDIR/$RUN_ID.eval.log"
[ -f "$RUN_DIR/eval_nominal.json" ] || { echo "FATAL: eval_nominal.json not produced"; exit 1; }
"$PY" - <<PY
import json
ev = json.load(open("$RUN_DIR/eval_nominal.json"))
print("evaluated checkpoints:", sorted(ev.keys()))
PY
echo "=== eval DONE $(date) ==="
