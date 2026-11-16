#!/bin/bash
# Finalize one run into the standard report layout (see colab_workflow/README.md):
#   manifest.json  provenance.json  summary.json  curve.csv  report.md
#   evaluation/{eval_nominal.json, per_clip.csv, summary.json}
#   checkpoints/   (rename of upstream ckpt/)
# Works with or without eval results (marks the run INCOMPLETE when eval is missing).
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== report start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
RUNS="${RUNS:-$WORK/runs}"
LOGDIR="$WORK/logs"
PY="$VENV/bin/python"
RUN_ID="${RUN_ID:-$(cat "$LOGDIR/active_run_id.txt" 2>/dev/null)}"
[ -n "$RUN_ID" ] || { echo "FATAL: RUN_ID not set and no active_run_id.txt"; exit 1; }
RUN_DIR="$RUNS/$RUN_ID"
DRIVE_RUN="${DRIVE_RUN:-$(cat "$LOGDIR/${RUN_ID}.drive_dir.txt" 2>/dev/null)}"

export LAFAN1_G1_DIR="$WORK/data/lafan1_g1"
export MOTION_CACHE_DIR="$WORK/data/cache"
export MENAGERIE_DIR="$REPO/third_party/mujoco_menagerie"

# finalize_run.py --live enriches provenance by building the model; it falls back to
# config-derived provenance when the nablamimic stack is not importable (e.g. local re-runs).
"$PY" /content/finalize_run.py "$RUN_DIR" --live \
  --git-sha "$(git -C "$REPO" rev-parse HEAD)" \
  --drive-path "${DRIVE_RUN:-}" --run-id "$RUN_ID"
for f in manifest.json summary.json curve.csv report.md provenance.json; do
  [ -f "$RUN_DIR/$f" ] || { echo "FATAL: $f not generated"; exit 1; }
done
echo "--- manifest ---"
cat "$RUN_DIR/manifest.json"
echo "=== report DONE $(date) ==="
