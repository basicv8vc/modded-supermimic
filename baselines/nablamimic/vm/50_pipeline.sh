#!/bin/bash
# One run's pipeline, executed inside tmux (independent of any local CLI connection):
#   train (configs/ours.args, effective 8 min via --hours) -> official eval -> finalize/report
# -> final Drive sync. Writes $WORK/logs/<RUN_ID>.status: RUNNING | COMPLETE | INCOMPLETE | FAIL.
# Launched by 50_train.sh; can also be re-run by hand for recovery (eval/report skip done work).
set -uo pipefail
RUN_ID="${RUN_ID:?RUN_ID required}"
SEED="${SEED:-0}"
BUDGET_MINUTES="${BUDGET_MINUTES:-8}"
WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
RUNS="${RUNS:-$WORK/runs}"
CACHE="${CACHE:-$WORK/cache}"
LOGDIR="$WORK/logs"
PY="$VENV/bin/python"
RUN_DIR="$RUNS/$RUN_ID"
STATUS_FILE="$LOGDIR/$RUN_ID.status"
PIPE_LOG="$LOGDIR/$RUN_ID.pipeline.log"

exec >> "$PIPE_LOG" 2>&1
echo "=== pipeline start $(date -u +%FT%TZ) run_id=$RUN_ID seed=$SEED budget=${BUDGET_MINUTES}min ==="
mkdir -p "$LOGDIR" "$RUN_DIR"
echo "RUNNING $(date -u +%FT%TZ)" > "$STATUS_FILE"

export LAFAN1_G1_DIR="$WORK/data/lafan1_g1"
export MOTION_CACHE_DIR="$WORK/data/cache"
export MENAGERIE_DIR="$REPO/third_party/mujoco_menagerie"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export DK_CACHE_DIR="$CACHE/dk"
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then export NVCC=/usr/local/cuda-12.9/bin/nvcc
elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then export NVCC=/usr/local/cuda-12.8/bin/nvcc
fi

fail() { echo "FAIL $(date -u +%FT%TZ) $*" > "$STATUS_FILE"; echo "PIPELINE_FAIL: $*"; exit 1; }

# ---- train ---------------------------------------------------------------
# Only two overrides vs configs/ours.args: the wall-clock budget (--hours) and --seed.
# ckpt-every 250 and every other algorithm parameter stay official.
HOURS="$(awk "BEGIN{printf \"%.5f\", $BUDGET_MINUTES/60}")"
echo "--- train: configs/ours.args --hours $HOURS --seed $SEED (only these two overridden) ---"
cd "$REPO"
rm -f "$RUN_DIR/train.log"
# early-fallback watchdog: if the run silently switched to MuJoCo Warp, kill it soon
( for _ in $(seq 1 36); do
    sleep 10
    if [ -f "$RUN_DIR/train.log" ]; then
      if grep -q "warp physics" "$RUN_DIR/train.log" && ! grep -q "engine physics" "$RUN_DIR/train.log"; then
        pkill -f "nablamimic.train" || true
        echo "FALLBACK_KILLED: MuJoCo Warp detected in train.log" >> "$PIPE_LOG"
        exit 0
      fi
      grep -q "engine physics" "$RUN_DIR/train.log" && exit 0
    fi
  done ) &
WD=$!
set +e
"$PY" -u -m nablamimic.train --config configs/ours.args --out "$RUN_DIR" \
  --hours "$HOURS" --seed "$SEED" 2>&1 | tee "$RUN_DIR/train.log"
train_rc=${PIPESTATUS[0]}
set -e
wait "$WD" 2>/dev/null || true
[ "$train_rc" -eq 0 ] || fail "training exit code $train_rc (checkpoints/logs kept in $RUN_DIR)"
grep -q "engine physics" "$RUN_DIR/train.log" || fail "training did not report NablaEngine backend"
grep -q "warp physics" "$RUN_DIR/train.log" && fail "training fell back to MuJoCo Warp"
echo "--- train done: $(tail -1 "$RUN_DIR/log.jsonl" 2>/dev/null | head -c 200) ---"

# ---- eval (official protocol, keeps going even if it fails -> INCOMPLETE) --
set +e
bash /content/70_eval.sh
eval_rc=$?
set -e
if [ "$eval_rc" -ne 0 ]; then
  echo "EVAL_FAILED rc=$eval_rc (checkpoints and logs kept)" | tee -a "$RUN_DIR/train.log"
fi

# ---- report / finalize (works with or without eval) -----------------------
set +e
bash /content/80_report.sh
report_rc=$?
set -e

# ---- final Drive sync (verifies source-vs-destination, writes the receipt) --
set +e
bash /content/90_sync.sh once
sync_rc=$?
set -e
sync_state="$(cat "$LOGDIR/$RUN_ID.sync_state" 2>/dev/null || echo none)"

# ---- status: compute completion and persistence are separate facts --------
#   PERSISTED            compute ok + destination verified against source
#   UNSYNCED             compute ok, nothing proves the data left this VM
#   INCOMPLETE           eval/report failed but everything that exists is verified
#   INCOMPLETE_UNSYNCED  eval/report failed and the archive is not verified
#   FAIL                 training failed
if [ "$train_rc" -ne 0 ]; then
  echo "FAIL $(date -u +%FT%TZ) train_rc=$train_rc" > "$STATUS_FILE"
  echo "PIPELINE_FAIL"
elif [ "$eval_rc" -ne 0 ] || [ "$report_rc" -ne 0 ]; then
  if [ "$sync_rc" -eq 0 ]; then
    echo "INCOMPLETE $(date -u +%FT%TZ) eval_rc=$eval_rc report_rc=$report_rc sync=$sync_state" > "$STATUS_FILE"
    echo "PIPELINE_INCOMPLETE"
  else
    echo "INCOMPLETE_UNSYNCED $(date -u +%FT%TZ) eval_rc=$eval_rc report_rc=$report_rc sync=$sync_state" > "$STATUS_FILE"
    echo "PIPELINE_INCOMPLETE_UNSYNCED"
  fi
else
  if [ "$sync_rc" -eq 0 ] && [ "$sync_state" = "ok" ]; then
    echo "PERSISTED $(date -u +%FT%TZ) sync=ok" > "$STATUS_FILE"
    echo "PIPELINE_PERSISTED"
  else
    # compute finished but the archive is NOT verified: never call this COMPLETE
    echo "UNSYNCED $(date -u +%FT%TZ) sync_rc=$sync_rc sync=$sync_state" > "$STATUS_FILE"
    echo "PIPELINE_UNSYNCED"
  fi
fi
echo "=== pipeline end $(date -u +%FT%TZ) ==="
