#!/bin/bash
# Progress snapshot for one run: tmux sessions, pipeline status, last training records,
# evaluation progress, Drive sync heartbeat, GPU utilization. Read-only, safe to re-run.
set -uo pipefail
WORK="${WORK:-/content/nabla}"
LOGDIR="$WORK/logs"
RUNS="${RUNS:-$WORK/runs}"
RUN_ID="${RUN_ID:-$(cat "$LOGDIR/active_run_id.txt" 2>/dev/null)}"
[ -n "$RUN_ID" ] || { echo "no active run (run 50_train.sh first)"; exit 0; }
RUN_DIR="$RUNS/$RUN_ID"

echo "--- run ---"
echo "run_id=$RUN_ID"
echo "status: $(cat "$LOGDIR/$RUN_ID.status" 2>/dev/null || echo '(no status file)')"
echo "drive:  $(cat "$LOGDIR/${RUN_ID}.drive_dir.txt" 2>/dev/null || echo '(not synced to Drive)')"
echo "--- tmux ---"
tmux ls 2>/dev/null || echo "(no sessions)"
echo "--- train (log.jsonl tail) ---"
if [ -f "$RUN_DIR/log.jsonl" ]; then
  python3 - "$RUN_DIR/log.jsonl" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
for r in recs[-3:]:
    print(f"iter={r.get('iter')} wall_s={r.get('wall_s', 0):.1f} "
          f"rew_mean={r.get('rew_mean', 0):.4f} term_rate={r.get('term_rate', 0):.5f} "
          f"iter_s={r.get('iter_s', 0):.3f}")
print(f"records={len(recs)}")
PY
else
  echo "(no log.jsonl yet)"
fi
echo "--- train.log tail ---"
tail -3 "$RUN_DIR/train.log" 2>/dev/null || echo "(no train.log)"
echo "--- evaluation ---"
if [ -f "$RUN_DIR/eval_nominal.json" ]; then
  python3 - "$RUN_DIR/eval_nominal.json" <<'PY'
import json, sys
ev = json.load(open(sys.argv[1]))
for k in sorted(ev, key=lambda s: int(s.split("/")[0][2:])):
    e = ev[k]
    print(f"{k}: full_norm_return={e.get('full_norm_return')} failed_frac={e.get('failed_frac')}")
PY
else
  echo "(no eval_nominal.json yet)"
fi
echo "--- sync ---"
DST="$(cat "$LOGDIR/${RUN_ID}.drive_dir.txt" 2>/dev/null)"
if [ -n "$DST" ] && [ -f "$DST/.sync_status" ]; then
  cat "$DST/.sync_status"
else
  echo "(no sync heartbeat)"
fi
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
