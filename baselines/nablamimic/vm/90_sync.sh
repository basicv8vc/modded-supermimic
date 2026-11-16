#!/bin/bash
# Drive sync for one run directory.
#   bash 90_sync.sh loop   # periodic sync until the pipeline leaves RUNNING (used in tmux)
#   bash 90_sync.sh once   # single sync + verification receipt (used by 50_pipeline.sh/tests)
#
# The receipt is an audit record, not a participation trophy: it is only written when every
# required file and every checkpoint produced by the run has been compared source-vs-destination
# (size + sha256) and matches. A failed rsync, a missing file or a hash mismatch makes the
# script exit non-zero and removes any previous receipt, so "receipt exists" really means
# "the data is on Drive".
#
# Receipt states written to .sync_receipt:
#   verified=ok       all required artifacts + all checkpoints match the source
#   verified=partial  training artifacts (log.jsonl + every checkpoint) match, but evaluation
#                     or report artifacts are missing/mismatched -> a failed run whose
#                     training output is still safely persisted
# A fully failed verification leaves .sync_receipt.invalid and removes .sync_receipt.
set -uo pipefail
MODE="${1:-loop}"
WORK="${WORK:-/content/nabla}"
RUN_ID="${RUN_ID:-$(cat "$WORK/logs/active_run_id.txt" 2>/dev/null)}"
SRC="${SRC:-$WORK/runs/$RUN_ID}"
# DST must default to the per-run Drive dir recorded by 50_train.sh: 50_pipeline.sh
# calls "90_sync.sh once" without env, and an empty DST aborted the final sync.
DST="${DST:-$(cat "$WORK/logs/${RUN_ID}.drive_dir.txt" 2>/dev/null || true)}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
LOG="${SYNC_LOG:-$WORK/logs/${RUN_ID:-sync}.sync.log}"

# Artifacts a finished run must have. Missing ones downgrade the receipt to partial.
KEY_FILES="manifest.json config.json log.jsonl summary.json curve.csv report.md evaluation/eval_nominal.json evaluation/per_clip.csv"

mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }
# GNU sha256sum on the VM, BSD shasum for local (macOS) test runs. A pipeline's exit code
# is the last command's, so test for the binary instead of relying on "||" after a pipe.
sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | cut -d' ' -f1
  else
    shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
  fi
}

sync_once() {
  mkdir -p "$DST"
  if command -v rsync >/dev/null 2>&1; then
    timeout 900 rsync -a --partial --no-perms --no-owner --no-group "$SRC/" "$DST/"
  else
    timeout 900 cp -ru "$SRC/." "$DST/"
  fi
}

heartbeat() {
  {
    date -u +%FT%TZ
    tail -1 "$SRC/log.jsonl" 2>/dev/null
    tail -1 "$SRC/eval_nominal.json" 2>/dev/null | head -c 200
    cat "$WORK/logs/${RUN_ID}.status" 2>/dev/null
  } > "$DST/.sync_status.tmp" 2>/dev/null && mv "$DST/.sync_status.tmp" "$DST/.sync_status" 2>/dev/null
}

# Compare one relative path. Prints "<status> <file> <size> <sha>" for the receipt.
# status: OK | MISSING_SRC | MISSING_DST | SIZE_MISMATCH | HASH_MISMATCH
compare_file() {
  local rel="$1" sa sb sza szb
  local srcp="$SRC/$rel" dstp="$DST/$rel"
  [ -f "$srcp" ] || { echo "MISSING_SRC $rel - -"; return 0; }
  [ -f "$dstp" ] || { echo "MISSING_DST $rel - -"; return 0; }
  sa=$(sha_of "$srcp"); sb=$(sha_of "$dstp")
  sza=$(size_of "$srcp"); szb=$(size_of "$dstp")
  if [ -z "$sa" ] || [ -z "$sb" ]; then echo "HASH_MISMATCH $rel $szb -"; return 0; fi
  if [ "$sza" != "$szb" ]; then echo "SIZE_MISMATCH $rel $szb $sb"; return 0; fi
  if [ "$sa" != "$sb" ]; then echo "HASH_MISMATCH $rel $szb $sb"; return 0; fi
  echo "OK $rel $szb $sb"
}

# Verify the destination against the source. Returns 0 only for verified=ok.
# Sets VERIFY_STATE to ok|partial|failed and writes/removes the receipt accordingly.
verify_receipt() {
  local receipt="$DST/.sync_receipt" tmp="$DST/.sync_receipt.tmp"
  local hard_fail=0 soft_fail=0 f line ck name n_ck=0 n_ok=0
  : > "$tmp" || return 1
  {
    echo "verified_at=$(date -u +%FT%TZ)"
    echo "src=$SRC"
    echo "dst=$DST"
  } >> "$tmp"

  for f in $KEY_FILES; do
    line=$(compare_file "$f")
    echo "$line" >> "$tmp"
    case "$line" in
      OK\ *) ;;
      *)
        # The training log and resolved config are required even for a failed run.
        # compare_file emits details after the filename (e.g. "MISSING_SRC log.jsonl - -"),
        # so matching the complete output against "MISSING_SRC log.jsonl" is unsafe.
        case "$f" in
          log.jsonl|config.json) hard_fail=1 ;;
          *) soft_fail=1 ;;
        esac
        ;;
    esac
  done

  # every checkpoint the run produced must be on the destination, byte-identical
  for ck in "$SRC"/checkpoints/*.pkl "$SRC"/ckpt/*.pkl; do
    [ -f "$ck" ] || continue
    n_ck=$((n_ck + 1))
    name=$(basename "$ck")
    line=$(compare_file "checkpoints/$name")
    echo "$line" >> "$tmp"
    case "$line" in
      OK\ *) n_ok=$((n_ok + 1)) ;;
      *) hard_fail=1 ;;
    esac
  done
  {
    echo "checkpoints_total=$n_ck"
    echo "checkpoints_verified=$n_ok"
  } >> "$tmp"

  if [ "$n_ck" -eq 0 ]; then
    # no checkpoint at all: the run cannot be resumed or reproduced from this archive
    echo "no_checkpoints=1" >> "$tmp"
    hard_fail=1
  fi

  if [ "$hard_fail" -eq 0 ] && [ "$soft_fail" -eq 0 ]; then
    echo "verified=ok" >> "$tmp"
    VERIFY_STATE=ok
    mv "$tmp" "$receipt"
    echo "$VERIFY_STATE" > "$WORK/logs/${RUN_ID}.sync_state" 2>/dev/null || true
    say "receipt written: $receipt (verified=ok, $n_ok/$n_ck checkpoints)"
    return 0
  fi
  if [ "$hard_fail" -eq 0 ]; then
    echo "verified=partial" >> "$tmp"
    VERIFY_STATE=partial
    mv "$tmp" "$receipt"
    echo "$VERIFY_STATE" > "$WORK/logs/${RUN_ID}.sync_state" 2>/dev/null || true
    say "receipt written: $receipt (verified=partial, $n_ok/$n_ck checkpoints; report files incomplete)"
    echo "WARN: verification partial - training artifacts and checkpoints are safe, run is INCOMPLETE"
    return 1
  fi

  echo "verified=failed" >> "$tmp"
  VERIFY_STATE=failed
  echo "$VERIFY_STATE" > "$WORK/logs/${RUN_ID}.sync_state" 2>/dev/null || true
  rm -f "$receipt"                    # a stale receipt must never outlive a failed check
  mv "$tmp" "$DST/.sync_receipt.invalid"
  say "verification FAILED: $DST/.sync_receipt.invalid"
  echo "FATAL: verification failed - source and destination differ (see .sync_receipt.invalid)"
  return 1
}

[ -n "$DST" ] || { say "FATAL: DST not set"; echo "FATAL: DST not set"; exit 1; }
[ -d "$SRC" ] || { say "FATAL: SRC $SRC missing"; echo "FATAL: SRC missing"; exit 1; }

case "$MODE" in
  once)
    say "sync once $SRC -> $DST"
    if ! sync_once >> "$LOG" 2>&1; then
      # a stale receipt next to stale files is exactly the failure mode we must not have
      rm -f "$DST/.sync_receipt"
      say "FATAL: rsync failed; previous receipt removed, no new receipt written"
      echo "FATAL: sync failed (rsync rc != 0)"
      exit 1
    fi
    heartbeat
    verify_receipt
    exit $?
    ;;
  loop)
    say "sync loop $SRC -> $DST every ${SYNC_INTERVAL}s"
    while true; do
      if ! sync_once >> "$LOG" 2>&1; then
        say "WARN: periodic sync failed (will retry; no receipt until it succeeds)"
      fi
      heartbeat
      st="$(cat "$WORK/logs/${RUN_ID}.status" 2>/dev/null || echo RUNNING)"
      case "$st" in
        RUNNING*) ;;
        *)
          if ! sync_once >> "$LOG" 2>&1; then
            rm -f "$DST/.sync_receipt"
            say "FATAL: final sync failed; previous receipt removed"
            echo "FATAL: final sync failed"
            exit 1
          fi
          heartbeat
          verify_receipt
          say "=== sync DONE (pipeline status: $st, verify=$VERIFY_STATE) ==="
          exit $?
          ;;
      esac
      sleep "$SYNC_INTERVAL"
    done
    ;;
  *)
    echo "usage: 90_sync.sh [loop|once] (env: SRC DST RUN_ID SYNC_INTERVAL)" >&2
    exit 2
    ;;
esac
