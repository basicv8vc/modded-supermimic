#!/bin/bash
# Drive a full NablaMimic-on-Colab experiment from this machine.
#
#   ./run.sh all          # up -> clone -> deps -> data -> smoke -> train -> wait -> collect
#   ./run.sh up           # create/reuse the GPU session and mount Drive
#   ./run.sh step NAME    # one VM step: clone|deps|data|smoke|train|eval|report
#   ./run.sh status       # training/eval/sync progress snapshot
#   ./run.sh wait         # block until the remote pipeline leaves RUNNING (then collect hint)
#   ./run.sh collect      # fetch out/<RUN_ID>/ and verify every byte against the source's own
#                         # size+sha256 snapshot (3 tries/file, atomic replace, resumable,
#                         # no false receipt); SOURCE=... recovers from a mounted Drive copy
#   ./run.sh report       # regenerate the local report from out/<RUN_ID>/ (offline)
#   ./run.sh aggregate    # mean/std across collected seeds (out/*/manifest.json)
#   ./run.sh down         # stop the session (refuses if results are not persisted, see below)
#
# Env: SESSION (default nablamimic) | GPU (default G4) | RUN (default ea) | SEED (default 0)
#      BUDGET_MINUTES (default 8) | DRIVE (default 1, 0 disables Drive) | COLLECT_CKPT
#      (latest|all|none, default latest) | PROXY (e.g. http://127.0.0.1:7897) | RUN_ID
#      (override for status/collect/wait) | FORCE=1 (down without persistence check)
#      | SOURCE=/content/drive/MyDrive/nm/<run_id> (collect from a mounted Drive copy instead
#        of /content/nabla/runs/<run_id>; needs RUN_ID, no symlink required)
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-nablamimic}"
GPU="${GPU:-G4}"
RUN="${RUN:-ea}"
SEED="${SEED:-0}"
BUDGET_MINUTES="${BUDGET_MINUTES:-8}"
DRIVE="${DRIVE:-1}"
COLLECT_CKPT="${COLLECT_CKPT:-latest}"
SOURCE="${SOURCE:-}"

if [ -n "${PROXY:-}" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
fi

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- validation (fails fast on bad input) ---------------------------------
case "$SEED" in ''|*[!0-9]*) die "SEED must be a non-negative integer (got '$SEED')" ;; esac
case "$BUDGET_MINUTES" in ''|*[!0-9.]*|.*|*.) die "BUDGET_MINUTES must be a positive number (got '$BUDGET_MINUTES')" ;; esac
awk "BEGIN{exit !($BUDGET_MINUTES > 0)}" || die "BUDGET_MINUTES must be > 0"
case "$RUN" in *[!A-Za-z0-9._-]*|'') die "RUN must match [A-Za-z0-9._-]+ (got '$RUN')" ;; esac
case "$COLLECT_CKPT" in
  none|latest|all) ;;
  *) die "COLLECT_CKPT must be none|latest|all (got '$COLLECT_CKPT')" ;;
esac
# SOURCE is a remote *path*, not shell text: it is re-validated inside collect_run.py (which
# also checks the run id and the source's own manifest), this only fails fast and locally.
if [ -n "$SOURCE" ]; then
  case "$SOURCE" in
    /*) ;;
    *) die "SOURCE must be an absolute path on the VM (got '$SOURCE')" ;;
  esac
  case "$SOURCE" in
    *[!A-Za-z0-9._/@+=,-]*) die "SOURCE may only contain [A-Za-z0-9._/@+=,-] (got '$SOURCE')" ;;
  esac
  [ -n "${RUN_ID:-}" ] || die "SOURCE needs RUN_ID: the recovery directory comes from Drive, so the run cannot be discovered on the VM"
fi

# --- remote helpers -------------------------------------------------------
remote_py() {  # remote_py <<'PY' ... PY   (short commands; one kernel connection at a time)
  colab exec -s "$SESSION" --timeout "${REMOTE_TIMEOUT:-120}"
}

drive_mounted() {
  printf 'import os\nprint(os.path.isdir("/content/drive/MyDrive"))\n' \
    | remote_py 2>/dev/null | tail -1 | grep -qx True
}

mount_drive() {
  if [ "$DRIVE" = "0" ]; then
    echo "Drive disabled (DRIVE=0), skip mount"
    return 0
  fi
  if drive_mounted; then
    echo "Drive already mounted at /content/drive/MyDrive"
    return 0
  fi
  say "Mounting Google Drive (browser authorization may be required)"
  echo "提示：若终端打印授权 URL，请在浏览器完成授权；若提示 Press Enter，直接回车继续。"
  if colab drivemount -s "$SESSION"; then
    if drive_mounted; then
      echo "Drive mounted: /content/drive/MyDrive"
    else
      echo "WARN: drivemount returned but /content/drive/MyDrive not found; continuing without Drive"
    fi
  else
    echo "WARN: Drive mount failed; continuing without data mirror / result sync"
  fi
}

# upload every VM script + tool so any step can call any other (small files, idempotent).
# Fails hard: silently running an older script left on the VM would record an experiment
# under a workflow commit whose code never executed.
upload_all() {
  local f dst ok failed=""
  for f in vm/*.sh tools/*.sh tools/*.py; do
    [ -f "$f" ] || continue
    dst="/content/$(basename "$f")"
    ok=0
    for _ in 1 2 3; do
      if colab upload -s "$SESSION" "$f" "$dst" >/dev/null 2>&1; then ok=1; break; fi
      sleep 3
    done
    [ "$ok" = 1 ] || { echo "ERROR: upload failed for $f (3 tries)"; failed="$failed $f"; }
  done
  [ -z "$failed" ] || return 1
  verify_remote_scripts || return 1
  return 0
}

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | cut -d' ' -f1
  else
    shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
  fi
}

size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }

# Confirm the VM really has the scripts this checkout has.
verify_remote_scripts() {
  local names="" f n r l bad=0 remote
  for f in vm/*.sh tools/*.sh tools/*.py; do
    [ -f "$f" ] || continue
    names="$names $(basename "$f")"
  done
  remote=$(printf 'import hashlib,os\nfor n in "%s".split():\n p="/content/"+n\n print(n, hashlib.sha256(open(p,"rb").read()).hexdigest() if os.path.exists(p) else "MISSING")\n' \
    "$names" | remote_py 2>/dev/null) || { echo "ERROR: cannot verify remote script hashes"; return 1; }
  [ -n "$remote" ] || { echo "ERROR: empty response while verifying remote script hashes"; return 1; }
  for f in vm/*.sh tools/*.sh tools/*.py; do
    [ -f "$f" ] || continue
    n=$(basename "$f")
    r=$(printf '%s\n' "$remote" | awk -v n="$n" '$1==n {print $2; exit}')
    l=$(sha_of "$f")
    [ "$r" = "$l" ] || { echo "ERROR: remote $n ($r) != local ($l)"; bad=1; }
  done
  [ "$bad" -eq 0 ] || return 1
  return 0
}

# run one VM step: upload -> clear old log -> remote detached Popen -> poll the log
# for DONE/LAUNCHED/FATAL (same convention as the SuperMimic workflow).
run_step() {
  local name="$1" script="$2"
  say "Step: $name ($script)"
  upload_all || die "upload/verification failed - refusing to run a step with possibly stale VM scripts"
  printf 'import os\np="/content/%s.log"\nos.path.exists(p) and os.remove(p)\nprint("log cleared")\n' \
    "${script%.sh}" | remote_py | tail -1 || echo "(log clear skipped)"
  # start_new_session: the step survives local disconnects and kernel churn
  printf 'import os,subprocess\nenv=dict(os.environ,RUN="%s",SEED="%s",BUDGET_MINUTES="%s",DRIVE_ROOT="/content/drive/MyDrive")\nsubprocess.Popen(["bash","/content/%s"],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nprint("started")\n' \
    "$RUN" "$SEED" "$BUDGET_MINUTES" "$script" | remote_py | tail -1
  local tries=0
  local out=""
  while true; do
    sleep 20
    out=$(printf 'import os\np="/content/%s.log"\nprint(open(p).read()[-2000:] if os.path.exists(p) else "(no log yet)")\n' \
      "${script%.sh}" | remote_py 2>/dev/null | tail -40 || true)
    # -E is required: BSD grep (macOS) does not treat BRE "\|" as alternation,
    # so the old form never matched and every step polled to its 45-min timeout.
    if grep -qE " DONE | LAUNCHED " <<<"$out"; then
      echo "$out" | grep -E " DONE | LAUNCHED "
      return 0
    fi
    if grep -qE "FATAL|=== FAIL " <<<"$out"; then
      echo "$out"
      die "step $name failed"
    fi
    tries=$((tries + 1))
    if [ "$tries" -ge "${STEP_TRIES:-135}" ]; then   # 135 * 20 s = 45 min
      echo "$out"
      die "step $name timed out (45 min)"
    fi
    if [ $((tries % 6)) -eq 0 ]; then
      echo "$out" | tail -3
      echo "... waiting ($((tries * 20))s)"
    fi
  done
}

# Resolve the active run id. Non-zero means "the VM could not be queried", which callers
# must treat as unknown rather than as "there is no run" (fail closed).
resolve_run_id() {
  if [ -n "${RUN_ID:-}" ]; then
    echo "$RUN_ID"
    return 0
  fi
  local out
  out=$(printf 'import os\np="/content/nabla/logs/active_run_id.txt"\nprint(open(p).read().strip() if os.path.exists(p) else "")\n' \
    | remote_py 2>&1) || return 1
  out=$(printf '%s\n' "$out" | tail -1)
  case "$out" in
    *Traceback*|*Error*|*Exception*|*"Connection"*) return 1 ;;
  esac
  printf '%s\n' "$out"
  return 0
}

# Pipeline state on the VM, or QUERY_FAILED. Never guesses.
remote_pipeline_state() {
  local out
  out=$(printf 'import os\np="/content/nabla/logs/active_run_id.txt"\nrid=open(p).read().strip() if os.path.exists(p) else ""\nif not rid:\n print("NO_RUN")\nelse:\n q="/content/nabla/logs/"+rid+".status"\n print(open(q).read().strip().split()[0] if os.path.exists(q) else "NO_STATUS")\n' \
    | remote_py 2>&1) || { echo "QUERY_FAILED"; return 1; }
  out=$(printf '%s\n' "$out" | tail -1)
  case "$out" in
    PERSISTED|UNSYNCED|INCOMPLETE|INCOMPLETE_UNSYNCED|FAIL|RUNNING|NO_RUN|NO_STATUS|COMPLETE)
      echo "$out" ;;
    *) echo "QUERY_FAILED"; return 1 ;;
  esac
}

# .sync_receipt state on the Drive copy: ok | partial | failed | none | query_failed
drive_verify_state() {
  local out
  out=$(printf 'import os\np="/content/nabla/logs/active_run_id.txt"\nrid=open(p).read().strip() if os.path.exists(p) else ""\nd=open("/content/nabla/logs/"+rid+".drive_dir.txt").read().strip() if rid else ""\nr=os.path.join(d,".sync_receipt") if d else ""\nt=open(r).read() if r and os.path.exists(r) else ""\nprint("ok" if "verified=ok" in t else ("partial" if "verified=partial" in t else ("failed" if "verified=FAILED" in t else "none")))\n' \
    | remote_py 2>&1) || { echo "query_failed"; return 1; }
  out=$(printf '%s\n' "$out" | tail -1)
  case "$out" in
    ok|partial|failed|none) echo "$out" ;;
    *) echo "query_failed"; return 1 ;;
  esac
}

# Counts in collect_manifest.txt are diagnostic, not proof that the downloaded
# weights are intact. Query the still-running VM for hashes before authorizing
# its destruction, and compare every source file against its local copy.
local_archive_complete() {
  local rid="$1" state="$2" dir="out/$1" exp loc remote
  [ -f "$dir/manifest.json" ] || return 1
  [ -f "$dir/collect_manifest.txt" ] || return 1
  exp=$(sed -n 's/^ckpt_expected=//p' "$dir/collect_manifest.txt" | head -1)
  case "${exp:-x}" in ''|*[!0-9]*) return 1 ;; esac
  [ "$exp" -gt 0 ] || return 1

  # The remote snapshot is freshly computed, not read back from a local receipt.
  # Normalise upstream's ckpt/ and checkpoints/ to the collector's checkpoints/.
  remote=$(printf 'import os,glob,hashlib\nroot="/content/nabla/runs/%s"\nfiles=["manifest.json","config.json","log.jsonl","summary.json","curve.csv","report.md","train.log","provenance.json","ea_report.json","evaluation/eval_nominal.json","evaluation/per_clip.csv","evaluation/summary.json"]\nprint("PERSISTENCE_BEGIN")\nfor rel in files:\n p=os.path.join(root,rel)\n if not os.path.isfile(p) and rel.startswith("evaluation/"):\n  p=os.path.join(root,os.path.basename(rel))\n if not os.path.isfile(p):\n  print("MISSING",rel)\n  continue\n h=hashlib.sha256()\n with open(p,"rb") as f:\n  for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)\n print("FILE",rel,os.path.getsize(p),h.hexdigest())\ncks=sorted(glob.glob(os.path.join(root,"checkpoints","it_*.pkl"))+glob.glob(os.path.join(root,"ckpt","it_*.pkl")))\nfor p in cks:\n h=hashlib.sha256()\n with open(p,"rb") as f:\n  for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)\n print("FILE","checkpoints/"+os.path.basename(p),os.path.getsize(p),h.hexdigest())\nprint("PERSISTENCE_END",len(cks))\n' "$rid" | REMOTE_TIMEOUT=600 remote_py 2>/dev/null) || return 1

  local started=0 ended=0 remote_count="" n_ck=0 seen="" line rel size hash extra
  while IFS= read -r line; do
    case "$line" in
      PERSISTENCE_BEGIN) started=1 ;;
      PERSISTENCE_END\ *)
        [ "$started" -eq 1 ] || return 1
        read -r _ remote_count extra <<<"$line"
        ended=1
        ;;
      FILE\ *)
        [ "$started" -eq 1 ] && [ "$ended" -eq 0 ] || return 1
        read -r _ rel size hash extra <<<"$line"
        [ -z "${extra:-}" ] || return 1
        case "$size" in ''|*[!0-9]*) return 1 ;; esac
        [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || return 1
        [ -f "$dir/$rel" ] || return 1
        [ "$(size_of "$dir/$rel")" = "$size" ] || return 1
        [ "$(sha_of "$dir/$rel")" = "$hash" ] || return 1
        seen="$seen $rel"
        case "$rel" in checkpoints/it_*.pkl) n_ck=$((n_ck + 1)) ;; esac
        ;;
      MISSING\ *)
        [ "$started" -eq 1 ] && [ "$ended" -eq 0 ] || return 1
        ;;
      *) [ "$started" -eq 0 ] || return 1 ;;
    esac
  done <<<"$remote"
  [ "$started" -eq 1 ] && [ "$ended" -eq 1 ] || return 1
  case "$remote_count" in ''|*[!0-9]*) return 1 ;; esac
  [ "$remote_count" -gt 0 ] && [ "$remote_count" -eq "$exp" ] && [ "$n_ck" -eq "$exp" ] || return 1
  loc=$(find "$dir/checkpoints" -type f -name 'it_*.pkl' 2>/dev/null | wc -l | tr -d ' ')
  [ "$loc" -eq "$exp" ] || return 1
  local required="manifest.json config.json log.jsonl"
  if [ "$state" = "PERSISTED" ] || [ "$state" = "COMPLETE" ]; then
    required="$required summary.json curve.csv report.md evaluation/eval_nominal.json evaluation/per_clip.csv"
  fi
  local need
  for need in $required; do
    case " $seen " in *" $need "*) ;; *) return 1 ;; esac
  done
  return 0
}

# --- commands -------------------------------------------------------------
up() {
  say "Creating session '$SESSION' (GPU $GPU)"
  local st
  if ! st=$(colab status -s "$SESSION" 2>&1); then
    sleep 3
    st=$(colab status -s "$SESSION" 2>&1) || die "cannot query session (network/proxy): $st"
  fi
  if grep -q "not found" <<<"$st"; then
    colab new -s "$SESSION" --gpu "$GPU"
  else
    echo "session exists, reusing"
  fi
  colab status -s "$SESSION"
  mount_drive
  upload_all || die "upload/verification failed - the VM may hold stale scripts"
}

status() {
  say "Status: $RUN"
  upload_all || die "upload/verification failed - status would describe stale VM scripts"
  printf 'import subprocess,os\nr=subprocess.run(["bash","/content/60_status.sh"],capture_output=True,text=True,env={**os.environ,"RUN":"%s","SEED":"%s"})\nprint(r.stdout); print(r.stderr)\n' "$RUN" "$SEED" \
    | remote_py --timeout 120 | tail -50
}

wait_run() {
  say "Waiting for the remote pipeline to finish"
  local st="" state="" tries=0 interval="${WAIT_INTERVAL:-60}"
  while true; do
    st=$(printf 'import os\np="/content/nabla/logs/active_run_id.txt"\nrid=open(p).read().strip() if os.path.exists(p) else ""\ns=open(f"/content/nabla/logs/{rid}.status").read().strip() if rid else "NO_RUN"\nprint(rid+"|"+s.split()[0] if rid else "NO_RUN")\n' \
      | remote_py 2>/dev/null | tail -1 || echo "POLL_ERROR")
    state="${st##*|}"
    case "$state" in
      # COMPUTE_COMPLETE and PERSISTED are different facts; both mean the pipeline stopped.
      PERSISTED|UNSYNCED|INCOMPLETE|INCOMPLETE_UNSYNCED|FAIL|COMPLETE)
        echo "pipeline finished: $st"
        [ "$state" = "PERSISTED" ] || echo "NOTE: state is $state - artifacts are NOT verified on Drive (see ./run.sh status)"
        return 0
        ;;
      *)
        sleep "$interval"
        tries=$((tries + 1))
        if [ "$tries" -ge "${WAIT_TRIES:-90}" ]; then
          die "wait timed out after $((tries * interval))s (status: $st)"
        fi
        ;;
    esac
  done
}

# tools/collect_run.py, optionally pointed at a mounted Drive copy (SOURCE=...).
collect_run() {
  if [ -n "$SOURCE" ]; then
    python3 tools/collect_run.py "$@" --source "$SOURCE"
  else
    python3 tools/collect_run.py "$@"
  fi
}

collect() {
  local rid out st
  # Fail closed at every step: a collection we cannot verify is worse than no collection,
  # because the persistence gate in `down` would otherwise treat it as a safe archive.
  if [ -n "$SOURCE" ]; then
    # Recovery source: the training VM is normally gone, so the run cannot be discovered
    # from it and its state file is not a witness any more. Requiring it would block a
    # recovery that is otherwise complete; the collector still verifies every byte and now
    # labels provenance from the source's own .sync_receipt instead of assuming it.
    rid="$RUN_ID"
  else
    rid=$(resolve_run_id) || die "cannot query the VM for the active run (network/proxy); refusing to guess what to collect"
    [ -n "$rid" ] || die "no run to collect (run step train first, or set RUN_ID)"
  fi
  st=$(remote_pipeline_state) || st="QUERY_FAILED"
  if [ -n "$SOURCE" ]; then
    case "$st" in
      RUNNING) die "pipeline is still RUNNING on the VM - artifacts are still being written. './run.sh wait' first, or FORCE=1 to collect a moving target" ;;
      QUERY_FAILED|NO_RUN|NO_STATUS)
        echo "NOTE: pipeline state is '$st' at the recovery source; the collector derives it from the source .sync_receipt and records state_source accordingly"
        st="" ;;
    esac
  else
    if [ "$st" = "QUERY_FAILED" ]; then
      die "cannot read the remote pipeline state; refusing to collect (unreachable VM or unreadable status). FORCE=1 does not override a query we could not run"
    fi
    if [ "$st" = "RUNNING" ] && [ -z "${FORCE:-}" ]; then
      die "pipeline is still RUNNING on the VM - artifacts are still being written. './run.sh wait' first, or FORCE=1 to collect a moving target"
    fi
  fi
  out="out/$rid"
  if [ -e "$out" ]; then
    # A verified archive is protected. A directory without a success receipt is the debris of
    # an interrupted collect: re-entering it is the whole point of a resumable collector, and
    # every local file in it is re-measured against a fresh source snapshot before it is
    # reused, so nothing unverified can be inherited.
    if [ -e "$out/collect_manifest.txt" ] && [ -z "${FORCE:-}" ]; then
      die "$out already holds a verified archive (collect_manifest.txt); set FORCE=1 to re-collect into it"
    fi
    if [ ! -e "$out/collect_manifest.txt" ]; then
      echo "resuming into existing $out (no verified receipt there; local files will be re-measured)"
    fi
  fi
  say "Collecting results -> $out (source=${SOURCE:-/content/nabla/runs/$rid}, pipeline=${st:-derived}, ckpt=$COLLECT_CKPT)"
  # tools/collect_run.py owns the transfer: source snapshot -> per-file retry -> atomic
  # replace -> collect_manifest.txt only when every required file is verified. Local files
  # that already match the source are skipped, so a re-run continues instead of restarting.
  collect_run \
    --session "$SESSION" --run-id "$rid" --out "$out" --ckpt-mode "$COLLECT_CKPT" \
    --state "$st" --run "$RUN" --seed "$SEED" --gpu "$GPU" \
    --budget-minutes "$BUDGET_MINUTES" \
    || die "collect incomplete: a file did not match the source size+sha256 after 3 attempts, or a local I/O error stopped the transfer (see the ERROR lines above; exit 2=usage/identity, 3=source query, 4=verification, 5=local filesystem). Nothing was published as verified; fix the transfer and re-run './run.sh collect' - verified files are kept and skipped"
  say "Done: $(ls "$out")"
}

report_local() {
  local rid out
  rid=$(resolve_run_id)
  out="out/${RUN_ID:-$rid}"
  [ -d "$out" ] || die "$out not found (collect first)"
  python3 tools/finalize_run.py "$out" --run-id "$rid"
  say "Local report regenerated: $out/report.md"
}

aggregate() {
  # Default to the current experiment series (RUN prefix) instead of scanning every run in
  # out/, which mixes unrelated experiments and makes the consistency gate refuse everything.
  python3 tools/aggregate_seeds.py --scan out --series "${RUN:-ea}"
}

down() {
  local rid="" st="" ds="" local_ok=0 persisted=0
  if [ "${FORCE:-}" != "1" ]; then
    # Persistence gate, fail-closed: an unreachable VM, an unreadable state or a receipt
    # we cannot verify all mean "we do not know that the data is safe", and unknown is
    # not a reason to destroy the only copy of 1.5 GB of checkpoints.
    rid=$(resolve_run_id) || die "cannot query the VM for the active run (network/proxy). Refusing to stop; FORCE=1 only if you accept data loss"
    if [ -z "$rid" ]; then
      # A missing active_run_id is not proof that all runs were collected.
      # Only a genuinely empty run directory can be stopped without a run receipt.
      local inventory
      inventory=$(printf 'import os\np="/content/nabla/runs"\nprint("HAS_RUNS" if os.path.isdir(p) and any(os.path.isdir(os.path.join(p,x)) for x in os.listdir(p)) else "EMPTY")\n' \
        | remote_py 2>/dev/null | tail -1) || die "cannot inspect remote runs; refusing to stop"
      [ "$inventory" = "EMPTY" ] || die "no active run ID but remote run inventory is '${inventory:-unknown}'; refusing to stop"
      echo "no active run and no run directories on VM - ok to stop"
    else
      st=$(remote_pipeline_state) || st="QUERY_FAILED"
      case "$st" in
        RUNNING) die "pipeline still RUNNING on the VM - results are not final. Wait/collect first, or FORCE=1 ./run.sh down" ;;
        NO_RUN|NO_STATUS) die "active run $rid exists but its status is $st; refusing to stop without persistence proof" ;;
        PERSISTED|UNSYNCED|INCOMPLETE|INCOMPLETE_UNSYNCED|FAIL|COMPLETE) ;;
        *) die "cannot confirm the remote pipeline state (got '${st:-<empty>}'); refusing to stop. FORCE=1 only if you accept data loss" ;;
      esac
      case "$st" in
        PERSISTED|UNSYNCED|INCOMPLETE|INCOMPLETE_UNSYNCED|FAIL|COMPLETE)
          ds=$(drive_verify_state) || ds="query_failed"
          case "$ds:$st" in
            ok:*) persisted=1 ;;
            # a failed/incomplete run may legitimately lack report.md; its training
            # artifacts + every checkpoint are still required to match
            partial:FAIL|partial:INCOMPLETE|partial:INCOMPLETE_UNSYNCED) persisted=1 ;;
          esac
          # If Drive lacks a verified copy, inspect the actual local bytes against
          # a fresh remote SHA256 snapshot while the VM is still available.
          if [ "$persisted" != "1" ]; then
            local_archive_complete "$rid" "$st" && local_ok=1
          fi
          [ "$local_ok" = "1" ] && persisted=1
          if [ "$persisted" != "1" ]; then
            die "results of $rid are not safely persisted (pipeline=$st drive=$ds local_complete=$local_ok). Run './run.sh collect', or wait for a verified sync; FORCE=1 overrides"
          fi
          echo "persistence check ok (pipeline=$st drive=$ds local_complete=$local_ok)"
          ;;
      esac
    fi
  fi
  say "Stopping session '$SESSION'"
  colab stop -s "$SESSION"
  colab usage | tail -3
}

all() {
  up
  run_step clone 10_clone.sh
  run_step deps  20_deps.sh
  run_step data  30_data.sh
  run_step smoke 40_smoke.sh
  run_step train 50_train.sh
  wait_run
  collect
  say "All done. Inspect out/<run_id>/report.md; stop the VM with: $0 down"
}

case "${1:-}" in
  all)      all ;;
  up)       up ;;
  status)   status ;;
  wait)     wait_run ;;
  collect)  collect ;;
  report)   report_local ;;
  aggregate) aggregate ;;
  down)     down ;;
  step)     case "${2:-}" in
              clone)  run_step clone  10_clone.sh ;;
              deps)   run_step deps   20_deps.sh ;;
              data)   run_step data   30_data.sh ;;
              smoke)  run_step smoke  40_smoke.sh ;;
              train)  run_step train  50_train.sh ;;
              eval)   run_step eval   70_eval.sh ;;
              report) run_step report 80_report.sh ;;
              *) die "unknown step: ${2:-} (clone|deps|data|smoke|train|eval|report)" ;;
            esac ;;
  *) cat <<EOF
Usage: $0 {all|up|step NAME|status|wait|collect|report|aggregate|down}
  step NAME: clone deps data smoke train eval report
Env: SESSION GPU RUN SEED BUDGET_MINUTES DRIVE(=0 to disable) COLLECT_CKPT(latest|all|none)
     PROXY RUN_ID FORCE(=1 to bypass overwrite/persistence guards)
     SOURCE=/content/drive/MyDrive/nm/<run_id> (collect from a mounted Drive copy; needs RUN_ID)
EOF
     exit 1 ;;
esac
