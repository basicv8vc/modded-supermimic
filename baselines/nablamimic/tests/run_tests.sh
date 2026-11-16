#!/bin/bash
# Non-GPU test suite for the NablaMimic Colab workflow.
#   ./run_tests.sh
# Covers: shell syntax + shellcheck, run.sh parameter validation, run_id format,
# finalize_run.py + aggregate_seeds.py against the archived Experiment A fixture,
# Drive-sync mock (local dirs), and the down-gate/status plumbing via a mock colab CLI.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # colab_workflow/

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "ok   - $*"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL - $*"; }
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }
OUT=/tmp/nm_test_out

echo "=== 1. shell syntax ==="
for f in run.sh vm/*.sh tools/*.sh tests/*.sh; do
  if bash -n "$f"; then ok "bash -n $f"; else bad "bash -n $f"; fi
done
if command -v shellcheck >/dev/null 2>&1; then
  if shellcheck -S warning run.sh vm/*.sh tools/*.sh; then ok "shellcheck"; else bad "shellcheck"; fi
else
  echo "skip - shellcheck not installed"
fi
check "py_compile finalize_run.py" "python3 -m py_compile tools/finalize_run.py"
check "py_compile aggregate_seeds.py" "python3 -m py_compile tools/aggregate_seeds.py"
check "py_compile collect_run.py" "python3 -m py_compile tools/collect_run.py"

echo "=== 2. run.sh parameter validation & usage ==="
./run.sh > "$OUT" 2>&1; rc=$?
check "usage exits 1" "[ $rc -eq 1 ]"
check "usage text lists steps" "grep -q 'step NAME' $OUT"
SEED=abc ./run.sh status > "$OUT" 2>&1; rc=$?
check "SEED=abc rejected" "[ $rc -ne 0 ] && grep -q 'SEED must be' $OUT"
BUDGET_MINUTES=0 ./run.sh status > "$OUT" 2>&1; rc=$?
check "BUDGET_MINUTES=0 rejected" "[ $rc -ne 0 ] && grep -q 'BUDGET_MINUTES' $OUT"
BUDGET_MINUTES=x ./run.sh status > "$OUT" 2>&1; rc=$?
check "BUDGET_MINUTES=x rejected" "[ $rc -ne 0 ]"
RUN='bad name' ./run.sh status > "$OUT" 2>&1; rc=$?
check "RUN with space rejected" "[ $rc -ne 0 ] && grep -q 'RUN must match' $OUT"
./run.sh step nosuch > "$OUT" 2>&1; rc=$?
check "unknown step rejected" "[ $rc -ne 0 ] && grep -q 'unknown step' $OUT"

echo "=== 3. run_id format ==="
rid=$(bash tools/run_id.sh ea 0)
check "run_id matches NAME_sSEED_UTC ($rid)" "grep -qE '^ea_s0_[0-9]{8}T[0-9]{6}Z$' <<<'$rid'"
bash tools/run_id.sh 'bad name' 0 > "$OUT" 2>&1; rc=$?
check "run_id rejects bad NAME" "[ $rc -ne 0 ]"
bash tools/run_id.sh ea x > "$OUT" 2>&1; rc=$?
check "run_id rejects non-numeric SEED" "[ $rc -ne 0 ]"

echo "=== 4. finalize_run.py on archived Experiment A fixture ==="
FIX=out/ea_s0_20261106T061000Z
rm -rf /tmp/nm_test_run && mkdir -p /tmp/nm_test_run
cp -R "$FIX/." /tmp/nm_test_run/
rm -f /tmp/nm_test_run/manifest.json /tmp/nm_test_run/summary.json /tmp/nm_test_run/report.md \
      /tmp/nm_test_run/curve.csv /tmp/nm_test_run/provenance.json
python3 tools/finalize_run.py /tmp/nm_test_run --run-id test_run \
  --git-sha bfbb2a6773b4776c17402a34a93ec7589d9c7c18 > "$OUT" 2>&1
check "finalize exits 0" "[ $? -eq 0 ]"
check "manifest.json written"   "[ -s /tmp/nm_test_run/manifest.json ]"
check "curve.csv written"       "[ -s /tmp/nm_test_run/curve.csv ]"
check "report.md written"       "[ -s /tmp/nm_test_run/report.md ]"
check "evaluation/per_clip.csv" "[ -s /tmp/nm_test_run/evaluation/per_clip.csv ]"
python3 - > "$OUT" 2>&1 <<'PY'
import json
m = json.load(open("/tmp/nm_test_run/manifest.json"))
assert m["status"] == "PASS", m["status"]
assert m["seed"] == 0, m["seed"]
assert m["updates"] == 1255, m["updates"]
assert abs(m["normalized_return"] - 0.81353465484338) < 1e-9, m["normalized_return"]
assert m["failed_clips"] == "1/40", m["failed_clips"]
assert m["budget_checkpoint"] == "it_1250", m["budget_checkpoint"]
assert m["last_checkpoint"] == "it_1255", m["last_checkpoint"]
assert m["git_sha"] == "bfbb2a6773b4776c17402a34a93ec7589d9c7c18"
assert m["config_fingerprint"], "fingerprint missing"
assert len(m["checkpoint_curve"]) == 6, m["checkpoint_curve"]
print("manifest fields ok")
PY
check "manifest fields (PASS, 0.8135, it_1250, 1/40, 1255)" "[ $? -eq 0 ]"
python3 - > "$OUT" 2>&1 <<'PY'
import csv
rows = list(csv.DictReader(open("/tmp/nm_test_run/evaluation/per_clip.csv")))
its = {r["checkpoint"] for r in rows}
assert its == {"it_250", "it_500", "it_750", "it_1000", "it_1250", "it_1255"}, its
n1250 = [r for r in rows if r["checkpoint"] == "it_1250"]
assert len(n1250) == 40, len(n1250)
fails = [r for r in n1250 if r["fall_s"]]
assert len(fails) == 1 and fails[0]["clip"] == "fightAndSports1_subject1", fails
print("per_clip ok")
PY
check "per_clip.csv (6 ckpt x 40 clips, 1 fall @ it_1250)" "[ $? -eq 0 ]"
python3 - > "$OUT" 2>&1 <<'PY'
import json
s = json.load(open("/tmp/nm_test_run/summary.json"))
assert s["final"]["budget_checkpoint"] == "it_1250"
assert s["final"]["last_evaluated_checkpoint"] == "it_1255"
assert abs(s["final"]["last_evaluated_return"] - 0.7942917616027125) < 1e-9
assert s["status"] == "PASS"
assert s["evaluation"]["it1250"]["full_norm_return"] is not None
print("summary ok")
PY
check "summary.json budget vs last checkpoint" "[ $? -eq 0 ]"

echo "=== 5. aggregate_seeds.py (consistency gate) ==="
rm -rf /tmp/nm_agg && mkdir -p /tmp/nm_agg/run_a /tmp/nm_agg/run_b /tmp/nm_agg/run_bad
python3 - <<'PY'
import json, pathlib
base = json.load(open("/tmp/nm_test_run/manifest.json"))
for name, seed, ret in (("run_a", 0, 0.81), ("run_b", 1, 0.83)):
    m = dict(base); m.update(experiment_id=name, seed=seed, normalized_return=ret,
                             failed_clips="1/40", status="PASS")
    pathlib.Path(f"/tmp/nm_agg/{name}/manifest.json").write_text(json.dumps(m))
m = dict(base); m.update(experiment_id="run_bad", seed=2, normalized_return=0.5,
                        config_fingerprint="deadbeefdeadbeef")
pathlib.Path("/tmp/nm_agg/run_bad/manifest.json").write_text(json.dumps(m))
PY
python3 tools/aggregate_seeds.py /tmp/nm_agg/run_a /tmp/nm_agg/run_b > "$OUT" 2>&1
check "aggregate exits 0" "[ $? -eq 0 ]"
check "aggregate writes aggregate.json" "[ -s /tmp/nm_agg/run_a/aggregate.json ]"
python3 - > "$OUT" 2>&1 <<'PY'
import json, statistics
a = json.load(open("/tmp/nm_agg/run_a/aggregate.json"))
assert a["n_pass"] == 2, a
assert abs(a["normalized_return_mean"] - round(statistics.mean([0.81, 0.83]), 4)) < 1e-9
assert abs(a["normalized_return_std"] - round(statistics.stdev([0.81, 0.83]), 4)) < 1e-9
assert a["consistency"]["git_sha"] == "bfbb2a6773b4776c17402a34a93ec7589d9c7c18"
print("aggregate ok")
PY
check "aggregate mean/std correct" "[ $? -eq 0 ]"
python3 tools/aggregate_seeds.py /tmp/nm_agg/run_a /tmp/nm_agg/run_bad > "$OUT" 2>&1; rc=$?
check "mixed configs refused (rc=3)" "[ $rc -eq 3 ] && grep -q 'REFUSE' $OUT"

echo "=== 6. Drive sync: source-vs-destination verification ==="
rm -rf /tmp/nm_sync && mkdir -p /tmp/nm_sync/src/evaluation /tmp/nm_sync/src/checkpoints \
  /tmp/nm_sync/dst /tmp/nm_sync/logs
echo '{"iter": 1}' > /tmp/nm_sync/src/log.jsonl
echo '{}' > /tmp/nm_sync/src/manifest.json
echo '{}' > /tmp/nm_sync/src/config.json
echo '{"k": 1}' > /tmp/nm_sync/src/summary.json
printf 'a,b\n1,2\n' > /tmp/nm_sync/src/curve.csv
echo '# r' > /tmp/nm_sync/src/report.md
echo '{}' > /tmp/nm_sync/src/evaluation/eval_nominal.json
printf 'c\nx\n' > /tmp/nm_sync/src/evaluation/per_clip.csv
head -c 8192 /dev/urandom > /tmp/nm_sync/src/checkpoints/it_000250.pkl
head -c 8192 /dev/urandom > /tmp/nm_sync/src/checkpoints/it_000500.pkl
echo COMPLETE > /tmp/nm_sync/logs/testrun.status
sync_run() {
  SYNC_LOG="/tmp/nm_sync/logs/$1.log" RUN_ID=testrun WORK=/tmp/nm_sync \
    SRC=/tmp/nm_sync/src DST=/tmp/nm_sync/dst bash vm/90_sync.sh once
}

sync_run ok > "$OUT" 2>&1; rc=$?
check "sync once exits 0" "[ $rc -eq 0 ]"
check "sync copied files" "[ -s /tmp/nm_sync/dst/manifest.json ]"
check "sync copied every checkpoint" "[ -s /tmp/nm_sync/dst/checkpoints/it_000500.pkl ]"
check "receipt written only after verification" "[ -s /tmp/nm_sync/dst/.sync_receipt ]"
check "receipt says verified=ok" "grep -q '^verified=ok' /tmp/nm_sync/dst/.sync_receipt"
check "receipt covers every checkpoint" "grep -q '^checkpoints_verified=2' /tmp/nm_sync/dst/.sync_receipt"
check "receipt verified=2 of total=2" "grep -q '^checkpoints_total=2' /tmp/nm_sync/dst/.sync_receipt"
check "receipt has no MISSING/MISMATCH rows" "! grep -qE 'MISSING|MISMATCH' /tmp/nm_sync/dst/.sync_receipt"
check "heartbeat written" "[ -s /tmp/nm_sync/dst/.sync_status ]"

# rsync failure must not leave a receipt behind: stale files + stale receipt is the
# exact combination that would let `down` destroy the only copy. An unreadable source
# file makes rsync exit non-zero while the destination stays writable.
echo secret > /tmp/nm_sync/src/secret.bin && chmod 000 /tmp/nm_sync/src/secret.bin
sync_run rsfail > "$OUT" 2>&1; rc=$?
check "sync failure exits non-zero" "[ $rc -ne 0 ]"
check "sync failure removes previous receipt" "[ ! -e /tmp/nm_sync/dst/.sync_receipt ]"
chmod 644 /tmp/nm_sync/src/secret.bin && rm -f /tmp/nm_sync/src/secret.bin
sync_run recovered > "$OUT" 2>&1
check "sync recovers after a transient failure" "[ -s /tmp/nm_sync/dst/.sync_receipt ]"

# report artifacts missing -> partial (training data + checkpoints still verified)
rm /tmp/nm_sync/src/report.md
sync_run partial > "$OUT" 2>&1; rc=$?
check "incomplete run exits non-zero" "[ $rc -ne 0 ]"
check "incomplete run is verified=partial" "grep -q '^verified=partial' /tmp/nm_sync/dst/.sync_receipt"

# The training log and resolved config are NOT soft/report artifacts. A mismatch
# or missing source must invalidate the receipt, never produce verified=partial.
mv /tmp/nm_sync/src/log.jsonl /tmp/nm_sync/log_saved
sync_run missing_log > "$OUT" 2>&1; rc=$?
check "missing source log is a hard sync failure" "[ $rc -ne 0 ] && [ ! -e /tmp/nm_sync/dst/.sync_receipt ]"
check "missing log cannot be downgraded to partial" "grep -q '^verified=failed' /tmp/nm_sync/dst/.sync_receipt.invalid"
mv /tmp/nm_sync/log_saved /tmp/nm_sync/src/log.jsonl
mv /tmp/nm_sync/src/config.json /tmp/nm_sync/config_saved
sync_run missing_config > "$OUT" 2>&1; rc=$?
check "missing resolved config is a hard failure" "[ $rc -ne 0 ] && [ ! -e /tmp/nm_sync/dst/.sync_receipt ]"
mv /tmp/nm_sync/config_saved /tmp/nm_sync/src/config.json

# no checkpoints at source -> archive cannot reproduce the run: hard failure, no receipt
mv /tmp/nm_sync/src/checkpoints /tmp/nm_sync/ckpt_away
sync_run nockpt > "$OUT" 2>&1; rc=$?
check "missing checkpoints exits non-zero" "[ $rc -ne 0 ]"
check "missing checkpoints leaves no receipt" "[ ! -e /tmp/nm_sync/dst/.sync_receipt ]"
check "missing checkpoints flagged invalid" "[ -s /tmp/nm_sync/dst/.sync_receipt.invalid ]"
check "invalid receipt records no_checkpoints" "grep -q 'no_checkpoints=1' /tmp/nm_sync/dst/.sync_receipt.invalid"
mv /tmp/nm_sync/ckpt_away /tmp/nm_sync/src/checkpoints

echo RUNNING > /tmp/nm_sync/logs/testrun.status
( sleep 3; echo COMPLETE > /tmp/nm_sync/logs/testrun.status ) &
SYNC_LOG=/tmp/nm_sync/logs/sync_loop.log RUN_ID=testrun WORK=/tmp/nm_sync SYNC_INTERVAL=1 \
  SRC=/tmp/nm_sync/src DST=/tmp/nm_sync/dst timeout 40 bash vm/90_sync.sh loop > "$OUT" 2>&1
check "loop mode exits after the pipeline leaves RUNNING" "[ $? -eq 0 ]"

echo "=== 7. mock colab CLI plumbing (status + fail-closed down gate) ==="
MOCK=/tmp/nm_mockbin
rm -rf "$MOCK" /tmp/nm_mock_state && mkdir -p "$MOCK" /tmp/nm_mock_state
cat > "$MOCK/colab" <<'MOCK'
#!/bin/bash
# mock colab CLI for plumbing tests: dispatches on args, then on the python code on stdin
STATE=/tmp/nm_mock_state
MOCK_RID="${MOCK_RID:-$(cat "$STATE/run_id" 2>/dev/null || echo mock_s1_20261107T000000Z)}"
FAKE_VM="${NM_FAKE_VM:-/tmp/nm_fakevm}"
case "${1:-}" in
  upload)
    if [ -f /tmp/nm_mock_state/upload_fail ]; then echo "mock: upload failed" >&2; exit 1; fi
    echo uploaded ;;
  stop)   echo "Session terminated" ;;
  usage)  echo "Active assignments: 0" ;;
  status)
    # `colab status -s SESSION`: a session probe for the pre-flight diagnostics record.
    if [ -f "$STATE/status_probe_fail" ]; then
      echo "mock: session not found" >&2
      exit 1
    fi
    if [ -f "$STATE/status_probe_out" ]; then cat "$STATE/status_probe_out"; else
      printf 'Session: mock-session-42\nGPU: none\nStatus: idle\n'
    fi
    exit 0 ;;
  download)
    # colab download -s SESSION SRC DST  (mirrors $FAKE_VM/content/... and injects faults)
    shift
    src=""; dst=""
    while [ $# -gt 0 ]; do
      case "$1" in
        -s) shift 2 ;;
        *)  if [ -z "$src" ]; then src="$1"; else dst="$1"; fi; shift ;;
      esac
    done
    base="$(basename "$src")"
    # Every download attempt is logged so a test can prove that a resumed collect asked
    # the VM for nothing (or only for the files that were actually missing/damaged).
    printf '%s\n' "$src" >> "$STATE/dl_calls.log"
    # Abrupt termination of the *collector* mid-transfer: ppid is the python3 process
    # running tools/collect_run.py. This models SIGKILL/SIGTERM, where no Python-level
    # failure handler runs at all.
    if [ -f "$STATE/dl_signal_$base" ]; then
      sig=$(cat "$STATE/dl_signal_$base")
      echo "mock: sending SIG$sig to the collector mid-transfer ($base)" >&2
      # The collector captures this child's stderr and dies with it, so leave a marker the
      # test can read afterwards to prove the run really ended by signal, not by fail().
      echo "$sig" > "$STATE/dl_killed_by"
      kill -s "$sig" "$PPID" 2>/dev/null
      sleep 1
      exit 1
    fi
    n=$(cat "$STATE/dl_fail_$base" 2>/dev/null || echo 0)
    if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
      echo $((n - 1)) > "$STATE/dl_fail_$base"
      echo "mock: transfer interrupted at 0% ($base)" >&2
      exit 1
    fi
    # /content/... on the "VM" is $FAKE_VM/... here, whichever directory the source names.
    case "$src" in
      /content/*) real="$FAKE_VM${src#/content}" ;;
      *) real="$src" ;;
    esac
    if [ ! -f "$real" ]; then echo "mock: remote file not found: $src" >&2; exit 1; fi
    mkdir -p "$(dirname "$dst")" || exit 1
    cp "$real" "$dst" || exit 1
    if [ -f "$STATE/dl_truncate_$base" ]; then
      sz=$(wc -c < "$dst" | tr -d ' ')
      head -c $((sz / 2)) "$dst" > "$dst.half" && mv "$dst.half" "$dst"
    fi
    if [ -f "$STATE/dl_corrupt_$base" ]; then
      # same size, different bytes: only a checksum can catch this
      printf 'Z' | dd of="$dst" bs=1 seek=0 conv=notrunc 2>/dev/null
    fi
    if [ -f "$STATE/dl_unreadable_$base" ]; then
      # the transfer "succeeded" but the local file cannot be read back: a local I/O error
      # inside the verification step, not a remote one
      chmod 000 "$dst" 2>/dev/null
    fi
    exit 0 ;;
  exec)
    code="$(cat)"
    case "$code" in
      *COLLECT_SNAPSHOT_BEGIN*)
        if [ -f "$STATE/snapshot_fail" ]; then
          echo "Traceback (most recent call last):" >&2
          echo "ConnectionError: kernel is not responding" >&2
          exit 1
        fi
        # The source directory is whatever the program was given (repr'd on one line),
        # so --source is exercised for real rather than assumed.
        root=$(printf '%s\n' "$code" | sed -n "s/^root = '\(.*\)'\$/\1/p" | head -1)
        [ -n "$root" ] || root="/content/nabla/runs/$MOCK_RID"
        case "$root" in
          /content/*) rootdir="$FAKE_VM${root#/content}" ;;
          *) rootdir="$root" ;;
        esac
        echo COLLECT_SNAPSHOT_BEGIN
        echo "ROOT $root"
        # An absent/unparseable experiment_id must come back as "IDENTITY -", exactly like
        # the real SNAPSHOT_PROGRAM does, so the strict --source check can refuse it as an
        # identity failure (exit 2) instead of a malformed-snapshot failure (exit 3).
        if [ -f "$STATE/identity" ]; then
          ident=$(cat "$STATE/identity")
        elif [ -f "$rootdir/manifest.json" ]; then
          ident=$(sed -n 's/.*"experiment_id" *: *"\([^"]*\)".*/\1/p' "$rootdir/manifest.json" | head -1)
        else
          ident="-"
        fi
        [ -n "$ident" ] || ident="-"
        echo "IDENTITY $ident"
        raw=0
        if [ -d "$rootdir" ]; then
          for f in $(cd "$rootdir" && find . -type f ! -name '.*' | sed 's|^\./||' | sort); do
            case "$f" in
              checkpoints/*.pkl|ckpt/*.pkl) rel="checkpoints/$(basename "$f")"; raw=$((raw + 1)) ;;
              *) rel="$f" ;;
            esac
            printf 'FILE %s %s %s %s\n' "$rel" "$(wc -c < "$rootdir/$f" | tr -d ' ')" \
              "$(shasum -a 256 "$rootdir/$f" | cut -d' ' -f1)" "$root/$f"
          done
          if [ -f "$STATE/snapshot_missing" ]; then
            while read -r rel; do
              [ -n "$rel" ] || continue
              [ -e "$rootdir/$rel" ] || echo "MISSING $rel"
            done < "$STATE/snapshot_missing"
          fi
          if [ -f "$rootdir/.sync_receipt" ]; then
            echo COLLECT_RECEIPT_BEGIN
            while IFS= read -r rline; do echo "RCPT $rline"; done < "$rootdir/.sync_receipt"
            echo COLLECT_RECEIPT_END
          fi
        fi
        uniq=$(cd "$rootdir" 2>/dev/null && ls checkpoints/*.pkl 2>/dev/null | wc -l | tr -d ' ')
        echo "COLLECT_SNAPSHOT_END ${raw:-0} ${uniq:-0}"
        ;;
      *PERSISTENCE_BEGIN*)
        cat /tmp/nm_mock_state/remote_snapshot 2>/dev/null || echo "PERSISTENCE_ERROR" ;;
      *hashlib*)
        if [ -f /tmp/nm_mock_state/bad_hash ]; then
          echo "vm/10_clone.sh deadbeefdeadbeef"
        else
          names=$(printf '%s' "$code" | sed -n 's/.*for n in "\(.*\)"\.split():.*/\1/p')
          for n in $names; do
            p=""
            [ -f "vm/$n" ] && p="vm/$n"
            [ -f "tools/$n" ] && p="tools/$n"
            if [ -n "$p" ]; then echo "$n $(shasum -a 256 "$p" | cut -d' ' -f1)"
            else echo "$n MISSING"; fi
          done
        fi ;;
      *'verified=ok'*)
        cat /tmp/nm_mock_state/drive_state 2>/dev/null || echo none ;;
      *HAS_RUNS*)
        cat /tmp/nm_mock_state/inventory 2>/dev/null || echo HAS_RUNS ;;
      *'.status'*)
        if [ -f /tmp/nm_mock_state/status_fail ]; then
          echo "mock: kernel died" >&2; exit 1
        fi
        if [ -f /tmp/nm_mock_state/garbage ]; then echo "Traceback (most recent call last):"; 
        else cat /tmp/nm_mock_state/pipeline_state 2>/dev/null || echo PERSISTED; fi ;;
      *active_run_id.txt*)
        if [ -f /tmp/nm_mock_state/runid_fail ]; then echo "mock: no session" >&2; exit 1; fi
        cat /tmp/nm_mock_state/run_id 2>/dev/null || echo "mock_s0_20261106T000000Z" ;;
      *60_status.sh*)
        printf '%s\n' "--- run ---" "run_id=mock_s0" "status: PERSISTED 2026-11-06T00:00:00Z" "--- gpu ---" "0 %, 1000 MiB" ;;
      *) echo "started" ;;
    esac ;;
  *) echo "mock colab: $*" ;;
esac
MOCK
chmod +x "$MOCK/colab"

RID=mock_s0_20261106T000000Z
echo PERSISTED > /tmp/nm_mock_state/pipeline_state
rm -rf "out/$RID"

echo RUNNING > /tmp/nm_mock_state/pipeline_state
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refused while RUNNING" "[ $rc -ne 0 ] && grep -q 'still RUNNING' $OUT"

# an unreadable remote state must never be read as "safe to stop"
touch /tmp/nm_mock_state/garbage
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refused when the state query is unreadable" "[ $rc -ne 0 ] && grep -q 'cannot confirm' $OUT"
rm -f /tmp/nm_mock_state/garbage

# manifest.json alone is not evidence: the checkpoints are what cannot be re-created
echo PERSISTED > /tmp/nm_mock_state/pipeline_state
echo none > /tmp/nm_mock_state/drive_state
mkdir -p "out/$RID"
echo '{}' > "out/$RID/manifest.json"
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refused with only a local manifest" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"

# A known active run with a missing state file must not slip past the gate.
echo NO_STATUS > /tmp/nm_mock_state/pipeline_state
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refuses NO_STATUS for a known run" "[ $rc -ne 0 ] && grep -q 'refusing to stop' $OUT"
echo NO_RUN > /tmp/nm_mock_state/pipeline_state
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refuses NO_RUN for a known run" "[ $rc -ne 0 ] && grep -q 'refusing to stop' $OUT"
echo PERSISTED > /tmp/nm_mock_state/pipeline_state

# Losing active_run_id.txt while run directories still exist is also unsafe.
: > /tmp/nm_mock_state/run_id
echo HAS_RUNS > /tmp/nm_mock_state/inventory
PATH="$MOCK:$PATH" ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refuses missing active ID with run directories present" "[ $rc -ne 0 ] && grep -q 'run inventory' $OUT"
echo EMPTY > /tmp/nm_mock_state/inventory
PATH="$MOCK:$PATH" ./run.sh down > "$OUT" 2>&1; rc=$?
check "down permits genuinely empty VM" "[ $rc -eq 0 ] && grep -q 'no run directories' $OUT"
rm -f /tmp/nm_mock_state/run_id /tmp/nm_mock_state/inventory

# A recorded count alone is not evidence: the files may be missing or corrupt.
printf 'ckpt_expected=2\nckpt_local=2\n' > "out/$RID/collect_manifest.txt"
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refuses invented checkpoint counts without real files" "[ $rc -ne 0 ]"

# Make a complete mock remote snapshot with two checkpoint hashes and all
# official key artifacts, and assert that the real local bytes must match.
mkdir -p "out/$RID/checkpoints" "out/$RID/evaluation"
for f in config.json log.jsonl summary.json curve.csv report.md \
         evaluation/eval_nominal.json evaluation/per_clip.csv; do
  printf 'content for %s\n' "$f" > "out/$RID/$f"
done
printf 'checkpoint A\n' > "out/$RID/checkpoints/it_250.pkl"
printf 'checkpoint B\n' > "out/$RID/checkpoints/it_500.pkl"
{
  echo PERSISTENCE_BEGIN
  for f in manifest.json config.json log.jsonl summary.json curve.csv report.md \
           evaluation/eval_nominal.json evaluation/per_clip.csv checkpoints/it_250.pkl checkpoints/it_500.pkl; do
    printf 'FILE %s %s %s\n' "$f" "$(wc -c < "out/$RID/$f" | tr -d ' ')" "$(shasum -a 256 "out/$RID/$f" | cut -d' ' -f1)"
  done
  echo 'PERSISTENCE_END 2'
} > /tmp/nm_mock_state/remote_snapshot
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down proceeds only when all local files match remote hashes" "[ $rc -eq 0 ] && grep -q 'local_complete=1' $OUT"

# Same count and same size, but different checkpoint content: must refuse.
printf 'checkpoint C\n' > "out/$RID/checkpoints/it_500.pkl"
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refuses corrupted checkpoint despite matching counts" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"
printf 'checkpoint B\n' > "out/$RID/checkpoints/it_500.pkl"

# ...but a partial local archive is not
printf 'ckpt_expected=7\nckpt_local=1\n' > "out/$RID/collect_manifest.txt"
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refused on a partial local archive" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"

# a verified Drive receipt authorises shutdown even with nothing collected locally
rm -rf "out/$RID"
echo ok > /tmp/nm_mock_state/drive_state
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down proceeds on a verified Drive receipt" "[ $rc -eq 0 ] && grep -q 'drive=ok' $OUT"

# UNSYNCED compute with an unverified Drive copy must be refused
echo UNSYNCED > /tmp/nm_mock_state/pipeline_state
echo none > /tmp/nm_mock_state/drive_state
PATH="$MOCK:$PATH" RUN_ID=$RID ./run.sh down > "$OUT" 2>&1; rc=$?
check "down refused for UNSYNCED runs" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"

rm -f /tmp/nm_mock_state/drive_state /tmp/nm_mock_state/pipeline_state
PATH="$MOCK:$PATH" RUN_ID=$RID FORCE=1 ./run.sh down > "$OUT" 2>&1; rc=$?
check "FORCE=1 down stops the session" "[ $rc -eq 0 ] && grep -q 'Session terminated' $OUT"

echo PERSISTED > /tmp/nm_mock_state/pipeline_state
PATH="$MOCK:$PATH" ./run.sh status > "$OUT" 2>&1; rc=$?
check "status shows remote snapshot" "[ $rc -eq 0 ] && grep -q 'run_id=mock_s0' $OUT"

echo "=== 8. stale or failed uploads must block the step ==="
touch /tmp/nm_mock_state/upload_fail
PATH="$MOCK:$PATH" ./run.sh step clone > "$OUT" 2>&1; rc=$?
check "step aborts when uploads keep failing" "[ $rc -ne 0 ] && grep -q 'upload/verification failed' $OUT"
rm -f /tmp/nm_mock_state/upload_fail

touch /tmp/nm_mock_state/bad_hash
PATH="$MOCK:$PATH" ./run.sh step clone > "$OUT" 2>&1; rc=$?
check "step aborts when the VM holds a different script" "[ $rc -ne 0 ] && grep -q 'upload/verification failed' $OUT"
rm -f /tmp/nm_mock_state/bad_hash

PATH="$MOCK:$PATH" ./run.sh status > "$OUT" 2>&1; rc=$?
check "status proceeds when hashes match" "[ $rc -eq 0 ]"

echo "=== 9. config fingerprint is normalised, protocol is separate ==="
python3 - > "$OUT" 2>&1 <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("fr", "tools/finalize_run.py")
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)

base = {"argv": ["--config", "configs/ours.args", "--hours", "0.13333", "--seed", "0",
                 "--out", "/content/nabla/runs/a"]}
# same algorithm, different seed / output dir / argument order -> identical fingerprint
other = {"argv": ["--out", "/content/nabla/runs/b", "--seed", "1", "--hours", "0.13333",
                  "--config", "configs/ours.args"]}
assert fr.config_fingerprint(base) == fr.config_fingerprint(other), "seed/out must not change the fp"

# repeated flags: last value wins, like the CLI itself
dup = {"argv": ["--config", "configs/ours.args", "--hours", "3", "--hours", "0.13333", "--seed", "0"]}
assert fr.config_fingerprint(dup) == fr.config_fingerprint(base), "override order must not change the fp"

# a different training budget is a different experiment family
longer = {"argv": ["--config", "configs/ours.args", "--hours", "0.5", "--seed", "0"]}
assert fr.config_fingerprint(longer) != fr.config_fingerprint(base), "budget must change the fp"

# the real config from the archived run is stable across two invocations
import json
cfg = json.load(open("out/ea_s0_20261106T061000Z/config.json"))
assert fr.config_fingerprint(cfg) == fr.config_fingerprint(cfg)

# Resolved argparse values include multi-argument reward/action parameters and
# defaults. Changing an inner element MUST change the fingerprint.
import copy
changed = copy.deepcopy(cfg)
changed["term_penalty"][1] = 0.5
assert fr.config_fingerprint(changed) != fr.config_fingerprint(cfg), "term penalty second value lost"
changed = copy.deepcopy(cfg)
changed["action_target"][2] = 0.2
assert fr.config_fingerprint(changed) != fr.config_fingerprint(cfg), "action target third value lost"
changed = copy.deepcopy(cfg)
changed.update(seed=1, out="/content/another_run", gpu="another GPU")
assert fr.config_fingerprint(changed) == fr.config_fingerprint(cfg), "run identity leaked into fingerprint"

# Historical seed-0 archives must be migrated along with the fingerprint code.
for rid in ("ea_s0_20261106T061000Z", "ea_s0_20261106T142202Z"):
    root = f"out/{rid}"
    manifest = json.load(open(f"{root}/manifest.json"))
    resolved = json.load(open(f"{root}/config.json"))
    assert manifest["config_fingerprint"] == fr.config_fingerprint(resolved), rid

# Even legacy argv-only records must retain the full multi-argument tuple.
old = {"argv": ["--term-penalty", "2", "0.08", "0.10", "--seed", "0"]}
new = {"argv": ["--term-penalty", "2", "0.20", "0.10", "--seed", "1"]}
assert fr.config_fingerprint(old) != fr.config_fingerprint(new), "argv tuple truncated"

p1, p2 = fr.eval_protocol_fingerprint(), fr.eval_protocol_fingerprint()
assert p1 == p2 and len(p1) == 16, "protocol fp must be stable"
assert p1 != fr.config_fingerprint(cfg), "protocol fp must be independent of the config fp"
print("fingerprints ok")
PY
check "fingerprint normalisation (seed/order/override/budget)" "[ $? -eq 0 ]"

echo "=== 10. PASS requires a complete evaluation at the budget point ==="
mk_run() {  # mk_run DIR  (python expression mutates ev)
  python3 - "$1" "$2" <<'PY'
import json, pathlib, sys, shutil
dst = pathlib.Path(sys.argv[1]); mut = sys.argv[2]
shutil.rmtree(dst, ignore_errors=True)
dst.mkdir(parents=True)
src = pathlib.Path("out/ea_s0_20261106T061000Z")
shutil.copyfile(src / "log.jsonl", dst / "log.jsonl")
shutil.copyfile(src / "config.json", dst / "config.json")
# the archive lost its weights, but the checkpoint set still defines the budget point
ck = dst / "checkpoints"; ck.mkdir()
for it in (250, 500, 750, 1000, 1250):
    (ck / f"it_{it:06d}.pkl").write_bytes(b"placeholder")
ev = json.loads((src / "eval_nominal.json").read_text())
head = ev["it1250/nominal"]
if mut == "half_clips":
    head["per_clip"] = head["per_clip"][:20]
    head["failed_frac"] = 0.0
elif mut == "wrong_actor":
    head["actor_key"] = "actor"
elif mut == "nan_return":
    head["full_norm_return"] = float("nan")
elif mut == "no_budget_eval":
    del ev["it1250/nominal"]
(dst / "eval_nominal.json").write_text(json.dumps(ev))
PY
}
for mut in half_clips wrong_actor nan_return no_budget_eval; do
  mk_run /tmp/nm_integ "$mut"
  python3 tools/finalize_run.py /tmp/nm_integ --run-id "t_$mut" > "$OUT" 2>&1
  st=$(python3 -c 'import json;print(json.load(open("/tmp/nm_integ/manifest.json"))["status"])' 2>/dev/null)
  if [ "$st" = "INCOMPLETE" ]; then ok "partial eval ($mut) is INCOMPLETE"
  else bad "partial eval ($mut) should be INCOMPLETE, got $st"; fi
done

mk_run /tmp/nm_integ ok
python3 tools/finalize_run.py /tmp/nm_integ --run-id t_ok > "$OUT" 2>&1
check "intact eval stays PASS" "python3 -c 'import json;m=json.load(open(\"/tmp/nm_integ/manifest.json\"));assert m[\"status\"]==\"PASS\", m[\"status\"];assert not m[\"integrity_problems\"]'"
check "manifest carries both fingerprints" "python3 -c 'import json;m=json.load(open(\"/tmp/nm_integ/manifest.json\"));assert len(m[\"config_fingerprint\"])==16;assert len(m[\"eval_protocol_fingerprint\"])==16'"

echo "=== 11. aggregate groups by series and refuses mixed protocols ==="
rm -rf /tmp/nm_agg2 && mkdir -p /tmp/nm_agg2
python3 - <<'PY'
import json, pathlib
base = json.load(open("/tmp/nm_integ/manifest.json"))
for rid, seed, ret in (("ea_s0_20261101T000000Z", 0, 0.83), ("ea_s1_20261101T001000Z", 1, 0.82),
                       ("eb_s0_20261101T002000Z", 0, 0.60)):
    d = pathlib.Path("/tmp/nm_agg2") / rid; d.mkdir()
    m = dict(base); m.update(experiment_id=rid, seed=seed, normalized_return=ret)
    (d / "manifest.json").write_text(json.dumps(m))
# a run scored with a different protocol must not be averaged in
d = pathlib.Path("/tmp/nm_agg2") / "ea_s2_20261101T003000Z"; d.mkdir()
m = dict(base); m.update(experiment_id="ea_s2_20261101T003000Z", seed=2, normalized_return=0.9,
                         eval_protocol_fingerprint="0000000000000000")
(d / "manifest.json").write_text(json.dumps(m))
PY
python3 tools/aggregate_seeds.py --scan /tmp/nm_agg2 --series eb > "$OUT" 2>&1
check "series filter selects one family" "[ $? -eq 0 ] && grep -q 'eb_s0_20261101T002000Z' $OUT && ! grep -q 'ea_s0_20261101T000000Z' $OUT"
python3 tools/aggregate_seeds.py /tmp/nm_agg2/ea_s0_20261101T000000Z /tmp/nm_agg2/ea_s1_20261101T001000Z > "$OUT" 2>&1
check "seed 0/1 share a fingerprint and aggregate" "[ $? -eq 0 ]"
python3 tools/aggregate_seeds.py /tmp/nm_agg2/ea_s0_20261101T000000Z /tmp/nm_agg2/ea_s2_20261101T003000Z > "$OUT" 2>&1; rc=$?
check "different eval protocol refused" "[ $rc -eq 3 ] && grep -q 'eval_protocol_fingerprint' $OUT"
python3 tools/aggregate_seeds.py --scan /tmp/nm_agg2 --series nope > "$OUT" 2>&1; rc=$?
check "empty series selection fails loudly" "[ $rc -ne 0 ]"

# two runs of the same seed are not two samples
mkdir -p /tmp/nm_agg2/ea_s0_20261101T004000Z
python3 - <<'PY'
import json, pathlib
m = json.load(open("/tmp/nm_agg2/ea_s0_20261101T000000Z/manifest.json"))
m["experiment_id"] = "ea_s0_20261101T004000Z"
pathlib.Path("/tmp/nm_agg2/ea_s0_20261101T004000Z/manifest.json").write_text(json.dumps(m))
PY
python3 tools/aggregate_seeds.py /tmp/nm_agg2/ea_s0_20261101T000000Z /tmp/nm_agg2/ea_s0_20261101T004000Z > "$OUT" 2>&1; rc=$?
check "duplicate seed refused (rc=4)" "[ $rc -eq 4 ] && grep -q 'not independent samples' $OUT"
python3 tools/aggregate_seeds.py --allow-duplicate-seeds /tmp/nm_agg2/ea_s0_20261101T000000Z /tmp/nm_agg2/ea_s0_20261101T004000Z > "$OUT" 2>&1
check "duplicate seed allowed only when explicit" "[ $? -eq 0 ]"

rm -rf "out/$RID"

echo "=== 12. --series is an exact series match, never a prefix (P1-1) ==="
# out/ can hold several experiment families at once (ea, ea_verify, exp-b2_run ...).
# Selecting by prefix mixes them into one mean; selecting by the parsed run id cannot.
rm -rf /tmp/nm_series && mkdir -p /tmp/nm_series
python3 - <<'PY'
import json, pathlib
base = json.load(open("/tmp/nm_integ/manifest.json"))   # PASS manifest built in section 10
def mk(rid, seed, ret, **kw):
    d = pathlib.Path("/tmp/nm_series") / rid
    d.mkdir()
    m = dict(base); m.update(experiment_id=rid, seed=seed, normalized_return=ret); m.update(kw)
    (d / "manifest.json").write_text(json.dumps(m))
mk("ea_s0_20261106T061000Z", 0, 0.8135)
mk("ea_s1_20261107T061000Z", 1, 0.8307)
mk("ea_verify_s0_20261107T085809Z", 0, 0.8307)
mk("exp-b2_run_s0_20261107T090000Z", 0, 0.79)
mk("exp-b2_run_s3_20261107T091500Z", 3, 0.80)
# run ids that do not follow <series>_s<seed>_<UTC timestamp>: must never be selected
for bad in ("ea_s0_garbage", "ea_s0_20261107T0858Z", "ea_s0_20261107T085809Z_extra",
            "ea_s0_20261107T085809"):
    mk(bad, 9, 0.10)   # seed 9 / 0.10 would visibly poison any mean they joined
PY
agg_mean() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["normalized_return_mean"])' "$1"; }
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series ea --out /tmp/nm_series/agg_ea.json > "$OUT" 2>&1; rc=$?
check "series ea selects ea_s0/ea_s1 only" "[ $rc -eq 0 ] && python3 -c 'import json;d=json.load(open(\"/tmp/nm_series/agg_ea.json\"));ids=sorted(r[\"run_id\"] for r in d[\"per_seed\"]);assert ids==[\"ea_s0_20261106T061000Z\",\"ea_s1_20261107T061000Z\"], ids'"
check "series ea mean is not polluted by ea_verify" "[ \"\$(agg_mean /tmp/nm_series/agg_ea.json)\" = \"0.8221\" ]"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series ea_verify --out /tmp/nm_series/agg_ev.json > "$OUT" 2>&1; rc=$?
check "series ea_verify selects the verify run only" "[ $rc -eq 0 ] && python3 -c 'import json;d=json.load(open(\"/tmp/nm_series/agg_ev.json\"));ids=[r[\"run_id\"] for r in d[\"per_seed\"]];assert ids==[\"ea_verify_s0_20261107T085809Z\"], ids'"
check "series ea_verify mean is its own" "[ \"\$(agg_mean /tmp/nm_series/agg_ev.json)\" = \"0.8307\" ]"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series exp-b2_run --out /tmp/nm_series/agg_b2.json > "$OUT" 2>&1; rc=$?
check "series with underscore/digits/hyphen matches" "[ $rc -eq 0 ] && python3 -c 'import json;d=json.load(open(\"/tmp/nm_series/agg_b2.json\"));ids=sorted(r[\"run_id\"] for r in d[\"per_seed\"]);assert ids==[\"exp-b2_run_s0_20261107T090000Z\",\"exp-b2_run_s3_20261107T091500Z\"], ids'"
check "series with underscore/digits/hyphen is not a prefix match" "python3 -c 'import json;d=json.load(open(\"/tmp/nm_series/agg_b2.json\"));assert not any(r[\"run_id\"].startswith(\"ea_s\") for r in d[\"per_seed\"]), d[\"per_seed\"]'"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series exp > "$OUT" 2>&1; rc=$?
check "series exp does not prefix-match exp-b2_run" "[ $rc -ne 0 ] && grep -q 'no run matches' $OUT"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series exp-b2 > "$OUT" 2>&1; rc=$?
check "series exp-b2 does not prefix-match exp-b2_run" "[ $rc -ne 0 ]"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series ea > "$OUT" 2>&1; rc=$?
check "invalid run ids are reported, not silently included" "grep -q \"SKIP: 'ea_s0_garbage'\" $OUT && grep -q \"SKIP: 'ea_s0_20261107T085809Z_extra'\" $OUT"
check "invalid run ids excluded from the aggregate" "python3 -c 'import json;d=json.load(open(\"/tmp/nm_series/agg_ea.json\"));ids=[r[\"run_id\"] for r in d[\"per_seed\"]];assert sorted(ids)==[\"ea_s0_20261106T061000Z\",\"ea_s1_20261107T061000Z\"], ids;assert len(d[\"excluded_run_ids\"])==4, d[\"excluded_run_ids\"];assert d[\"series\"]==\"ea\"'"
# a / in --series is not a series name
python3 tools/aggregate_seeds.py --scan /tmp/nm_series --series 'ea/x' > "$OUT" 2>&1; rc=$?
check "malformed --series rejected (rc=2)" "[ $rc -eq 2 ] && grep -q 'not a valid series name' $OUT"
# the numeric suffix must not leak: ea_s10 is series ea seed 10, ea_s1_s0 is series ea_s1
rm -rf /tmp/nm_series2 && mkdir -p /tmp/nm_series2
python3 - <<'PY'
import json, pathlib
base = json.load(open("/tmp/nm_integ/manifest.json"))
for rid, seed, ret in (("ea_s10_20261107T092000Z", 10, 0.70), ("ea_s1_s0_20261107T093000Z", 0, 0.90)):
    d = pathlib.Path("/tmp/nm_series2") / rid; d.mkdir()
    m = dict(base); m.update(experiment_id=rid, seed=seed, normalized_return=ret)
    (d / "manifest.json").write_text(json.dumps(m))
PY
python3 tools/aggregate_seeds.py --scan /tmp/nm_series2 --series ea --out /tmp/nm_series2/agg.json > "$OUT" 2>&1
check "ea_s10 belongs to series ea, not ea_s1" "[ $? -eq 0 ] && grep -q 'ea_s10_20261107T092000Z' $OUT && ! grep -q 'ea_s1_s0' $OUT"
python3 tools/aggregate_seeds.py --scan /tmp/nm_series2 --series ea_s1 --out /tmp/nm_series2/agg2.json > "$OUT" 2>&1
check "ea_s1 selects ea_s1_s0 but not ea_s10" "[ $? -eq 0 ] && grep -q 'ea_s1_s0_20261107T093000Z' $OUT && ! grep -q 'ea_s10_20261107T092000Z' $OUT"
# a manifest whose embedded seed disagrees with the run id is not selectable
rm -rf /tmp/nm_series3 && mkdir -p /tmp/nm_series3/ea_s0_20261107T094000Z
python3 - <<'PY'
import json, pathlib
m = json.load(open("/tmp/nm_integ/manifest.json"))
m.update(experiment_id="ea_s0_20261107T094000Z", seed=3, normalized_return=0.11)
pathlib.Path("/tmp/nm_series3/ea_s0_20261107T094000Z/manifest.json").write_text(json.dumps(m))
PY
python3 tools/aggregate_seeds.py --scan /tmp/nm_series3 --series ea > "$OUT" 2>&1; rc=$?
check "run-id/manifest seed mismatch excluded" "[ $rc -ne 0 ] && grep -q 'SKIP' $OUT && grep -q 'manifest seed is 3' $OUT"
# the real archives: ea and ea_verify must not contaminate each other. Scanned from a copy of
# the committed manifests, so a developer's uncommitted run of the same series (which is real
# data, just not part of this tree) cannot turn a code check into a data check.
rm -rf /tmp/nm_real && mkdir -p /tmp/nm_real
for r in ea_s0_20261106T061000Z ea_s0_20261106T142202Z ea_verify_s0_20261107T085809Z; do
  mkdir -p "/tmp/nm_real/$r" && cp "out/$r/manifest.json" "/tmp/nm_real/$r/"
done
python3 tools/aggregate_seeds.py --scan /tmp/nm_real --series ea --allow-duplicate-seeds --out /tmp/nm_real_ea.json > "$OUT" 2>&1; rc=$?
check "real archives: --series ea stays at 0.8238" "[ $rc -eq 0 ] && [ \"\$(agg_mean /tmp/nm_real_ea.json)\" = \"0.8238\" ]"
check "real archives: ea aggregate lists exactly the two Experiment A runs" "python3 -c 'import json;d=json.load(open(\"/tmp/nm_real_ea.json\"));ids=sorted(r[\"run_id\"] for r in d[\"per_seed\"]);assert ids==[\"ea_s0_20261106T061000Z\",\"ea_s0_20261106T142202Z\"], ids'"
python3 tools/aggregate_seeds.py --scan /tmp/nm_real --series ea_verify --out /tmp/nm_real_ev.json > "$OUT" 2>&1; rc=$?
check "real archives: --series ea_verify stays at 0.8307" "[ $rc -eq 0 ] && [ \"\$(agg_mean /tmp/nm_real_ev.json)\" = \"0.8307\" ]"

echo "=== 13. collect: per-file retry + source-sha256 verification (P1-2) ==="
export NM_FAKE_VM=/tmp/nm_fakevm
RID2=mock_s1_20261107T000000Z
VMRUN="$NM_FAKE_VM/nabla/runs/$RID2"
OUT2="out/$RID2"
vm_build() {  # a fake VM run directory: 10 artifacts + 3 checkpoints (unpadded, unsorted)
  rm -rf "$NM_FAKE_VM"
  mkdir -p "$VMRUN/checkpoints" "$VMRUN/evaluation"
  printf '{"experiment_id":"%s","status":"PASS","drive_path":"/content/drive/MyDrive/nm/%s"}\n' \
    "$RID2" "$RID2" > "$VMRUN/manifest.json"
  printf '{"argv":["--config","configs/ours.args","--seed","0"],"seed":0}\n' > "$VMRUN/config.json"
  printf '{"iter":250,"wall_s":90.0}\n' > "$VMRUN/log.jsonl"
  echo '{"mode":"live"}' > "$VMRUN/provenance.json"
  echo '{"status":"PASS"}' > "$VMRUN/summary.json"
  printf 'iter,return\n250,0.4\n500,0.7\n1000,0.83\n' > "$VMRUN/curve.csv"
  echo '# mock report' > "$VMRUN/report.md"
  echo 'mock training log' > "$VMRUN/train.log"
  echo '{}' > "$VMRUN/evaluation/eval_nominal.json"
  printf 'clip,fall\nx,\n' > "$VMRUN/evaluation/per_clip.csv"
  echo '{}' > "$VMRUN/evaluation/summary.json"
  head -c 8192 /dev/urandom > "$VMRUN/checkpoints/it_250.pkl"
  head -c 8192 /dev/urandom > "$VMRUN/checkpoints/it_500.pkl"
  head -c 8192 /dev/urandom > "$VMRUN/checkpoints/it_1000.pkl"
}
fault_reset() {
  rm -rf "$OUT2"
  rm -f /tmp/nm_mock_state/dl_fail_* /tmp/nm_mock_state/dl_truncate_* \
        /tmp/nm_mock_state/dl_corrupt_* /tmp/nm_mock_state/dl_unreadable_* \
        /tmp/nm_mock_state/snapshot_fail \
        /tmp/nm_mock_state/snapshot_missing /tmp/nm_mock_state/status_fail \
        /tmp/nm_mock_state/runid_fail /tmp/nm_mock_state/dl_calls.log \
        /tmp/nm_mock_state/dl_signal_* /tmp/nm_mock_state/dl_killed_by \
        /tmp/nm_mock_state/identity \
        /tmp/nm_mock_state/status_probe_fail /tmp/nm_mock_state/status_probe_out
  echo PERSISTED > /tmp/nm_mock_state/pipeline_state
  echo "$RID2" > /tmp/nm_mock_state/run_id
}
collect_try() {  # collect_try MODE [extra env assignments...]
  local mode="$1"; shift
  env PATH="$MOCK:$PATH" RUN_ID="$RID2" COLLECT_CKPT="$mode" "$@" ./run.sh collect
}
no_receipt() { [ ! -e "$OUT2/collect_manifest.txt" ]; }

fault_reset; vm_build
collect_try latest > "$OUT" 2>&1; rc=$?
check "collect(latest) exits 0" "[ $rc -eq 0 ]"
check "collect(latest) writes a verified receipt" "grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "collect(latest) records every VM checkpoint as expected" "grep -qx 'ckpt_expected=3' $OUT2/collect_manifest.txt"
check "collect(latest) verified exactly one checkpoint" "grep -qx 'ckpt_local=1' $OUT2/collect_manifest.txt"
check "collect(latest) counts verified files, not attempts" "grep -qE '^files_verified=[0-9]+' $OUT2/collect_manifest.txt"
check "collect(latest) picks it_1000 (numeric, not lexicographic)" "[ -f $OUT2/checkpoints/it_1000.pkl ] && [ ! -e $OUT2/checkpoints/it_250.pkl ] && [ ! -e $OUT2/checkpoints/it_500.pkl ]"
check "collect(latest) downloaded every artifact" "[ -s $OUT2/evaluation/per_clip.csv ] && [ -s $OUT2/train.log ] && [ -s $OUT2/report.md ]"
check "collect leaves no .part debris" "! find $OUT2 -name '*.part*' | grep -q ."

fault_reset; vm_build
collect_try all > "$OUT" 2>&1; rc=$?
check "collect(all) exits 0" "[ $rc -eq 0 ]"
check "collect(all) verifies all 3 checkpoints" "grep -qx 'ckpt_local=3' $OUT2/collect_manifest.txt"
check "collect(all) has the complete weight set" "[ -f $OUT2/checkpoints/it_250.pkl ] && [ -f $OUT2/checkpoints/it_500.pkl ] && [ -f $OUT2/checkpoints/it_1000.pkl ]"

fault_reset; vm_build
collect_try none > "$OUT" 2>&1; rc=$?
check "collect(none) exits 0 without weights" "[ $rc -eq 0 ] && grep -qx 'ckpt_local=0' $OUT2/collect_manifest.txt && grep -qx 'ckpt_expected=3' $OUT2/collect_manifest.txt"
check "collect(none) downloads no checkpoint" "! find $OUT2/checkpoints -name '*.pkl' 2>/dev/null | grep -q ."
check "collect(none) still verifies the metadata" "[ -s $OUT2/manifest.json ] && [ -s $OUT2/log.jsonl ]"

# a dropped connection on the first attempt must be retried, not archived
fault_reset; vm_build; echo 1 > /tmp/nm_mock_state/dl_fail_manifest.json
collect_try all > "$OUT" 2>&1; rc=$?
check "first-attempt failure is retried and succeeds" "[ $rc -eq 0 ] && grep -q 'retry 1/2 for manifest.json' $OUT"
check "retried file is verified before publishing" "grep -qx 'verify=ok' $OUT2/collect_manifest.txt && [ -s $OUT2/manifest.json ]"

# three failures -> non-zero, and nothing that looks like success
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_log.jsonl
collect_try all > "$OUT" 2>&1; rc=$?
check "3 failed attempts make collect exit non-zero" "[ $rc -ne 0 ]"
check "3 failed attempts write no success receipt" "no_receipt"
check "3 failed attempts record a failed receipt instead" "grep -qx 'verify=FAILED' $OUT2/collect_manifest.txt.failed"
check "3 failed attempts never publish the file" "[ ! -e $OUT2/log.jsonl ]"
check "3 failed attempts report the file" "grep -q 'log.jsonl' $OUT"

# same size, different bytes: only the checksum can catch it
fault_reset; vm_build; touch /tmp/nm_mock_state/dl_corrupt_summary.json
collect_try all > "$OUT" 2>&1; rc=$?
check "sha256 mismatch fails the collection" "[ $rc -ne 0 ] && no_receipt"
check "sha256 mismatch is reported as a checksum error" "grep -q 'sha256' $OUT"
check "corrupted file is not published" "[ ! -e $OUT2/summary.json ]"

# truncated transfer
fault_reset; vm_build; touch /tmp/nm_mock_state/dl_truncate_log.jsonl
collect_try all > "$OUT" 2>&1; rc=$?
check "truncated download fails the collection" "[ $rc -ne 0 ] && no_receipt"
check "truncated transfer is reported as a size mismatch" "grep -q 'size .* != source' $OUT"
check "truncated file is not published" "[ ! -e $OUT2/log.jsonl ]"

# --- local filesystem failures are part of the fail-closed contract (review P1-3) ---
# A disk-side error used to escape as a traceback: fail() never ran, so the previous success
# receipt survived a FORCE=1 re-collect and kept claiming the archive was verified.

# (a) the downloaded file cannot be read back locally -> local i/o error, retried, then failed
fault_reset; vm_build
touch /tmp/nm_mock_state/dl_unreadable_summary.json
collect_try all > "$OUT" 2>&1; rc=$?
check "an unreadable download is a local i/o error" "[ $rc -ne 0 ] && grep -qi 'local i/o error' $OUT"
check "an unreadable download is retried like any other attempt" "grep -q 'retry 1/2 for summary.json' $OUT"
check "an unreadable download is not published" "no_receipt && [ ! -e $OUT2/summary.json ]"
check "an unreadable download never produces a traceback" "! grep -q 'Traceback' $OUT"

# (b) a verified download that cannot be published (destination is a non-empty directory)
fault_reset; vm_build
collect_try all > "$OUT" 2>&1                       # a good archive with a verified receipt
check "(setup) the archive starts out verified" "grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
rm -rf "$OUT2/checkpoints/it_1000.pkl"; mkdir -p "$OUT2/checkpoints/it_1000.pkl/blocker"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "an unpublishable file fails the re-collect" "[ $rc -ne 0 ]"
check "an unpublishable file revokes the previous success receipt" "no_receipt"
check "an unpublishable file is reported as a local i/o error" "grep -qi 'local i/o error' $OUT"
check "the other verified files survive the failed re-collect" "[ -s $OUT2/checkpoints/it_500.pkl ] && [ -s $OUT2/checkpoints/it_250.pkl ]"
check "the failed re-collect leaves a failed marker" "grep -qx 'verify=FAILED' $OUT2/collect_manifest.txt.failed"
rm -rf "$OUT2/checkpoints/it_1000.pkl"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "clearing the blocker lets the archive become verified again" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"

# (c) an OSError that escapes the per-file loop (a directory that cannot be created)
fault_reset; vm_build
collect_try all > "$OUT" 2>&1
rm -rf "$OUT2/evaluation"; : > "$OUT2/evaluation"   # a regular file where a dir must be
env PATH="$MOCK:$PATH" python3 tools/collect_run.py --session mock --run-id "$RID2" \
  --out "$OUT2" --ckpt-mode none --state PERSISTED > "$OUT" 2>&1; rc=$?
check "an escaping local i/o error exits 5 (distinct from 3/4)" "[ $rc -eq 5 ]"
check "an escaping local i/o error revokes the stale receipt" "no_receipt"
check "an escaping local i/o error is named, not traced back" "grep -q 'local filesystem error' $OUT && ! grep -q 'Traceback' $OUT"
check "an escaping local i/o error leaves the failed marker" "grep -qx 'verify=FAILED' $OUT2/collect_manifest.txt.failed"
rm -f "$OUT2/evaluation"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "the archive recovers once the filesystem problem is gone" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt && [ ! -e $OUT2/collect_manifest.txt.failed ]"

# the VM cannot be asked for hashes -> fail closed, no guessing
fault_reset; vm_build; touch /tmp/nm_mock_state/snapshot_fail
collect_try all > "$OUT" 2>&1; rc=$?
check "snapshot query failure aborts the collection" "[ $rc -ne 0 ] && no_receipt"
check "snapshot failure is named in the error" "grep -q 'source snapshot' $OUT"

# the pipeline state cannot be read -> fail closed before transferring anything
fault_reset; vm_build; touch /tmp/nm_mock_state/status_fail
collect_try all > "$OUT" 2>&1; rc=$?
check "unreadable pipeline state aborts the collection" "[ $rc -ne 0 ] && no_receipt && grep -q 'pipeline state' $OUT"
fault_reset; vm_build; touch /tmp/nm_mock_state/runid_fail
env PATH="$MOCK:$PATH" COLLECT_CKPT=all ./run.sh collect > "$OUT" 2>&1; rc=$?
check "unreadable active-run id aborts the collection" "[ $rc -ne 0 ] && grep -q 'cannot query the VM' $OUT"

# RUNNING pipeline: results are still moving
fault_reset; vm_build; echo RUNNING > /tmp/nm_mock_state/pipeline_state
collect_try all > "$OUT" 2>&1; rc=$?
check "a RUNNING pipeline is refused" "[ $rc -ne 0 ] && no_receipt && grep -q 'still RUNNING' $OUT"

# all-mode partial: 2 of 3 checkpoints is not a success
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_it_500.pkl
collect_try all > "$OUT" 2>&1; rc=$?
check "all-mode with a failing checkpoint fails" "[ $rc -ne 0 ] && no_receipt"
check "all-mode partial download is reported as partial" "grep -q 'partial archive' $OUT"
check "all-mode partial download is visible on disk" "[ -f $OUT2/checkpoints/it_250.pkl ] && [ ! -e $OUT2/checkpoints/it_500.pkl ]"

# latest-mode must not fall back to an older weight when the newest fails
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_it_1000.pkl
collect_try latest > "$OUT" 2>&1; rc=$?
check "latest-mode fails when the newest checkpoint fails" "[ $rc -ne 0 ] && no_receipt"
check "latest-mode does not silently fall back to an older checkpoint" "[ ! -e $OUT2/checkpoints/it_500.pkl ] && [ ! -e $OUT2/checkpoints/it_250.pkl ]"

# a PERSISTED run missing a report artifact is not archivable
fault_reset; vm_build; rm -f "$VMRUN/report.md"
collect_try all > "$OUT" 2>&1; rc=$?
check "PERSISTED run missing report.md is refused" "[ $rc -ne 0 ] && no_receipt && grep -q 'report.md' $OUT"
# ...but an older UNPERSISTED run may legitimately lack report artifacts
fault_reset; vm_build; rm -f "$VMRUN/report.md"; echo INCOMPLETE > /tmp/nm_mock_state/pipeline_state
collect_try all > "$OUT" 2>&1; rc=$?
check "INCOMPLETE run may lack report artifacts" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"

# a failed re-collect must not damage verified bytes from the previous run. The local copy of
# manifest.json is removed first: a resumable collect would otherwise skip a file that is
# already verified, and the injected download fault would never be reached.
fault_reset; vm_build
collect_try all > "$OUT" 2>&1
rm -f "$OUT2/manifest.json"
( cd "$OUT2" && find . -type f ! -name 'collect_manifest.txt*' ! -name 'collect_diagnostics.txt' \
    ! -name '.collect_partial.json' | sort | xargs shasum -a 256 ) > /tmp/nm_keep.sha 2>/dev/null
echo 9 > /tmp/nm_mock_state/dl_fail_manifest.json
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "failed re-collect exits non-zero" "[ $rc -ne 0 ]"
check "failed re-collect leaves no success receipt" "no_receipt"
check "failed re-collect leaves the failed marker" "[ -s $OUT2/collect_manifest.txt.failed ]"
if ( cd "$OUT2" && shasum -a 256 -c /tmp/nm_keep.sha ) > /tmp/nm_keep.log 2>&1; then ok "verified files survive a failed re-collect"; else bad "verified files survive a failed re-collect"; fi
# ...and a re-collect that succeeds repairs the archive
rm -f /tmp/nm_mock_state/dl_fail_manifest.json
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "a later successful re-collect restores the receipt" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt && [ ! -e $OUT2/collect_manifest.txt.failed ]"

# no checkpoints on the VM at all
fault_reset; vm_build; rm -rf "$VMRUN/checkpoints"
collect_try none > "$OUT" 2>&1; rc=$?
check "logs-only collection works without any checkpoint" "[ $rc -eq 0 ] && grep -qx 'ckpt_expected=0' $OUT2/collect_manifest.txt"
rm -rf "$OUT2"
collect_try all > "$OUT" 2>&1; rc=$?
check "all-mode without any checkpoint fails closed" "[ $rc -ne 0 ] && no_receipt && grep -q 'no checkpoints' $OUT"

# COLLECT_CKPT validation
env COLLECT_CKPT=sometimes ./run.sh collect > "$OUT" 2>&1; rc=$?
check "invalid COLLECT_CKPT rejected" "[ $rc -ne 0 ] && grep -q 'COLLECT_CKPT must be' $OUT"
# the run id is interpolated into the remote program: reject anything that is not one
env PATH="$MOCK:$PATH" RUN_ID='bad id; rm -rf /' ./run.sh collect > "$OUT" 2>&1; rc=$?
check "unsafe RUN_ID rejected before it reaches the VM" "[ $rc -ne 0 ] && grep -q 'run-id must match' $OUT"

# The new receipt must still feed the (unchanged) down gate: a complete `all` archive
# authorises shutdown, a `latest` one does not pretend to be complete.
persist_snapshot() {  # PERSISTENCE snapshot built from the fake VM (the source of truth)
  { echo PERSISTENCE_BEGIN
    for rel in manifest.json config.json log.jsonl summary.json curve.csv report.md train.log \
               provenance.json evaluation/eval_nominal.json evaluation/per_clip.csv \
               evaluation/summary.json; do
      [ -f "$VMRUN/$rel" ] || continue
      printf 'FILE %s %s %s\n' "$rel" "$(wc -c < "$VMRUN/$rel" | tr -d ' ')" \
        "$(shasum -a 256 "$VMRUN/$rel" | cut -d' ' -f1)"
    done
    for f in "$VMRUN"/checkpoints/it_*.pkl; do
      [ -f "$f" ] || continue
      printf 'FILE checkpoints/%s %s %s\n' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" \
        "$(shasum -a 256 "$f" | cut -d' ' -f1)"
    done
    echo 'PERSISTENCE_END 3'
  } > /tmp/nm_mock_state/remote_snapshot
}
fault_reset; vm_build
collect_try all > "$OUT" 2>&1
persist_snapshot; echo none > /tmp/nm_mock_state/drive_state
PATH="$MOCK:$PATH" RUN_ID="$RID2" ./run.sh down > "$OUT" 2>&1; rc=$?
check "a verified collect(all) satisfies the down gate" "[ $rc -eq 0 ] && grep -q 'local_complete=1' $OUT"
fault_reset; vm_build
collect_try latest > "$OUT" 2>&1
persist_snapshot; echo none > /tmp/nm_mock_state/drive_state
PATH="$MOCK:$PATH" RUN_ID="$RID2" ./run.sh down > "$OUT" 2>&1; rc=$?
check "collect(latest) is still not a complete local archive for down" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"
rm -f /tmp/nm_mock_state/drive_state /tmp/nm_mock_state/remote_snapshot

echo "=== 14. resumable collect (C-1), --source recovery (C-2), diagnostics (C-3) ==="
DRIVE_SRC="/content/drive/MyDrive/nm/$RID2"
DRIVE_DIR="$NM_FAKE_VM/drive/MyDrive/nm/$RID2"
DL_LOG=/tmp/nm_mock_state/dl_calls.log
dl_count() { wc -l < "$DL_LOG" 2>/dev/null | tr -d ' ' ; }
fsha() { shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1; }
n_verified() { sed -n 's/^files_verified=//p' "$OUT2/collect_manifest.txt" 2>/dev/null | head -1; }
drv_build() {  # a Drive copy of the same run, without the training VM's receipt
  rm -rf "$DRIVE_DIR"
  mkdir -p "$DRIVE_DIR"
  cp -R "$VMRUN/." "$DRIVE_DIR/"
  rm -f "$DRIVE_DIR/.sync_receipt"
}
receipt_build() {  # receipt_build STATE DST  == what vm/90_sync.sh writes on the training VM
  local state="$1" dst="$2" nck
  nck=$(ls "$DRIVE_DIR"/checkpoints/it_*.pkl 2>/dev/null | wc -l | tr -d ' ')
  {
    echo "verified_at=2026-11-07T15:00:00Z"
    echo "src=/content/nabla/runs/$RID2"
    echo "dst=$dst"
    for rel in manifest.json config.json log.jsonl summary.json curve.csv report.md \
               train.log provenance.json evaluation/eval_nominal.json \
               evaluation/per_clip.csv evaluation/summary.json; do
      [ -f "$DRIVE_DIR/$rel" ] || continue
      printf 'OK %s %s %s\n' "$rel" "$(wc -c < "$DRIVE_DIR/$rel" | tr -d ' ')" "$(fsha "$DRIVE_DIR/$rel")"
    done
    for f in "$DRIVE_DIR"/checkpoints/it_*.pkl; do
      [ -f "$f" ] || continue
      printf 'OK checkpoints/%s %s %s\n' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" "$(fsha "$f")"
    done
    echo "checkpoints_total=$nck"
    echo "checkpoints_verified=$nck"
    echo "verified=$state"
  } > "$DRIVE_DIR/.sync_receipt"
}
partial_ok() {  # the progress file is a collect-partial document and never says ok
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("kind") == "collect-partial", d.get("kind")
assert d.get("verify") == "partial", d.get("verify")
assert d.get("files"), "no per-file progress recorded"
print("partial ok")
PY
}
drive_case() { fault_reset; vm_build; drv_build; }
src_collect() {  # src_collect MODE [extra env assignments...]  -- recovery from DRIVE_SRC
  local mode="$1"; shift
  env PATH="$MOCK:$PATH" RUN_ID="$RID2" COLLECT_CKPT="$mode" SOURCE="$DRIVE_SRC" "$@" ./run.sh collect
}

# --- C-1: first collect, then a resume that asks the source for nothing ------
fault_reset; vm_build; : > "$DL_LOG"
collect_try all > "$OUT" 2>&1; rc=$?
NFILES=$(n_verified)
check "first collect exits 0 with a verified receipt" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "first collect downloads every planned file ($NFILES)" "[ \"\$(dl_count)\" -eq ${NFILES:-0} ]"
check "first collect reports no resumed files" "grep -qx 'files_resumed=0' $OUT2/collect_manifest.txt"
check "a finished collect leaves no partial manifest behind" "[ ! -e $OUT2/.collect_partial.json ]"
: > "$DL_LOG"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "second run exits 0" "[ $rc -eq 0 ]"
check "second run downloads nothing at all" "[ \"\$(dl_count)\" -eq 0 ]"
check "second run records files_downloaded=0" "grep -qx 'files_downloaded=0' $OUT2/collect_manifest.txt"
check "second run counts every file as resumed" "grep -qx \"files_resumed=$NFILES\" $OUT2/collect_manifest.txt"
check "the receipt still states verify=ok exactly once" "[ \"\$(grep -c '^verify=ok$' $OUT2/collect_manifest.txt)\" -eq 1 ]"

# --- C-1: a damaged local file is re-downloaded, the rest is kept -----------
fault_reset; vm_build; collect_try all > "$OUT" 2>&1
printf 'X' | dd of="$OUT2/checkpoints/it_500.pkl" bs=1 seek=16 conv=notrunc 2>/dev/null
: > "$DL_LOG"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "a corrupted local file is re-downloaded" "[ $rc -eq 0 ] && [ \"\$(dl_count)\" -eq 1 ] && grep -q 'it_500.pkl$' $DL_LOG"
check "the corrupted file is repaired byte for byte" "[ \"\$(fsha $OUT2/checkpoints/it_500.pkl)\" = \"\$(fsha $VMRUN/checkpoints/it_500.pkl)\" ]"
check "only the damaged file was fetched" "grep -qx 'files_downloaded=1' $OUT2/collect_manifest.txt"

# --- C-1: a changed source hash invalidates the local copy ------------------
fault_reset; vm_build; collect_try all > "$OUT" 2>&1
head -c 8192 /dev/urandom > "$VMRUN/checkpoints/it_250.pkl"
: > "$DL_LOG"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "a changed source sha256 forces a re-download" "[ $rc -eq 0 ] && [ \"\$(dl_count)\" -eq 1 ] && grep -q 'it_250.pkl$' $DL_LOG"
check "the archive follows the new source hash" "[ \"\$(fsha $OUT2/checkpoints/it_250.pkl)\" = \"\$(fsha $VMRUN/checkpoints/it_250.pkl)\" ]"
check "the untouched files were not re-fetched" "grep -qx 'files_downloaded=1' $OUT2/collect_manifest.txt"

# --- C-1: an interrupted collect, and a resume that continues it ------------
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_it_500.pkl
collect_try all > "$OUT" 2>&1; rc=$?
check "an interrupted collect fails closed" "[ $rc -ne 0 ] && no_receipt"
check "the interruption leaves per-file progress" "[ -s $OUT2/.collect_partial.json ]"
check "the progress file is a collect-partial document" "partial_ok $OUT2/.collect_partial.json"
check "the progress file never claims verify=ok" "! grep -q 'verify=ok' $OUT2/.collect_partial.json"
check "the interrupted archive keeps what it verified" "[ -s $OUT2/checkpoints/it_250.pkl ] && [ ! -e $OUT2/checkpoints/it_500.pkl ]"
rm -f /tmp/nm_mock_state/dl_fail_it_500.pkl; : > "$DL_LOG"
collect_try all > "$OUT" 2>&1; rc=$?
check "a resume re-enters the directory without FORCE" "[ $rc -eq 0 ]"
check "the resume produces the success receipt" "grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "the resume fetches only what is missing" "[ \"\$(dl_count)\" -eq 1 ] && grep -q 'it_500.pkl$' $DL_LOG"
check "the resume reports the files it skipped" "grep -qE '^files_resumed=[1-9]' $OUT2/collect_manifest.txt"
check "a finished archive drops the progress file" "[ ! -e $OUT2/.collect_partial.json ]"

# --- C-1: a resume that fails again stays fail-closed -----------------------
# The first attempt stops on it_500 (and has already fetched the later checkpoints), so the
# second one is given a different missing file to fail on: the resume repairs it_500 and then
# dies on it_1000, which is exactly the "one step forward, one step back" case.
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_it_500.pkl
collect_try all > "$OUT" 2>&1
rm -f /tmp/nm_mock_state/dl_fail_it_500.pkl
rm -f "$OUT2/checkpoints/it_1000.pkl"
echo 9 > /tmp/nm_mock_state/dl_fail_it_1000.pkl
collect_try all > "$OUT" 2>&1; rc=$?
check "a resume that fails again exits non-zero" "[ $rc -ne 0 ]"
check "a resume that fails again leaves no success receipt" "[ ! -e $OUT2/collect_manifest.txt ]"
check "a resume that fails again records the failure" "grep -qx 'verify=FAILED' $OUT2/collect_manifest.txt.failed"
check "a resume that fails again keeps every verified byte" "[ -s $OUT2/checkpoints/it_250.pkl ] && [ -s $OUT2/checkpoints/it_500.pkl ] && [ ! -e $OUT2/checkpoints/it_1000.pkl ]"
rm -f /tmp/nm_mock_state/dl_fail_it_1000.pkl; : > "$DL_LOG"
collect_try all > "$OUT" 2>&1; rc=$?
check "the next resume finishes the archive" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "the next resume fetches only the last missing file" "[ \"\$(dl_count)\" -eq 1 ] && grep -q 'it_1000.pkl$' $DL_LOG"

# --- C-1: a failing re-collect revokes the previous success receipt ---------
fault_reset; vm_build; collect_try all > "$OUT" 2>&1
check "(setup) the archive starts out verified" "grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
rm -f "$OUT2/checkpoints/it_1000.pkl"
echo 9 > /tmp/nm_mock_state/dl_fail_it_1000.pkl
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "a failing re-collect revokes the old success receipt" "[ $rc -ne 0 ] && no_receipt"
check "the revoked archive keeps the surviving files" "[ -s $OUT2/checkpoints/it_250.pkl ] && [ -s $OUT2/checkpoints/it_500.pkl ]"
check "the revoked archive leaves the failed marker" "grep -qx 'verify=FAILED' $OUT2/collect_manifest.txt.failed"
rm -f /tmp/nm_mock_state/dl_fail_it_1000.pkl
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "the archive becomes verified again once the transfer works" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt && [ ! -e $OUT2/collect_manifest.txt.failed ]"

# --- C-1: a broken progress file is ignored, never trusted ------------------
fault_reset; vm_build; echo 9 > /tmp/nm_mock_state/dl_fail_it_500.pkl
collect_try all > "$OUT" 2>&1
printf '{ this is not json at all' > "$OUT2/.collect_partial.json"
rm -f /tmp/nm_mock_state/dl_fail_it_500.pkl; : > "$DL_LOG"
collect_try all > "$OUT" 2>&1; rc=$?
check "a corrupt progress file does not block the resume" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "a corrupt progress file is reported as ignored" "grep -q 'ignoring an unreadable partial manifest' $OUT"
check "a corrupt progress file licenses no skip" "[ \"\$(dl_count)\" -eq 1 ] && grep -q 'it_500.pkl$' $DL_LOG"

# --- C-1: a progress file that *claims* verify=ok is still not evidence -----
fault_reset; vm_build; collect_try all > "$OUT" 2>&1
printf 'X' | dd of="$OUT2/summary.json" bs=1 seek=0 conv=notrunc 2>/dev/null
python3 - "$OUT2" "$RID2" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
files = {}
for p in sorted(out.rglob("*")):
    if p.is_file() and p.name != "collect_manifest.txt":
        files[p.relative_to(out).as_posix()] = {
            "size": p.stat().st_size, "sha256": "0" * 64, "origin": "downloaded"}
(out / ".collect_partial.json").write_text(json.dumps({
    "schema": 1, "kind": "collect-partial", "verify": "ok", "run_id": sys.argv[2],
    "files": files, "updated_at": "2026-11-07T00:00:00Z"}))
PY
: > "$DL_LOG"
collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
check "a progress file claiming verify=ok is not believed" "[ $rc -eq 0 ] && [ \"\$(dl_count)\" -eq 1 ] && grep -q 'summary.json$' $DL_LOG"
check "the repaired file matches the source, not the progress file" "[ \"\$(fsha $OUT2/summary.json)\" = \"\$(fsha $VMRUN/summary.json)\" ]"

# --- C-3: pre-flight diagnostics -------------------------------------------
fault_reset; vm_build; collect_try all > "$OUT" 2>&1; rc=$?
DG="$OUT2/collect_diagnostics.txt"
check "diagnostics are written next to the archive" "[ $rc -eq 0 ] && [ -s $DG ]"
check "diagnostics record a UTC start timestamp" "grep -qE '^started_at_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' $DG"
check "diagnostics record the run id and the source path" "grep -qx 'run_id=$RID2' $DG && grep -qx 'source_path=/content/nabla/runs/$RID2' $DG"
check "diagnostics record the data source" "grep -qx 'data_source=training-vm' $DG"
check "diagnostics record the colab status output" "grep -qx 'colab_status=ok' $DG && grep -qx 'colab_status_line=Session: mock-session-42' $DG"
check "diagnostics record the colab session id" "grep -qx 'colab_session_id=mock-session-42' $DG"
check "diagnostics record how the collection ended" "grep -qx 'outcome=verified' $DG"
fault_reset; vm_build; touch /tmp/nm_mock_state/status_probe_fail
collect_try all > "$OUT" 2>&1; rc=$?
check "a failed status probe does not block a verified archive" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "a failed status probe is recorded as a failure" "grep -qx 'colab_status=FAILED' $OUT2/collect_diagnostics.txt && grep -qx 'colab_status_rc=1' $OUT2/collect_diagnostics.txt"
check "a failed status probe never claims success" "! grep -qx 'colab_status=ok' $OUT2/collect_diagnostics.txt"
check "the status error itself is kept" "grep -qx 'colab_status_stderr=mock: session not found' $OUT2/collect_diagnostics.txt"
rm -f /tmp/nm_mock_state/status_probe_fail

# --- C-2: recovery from a mounted Drive copy, with the VM's own evidence -----
drive_case; receipt_build ok "$DRIVE_SRC"; echo NO_RUN > /tmp/nm_mock_state/pipeline_state
: > "$DL_LOG"
src_collect all > "$OUT" 2>&1; rc=$?
check "recovery from a mounted Drive copy succeeds" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "recovery records the data source and path" "grep -qx 'data_source=drive-recovery' $OUT2/collect_manifest.txt && grep -qx 'source_path=$DRIVE_SRC' $OUT2/collect_manifest.txt"
check "every byte came from the Drive copy, not the VM run dir" "[ \"\$(dl_count)\" -eq ${NFILES:-0} ] && grep -q 'drive/MyDrive' $DL_LOG && ! grep -q 'nabla/runs' $DL_LOG"
check "a VM sync receipt gives full provenance" "grep -qx 'provenance=drive-recovery-verified' $OUT2/collect_manifest.txt && grep -qx 'provenance_degraded=0' $OUT2/collect_manifest.txt"
check "the receipt vouches for it with its verified_at and checkpoint counts" "grep -qx 'provenance_receipt_verified=ok' $OUT2/collect_manifest.txt && grep -qx 'provenance_evidence=vm-sync-receipt' $OUT2/collect_manifest.txt"
check "the pipeline state is derived from the receipt" "grep -qx 'pipeline_state=PERSISTED' $OUT2/collect_manifest.txt && grep -qx 'state_source=sync receipt' $OUT2/collect_manifest.txt"
check "recovery needs no symlink: the VM run dir is never read" "! grep -q \"nabla/runs/$RID2\" $DL_LOG"

# --- C-2: Drive recovery without source evidence degrades honestly ----------
drive_case; echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "recovery without a receipt still verifies the bytes" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "missing source evidence degrades provenance" "grep -qx 'provenance=drive-copy-only' $OUT2/collect_manifest.txt && grep -qx 'provenance_degraded=1' $OUT2/collect_manifest.txt"
check "degradation is stated loudly, not hidden" "grep -q 'NOT evidence that' $OUT"
check "degradation is recorded in the diagnostics" "grep -qx 'provenance_degraded=1' $OUT2/collect_diagnostics.txt"
# a receipt that does not cover a file is not evidence for that file
drive_case; receipt_build ok "$DRIVE_SRC"
python3 - "$DRIVE_DIR/.sync_receipt" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
out = []
for line in p.read_text().splitlines():
    if line.startswith("OK summary.json "):
        line = "OK summary.json 1 " + "0" * 64
    out.append(line)
p.write_text("\n".join(out) + "\n")
PY
echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "a receipt that contradicts a file degrades provenance" "[ $rc -eq 0 ] && grep -qx 'provenance=drive-copy-only' $OUT2/collect_manifest.txt"
check "the contradicted file is named" "grep -q 'different size/sha256' $OUT"
# a receipt for a different directory is not evidence for this one
drive_case; receipt_build ok "/content/drive/MyDrive/nm/some_other_run"
echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "a receipt written for another directory degrades provenance" "[ $rc -eq 0 ] && grep -qx 'provenance=drive-copy-only' $OUT2/collect_manifest.txt"
check "the mismatched receipt dst is named" "grep -q 'was written for dst=' $OUT"

# --- C-2: invalid sources and a mismatched run identity --------------------
fault_reset; vm_build
env PATH="$MOCK:$PATH" RUN_ID="$RID2" SOURCE="relative/path/$RID2" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "a relative SOURCE is rejected" "[ $rc -ne 0 ] && grep -q 'must be an absolute path' $OUT"
env PATH="$MOCK:$PATH" RUN_ID="$RID2" SOURCE="/content/drive/../etc/$RID2" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "a SOURCE with .. is rejected" "[ $rc -ne 0 ] && grep -q \"must not contain '..'\" $OUT"
env PATH="$MOCK:$PATH" RUN_ID="$RID2" SOURCE="/content/drive/./nm/$RID2" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "a non-canonical SOURCE is rejected" "[ $rc -ne 0 ] && grep -q 'not canonical' $OUT"
env PATH="$MOCK:$PATH" RUN_ID="$RID2" SOURCE="/content/drive/x; rm -rf /$RID2" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "a SOURCE carrying shell metacharacters is rejected" "[ $rc -ne 0 ] && grep -q 'only contain' $OUT"
env PATH="$MOCK:$PATH" RUN_ID="$RID2" SOURCE="/content/drive/MyDrive/nm/other_s1_20260101T000000Z" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "a SOURCE naming another run is rejected" "[ $rc -ne 0 ] && grep -q 'must end in the run id' $OUT"
check "a rejected SOURCE writes no receipt" "no_receipt"
env PATH="$MOCK:$PATH" SOURCE="$DRIVE_SRC" ./run.sh collect > "$OUT" 2>&1; rc=$?
check "SOURCE without RUN_ID is rejected" "[ $rc -ne 0 ] && grep -q 'SOURCE needs RUN_ID' $OUT"
drive_case; receipt_build ok "$DRIVE_SRC"
echo "some_other_run_s0_20260101T000000Z" > /tmp/nm_mock_state/identity
echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "a source whose manifest names another run is refused" "[ $rc -ne 0 ] && grep -q 'run identity mismatch' $OUT"
check "an identity mismatch writes no receipt" "no_receipt"
rm -f /tmp/nm_mock_state/identity

# --- C-2: the old CLI keeps working unchanged ------------------------------
fault_reset; vm_build; : > "$DL_LOG"
env PATH="$MOCK:$PATH" python3 tools/collect_run.py --session mock --run-id "$RID2" \
  --out "$OUT2" --ckpt-mode all --state PERSISTED --run ea --seed 1 --gpu G4 \
  --budget-minutes 8 > "$OUT" 2>&1; rc=$?
check "the old CLI invocation (no --source) still works" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "the old CLI keeps the original VM source path" "grep -qx 'source_path=/content/nabla/runs/$RID2' $OUT2/collect_manifest.txt && grep -qx 'data_source=training-vm' $OUT2/collect_manifest.txt"
check "the old CLI reports training-VM provenance" "grep -qx 'provenance=training-vm' $OUT2/collect_manifest.txt && grep -qx 'provenance_degraded=0' $OUT2/collect_manifest.txt"
check "the old CLI still fills the historical receipt fields" "grep -qx 'ckpt_expected=3' $OUT2/collect_manifest.txt && grep -qx 'ckpt_local=3' $OUT2/collect_manifest.txt && grep -qx 'integrity=source-sha256-snapshot' $OUT2/collect_manifest.txt"
check "--source is documented in --help" "python3 tools/collect_run.py --help 2>&1 | grep -q -- '--source'"

# --- C-2: recovery must not become a way around the down gate --------------
drive_case; echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1
PATH="$MOCK:$PATH" RUN_ID="$RID2" ./run.sh down > "$OUT" 2>&1; rc=$?
check "a recovered archive does not open the down gate by itself" "[ $rc -ne 0 ] && grep -q 'refusing to stop' $OUT"

echo "=== 15. review P1/P2: abrupt revocation, malformed progress, strict source identity ==="
PART="$OUT2/.collect_partial.json"
SUPER="$OUT2/collect_manifest.txt.superseded"
partial_state() {  # archive_state / verify / verdict of the receipt this run superseded
  python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));r=d.get("receipt_superseded") or {};print(d.get("archive_state"),d.get("verify"),r.get("verdict"))' \
    "$1" 2>/dev/null
}
abrupt_case() {  # abrupt_case SIG -> a FORCE=1 re-collect killed in the middle of a download
  fault_reset; vm_build; collect_try all > "$OUT" 2>&1
  rm -f "$OUT2/checkpoints/it_1000.pkl"   # force a real download so the mock can kill us
  echo "$1" > /tmp/nm_mock_state/dl_signal_it_1000.pkl
  collect_try all FORCE=1 > "$OUT" 2>&1
  local rc=$?
  rm -f /tmp/nm_mock_state/dl_signal_it_1000.pkl
  return "$rc"
}
for sig in KILL TERM; do
  abrupt_case "$sig"; rc=$?
  check "SIG$sig during a FORCE re-collect exits non-zero" "[ $rc -ne 0 ]"
  check "SIG$sig leaves no success receipt in the half-replaced archive" "no_receipt"
  check "SIG$sig keeps the old verdict only under an untrusted name" "grep -qx 'verify=ok' $SUPER"
  check "SIG$sig records an explicit collecting state" "[ \"\$(partial_state $PART)\" = 'collecting partial verify=ok' ]"
  check "SIG$sig does not damage the already verified bytes" "[ -s $OUT2/checkpoints/it_250.pkl ]"
  check "SIG$sig came from a real signal, not a normal failure path" "[ \"\$(cat /tmp/nm_mock_state/dl_killed_by 2>/dev/null)\" = '$sig' ]"
done
echo none > /tmp/nm_mock_state/drive_state
PATH="$MOCK:$PATH" RUN_ID="$RID2" ./run.sh down > "$OUT" 2>&1; rc=$?
check "an interrupted re-collect does not open the down gate" "[ $rc -ne 0 ] && grep -q 'not safely persisted' $OUT"
collect_try all > "$OUT" 2>&1; rc=$?
check "the next run resumes from the abrupt kill and re-issues the verdict" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "the re-issued verdict retires the superseded copy" "[ ! -e $SUPER ]"
check "the re-issued verdict drops the collecting marker" "[ ! -e $PART ]"

# --- P1: no shape of a partial manifest may raise a traceback --------------
# The old `len(old.get("files") or {})` raised TypeError for files=42 / files=true and
# escaped the OSError/ValueError handler, killing the collect and keeping any old receipt.
i=0
for bad in \
 '{"kind":"collect-partial","verify":"partial","files":42,"run_id":"__RID__","updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":true,"run_id":"__RID__","updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":"all","run_id":"__RID__","updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":{"a":1},"run_id":"__RID__","updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":{"a":{"size":"8","sha256":"x"}},"run_id":"__RID__","updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":{},"run_id":42,"updated_at":"t"}' \
 '{"kind":"collect-partial","verify":"partial","files":{},"run_id":"__RID__","updated_at":null}' \
 '[1,2,3]'
do
  i=$((i + 1))
  fault_reset; vm_build; collect_try all > "$OUT" 2>&1
  printf '%s\n' "$bad" | sed "s/__RID__/$RID2/" > "$PART"
  collect_try all FORCE=1 > "$OUT" 2>&1; rc=$?
  check "malformed progress file #$i does not abort the collect" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
  check "malformed progress file #$i is reported as ignored" "grep -q 'ignoring ' $OUT"
  check "malformed progress file #$i produces no traceback" "! grep -q 'Traceback' $OUT"
done
check "all 8 malformed progress shapes were exercised" "[ $i -eq 8 ]"

# --- P2: an explicit --source must prove which run it holds ----------------
drive_case; receipt_build ok "$DRIVE_SRC"
printf '{}\n' > "$DRIVE_DIR/manifest.json"   # no experiment_id at all
: > "$DL_LOG"; echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "a Drive source without experiment_id is refused" "[ $rc -ne 0 ] && grep -q 'cannot confirm run identity' $OUT"
check "an unidentified Drive source writes no receipt" "no_receipt"
check "an unidentified Drive source downloads nothing" "[ \"\$(dl_count)\" -eq 0 ]"
env PATH="$MOCK:$PATH" python3 tools/collect_run.py --session mock --run-id "$RID2" \
  --out "$OUT2" --source "$DRIVE_SRC" --ckpt-mode all > "$OUT" 2>&1; rc=$?
check "the collector returns exit 2 for a missing run identity" "[ $rc -eq 2 ]"
printf '{ not json at all' > "$DRIVE_DIR/manifest.json"
env PATH="$MOCK:$PATH" python3 tools/collect_run.py --session mock --run-id "$RID2" \
  --out "$OUT2" --source "$DRIVE_SRC" --ckpt-mode all > "$OUT" 2>&1; rc=$?
check "an unparseable source manifest is also an identity failure (rc=2)" "[ $rc -eq 2 ] && grep -q 'cannot confirm run identity' $OUT"
drive_case; receipt_build ok "$DRIVE_SRC"
echo "some_other_run_s0_20260101T000000Z" > /tmp/nm_mock_state/identity
env PATH="$MOCK:$PATH" python3 tools/collect_run.py --session mock --run-id "$RID2" \
  --out "$OUT2" --source "$DRIVE_SRC" --ckpt-mode all > "$OUT" 2>&1; rc=$?
check "a foreign experiment_id is refused with exit 2" "[ $rc -eq 2 ] && grep -q 'run identity mismatch' $OUT"
rm -f /tmp/nm_mock_state/identity
: > "$DL_LOG"; echo NO_RUN > /tmp/nm_mock_state/pipeline_state
src_collect all > "$OUT" 2>&1; rc=$?
check "a Drive source with a matching experiment_id still succeeds" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "the matching identity is recorded in the receipt" "grep -qx 'run_identity=$RID2' $OUT2/collect_manifest.txt"
# the default VM source is not operator-supplied: keep its historical leniency
fault_reset; vm_build
printf '{"status":"PASS"}\n' > "$VMRUN/manifest.json"
collect_try all > "$OUT" 2>&1; rc=$?
check "the default VM source stays lenient about experiment_id" "[ $rc -eq 0 ] && grep -qx 'verify=ok' $OUT2/collect_manifest.txt"
check "an absent identity is recorded, not invented" "grep -qx 'run_identity=-' $OUT2/collect_manifest.txt"
fault_reset

rm -rf "$OUT2" "$NM_FAKE_VM" /tmp/nm_keep.sha /tmp/nm_keep.log
unset NM_FAKE_VM

echo
echo "=============================="
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
