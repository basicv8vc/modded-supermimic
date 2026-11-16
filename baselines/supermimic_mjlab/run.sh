#!/bin/bash
# Local driver for Official HybridRobotics/supermimic + V2 Evaluation on Colab + Google Drive.
#
# Usage:
#   ./run.sh all          # Create G4 session -> mount Drive -> setup/restore data -> launch tmux(train+eval)
#   ./run.sh up           # Create/reuse GPU session + mount Drive (fails loudly if Drive not mounted)
#   ./run.sh step NAME    # Run a single step: setup | train | eval
#   ./run.sh status       # Show tmux, GPU, snapshots, and Drive sync STATUS.txt
#   ./run.sh ls-drive     # List saved runs on Google Drive
#   ./run.sh collect      # Download summary JSONs + log from active session into ./out/<RUN_ID>/
#   ./run.sh down         # Stop Colab session and print compute unit balance
#
# Environment overrides:
#   SESSION=supermimic-official  GPU=G4  RUN=official_35k  ITERS=35001  SAVE_INTERVAL=500
#   REQUIRE_DRIVE=1 (default)    REQUIRE_G4=1 (default)   GH_TOKEN=...  PROXY=http://127.0.0.1:7897
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-supermimic-official}"
GPU="${GPU:-G4}"
RUN="${RUN:-official_35k}"
ITERS="${ITERS:-35001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
RETURN_STEEPNESS="${RETURN_STEEPNESS:-}"
RETURN_EXPLORE="${RETURN_EXPLORE:-}"
REQUIRE_DRIVE="${REQUIRE_DRIVE:-1}"
REQUIRE_G4="${REQUIRE_G4:-1}"

if [ -n "${PROXY:-}" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
fi

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

drive_mounted() {
  printf 'import os\nprint(os.path.isdir("/content/drive/MyDrive"))\n' \
    | colab exec -s "$SESSION" 2>/dev/null | tail -1 | grep -qx True
}

mount_drive() {
  if drive_mounted; then
    echo "Drive already mounted at /content/drive/MyDrive"
    return 0
  fi
  say "Mounting Google Drive (complete browser OAuth if prompted)"
  colab drivemount -s "$SESSION" || true
  if ! drive_mounted; then
    if [ "$REQUIRE_DRIVE" = "1" ]; then
      die "Google Drive failed to mount (REQUIRE_DRIVE=1). Aborting to avoid losing artifacts."
    fi
    echo "WARN: Drive not mounted; continuing only because REQUIRE_DRIVE=0"
  else
    echo "Drive mounted at /content/drive/MyDrive"
  fi
}

upload_vm_scripts() {
  printf 'import pathlib\npathlib.Path("/content/vm").mkdir(parents=True, exist_ok=True)\n' \
    | colab exec -s "$SESSION" >/dev/null
  for f in vm/*.py; do
    colab upload -s "$SESSION" "$f" "/content/$f" >/dev/null
  done
}

up() {
  say "Session '$SESSION' (GPU $GPU)"
  local st
  if ! st=$(colab status -s "$SESSION" 2>&1); then
    sleep 3
    st=$(colab status -s "$SESSION" 2>&1) || die "cannot query session: $st"
  fi
  if grep -q "not found" <<<"$st"; then
    colab new -s "$SESSION" --gpu "$GPU"
  else
    echo "Session exists, reusing."
  fi
  colab status -s "$SESSION"
  mount_drive
  upload_vm_scripts
}

run_blocking_step() {
  local script="$1"
  local log_name="$2"
  upload_vm_scripts
  say "Running /content/vm/$script (log: /content/$log_name)"
  local tmp_log
  tmp_log=$(mktemp)
  python3 - "$RUN" "${RUN_ID:-}" "$ITERS" "$SAVE_INTERVAL" "$REQUIRE_DRIVE" "$REQUIRE_G4" "${GH_TOKEN:-}" "$log_name" "$script" <<'GEN' | colab exec -s "$SESSION" --timeout 1800 | tee "$tmp_log"
import sys
run, rid, iters, save_iv, req_drv, req_g4, tok, log_name, script = sys.argv[1:10]
print(f"""import os, subprocess
env = {{
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    "RUN": {run!r},
    "RUN_ID": {rid!r},
    "ITERS": {iters!r},
    "SAVE_INTERVAL": {save_iv!r},
    "REQUIRE_DRIVE": {req_drv!r},
    "REQUIRE_G4": {req_g4!r},
    "GH_TOKEN": {tok!r},
}}
with open({("/content/" + log_name)!r}, "w") as lf:
    p = subprocess.Popen(["python3", "-u", {("/content/vm/" + script)!r}], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
        lf.write(line)
        lf.flush()
    rc = p.wait()
print(f"__STEP_RC__={{rc}}", flush=True)
""")
GEN
  local ok=0
  grep -qx "__STEP_RC__=0" "$tmp_log" && ok=1
  rm -f "$tmp_log"
  [ "$ok" = "1" ] || die "/content/vm/$script failed (see /content/$log_name on VM)"
}

launch_tmux_pipeline() {
  local mode="$1"  # train | eval | both | bisect
  upload_vm_scripts
  local rid="${RUN_ID:-${RUN}_$(date -u +%Y%m%dT%H%M%SZ)}"
  say "Launching tmux 'official_pipeline' (mode=$mode, RUN_ID=$rid)"
  python3 - "$rid" "$RUN" "$ITERS" "$SAVE_INTERVAL" "$LOG_INTERVAL" "$RETURN_STEEPNESS" "$RETURN_EXPLORE" "$REQUIRE_DRIVE" "$REQUIRE_G4" "$mode" <<'GEN' | colab exec -s "$SESSION"
import shlex, sys
rid, run, iters, save_iv, log_iv, steep, explore, req_drv, req_g4, mode = sys.argv[1:11]
if mode == "both":
    body = "python3 -u /content/vm/20_train.py && python3 -u /content/vm/30_eval.py"
elif mode == "train":
    body = "python3 -u /content/vm/20_train.py"
elif mode == "bisect":
    body = "python3 -u /content/vm/35_bisect_earliest.py"
else:
    body = "python3 -u /content/vm/30_eval.py"
exports = " ".join([
    "PYTHONUNBUFFERED=1",
    f"RUN={shlex.quote(run)}",
    f"RUN_ID={shlex.quote(rid)}",
    f"ITERS={shlex.quote(iters)}",
    f"SAVE_INTERVAL={shlex.quote(save_iv)}",
    f"LOG_INTERVAL={shlex.quote(log_iv)}",
    f"RETURN_STEEPNESS={shlex.quote(steep)}",
    f"RETURN_EXPLORE={shlex.quote(explore)}",
    f"REQUIRE_DRIVE={shlex.quote(req_drv)}",
    f"REQUIRE_G4={shlex.quote(req_g4)}",
])
shell_cmd = f"export {exports}; ({body}) 2>&1 | tee -a /content/pipeline.log"
print(f"""import pathlib, subprocess
pathlib.Path("/content/.current_run_id").write_text({(rid + "\\n")!r})
subprocess.run(["tmux", "kill-session", "-t", "official_pipeline"], capture_output=True)
subprocess.run(["tmux", "new-session", "-d", "-s", "official_pipeline", "bash", "-lc", {shell_cmd!r}], check=True)
print(subprocess.check_output(["tmux", "ls"], text=True))
""")
GEN
}

status() {
  upload_vm_scripts
  say "Status ($SESSION)"
  python3 - "$RUN" "${RUN_ID:-}" <<'GEN' | colab exec -s "$SESSION"
import sys
run, rid = sys.argv[1:3]
print(f"""import os, subprocess
env = {{**os.environ, "RUN": {run!r}, "RUN_ID": {rid!r}}}
r = subprocess.run(["python3", "/content/vm/40_status.py"], capture_output=True, text=True, env=env)
print(r.stdout)
if r.stderr:
    print(r.stderr)
""")
GEN
}

ls_drive() {
  say "Google Drive: supermimic_v2"
  printf 'import pathlib
base = pathlib.Path("/content/drive/MyDrive/supermimic_v2")
if not base.exists():
    print("(not found:", base, ")")
else:
    data = base / "data"
    print("data_ready:", (data / "DATA_READY").exists())
    runs = sorted((base / "runs").iterdir()) if (base / "runs").is_dir() else []
    print("runs:", len(runs))
    for r in runs:
        st = (r / "STATUS.txt").read_text().splitlines() if (r / "STATUS.txt").exists() else ["(no STATUS.txt)"]
        print(" -", r.name, "|", " ; ".join(st[:5]))
' | colab exec -s "$SESSION"
}

collect() {
  local rid
  rid=$(printf 'import pathlib\np=pathlib.Path("/content/.current_run_id")\nprint(p.read_text().strip() if p.exists() else "latest")\n' \
    | colab exec -s "$SESSION" 2>/dev/null | tail -1)
  local out="out/${rid}"
  mkdir -p "$out"
  say "Downloading summary artifacts -> $out"
  for remote in /content/snap_evals/sweep_summary.json /content/snap_evals/selected.json /content/snap_evals/bisect_report.json /content/pipeline.log; do
    colab download -s "$SESSION" "$remote" "$out/$(basename "$remote")" || echo "skip $remote"
  done
  say "All checkpoints & per-snapshot evals remain safely stored in Google Drive under MyDrive/supermimic_v2/runs/${rid}/"
}

down() {
  say "Stopping session '$SESSION'"
  colab stop -s "$SESSION"
  colab usage | tail -3
}

case "${1:-}" in
  all)
    up
    run_blocking_step "10_setup.py" "setup.log"
    launch_tmux_pipeline "both"
    say "Pipeline running in tmux on Colab + syncing to Google Drive. Poll with: $0 status"
    ;;
  up) up ;;
  step)
    case "${2:-}" in
      setup)  run_blocking_step "10_setup.py" "setup.log" ;;
      train)  launch_tmux_pipeline "train" ;;
      eval)   launch_tmux_pipeline "eval" ;;
      both)   launch_tmux_pipeline "both" ;;
      bisect) launch_tmux_pipeline "bisect" ;;
      *) die "unknown step: ${2:-} (setup|train|eval|both|bisect)" ;;
    esac
    ;;
  status)   status ;;
  ls-drive) ls_drive ;;
  collect)  collect ;;
  down)     down ;;
  *)
    cat <<EOF
Usage: $0 {all|up|step NAME|status|ls-drive|collect|down}
  step NAME: setup | train | eval
Env: SESSION GPU RUN RUN_ID ITERS SAVE_INTERVAL REQUIRE_DRIVE(=1) REQUIRE_G4(=1) GH_TOKEN PROXY
EOF
    exit 1
    ;;
esac
