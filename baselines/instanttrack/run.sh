#!/bin/bash
# Local driver for InstantTrack v1.4 / BlinkMimic + V2 Evaluation on Colab G4 + Google Drive.
#
# Usage:
#   ./run.sh all          # Create G4 session -> mount Drive -> setup -> tmux(train+eval)
#   ./run.sh up           # Create/reuse GPU session + mount Drive
#   ./run.sh step NAME    # setup | train | eval | both | bisect
#   ./run.sh status       # Show tmux, GPU, local checkpoints, and Drive sync STATUS.txt
#   ./run.sh ls-drive     # List saved runs on Google Drive
#   ./run.sh collect      # Download V2 eval JSONs + logs into ./out/<RUN_ID>/
#   ./run.sh down         # Stop Colab session and print compute unit balance
#
# Environment overrides:
#   SESSION=instanttrack  GPU=G4  UPSTREAM=instanttrack  PRESET=instanttrack
#   RUN=instanttrack_v14  SEED=0  BUDGET_MINUTES=15  CKPT_EVERY=100  CKPT_MINUTES=1.0
#   FAST_SCREEN=1  NCONMAX=256  NJMAX=1024  REQUIRE_DRIVE=1  REQUIRE_G4=1
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SESSION:-instanttrack}"
GPU="${GPU:-G4}"
UPSTREAM="${UPSTREAM:-instanttrack}"
PRESET="${PRESET:-$([ "$UPSTREAM" = "blinkmimic" ] && echo "main" || echo "instanttrack")}"
RUN="${RUN:-$([ "$UPSTREAM" = "blinkmimic" ] && echo "blinkmimic" || echo "instanttrack_v14")}"
SEED="${SEED:-0}"
BUDGET_MINUTES="${BUDGET_MINUTES:-15}"
CKPT_EVERY="${CKPT_EVERY:-100}"
CKPT_MINUTES="${CKPT_MINUTES:-1.0}"
FAST_SCREEN="${FAST_SCREEN:-1}"
NCONMAX="${NCONMAX:-256}"
NJMAX="${NJMAX:-1024}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
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

sync_local_policy_adapter() {
  # Ensure local policy.py is present inside /content/modded-supermimic even before pushing a branch.
  printf 'import pathlib\npathlib.Path("/content/modded-supermimic/baselines/instanttrack").mkdir(parents=True, exist_ok=True)\n' \
    | colab exec -s "$SESSION" >/dev/null
  colab upload -s "$SESSION" "policy.py" "/content/modded-supermimic/baselines/instanttrack/policy.py" >/dev/null
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
  colab usage
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
  python3 - "$UPSTREAM" "$PRESET" "$RUN" "${RUN_ID:-}" "$SEED" "$BUDGET_MINUTES" "$CKPT_EVERY" "$CKPT_MINUTES" "$NCONMAX" "$NJMAX" "$REQUIRE_DRIVE" "$REQUIRE_G4" "${GH_TOKEN:-}" "$log_name" "$script" <<'GEN' | colab exec -s "$SESSION" --timeout 1800 | tee "$tmp_log"
import sys
up, preset, run, rid, seed, budget, ck_ev, ck_min, ncon, njmax, req_drv, req_g4, tok, log_name, script = sys.argv[1:16]
print(f"""import os, subprocess
env = {{
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    "UPSTREAM": {up!r},
    "PRESET": {preset!r},
    "RUN": {run!r},
    "RUN_ID": {rid!r},
    "SEED": {seed!r},
    "BUDGET_MINUTES": {budget!r},
    "CKPT_EVERY": {ck_ev!r},
    "CKPT_MINUTES": {ck_min!r},
    "NCONMAX": {ncon!r},
    "NJMAX": {njmax!r},
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
  sync_local_policy_adapter
  local rid="${RUN_ID:-${RUN}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
  say "Launching tmux 'instanttrack_pipeline' (mode=$mode, RUN_ID=$rid)"
  python3 - "$rid" "$UPSTREAM" "$PRESET" "$RUN" "$SEED" "$BUDGET_MINUTES" "$CKPT_EVERY" "$CKPT_MINUTES" "$FAST_SCREEN" "$EXTRA_ARGS" "$REQUIRE_DRIVE" "$REQUIRE_G4" "$mode" <<'GEN' | colab exec -s "$SESSION"
import shlex, sys
rid, up, preset, run, seed, budget, ck_ev, ck_min, fscreen, extra, req_drv, req_g4, mode = sys.argv[1:14]
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
    f"UPSTREAM={shlex.quote(up)}",
    f"PRESET={shlex.quote(preset)}",
    f"RUN={shlex.quote(run)}",
    f"RUN_ID={shlex.quote(rid)}",
    f"SEED={shlex.quote(seed)}",
    f"BUDGET_MINUTES={shlex.quote(budget)}",
    f"CKPT_EVERY={shlex.quote(ck_ev)}",
    f"CKPT_MINUTES={shlex.quote(ck_min)}",
    f"FAST_SCREEN={shlex.quote(fscreen)}",
    f"EXTRA_ARGS={shlex.quote(extra)}",
    f"REQUIRE_DRIVE={shlex.quote(req_drv)}",
    f"REQUIRE_G4={shlex.quote(req_g4)}",
])
shell_cmd = f"export {exports}; ({body}) 2>&1 | tee -a /content/pipeline_instanttrack.log"
print(f"""import pathlib, subprocess
pathlib.Path("/content/.current_instanttrack_run_id").write_text({(rid + "\\n")!r})
subprocess.run(["tmux", "kill-session", "-t", "instanttrack_pipeline"], capture_output=True)
subprocess.run(["tmux", "new-session", "-d", "-s", "instanttrack_pipeline", "bash", "-lc", {shell_cmd!r}], check=True)
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
    runs = sorted((base / "runs").iterdir()) if (base / "runs").is_dir() else []
    print("runs:", len(runs))
    for r in runs:
        st = (r / "STATUS.txt").read_text().splitlines() if (r / "STATUS.txt").exists() else ["(no STATUS.txt)"]
        print(" -", r.name, "|", " ; ".join(st[:5]))
' | colab exec -s "$SESSION"
}

collect() {
  local rid
  rid=$(printf 'import pathlib\np=pathlib.Path("/content/.current_instanttrack_run_id")\nprint(p.read_text().strip() if p.exists() else "latest")\n' \
    | colab exec -s "$SESSION" 2>/dev/null | tail -1)
  local out="out/${rid}"
  mkdir -p "$out"
  say "Downloading summary artifacts -> $out"
  for remote in \
    "/content/instanttrack_runs/${rid}/progress.csv" \
    "/content/instanttrack_runs/${rid}/args.json" \
    "/content/instanttrack_runs/${rid}/launcher_meta.json" \
    "/content/instanttrack_runs/${rid}/eval.csv" \
    "/content/instanttrack_runs/${rid}/v2_evals/selected.json" \
    "/content/instanttrack_runs/${rid}/v2_evals/bisect_report.json" \
    "/content/pipeline_instanttrack.log"; do
    colab download -s "$SESSION" "$remote" "$out/$(basename "$remote")" || echo "skip $remote"
  done
  say "Checkpoints and full per-snapshot V2 JSONs remain in Google Drive: MyDrive/supermimic_v2/runs/${rid}/"
}

down() {
  say "Stopping session '$SESSION'"
  colab stop -s "$SESSION"
  colab usage | tail -3
}

case "${1:-}" in
  all)
    up
    run_blocking_step "10_setup.py" "setup_instanttrack.log"
    launch_tmux_pipeline "both"
    say "Pipeline running in tmux on Colab + syncing to Drive. Poll with: $0 status"
    ;;
  up) up ;;
  step)
    case "${2:-}" in
      setup)  run_blocking_step "10_setup.py" "setup_instanttrack.log" ;;
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
  step NAME: setup | train | eval | both | bisect
Env: SESSION GPU UPSTREAM(instanttrack|blinkmimic) PRESET SEED BUDGET_MINUTES CKPT_EVERY CKPT_MINUTES FAST_SCREEN REQUIRE_DRIVE REQUIRE_G4
EOF
    exit 1
    ;;
esac
