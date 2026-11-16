#!/usr/bin/env bash
# Benchmark V2 - Colab G4 (RTX PRO 6000 Blackwell / sm_120) acceptance runbook.
#
# In a Colab notebook with a **G4** runtime, run:
#
#   !git clone --depth 1 https://github.com/basicv8vc/modded-supermimic.git /content/modded-supermimic
#   !bash /content/modded-supermimic/scripts/v2/colab_g4_runbook.sh
#
# If the repo is already cloned at $ROOT, the script fetches and checks out $BRANCH.
#
# Optional env: CLIPS="walk1_subject1 dance1_subject1"  STEPS=200  BRANCH=main
#               REPO_SLUG=basicv8vc/modded-supermimic  SUBDIR=.
#               GH_TOKEN=<PAT>  SKIP_INSTALL=1 (reuse a provisioned VM)
#
# The script fails fast and loudly on: wrong GPU, wrong compute capability,
# dependency drift, or a smoke that is not SMOKE_VALID.  It never upgrades the
# frozen benchmark stack to "make it work" -- see the drift check below.
set -euo pipefail

BRANCH="${BRANCH:-main}"
REPO_SLUG="${REPO_SLUG:-basicv8vc/modded-supermimic}"
ROOT="${ROOT:-/content/modded-supermimic}"
SUBDIR="${SUBDIR:-.}"
CLIPS="${CLIPS:-walk1_subject1 dance1_subject1}"
STEPS="${STEPS:-200}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
MJLAB_COMMIT="${MJLAB_COMMIT:-0fb8a681136be94ffc636a3dd423cabb97d91f10}"

GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -n "$GH_TOKEN" ]; then
  REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO_SLUG}.git"
else
  REPO_URL="https://github.com/${REPO_SLUG}.git"
fi

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
say "1/7 hardware (must be RTX PRO 6000 / sm_120)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null \
  || nvidia-smi --query-gpu=name,memory.total,driver_version \
                --format=csv,noheader 2>/dev/null \
  || die "nvidia-smi unavailable: no GPU runtime attached"
python - <<'PY' || exit 1
import sys, torch
ok = True
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() is False"); ok = False
else:
    name = torch.cuda.get_device_name(0)
    cap = tuple(torch.cuda.get_device_capability(0))
    print(f"torch                : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"device               : {name}")
    print(f"compute capability   : {cap} (sm_{cap[0]}{cap[1]})")
    if cap != (12, 0):
        print(f"FATAL: expected sm_120 for RTX PRO 6000, got sm_{cap[0]}{cap[1]}. "
              "Reconnect with a G4 runtime; do NOT proceed on L4/T4/H100.")
        ok = False
    if "RTX PRO 6000" not in name and "RTX 6000" not in name:
        print(f"FATAL: expected an RTX PRO 6000, got {name!r}. Reconnect with G4.")
        ok = False
sys.exit(0 if ok else 1)
PY

# --------------------------------------------------------------------------
say "2/7 torch / CUDA must understand sm_120"
# torch is NOT part of the frozen benchmark stack (mujoco / mujoco-warp /
# warp-lang / mjlab); it is environment provisioning.  Change it only if the
# image's torch predates Blackwell support, and record the change.
if ! python - <<'PY'
import sys, torch
cap = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0)
cuda = torch.version.cuda or "0"
major = tuple(int(x) for x in cuda.split(".")[:2])
# sm_120 needs a CUDA 12.8+ build; older wheels will fail at kernel launch.
sys.exit(0 if (cap == (12, 0) and major >= (12, 8)) else 1)
PY
then
  echo "installing a CUDA 13 torch build for sm_120 (provisioning step, logged)"
  pip -q install --index-url https://download.pytorch.org/whl/cu130 torch --upgrade
  python -c "import torch;print('torch now', torch.__version__, torch.version.cuda)"
fi

# --------------------------------------------------------------------------
say "3/7 frozen benchmark stack (exact versions, no substitutions)"
if [ "$SKIP_INSTALL" != "1" ]; then
  pip -q install "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0" pytest
  pip -q install "mjlab @ git+https://github.com/mujocolab/mjlab.git@${MJLAB_COMMIT}"
  pip -q install --no-deps --force-reinstall "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0"
fi
python - <<'PY' || exit 1
import importlib.metadata as md
want = {"mujoco": "3.11.0", "mujoco-warp": "3.11.0",
        "warp-lang": "1.16.0", "mjlab": "1.6.0"}
bad = []
for dist, v in want.items():
    try:
        got = md.version(dist)
    except md.PackageNotFoundError:
        got = "missing"
    mark = "ok " if got == v else "DRIFT"
    print(f"{mark} {dist:12s} {got:10s} (want {v})")
    if got != v:
        bad.append(dist)
if bad:
    print("FATAL: frozen dependency drift:", bad)
    print("Fix the environment; do not relax the frozen versions.")
    raise SystemExit(1)
PY

# --------------------------------------------------------------------------
say "4/7 repository @ ${BRANCH}"
if [ -d "$ROOT/.git" ]; then
  git -C "$ROOT" fetch --depth 1 origin "$BRANCH" -q
  git -C "$ROOT" checkout -q FETCH_HEAD
else
  git clone -q --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT" \
    || die "clone failed (if the repo is private, export GH_TOKEN=<read-only PAT>)"
fi
EVAL_SHA="$(git -C "$ROOT" rev-parse HEAD)"
export V2_EVALUATOR_GIT_SHA="$EVAL_SHA"
echo "evaluator git sha: $EVAL_SHA"
BM_DIR="${ROOT}"
if [ -n "$SUBDIR" ] && [ "$SUBDIR" != "." ]; then
  BM_DIR="${ROOT}/${SUBDIR}"
fi
cd "$BM_DIR"

# --------------------------------------------------------------------------
say "5/7 LAFAN1 data + manifest (not in git; hashes must match)"
python scripts/fetch_lafan1.py || die "dataset download failed"
python - <<'PY' || exit 1
import sys; sys.path.insert(0, ".")
from benchmark.evaluator.manifest import load_manifest
m = load_manifest()
print("dataset manifest sha256 :", m["dataset"]["manifest_sha256"])
print("robot model sha256      :", m["robot_model"]["sha256"])
PY

# --------------------------------------------------------------------------
say "6/7 GPU runtime parity (${STEPS} steps, real CUDA)"
set +e
python scripts/v2/gpu_runtime_check.py \
  --clip walk1_subject1 --steps "$STEPS" \
  --scope acceptance --out reports/v2/gpu_runtime_status.json
RC_CHECK=$?
set -e
echo "gpu_runtime_check exit=$RC_CHECK (0=PASS on target, 4=parity ok wrong GPU, 2=FAIL)"
[ "$RC_CHECK" -eq 0 ] || die "GPU runtime acceptance did not pass (exit $RC_CHECK)"

# --------------------------------------------------------------------------
say "7/7 single-clip smoke: ${CLIPS}"
for clip in $CLIPS; do
  out="reports/v2/smoke_gpu_${clip}.json"
  python scripts/v2/run_eval.py --smoke-policy replay --clips "$clip" \
    --cache-references --out "$out" || die "smoke failed for $clip"
  python - "$out" <<'PY' || exit 1
import json, sys
sys.path.insert(0, ".")
from benchmark.v2 import spec as S
d = json.load(open(sys.argv[1]))
a = S.audit_v2_smoke(d)
row = d["checkpoints"][0]["per_clip"][0]
print(f"  smoke {row['clip']}: {a['status']} eligible={S.is_submission_eligible(d)} "
      f"scope={d.get('report_scope')} return={row['normalized_return']:+.5f}")
if a["status"] != "SMOKE_VALID":
    print("  errors:", a["errors"]); raise SystemExit(1)
if d.get("report_scope") != "acceptance":
    print("  FATAL: record is not scoped as acceptance (wrong GPU?)"); raise SystemExit(1)
PY
done

# --------------------------------------------------------------------------
say "artifacts"
python scripts/v2/gpu_performance_report.py || true
python - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("reports/v2/gpu_runtime_status.json").read_text())
print("acceptance_status :", d.get("acceptance_status"))
print("gpu               :", (d.get("hardware") or "").splitlines()[1].strip())
perf = d.get("performance") or {}
print(f"fps (V2 step)     : {perf.get('steady_simulation_fps', 0):.1f}")
print(f"fps (raw physics) : {perf.get('sim_step_fps', 0):.1f}")
print(f"bottleneck        : {perf.get('bottleneck')}")
PY
(cd "$BM_DIR" && zip -q -r /content/v2_g4_artifacts.zip reports/v2) \
  && echo "wrote /content/v2_g4_artifacts.zip"
say "DONE - download /content/v2_g4_artifacts.zip and commit the three JSONs"
