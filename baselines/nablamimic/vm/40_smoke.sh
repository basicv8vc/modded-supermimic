#!/bin/bash
# Preflight before real training: (1) upstream tests incl. model counts, (2) NablaEngine CUDA
# library build on the GPU + engine identity, (3) 2-iteration scratch run through the real
# train.py engine path with the real config. Fails hard if physics silently falls back to
# MuJoCo Warp. Idempotent-ish: re-running is cheap after the first engine build (~1 min).
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== smoke start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
RUNS="${RUNS:-$WORK/runs}"
CACHE="${CACHE:-$WORK/cache}"
LOGDIR="$WORK/logs"
PY="$VENV/bin/python"

export LAFAN1_G1_DIR="$WORK/data/lafan1_g1"
export MOTION_CACHE_DIR="$WORK/data/cache"
export MENAGERIE_DIR="$REPO/third_party/mujoco_menagerie"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export DK_CACHE_DIR="$CACHE/dk"
if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then export NVCC=/usr/local/cuda-12.9/bin/nvcc
elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then export NVCC=/usr/local/cuda-12.8/bin/nvcc
fi
mkdir -p "$LOGDIR" "$CACHE/jax" "$CACHE/dk"
[ -x "$PY" ] || { echo "FATAL: venv missing, run 20_deps.sh first"; exit 1; }
cd "$REPO"

echo "--- (1) upstream tests (model counts nq/nv/nu/ngeom/npair == 36/35/29/69/502) ---"
"$PY" -m pip -q install pytest
"$PY" -m pytest -q tests/ 2>&1 | tee "$LOGDIR/tests.log" | tail -3

echo "--- (2) NablaEngine CUDA build + identity (no fallback) ---"
"$PY" - <<'PY' 2>&1 | tee "$LOGDIR/engine_smoke.log"
from nablamimic.env import EnvConfig, build_model
from nablamimic.sim import make_engine, engine_info
import jax
m, kp = build_model(EnvConfig())
counts = dict(nbody=int(m.nbody), njnt=int(m.njnt), ngeom=int(m.ngeom),
              npair=int(m.npair), nu=int(m.nu))
print("model:", counts)
assert counts["ngeom"] == 69 and counts["npair"] == 502, counts
class _E:
    pass
e = _E()
e.m = m
e.foot_pairs = [i for i in range(m.npair)
                if "_foot" in m.pair(i).name and m.pair(i).name.endswith("_floor")]
e.torso_id = m.body("torso_link").id
# exactly train.py's engine gradient options
phys = make_engine(e, [16], clamp_kv=0, normal=3, clamp_soft=None)
print("engine_info:", engine_info(phys[16]))
print("jax devices:", jax.devices())
print("ENGINE_PREFLIGHT_OK")
PY

echo "--- (3) 2-iteration scratch run through the real train.py engine path ---"
rm -rf "$RUNS/smoke"
"$PY" -u -m nablamimic.train --config configs/ours.args --out "$RUNS/smoke" \
  --hours 3 --iters 2 2>&1 | tee "$LOGDIR/smoke_train.log" | grep -E "engine|Engine|compiled|iter|warp" | head -20
# the backend line must say engine; warp here means a silent fallback and is fatal
grep -q "engine physics" "$LOGDIR/smoke_train.log" \
  || { echo "FATAL: training did not use NablaEngine (no 'engine physics' in log)"; exit 1; }
grep -q "warp physics" "$LOGDIR/smoke_train.log" \
  && { echo "FATAL: silent fallback to MuJoCo Warp detected"; exit 1; }
"$PY" - <<'PY'
import json
c = json.load(open("/content/nabla/runs/smoke/config.json"))
argv = c["argv"]
assert "--hours" in argv and argv[argv.index("--hours") + 1] == "3", argv
assert "--residual-action" in argv and "--physics" in argv and "engine" in argv, argv
print("locked argv ok:", " ".join(argv[:6]), "...")
print("SMOKE_OK")
PY

echo "=== smoke DONE $(date) ==="
