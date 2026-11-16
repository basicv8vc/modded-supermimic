#!/bin/bash
# CUDA 12.9 toolkit (nvcc >= 12.8 for Blackwell sm_120) + Python 3.12 venv + the pinned
# NablaMimic dependency stack (scripts/setup_env.sh: requirements-lock, warp cu12 wheel,
# patched MuJoCo Warp, Menagerie G1, NablaEngine at the pinned commit). Idempotent.
# Verified on Colab G4 (RTX PRO 6000 Blackwell, Ubuntu 24.04 noble) 2026-11-06.
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== deps start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
PY="$VENV/bin/python"
CACHE="${CACHE:-$WORK/cache}"
# Same pin as 10_clone.sh / scripts/setup_env.sh default; passed explicitly to avoid drift.
NABLAENGINE_SHA="c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c"
export ENGINE_SPEC="nablaengine[jax] @ git+https://github.com/OamIcs/NablaEngine.git@${NABLAENGINE_SHA}"

[ -d "$REPO/.git" ] || { echo "FATAL: repo missing, run 10_clone.sh first"; exit 1; }

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

echo "--- CUDA 12.x toolkit ---"
if ! ls /usr/local/cuda-12.*/bin/nvcc >/dev/null 2>&1; then
  echo "installing cuda-toolkit-12-9 ..."
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y wget build-essential git curl ca-certificates >/dev/null
  . /etc/os-release
  case "$VERSION_CODENAME" in
    jammy) REPO_CODENAME=ubuntu2204 ;;
    noble) REPO_CODENAME=ubuntu2404 ;;
    focal) REPO_CODENAME=ubuntu2004 ;;
    *)     REPO_CODENAME=ubuntu2404 ;;
  esac
  echo "ubuntu release: $VERSION_CODENAME -> $REPO_CODENAME"
  wget -O /tmp/cuda-keyring.deb \
    "https://developer.download.nvidia.com/compute/cuda/repos/${REPO_CODENAME}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/cuda-keyring.deb
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cuda-toolkit-12-9 >/dev/null
fi
cuda_home="$(ls -d /usr/local/cuda-12.* 2>/dev/null | head -1)"
[ -n "$cuda_home" ] || { echo "ERROR: no /usr/local/cuda-12.* toolkit found"; exit 1; }
echo "$cuda_home"
"$cuda_home/bin/nvcc" --version | tail -3

echo "--- python 3.12 venv ---"
if ! command -v uv >/dev/null 2>&1; then python3 -m pip -q install uv; fi
uv python install 3.12
if [ ! -x "$PY" ]; then uv venv --python 3.12 --seed "$VENV"; fi

echo "--- pinned dependency stack (scripts/setup_env.sh) ---"
if ! "$PY" -c "import nablamimic, mujoco_warp, warp, jax, nablaengine" 2>/dev/null; then
  ( cd "$REPO" && PYTHON="$PY" bash scripts/setup_env.sh "$VENV" )
fi

echo "--- import check (no fallback: engine must be importable) ---"
"$PY" - <<'PY'
import jax, mujoco, warp, mujoco_warp, nablaengine, nablamimic
print("python", __import__("sys").version.split()[0])
print("jax", jax.__version__, "devices:", jax.devices())
print("mujoco", mujoco.__version__, "warp", warp.__version__)
print("nablaengine", getattr(nablaengine, "__version__", "?"), nablaengine.__file__)
# jax 0.12 returns a unified jaxlib Device whose __name__ is "Device" (the repr still
# says CudaDevice), so match on platform/backend instead of the class name.
assert jax.default_backend() == "gpu", (jax.default_backend(), jax.devices())
assert all(getattr(d, "platform", "") == "gpu" for d in jax.devices()), jax.devices()
print("IMPORTS_OK")
PY

echo "--- dependency manifest ---"
mkdir -p "$WORK/logs" "$CACHE/jax" "$CACHE/dk"
{
  echo "collected_at=$(date -u +%FT%TZ)"
  echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)"
  echo "nvcc=$("$VENV/bin/python" -c 'import glob; print(glob.glob("/usr/local/cuda-12.*/bin/nvcc")[0])')"
  "$PY" - <<'PY'
import importlib.metadata as m, platform, sys
print("python", sys.version.split()[0], platform.machine())
for p in ("jax", "jax-cuda12-plugin", "mujoco", "warp-lang", "optax", "nablaengine", "nablamimic", "mujoco-warp", "numpy"):
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, "(not found)")
PY
} > "$WORK/logs/env_manifest.txt"
cat "$WORK/logs/env_manifest.txt"

echo "=== deps DONE $(date) ==="
