#!/bin/bash
# LAFAN1 G1 data (40 CSVs) + the 50 Hz reference cache used by training.
# Google Drive mirror: restore data_mirror.tar.gz when present (sha256-verified), otherwise
# download with the official scripts (scripts/download_lafan1.sh verifies scripts/lafan1_g1.sha256)
# and build the cache with nablamimic.motion; then upload the mirror atomically. Idempotent.
# LAFAN1 G1 is CC BY-NC-ND 4.0 (research/non-commercial only) - never commit data to git.
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== data start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
VENV="${VENV:-$WORK/venv}"
DATA="${DATA:-$WORK/data}"
PY="$VENV/bin/python"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"
MIRROR_DIR="$DRIVE_ROOT/supermimic/nablamimic/data"
MIRROR="$MIRROR_DIR/data_mirror.tar.gz"
MIRROR_SHA="$MIRROR_DIR/data_mirror.tar.gz.sha256"

[ -x "$PY" ] || { echo "FATAL: venv missing, run 20_deps.sh first"; exit 1; }
mkdir -p "$DATA"

data_complete() {
  [ "$(find "$DATA/lafan1_g1" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)" -eq 40 ] \
    && [ "$(find "$DATA/cache" -maxdepth 1 -type f 2>/dev/null | wc -l)" -ge 40 ]
}

# 0) prefer the Drive mirror (checksum-verified before extraction)
if ! data_complete && [ -d "$DRIVE_ROOT" ] && [ -f "$MIRROR" ] && [ -f "$MIRROR_SHA" ]; then
  echo "restoring data from Drive mirror: $MIRROR"
  if (cd "$MIRROR_DIR" && sha256sum -c data_mirror.tar.gz.sha256) >/dev/null 2>&1; then
    if tar -xzf "$MIRROR" -C "$WORK" --no-same-owner; then
      echo "mirror restored: $(du -sh "$DATA" | cut -f1)"
    else
      echo "WARN: mirror extraction failed, falling back to download"
    fi
  else
    echo "WARN: mirror checksum mismatch, falling back to download"
  fi
fi

# 1) official download + official sha256 manifest
if [ "$(find "$DATA/lafan1_g1" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)" -ne 40 ]; then
  export LAFAN1_G1_DIR="$DATA/lafan1_g1"
  PATH="$VENV/bin:$PATH" bash "$REPO/scripts/download_lafan1.sh" "$DATA/lafan1_g1"
fi
n=$(find "$DATA/lafan1_g1" -maxdepth 1 -name '*.csv' | wc -l)
[ "$n" -eq 40 ] || { echo "FATAL: expected 40 CSVs, got $n"; exit 1; }
# verify against the official manifest (also covers mirror-restored files)
( cd "$DATA/lafan1_g1" && grep -v '^#' "$REPO/scripts/lafan1_g1.sha256" | sha256sum -c --quiet - ) \
  || { echo "FATAL: LAFAN1 CSVs fail scripts/lafan1_g1.sha256"; exit 1; }
echo "motions: 40 CSVs ok (sha256 verified)"

# 2) 50 Hz reference cache (build is deterministic; restores skip this)
if [ "$(find "$DATA/cache" -maxdepth 1 -type f 2>/dev/null | wc -l)" -lt 40 ]; then
  export LAFAN1_G1_DIR="$DATA/lafan1_g1"
  export MOTION_CACHE_DIR="$DATA/cache"
  export MENAGERIE_DIR="$REPO/third_party/mujoco_menagerie"
  PATH="$VENV/bin:$PATH" "$PY" -m nablamimic.motion
fi
ncache=$(find "$DATA/cache" -maxdepth 1 -type f | wc -l)
[ "$ncache" -ge 40 ] || { echo "FATAL: cache build incomplete ($ncache files)"; exit 1; }
echo "cache: $ncache files ok"
du -sh "$DATA/lafan1_g1" "$DATA/cache"

# 3) upload the mirror atomically (pack -> .part -> size check -> rename -> sha256)
mirror_up() {
  [ -d "$DRIVE_ROOT" ] || { echo "drive not mounted, skip mirror upload"; return 0; }
  mkdir -p "$MIRROR_DIR" || return 1
  local local_tar=/content/nabla_data_mirror.tar.gz
  echo "packing data mirror ..."
  tar -czf "$local_tar" -C "$WORK" data || return 1
  local size
  size=$(stat -c %s "$local_tar")
  cp "$local_tar" "$MIRROR.part" || { rm -f "$MIRROR.part"; return 1; }
  if [ "$(stat -c %s "$MIRROR.part" 2>/dev/null || echo 0)" -ne "$size" ]; then
    rm -f "$MIRROR.part"; echo "WARN: mirror size mismatch after copy"; return 1
  fi
  mv "$MIRROR.part" "$MIRROR" || return 1
  ( cd "$MIRROR_DIR" \
      && sha256sum data_mirror.tar.gz > data_mirror.tar.gz.sha256.tmp \
      && mv data_mirror.tar.gz.sha256.tmp data_mirror.tar.gz.sha256 ) || return 1
  rm -f "$local_tar"
  echo "mirror uploaded: $MIRROR ($size bytes)"
}
if [ -d "$DRIVE_ROOT" ] && [ -f "$MIRROR" ] && [ -f "$MIRROR_SHA" ]; then
  echo "Drive mirror already present, skip re-upload"
else
  mirror_up || echo "WARN: data mirror upload failed (non-fatal; data lives on VM)"
fi

echo "=== data DONE $(date) ==="
