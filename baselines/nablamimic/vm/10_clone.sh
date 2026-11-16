#!/bin/bash
# Clone NablaMimic at the pinned commit (upstream stays external; no copy in gptfiles).
# Also records the pinned NablaEngine requirement for provenance. Idempotent.
set -euo pipefail
LOG="/content/$(basename "$0" .sh).log"
exec > "$LOG" 2>&1
trap 'echo "=== FAIL rc=$? $(date) ==="' ERR
echo "=== clone start $(date) ==="

WORK="${WORK:-/content/nabla}"
REPO="${REPO:-$WORK/NablaMimic}"
# Pinned revisions (Experiment A, 2026-11-06). NablaEngine itself is installed from
# this spec by scripts/setup_env.sh (20_deps.sh passes it explicitly).
NABLAMIMIC_SHA="bfbb2a6773b4776c17402a34a93ec7589d9c7c18"
NABLAENGINE_SHA="c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c"
NABLAENGINE_SPEC="nablaengine[jax] @ git+https://github.com/OamIcs/NablaEngine.git@${NABLAENGINE_SHA}"

mkdir -p "$WORK"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/sailab-ai/NablaMimic.git "$REPO"
fi
git -C "$REPO" fetch -q origin "$NABLAMIMIC_SHA" 2>/dev/null || git -C "$REPO" fetch -q origin
git -C "$REPO" checkout -q "$NABLAMIMIC_SHA"
actual="$(git -C "$REPO" rev-parse HEAD)"
[ "$actual" = "$NABLAMIMIC_SHA" ] || { echo "FATAL: NablaMimic checkout $actual != $NABLAMIMIC_SHA"; exit 1; }

# Engine pin sanity: setup_env.sh must carry the same default (guards upstream drift).
grep -q "$NABLAENGINE_SHA" "$REPO/scripts/setup_env.sh" \
  || { echo "FATAL: scripts/setup_env.sh no longer pins NablaEngine $NABLAENGINE_SHA"; exit 1; }

mkdir -p "$WORK/logs"
{
  echo "nablamimic_url=https://github.com/sailab-ai/NablaMimic"
  echo "nablamimic_commit=$actual"
  echo "nablamimic_commit_date=$(git -C "$REPO" log -1 --format=%ad --date=short)"
  echo "nablaengine_spec=$NABLAENGINE_SPEC"
  echo "nablaengine_commit=$NABLAENGINE_SHA"
  echo "cloned_at=$(date -u +%FT%TZ)"
} > "$WORK/logs/clone_env.txt"
cat "$WORK/logs/clone_env.txt"

echo "nablamimic: $actual"
echo "=== clone DONE $(date) ==="
