#!/bin/bash
# run_id.sh NAME SEED -> <NAME>_s<SEED>_<UTC timestamp>  (e.g. ea_s0_20261106T061500Z)
# Unique per launch; the UTC timestamp makes collisions across re-runs impossible.
set -euo pipefail
NAME="${1:?usage: run_id.sh NAME SEED}"
SEED="${2:?usage: run_id.sh NAME SEED}"
case "$NAME" in
  *[!A-Za-z0-9._-]*|'') echo "FATAL: NAME must match [A-Za-z0-9._-]+" >&2; exit 2 ;;
esac
case "$SEED" in
  ''|*[!0-9]*) echo "FATAL: SEED must be a non-negative integer" >&2; exit 2 ;;
esac
printf '%s_s%s_%s\n' "$NAME" "$SEED" "$(date -u +%Y%m%dT%H%M%SZ)"
