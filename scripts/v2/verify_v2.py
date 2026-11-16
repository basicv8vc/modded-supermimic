#!/usr/bin/env python3
"""V2 parity verifier / frozen oracle snapshot generator.

    python scripts/v2/verify_v2.py                    # report only (GPU-free)
    python scripts/v2/verify_v2.py --write-snapshot  # regenerate the committed snapshot

The snapshot is measured from the *real* pinned upstream source (never from a
hand-copied formula) so that ``tests/test_benchmark_v2.py`` can assert parity
on machines without the upstream checkouts or jax/torch installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import v2_support as O  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-snapshot", action="store_true")
    args = ap.parse_args()

    print("source integrity:")
    for name, status in O.source_integrity().items():
        print(f"  {name:32s} {status}")
    print(f"heavy deps (torch/jax/mjx): {O.heavy_deps_available()}")
    if not O.real_oracle_available():
        print("\nReal oracles unavailable; nothing regenerated. The committed "
              "snapshot still exercises the parity assertions.")
        return 0

    snapshot = O.build_snapshot()
    print("\nreward parity (per-step totals):")
    print(f"  {'case':22s} {'SuperMimic':>14s} {'NablaMimic':>14s} {'V2':>14s} {'max|NM-V2| term':>18s}")
    for kind, rec in snapshot["cases"].items():
        v2, nm = rec["v2"]["terms"], rec["nablamimic"]["terms"]
        worst = max(abs(v2[n] - nm[k]) for k, n in O.NM_TO_V2.items())
        print(f"  {kind:22s} {rec['supermimic']['total']:14.9f} "
              f"{rec['nablamimic']['total']:14.9f} {rec['v2']['total']:14.9f} {worst:18.3e}")

    print("\ntermination parity:")
    d = O.make_case("drift")
    sm, nm = O.load_supermimic(), O.load_nablamimic()
    cases = [("normal", {}), ("height 0.249", {"dz": 0.249}),
             ("height 0.250", {"dz": 0.25}), ("height 0.251", {"dz": 0.251}),
             ("tilt 0.6", {"tilt": 0.6}), ("tilt 1.7", {"tilt": 1.7}),
             ("nonfinite", {"nan": True})]
    for label, kw in cases:
        a = O.sm_termination(d, mdp=sm, **kw)
        b = O.nm_termination(d, nenv=nm, **kw)
        c = O.v2_termination(d, **kw).terminated
        print(f"  {label:16s} SM={str(a):5s} NM={str(b):5s} V2={str(c):5s}")

    if args.write_snapshot:
        path = O.SNAPSHOT_PATH
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
