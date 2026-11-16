#!/usr/bin/env python
"""Freeze / verify the evaluator asset + dataset hashes.

    python scripts/freeze_manifest.py            # write benchmark/evaluator/manifest.json
    python scripts/freeze_manifest.py --check    # verify on-disk hashes still match
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import manifest  # noqa: E402


def main() -> int:
    check = "--check" in sys.argv[1:]
    current = manifest.build_manifest()
    if check:
        if not C.MANIFEST_PATH.exists():
            print(f"FAIL: {C.MANIFEST_PATH} does not exist")
            return 1
        frozen = json.loads(C.MANIFEST_PATH.read_text(encoding="utf-8"))
        mismatched = [
            key
            for key in ("robot_model", "dataset", "mjlab")
            if frozen.get(key) != current.get(key)
        ]
        if mismatched:
            print(f"FAIL: hash mismatch in {mismatched}")
            print(json.dumps({"frozen": frozen, "current": current}, indent=2))
            return 1
        print("OK: manifest matches the frozen hashes")
        print(json.dumps(current, indent=2))
        return 0

    C.MANIFEST_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {C.MANIFEST_PATH}")
    print(json.dumps(current, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
