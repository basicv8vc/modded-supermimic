#!/usr/bin/env python3
"""Re-label already-generated V2 records as preflight (non-target hardware).

Why this exists
---------------
``scripts/v2/run_eval.py`` now stamps ``report_scope`` / ``target_hardware_met``
on every record it writes.  Records produced *before* that stamp existed (the
NVIDIA L4 / sm_89 smoke runs) carry the same hardware fields but no top-level
verdict, so a reader could mistake them for the RTX PRO 6000 acceptance.

This tool adds **only** those two metadata fields plus a note.  It never touches
a score, a provenance value or a hardware field, and it refuses to run on a
record whose detected GPU is already the target.  Idempotent.

    python scripts/v2/mark_preflight.py reports/v2/smoke_gpu*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TARGET_MARKERS = ("RTX PRO 6000", "RTX 6000")


def is_target(record: dict) -> bool:
    prov = (record.get("evaluation") or {}).get("provenance") or {}
    hw = record.get("hardware") or {}
    name = str(prov.get("gpu_name") or hw.get("gpu_name") or "")
    sm120 = bool(prov.get("sm_120_confirmed") or hw.get("sm_120_confirmed"))
    return sm120 and any(m in name for m in TARGET_MARKERS)


def annotate(path: Path) -> str:
    record = json.loads(path.read_text())
    if is_target(record):
        return "skipped: already target hardware"
    if record.get("report_scope") == "preflight_non_target_hardware":
        return "already annotated"
    prov = (record.get("evaluation") or {}).get("provenance") or {}
    hw = record.get("hardware") or {}
    gpu = prov.get("gpu_name") or hw.get("gpu_name") or "unknown"
    cc = prov.get("gpu_compute_capability") or hw.get("gpu_compute_capability") or "unknown"
    record["target_hardware_met"] = False
    record["report_scope"] = "preflight_non_target_hardware"
    record["_preflight_note"] = (
        f"Generated on {gpu} ({cc}), NOT the task's RTX PRO 6000 / sm_120 target. "
        "Valid evidence for CUDA runtime + V2 parity; it is not the Blackwell "
        "acceptance artifact. Re-run scripts/v2/run_eval.py on the target GPU to "
        "produce a record with report_scope=acceptance.")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return f"annotated ({gpu} / {cc})"


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or []
    if not paths:
        print(__doc__)
        return 2
    rc = 0
    for p in paths:
        if not p.exists():
            print(f"{p}: MISSING")
            rc = 1
            continue
        print(f"{p.name}: {annotate(p)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
