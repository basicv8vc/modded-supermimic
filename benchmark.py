#!/usr/bin/env python3
"""Offline benchmark record checker; *not* a physics simulator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from benchmark.core import load_and_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    validate = subs.add_parser("validate", help="validate one 40-clip result")
    validate.add_argument("record", type=Path)
    validate.add_argument("--json-out", type=Path)
    leaderboard = subs.add_parser("leaderboard", help="rank provisional qualifying records")
    leaderboard.add_argument("submissions", type=Path)
    leaderboard.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    if args.command == "validate":
        report = load_and_audit(args.record)
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        print(payload)
        if args.json_out:
            args.json_out.write_text(payload + "\n", encoding="utf-8")
        return 2 if report["status"] == "INVALID" else 0

    if not args.submissions.is_dir():
        parser.error("submissions must be a directory")
    rows = []
    invalid = []
    for path in sorted(args.submissions.glob("*.json")):
        report = load_and_audit(path)
        if report["status"] == "INVALID":
            invalid.append(path.name)
        elif report["first_qualifying_checkpoint"]:
            first = report["first_qualifying_checkpoint"]
            rows.append({
                "algorithm": report["algorithm"],
                "training_commit": report["training_commit"],
                "seed": report["seed"],
                "training_wall_s": first["training_wall_s"],
                "normalized_return": first["normalized_return"],
                "failed_clips": first["failed_clips"],
                "checkpoint_sha256": first["checkpoint_sha256"],
                "verification": "PROVISIONAL / PHYSICS NOT AUDITED",
            })
    rows.sort(key=lambda x: (x["training_wall_s"], x["algorithm"], x["seed"]))
    cols = ["algorithm", "training_commit", "seed", "training_wall_s", "normalized_return", "failed_clips", "checkpoint_sha256", "verification"]
    writer = csv.DictWriter(sys.stdout, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    if args.csv_out:
        with args.csv_out.open("w", encoding="utf-8", newline="") as f:
            file_writer = csv.DictWriter(f, fieldnames=cols)
            file_writer.writeheader()
            file_writer.writerows(rows)
    if invalid:
        print(f"Skipped {len(invalid)} INVALID submissions: {', '.join(invalid)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
