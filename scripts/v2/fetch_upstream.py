#!/usr/bin/env python3
"""Fetch the two pinned upstream evaluations for the V2 parity tests.

Nothing is vendored into the repository by default: this writes into
``research/modded-supermimic/.tmp/v2_upstream`` (gitignored).  Alternatively
set ``V2_SUPERMIMIC_ROOT`` / ``V2_NABLAMIMIC_ROOT`` to existing checkouts.

    python scripts/v2/fetch_upstream.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMMITS = {
    "supermimic_vibed": ("https://github.com/zita-ch/supermimic_vibed.git",
                         "8bfc1158bd968f729fcc65d5e897a05b05ea3d36"),
    "NablaMimic": ("https://github.com/sailab-ai/NablaMimic.git",
                   "bfbb2a6773b4776c17402a34a93ec7589d9c7c18"),
}

ROOT = Path(__file__).resolve().parents[2] / ".tmp/v2_upstream"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in COMMITS.items():
        dest = ROOT / name
        if (dest / ".git").is_dir():
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                                  capture_output=True, text=True, check=True).stdout.strip()
            if head == sha:
                print(f"{name}: already at {sha}")
                continue
        if dest.exists():
            print(f"{name}: exists but at a different revision, leaving it alone ({dest})")
            continue
        dest.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=dest, check=True)
        subprocess.run(["git", "fetch", "--depth=1", "origin", sha], cwd=dest, check=True)
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest, check=True)
        print(f"{name}: fetched {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
