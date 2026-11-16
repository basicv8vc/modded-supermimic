#!/usr/bin/env python
"""Download the 40 LAFAN1 G1 retarget CSVs at the frozen revision.

    python scripts/fetch_lafan1.py                  # uses hf-mirror
    python scripts/fetch_lafan1.py --base https://huggingface.co

Writes ``data/lafan1_g1/<clip>.csv`` plus the ``data/lafan1_g1.sha256.json``
manifest that ``benchmark/evaluator/manifest.py`` freezes. Files are not
committed to git; the hashes are.

Source: HuggingFace ``lvhaidong/LAFAN1_Retargeting_Dataset``, ``g1/`` subtree,
revision ``07ff8845a94d59d4195a8ac04fe918d4e6338239`` (see
``constants.LAFAN1_HF_REVISION``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator.protocol import protocol_clip_ids  # noqa: E402

DEFAULT_BASE = "https://hf-mirror.com/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve"


def download(url: str, destination: Path, retries: int = 3) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                destination.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries}: {exc}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", default=str(C.DATA_DIR))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    clips = protocol_clip_ids()
    for index, clip in enumerate(clips, start=1):
        destination = out / f"{clip}.csv"
        if not destination.exists():
            download(f"{args.base}/{C.LAFAN1_HF_REVISION}/g1/{clip}.csv", destination)
        manifest[clip] = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(
            f"{index:2d}/{len(clips)} {clip:24s} {destination.stat().st_size:8d} "
            f"{manifest[clip][:12]}",
            flush=True,
        )

    manifest_path = out.parent / "lafan1_g1.sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
