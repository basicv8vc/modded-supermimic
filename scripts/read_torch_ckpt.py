"""Inventory a ``torch.save`` checkpoint without importing torch.

    python scripts/read_torch_ckpt.py path/to/ckpt.pt
    python scripts/read_torch_ckpt.py path/to/ckpt.pt --heads pi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.evaluator.torchzip import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--heads", default=None, help="only keys with this prefix")
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    tensors = load_checkpoint(args.checkpoint)
    print(f"{args.checkpoint}  ({args.checkpoint.stat().st_size / 1e6:.1f} MB)")
    print(f"{len(tensors)} tensors")
    shown = 0
    for name, arr in tensors.items():
        if args.heads and not name.startswith(args.heads):
            continue
        print(f"  {name:44s} {str(arr.shape):22s} {arr.dtype}")
        shown += 1
        if shown >= args.max:
            print("  ...")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
