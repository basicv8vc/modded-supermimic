#!/usr/bin/env python
"""Layer-2 data-path comparison: v1 numpy/mujoco-C FK vs official mjlab FK.

For each clip, build the 50 Hz reference three ways and diff every array:

1. ``cache``   -- the npz currently in ``data/lafan1_g1_npz`` (provenance as-is);
2. ``v1``      -- the pre-refactor converter (git HEAD ``motion.py``: numpy
                  resampling + ``mj_forward``/``xpos`` FK), run from a snapshot;
3. ``official``-- the refactored converter (``OfficialConverter``: mjlab
                  ``MotionLoader`` resampling + ``Scene``/``Simulation`` FK).

This quantifies the FK/resampling data-path delta between the historical
implementation and the official one, without touching the cache or the physics.
The v1 converter is extracted from git history (the pre-refactor tree) so no
second implementation is maintained in the working tree.

    python scripts/alignment/compare_fk_paths.py --clips fallAndGetUp3_subject1 walk4_subject1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.evaluator import spec  # noqa: E402
from benchmark.evaluator import motion as official_motion  # noqa: E402

# The last commit whose tree still carries the pre-refactor motion.py.
V1_REV = "ac97315"
V1_MOTION_IN_REPO = "research/modded-supermimic/benchmark/evaluator/motion.py"

ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def load_v1_module(tmp_dir: str):
    """Import the pre-refactor motion.py snapshot from git history."""
    blob = subprocess.check_output(
        ["git", "show", f"{V1_REV}:{V1_MOTION_IN_REPO}"],
        cwd=REPO_ROOT.parent.parent,
    )
    path = Path(tmp_dir) / "v1_motion_snapshot.py"
    path.write_bytes(blob)
    module_name = "benchmark.evaluator._v1_motion_snapshot"
    spec_ = importlib.util.spec_from_file_location(module_name, path)
    if spec_ is None or spec_.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec_)
    mod.__package__ = "benchmark.evaluator"
    sys.modules[module_name] = mod
    spec_.loader.exec_module(mod)
    return mod


def arrays_of(ref) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(ref, name)) for name in ARRAYS}


def diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict:
    out = {}
    for name in ARRAYS:
        x, y = a[name], b[name]
        if x.shape != y.shape:
            out[name] = {"shape_a": list(x.shape), "shape_b": list(y.shape)}
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        out[name] = {
            "max_abs": float(d.max()) if d.size else 0.0,
            "mean_abs": float(d.mean()) if d.size else 0.0,
            "first_frame_max_abs": float(d[0].max()) if d.size else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", nargs="+", default=["fallAndGetUp3_subject1", "walk4_subject1"])
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "alignment" / "fk_paths.json"))
    args = ap.parse_args()

    v1 = load_v1_module(tempfile.mkdtemp(prefix="v1_snapshot_"))
    info = spec.build_model()
    report = {"clips": {}}

    for clip in args.clips:
        csv_path = REPO_ROOT / "data" / "lafan1_g1" / f"{clip}.csv"
        cache_path = official_motion.NPZ_CACHE_DIR / f"{clip}.npz"

        ref_v1 = v1.load_reference(csv_path, info, clip_id=clip)
        with tempfile.TemporaryDirectory() as td:
            tmp_npz = Path(td) / f"{clip}.npz"
            ref_off = official_motion._get_converter().convert(csv_path, tmp_npz)

        entry = {
            "num_frames": {
                "v1": int(ref_v1.num_frames),
                "official": int(ref_off.num_frames),
            }
        }
        entry["v1_vs_official"] = diff(arrays_of(ref_v1), arrays_of(ref_off))

        if cache_path.exists():
            ref_cache = official_motion.load_reference_npz(cache_path)
            entry["num_frames"]["cache"] = int(ref_cache.num_frames)
            entry["cache_vs_v1"] = diff(arrays_of(ref_cache), arrays_of(ref_v1))
            entry["cache_vs_official"] = diff(arrays_of(ref_cache), arrays_of(ref_off))
        else:
            entry["cache"] = "missing"

        report["clips"][clip] = entry
        print(f"\n=== {clip}")
        for pair, rows in entry.items():
            if not isinstance(rows, dict) or "max_abs" not in str(rows):
                continue
            for name, stats in rows.items():
                if "max_abs" in stats:
                    print(f"  [{pair}] {name:15s} max|d|={stats['max_abs']:.3e}")
                else:
                    print(f"  [{pair}] {name:15s} SHAPE MISMATCH {stats}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
