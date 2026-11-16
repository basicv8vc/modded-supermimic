"""Reference-motion conversion audit: original NumPy resampler vs mjlab's Torch one.

The original (``src/nablamimic/motion.py::_resample``) and the canonical
(``mjlab.scripts.csv_to_npz.MotionLoader``) resamplers are the same idea -- sample
the 30 Hz clip on a 50 Hz grid, lerp positions and joint angles, slerp the root
quaternion -- but they compute the sample grid in different floating-point
regimes::

    original : np.arange(0.0, (N-1)/30, 0.02)                  # float64 grid
    canonical: torch.arange(0, (N-1)/30.0, 0.02, dtype=float32)

This script separates two questions that the existing documentation conflates:

1. **Can the output length differ at all?** Yes, and only for one residue class.
   With ``duration = (N-1)/30`` and ``step = 1/50``, the exact ratio is
   ``duration/step = (N-1)*5/3``. ``arange`` emits ``ceil(ratio)`` samples when
   the ratio is not an integer, so a rounding difference of one ulp can only
   flip the count when the ratio is *within an ulp of an integer* -- i.e. when
   ``(N-1) % 3 == 0``. For every other N the ratio sits at least 1/3 from an
   integer and no float32/float64 difference can change the ceiling. This is a
   proof, not a measurement, and it holds regardless of which pipeline rounds
   which way.

2. **Which way does it round, and does the extra frame shift the interior?**
   This needs the actual clips and torch. :func:`compare_clips` does that and
   writes a per-clip table. It also checks the thing that actually matters for
   scoring: an extra terminal frame changes ``T`` (the normalisation
   denominator) and the last ``FUTURE`` goal windows, but **not** the interior
   sample times, because both grids start at 0 with the same 0.02 step.

The floor-alignment difference is reported separately
(:func:`check_floor_alignment`): the original shifts every clip's z by the 5th
percentile of the lowest foot point, mjlab's converter does not.

Usage
-----
    # mechanism only, no data needed (always runnable)
    python -m scripts.audit.reference_conversion --mechanism

    # per-clip table (needs the CSVs + `mjlab` installed)
    python -m scripts.audit.reference_conversion --clips data/lafan1_g1 --out reports/audit/reference_conversion.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

FPS_IN, FPS_OUT = 30.0, 50.0
STEP_OUT = 1.0 / FPS_OUT


def numpy_grid_len(n_frames: int) -> int:
    """Length of the original's float64 sample grid."""
    return len(np.arange(0.0, (n_frames - 1) / FPS_IN, STEP_OUT))


def exact_grid_len(n_frames: int) -> int:
    """``ceil(duration/step)`` in exact rational arithmetic."""
    # duration/step = (N-1)/30 / (1/50) = (N-1)*5/3, an exact rational.
    return math.ceil((n_frames - 1) * 5 / 3)


def grid_len_is_ambiguous(n_frames: int) -> bool:
    """True when the count depends on floating-point rounding.

    ``arange`` emits ``ceil(ratio)`` elements when ``ratio`` is not an integer.
    The count can therefore differ by one between two float regimes only when
    ``ratio`` is an integer in exact arithmetic -- otherwise it is at least
    ``1/3`` away from the boundary and every reasonable rounding agrees.
    """
    return ((n_frames - 1) * 5) % 3 == 0


def mechanism_report(n_min: int = 100, n_max: int = 4000) -> dict:
    """Classify every plausible clip length; no data required."""
    ambiguous = [n for n in range(n_min, n_max + 1) if grid_len_is_ambiguous(n)]
    agree = 0
    mismatched = []
    for n in range(n_min, n_max + 1):
        if numpy_grid_len(n) == exact_grid_len(n):
            agree += 1
        else:
            mismatched.append({"n_frames": n, "numpy": numpy_grid_len(n), "exact": exact_grid_len(n)})
    return {
        "claim": "output frame counts can differ only when (N-1) % 3 == 0",
        "n_frames_range": [n_min, n_max],
        "ambiguous_lengths": len(ambiguous),
        "ambiguous_fraction": len(ambiguous) / (n_max - n_min + 1),
        "numpy_matches_exact_ceil": agree,
        "numpy_deviations_from_exact": mismatched[:10],
        "interior_times_identical": True,
        "note": (
            "Both grids are 0, 0.02, 0.04, ...; only the final sample (t == duration) "
            "may be present or absent. Interior reference frames therefore coincide, so "
            "the divergence can only affect the last FUTURE goal windows and T itself."
        ),
    }


# ---------------------------------------------------------------------------
# Data-dependent comparison (needs the CSVs and torch+mjlab)
# ---------------------------------------------------------------------------


def _original_reference(csv_path: Path, upstream: Path) -> dict:
    """Run the original NumPy resampler on one clip (no MuJoCo needed for poses)."""
    import sys

    # ``upstream`` may be the repository root or its ``src`` directory.
    src = upstream / "src" if (upstream / "src/nablamimic").is_dir() else upstream
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from nablamimic.motion import _resample  # noqa: PLC0415

    raw = np.loadtxt(csv_path, delimiter=",")
    out = _resample(raw, FPS_IN, FPS_OUT)  # (T, 3 + 4 + 29) with root quat in wxyz
    return {
        "input_frames": int(raw.shape[0]),
        "n_frames": int(out.shape[0]),
        "pos": out[:, :3],
        "quat": out[:, 3:7],
        "joint": out[:, 7:],
    }


def _canonical_reference(csv_path: Path) -> dict:
    """Run mjlab's Torch resampler only (``MotionLoader``, before the FK loop)."""
    import torch
    from mjlab.scripts.csv_to_npz import MotionLoader  # noqa: PLC0415

    motion = MotionLoader(
        motion_file=str(csv_path), input_fps=int(FPS_IN), output_fps=int(FPS_OUT), device="cpu"
    )
    return {
        "n_frames": int(motion.output_frames),
        "pos": motion.motion_base_poss.cpu().numpy(),
        "quat": motion.motion_base_rots.cpu().numpy(),
        "joint": motion.motion_dof_poss.cpu().numpy(),
    }


def _quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between unit quaternions (wxyz), per row."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def compare_one(csv_path: Path, upstream: Path) -> dict:
    tag = csv_path.stem
    try:
        orig = _original_reference(csv_path, upstream)
        canon = _canonical_reference(csv_path)
    except Exception as exc:  # noqa: BLE001 - the table must degrade, not crash
        return {"clip": tag, "error": f"{type(exc).__name__}: {exc}"}

    n0, n1 = orig["n_frames"], canon["n_frames"]
    n = min(n0, n1)
    # Joint values are those the reference ships; the 30->50 Hz grid is compared
    # on the overlapping prefix, which is where the interior must agree.
    results = {
        "clip": tag,
        "input_frames": orig["input_frames"],
        "original_frames": n0,
        "canonical_frames": n1,
        "frame_delta": n1 - n0,
        "length_ambiguous_by_construction": grid_len_is_ambiguous(orig["input_frames"]),
        "max_joint_pos_error": float(np.abs(orig["joint"][:n] - canon["joint"][:n]).max()),
        "max_root_pos_error": float(np.abs(orig["pos"][:n] - canon["pos"][:n]).max()),
        "max_root_ori_error_rad": float(_quat_angle(orig["quat"][:n], canon["quat"][:n]).max()),
        "affects_final_goal_window": n0 != n1,
    }
    return results


def compare_clips(data_dir: Path, out: Path, upstream: Path) -> dict:
    clips = sorted(p for p in data_dir.glob("*.csv"))
    if not clips:
        raise SystemExit(f"no CSVs in {data_dir}")
    rows = [compare_one(p, upstream) for p in clips]
    ok = [r for r in rows if "error" not in r]
    payload = {
        "n_clips": len(rows),
        "n_compared": len(ok),
        "n_frame_delta_nonzero": int(sum(r["frame_delta"] != 0 for r in ok)),
        "max_abs_joint_error": float(max((r["max_joint_pos_error"] for r in ok), default=0.0)),
        "rows": rows,
        "note": (
            "Frame delta is measured, not inferred. Non-zero deltas must all satisfy "
            "(N-1) % 3 == 0; anything else falsifies the mechanism analysis above."
        ),
    }
    violations = [
        r["clip"]
        for r in ok
        if r["frame_delta"] != 0
        and not grid_len_is_ambiguous(r["input_frames"])
    ]
    payload["mechanism_violations"] = violations
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def check_floor_alignment(canonical_npz_dir: Path, original_cache_dir: Path | None = None) -> dict:
    """Compare the reference root height of both pipelines for the same clip.

    The original subtracts ``percentile(lowest foot point over the clip, 5)`` from
    every z (``motion.py::process_clip``). mjlab's converter has that line
    commented out. If the CSVs are not already floor-aligned, the two references
    differ by a per-clip constant in the root height feature (goal dim 67) and in
    the absolute root position the reward compares against.
    """
    diffs = {}
    for npz in sorted(canonical_npz_dir.glob("*.npz")):
        with np.load(npz) as data:
            z = np.asarray(data["body_pos_w"][:, 0, 2], dtype=np.float64)
        diffs[npz.stem] = {"canonical_root_z_min": float(z.min()), "canonical_root_z_first": float(z[0])}
    return {
        "n_clips": len(diffs),
        "per_clip": diffs,
        "verdict": (
            "Compare canonical_root_z_first against the original's qpos[0,2] for the same "
            "clip. A non-zero constant offset means the canonical reference is not "
            "floor-aligned and the height feature / root-position reward differ by that "
            "constant."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reference_conversion", description=__doc__)
    parser.add_argument("--mechanism", action="store_true", help="run the data-free analysis")
    parser.add_argument("--clips", type=Path, default=None, help="directory of LAFAN1 CSVs")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(os.environ.get("NABLAMIMIC_UPSTREAM", "NablaMimic")),
        help="checkout of sailab-ai/NablaMimic (the original resampler is imported from it)",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/audit/reference_conversion.json"))
    args = parser.parse_args(argv)

    if args.mechanism:
        report = mechanism_report()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "numpy_deviations_from_exact"}, indent=2))
        print(f"wrote {args.out}")
        return 0

    if args.clips is None:
        parser.error("pass --mechanism or --clips <dir>")
    if not (args.upstream / "src" / "nablamimic" / "motion.py").exists():
        parser.error(
            f"--upstream {args.upstream} does not contain src/nablamimic/motion.py "
            "(set NABLAMIMIC_UPSTREAM to the checkout of sailab-ai/NablaMimic)"
        )
    payload = compare_clips(args.clips, args.out, args.upstream / "src")
    print(
        f"{payload['n_compared']} clips compared, "
        f"{payload['n_frame_delta_nonzero']} with a non-zero frame delta, "
        f"max joint error {payload['max_abs_joint_error']:.3e}"
    )
    if payload["mechanism_violations"]:
        print(f"!! mechanism violations: {payload['mechanism_violations']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
