#!/usr/bin/env python
"""Compare two rollout-trace dumps from ``scripts/alignment/dump_traces.py``.

Writes a machine-readable report (JSON) plus a human summary. The comparison is
per (clip, action sequence) pair and reports, per category, the *first* timestep
at which the two implementations diverge -- both unconditionally (any non-zero
difference) and beyond a meaningful tolerance.

    python scripts/alignment/compare_traces.py \\
        --left reports/alignment/traces_v1 \\
        --right reports/alignment/traces_mjlab_l1 \\
        --label-left v1-mujoco-C --label-right mjlab-official \\
        --out reports/alignment_v1_vs_mjlab.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATE_TOL = 1e-5  # m / rad; below this a state difference is float noise
REWARD_TOL = 1e-4
TERM_TOL = 1e-5


def load_trace(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = rows[-1]["summary"]
    return rows[:-1], summary


def first_diff(rows_l: list[dict], rows_r: list[dict], getter, tol: float) -> dict:
    """First step where getter(rows) differs by more than tol (or at all if 0)."""
    out = {
        "first_any_diff_step": None,
        "first_over_tol_step": None,
        "max_abs_diff": 0.0,
        "max_abs_diff_step": None,
    }
    for step in range(min(len(rows_l), len(rows_r))):
        a = np.asarray(getter(rows_l[step]), dtype=np.float64)
        b = np.asarray(getter(rows_r[step]), dtype=np.float64)
        if a.shape != b.shape:
            out["first_any_diff_step"] = step
            out["first_over_tol_step"] = step
            out["shape_left"] = list(a.shape)
            out["shape_right"] = list(b.shape)
            return out
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        if d > 0.0 and out["first_any_diff_step"] is None:
            out["first_any_diff_step"] = step
        if d > tol and out["first_over_tol_step"] is None:
            out["first_over_tol_step"] = step
        if d > out["max_abs_diff"]:
            out["max_abs_diff"] = d
            out["max_abs_diff_step"] = step
    return out


def compare_pair(rows_l, rows_r, sum_l, sum_r) -> dict:
    n = min(len(rows_l), len(rows_r))
    result = {
        "steps_left": sum_l["steps_run"],
        "steps_right": sum_r["steps_run"],
        "total_reward_left": sum_l["total_reward"],
        "total_reward_right": sum_r["total_reward"],
        "total_reward_abs_diff": abs(sum_l["total_reward"] - sum_r["total_reward"]),
        "failed_left": sum_l["failed"],
        "failed_right": sum_r["failed"],
        "actions_sha256_left": sum_l["actions_sha256"],
        "actions_sha256_right": sum_r["actions_sha256"],
    }

    # Reference frame index at score time.
    ref_l = [r["ref_index"] for r in rows_l]
    ref_r = [r["ref_index"] for r in rows_r]
    result["ref_index"] = {
        "left": ref_l[: min(6, len(ref_l))],
        "right": ref_r[: min(6, len(ref_r))],
        "constant_offset": (
            sorted(set(r - l for l, r in zip(ref_l[:n], ref_r[:n])))[0]
            if n and len({r - l for l, r in zip(ref_l[:n], ref_r[:n])}) == 1
            else None
        ),
        "first_mismatch_step": next(
            (i for i in range(n) if ref_l[i] != ref_r[i]), None
        ),
    }

    result["reward"] = first_diff(rows_l, rows_r, lambda r: [r["reward"]], REWARD_TOL)
    term_names = sorted(
        set(rows_l[0].get("terms", {})) | set(rows_r[0].get("terms", {}))
    )
    result["terms"] = {
        name: first_diff(
            rows_l,
            rows_r,
            lambda r, name=name: [r.get("terms", {}).get(name, 0.0)],
            TERM_TOL,
        )
        for name in term_names
    }
    state_fields = sorted(
        set(rows_l[0].get("state", {})) | set(rows_r[0].get("state", {}))
    )
    result["state"] = {
        name: first_diff(
            rows_l,
            rows_r,
            lambda r, name=name: r.get("state", {}).get(name, []),
            STATE_TOL,
        )
        for name in state_fields
    }
    input_fields = sorted(
        set(rows_l[0].get("input", {})) | set(rows_r[0].get("input", {}))
    )
    result["input"] = {
        name: first_diff(
            rows_l,
            rows_r,
            lambda r, name=name: r.get("input", {}).get(name, []),
            STATE_TOL,
        )
        for name in input_fields
    }

    term_l = next((r for r in reversed(rows_l) if r.get("terminated")), None)
    term_r = next((r for r in reversed(rows_r) if r.get("terminated")), None)
    result["termination"] = {
        "left_step": None if term_l is None else term_l["step"],
        "right_step": None if term_r is None else term_r["step"],
        "left_terms": [] if term_l is None else term_l["terminated_terms"],
        "right_terms": [] if term_r is None else term_r["terminated_terms"],
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--label-left", default="left")
    ap.add_argument("--label-right", default="right")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    left_dir, right_dir = Path(args.left), Path(args.right)
    pairs = {}
    for left_path in sorted(left_dir.glob("*.jsonl")):
        right_path = right_dir / left_path.name
        if not right_path.exists():
            continue
        rows_l, sum_l = load_trace(left_path)
        rows_r, sum_r = load_trace(right_path)
        pairs[left_path.stem] = compare_pair(rows_l, rows_r, sum_l, sum_r)

    report = {
        "left": {"label": args.label_left, "dir": str(left_dir)},
        "right": {"label": args.label_right, "dir": str(right_dir)},
        "tolerances": {
            "state": STATE_TOL,
            "reward": REWARD_TOL,
            "termination": TERM_TOL,
        },
        "pairs": pairs,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"== {args.label_left}  vs  {args.label_right} ==")
    for name, p in pairs.items():
        print(f"\n--- {name}")
        print(
            f"  steps: {p['steps_left']} vs {p['steps_right']}   "
            f"total_reward: {p['total_reward_left']:.6f} vs {p['total_reward_right']:.6f} "
            f"(d={p['total_reward_abs_diff']:.6f})   "
            f"failed: {p['failed_left']} vs {p['failed_right']}"
        )
        ref = p["ref_index"]
        print(
            f"  ref_index offset: {ref['constant_offset']}  "
            f"first mismatch step: {ref['first_mismatch_step']}"
        )
        term = p["termination"]
        print(
            f"  termination: step {term['left_step']} {term['left_terms']} vs "
            f"step {term['right_step']} {term['right_terms']}"
        )
        r = p["reward"]
        print(
            f"  reward: first-any-diff step {r['first_any_diff_step']}, "
            f"first >{REWARD_TOL} step {r['first_over_tol_step']}, "
            f"max |d| {r['max_abs_diff']:.3e} @ step {r['max_abs_diff_step']}"
        )
        for group in ("state", "input"):
            for fname, d in p[group].items():
                print(
                    f"  {group}.{fname}: first-any-diff step {d['first_any_diff_step']}, "
                    f"first >{STATE_TOL} step {d['first_over_tol_step']}, "
                    f"max |d| {d['max_abs_diff']:.3e} @ step {d['max_abs_diff_step']}"
                )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
