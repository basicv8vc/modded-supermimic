#!/usr/bin/env python3
"""P2: per-clip numerical parity between two V2 records (serial vs fast).

Compares the fields the correctness gate names: per-clip reward (normalized
return), termination (failed / fall_step / fall_terms), alive steps, and the
aggregate return.  No threshold is loosened here -- the report shows the raw
deltas and an exact-match flag; tolerances are explicit CLI inputs.

    python scripts/v2/compare_evals.py --a serial.json --b parallel.json \\
        --out reports/v2/parity_serial_vs_parallel.json \\
        --markdown reports/v2/parity_serial_vs_parallel.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="reference record (e.g. serial)")
    ap.add_argument("--b", required=True, help="test record (e.g. fast/parallel)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--tol-return", type=float, default=0.0,
                    help="max |Δ normalized_return| per clip (default exact)")
    ap.add_argument("--tol-terms", type=float, default=1e-9,
                    help="max |Δ mean_term_score| per term")
    args = ap.parse_args(argv)

    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    rows_a = {r["clip"]: r for r in a["checkpoints"][0]["per_clip"]}
    rows_b = {r["clip"]: r for r in b["checkpoints"][0]["per_clip"]}
    diags_a = a["evaluation"].get("per_clip_diagnostics", {})
    diags_b = b["evaluation"].get("per_clip_diagnostics", {})

    clips = sorted(set(rows_a) | set(rows_b))
    per_clip = []
    n_fail = 0
    max_ret = 0.0
    max_term = 0.0
    for clip in clips:
        ra, rb = rows_a.get(clip), rows_b.get(clip)
        if ra is None or rb is None:
            per_clip.append({"clip": clip, "missing_in": "b" if ra is None else "a"})
            n_fail += 1
            continue
        da, db = diags_a.get(clip, {}), diags_b.get(clip, {})
        d_ret = abs(float(ra["normalized_return"]) - float(rb["normalized_return"]))
        term_diff = 0.0
        ta, tb = da.get("mean_term_score", {}), db.get("mean_term_score", {})
        for k in set(ta) | set(tb):
            term_diff = max(term_diff, abs(float(ta.get(k, 0.0)) - float(tb.get(k, 0.0))))
        same_failed = bool(ra["failed"]) == bool(rb["failed"])
        same_alive = int(da.get("alive_steps", -1)) == int(db.get("alive_steps", -2))
        same_fall = da.get("fall_step") == db.get("fall_step")
        same_terms = list(da.get("fall_terms", [])) == list(db.get("fall_terms", []))
        ok = (d_ret <= args.tol_return and term_diff <= args.tol_terms
              and same_failed and same_alive and same_fall and same_terms)
        if not ok:
            n_fail += 1
        max_ret = max(max_ret, d_ret)
        max_term = max(max_term, term_diff)
        per_clip.append({
            "clip": clip,
            "return_a": ra["normalized_return"], "return_b": rb["normalized_return"],
            "abs_return_diff": d_ret,
            "failed_a": ra["failed"], "failed_b": rb["failed"],
            "alive_steps_a": da.get("alive_steps"), "alive_steps_b": db.get("alive_steps"),
            "fall_step_a": da.get("fall_step"), "fall_step_b": db.get("fall_step"),
            "fall_terms_a": da.get("fall_terms"), "fall_terms_b": db.get("fall_terms"),
            "max_abs_term_diff": term_diff,
            "match": ok,
        })

    agg_a = float(a["evaluation"]["aggregate_normalized_return"])
    agg_b = float(b["evaluation"]["aggregate_normalized_return"])
    verdict = "PASS" if n_fail == 0 and abs(agg_a - agg_b) <= args.tol_return else "FAIL"
    report = {
        "a": {"path": args.a, "algorithm": a.get("algorithm"),
              "workers": a.get("evaluation", {}).get("fast_eval", {}).get("workers", "serial")},
        "b": {"path": args.b, "algorithm": b.get("algorithm"),
              "workers": b.get("evaluation", {}).get("fast_eval", {}).get("workers", "serial")},
        "tol_return": args.tol_return, "tol_terms": args.tol_terms,
        "clips_compared": len(clips),
        "clips_mismatched": n_fail,
        "max_abs_return_diff": max_ret,
        "max_abs_term_diff": max_term,
        "aggregate_return_a": agg_a, "aggregate_return_b": agg_b,
        "abs_aggregate_diff": abs(agg_a - agg_b),
        "verdict": verdict,
        "per_clip": per_clip,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Serial vs fast evaluation — numerical parity", "",
            f"- A: `{args.a}` ({report['a']['workers']})",
            f"- B: `{args.b}` ({report['b']['workers']})",
            f"- clips compared: {len(clips)}, mismatched: {n_fail}",
            f"- max |Δ return|: {max_ret:.3e}, max |Δ term|: {max_term:.3e}",
            f"- aggregate: A={agg_a:+.6f} B={agg_b:+.6f} |Δ|={abs(agg_a - agg_b):.3e}",
            f"- verdict: **{verdict}**", "",
            "| clip | ret A | ret B | Δ | failed | alive A/B | fall A/B | match |",
            "|---|---:|---:|---:|:---:|:---:|:---:|:---:|",
        ]
        for r in per_clip:
            if "missing_in" in r:
                lines.append(f"| {r['clip']} | - | - | - | - | - | - | MISSING in {r['missing_in']} |")
                continue
            lines.append(
                f"| {r['clip']} | {r['return_a']:+.5f} | {r['return_b']:+.5f} "
                f"| {r['abs_return_diff']:.2e} | {r['failed_a']}/{r['failed_b']} "
                f"| {r['alive_steps_a']}/{r['alive_steps_b']} "
                f"| {r['fall_step_a']}/{r['fall_step_b']} | {'ok' if r['match'] else 'DIFF'} |")
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {md}")
    print(f"verdict={verdict} mismatched={n_fail}/{len(clips)} "
          f"max|Δret|={max_ret:.3e} |Δagg|={abs(agg_a - agg_b):.3e}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
