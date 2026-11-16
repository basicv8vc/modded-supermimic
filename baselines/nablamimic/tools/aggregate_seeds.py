#!/usr/bin/env python3
"""Aggregate finalized NablaMimic runs (out/<run_id>/) across seeds.

Refuses to mix runs with different git SHAs, algorithm-config fingerprints, evaluation
protocol fingerprints, models or engines (section 8 of the workflow README: never average
across versions/configs). Seeds are expected to differ; everything else must match.

Series selection is an *exact series match* on the full run-id grammar, never a prefix:
a run id is ``<series>_s<seed>_<UTC timestamp>`` (tools/run_id.sh), e.g.
``ea_s0_20261106T061000Z``.  ``--series ea`` therefore selects ``ea_s0_...`` / ``ea_s1_...``
but never ``ea_verify_s0_...`` (that is the ``ea_verify`` series), and ``--series ea_verify``
selects only the verify family.  Run ids that do not parse (truncated timestamps, trailing
junk, an ID whose embedded seed disagrees with the manifest) are excluded from series
selection with an explicit message - they are never silently averaged in.

Usage:
  python3 aggregate_seeds.py OUT_DIR [OUT_DIR ...]        # dirs containing manifest.json
  python3 aggregate_seeds.py --scan ROOT [--series ea]    # ROOT/*/manifest.json, series-filtered

Prints a mean/std table and writes aggregate.json next to the first OUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Fields that must be identical across seeds. Seeds themselves are expected to differ.
CONSISTENCY_KEYS = (
    "git_sha", "config_fingerprint", "eval_protocol_fingerprint",
    "model", "engine", "algorithm",
)

# <series>_s<seed>_<UTC timestamp>  (tools/run_id.sh: printf '%s_s%s_%s' NAME SEED date)
# The series group is greedy so that a series which itself contains '_s<digits>'
# (e.g. 'ea_s1') wins over a shorter series plus a longer seed.
RUN_ID_RE = re.compile(r"^(?P<series>.+)_s(?P<seed>\d+)_(?P<ts>\d{8}T\d{6}Z)$")
# Same character set run.sh/run_id.sh allow for RUN: letters, digits, '.', '_', '-'.
SERIES_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RunIdError(ValueError):
    """The value is not a <series>_s<seed>_<UTC timestamp> run id."""


def parse_run_id(run_id) -> tuple[str, int]:
    """Split a run id into (series, seed) or raise RunIdError."""
    if not isinstance(run_id, str) or not run_id:
        raise RunIdError("empty run id")
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise RunIdError("not <series>_s<seed>_<UTC timestamp> (e.g. ea_s0_20261106T061000Z)")
    series = m.group("series")
    if not SERIES_RE.match(series):
        raise RunIdError(f"series {series!r} must match [A-Za-z0-9._-]+")
    return series, int(m.group("seed"))


def run_id_of(run) -> str:
    """The run id a manifest claims, falling back to its directory name."""
    rid = run["manifest"].get("experiment_id")
    if isinstance(rid, str) and rid:
        return rid
    return Path(run["dir"]).name


def load_run(d: Path):
    m = json.loads((d / "manifest.json").read_text())
    s = {}
    if (d / "summary.json").exists():
        s = json.loads((d / "summary.json").read_text())
    return {"dir": str(d), "manifest": m, "summary": s}


def select_series(runs, series: str):
    """Keep only runs whose run id is exactly in `series`.

    Returns (kept, excluded) where excluded is [(run_id, reason)]. Every exclusion is
    reported on stderr: an invalid run id must be visible, never silently dropped.
    """
    kept, excluded = [], []
    for r in runs:
        rid = run_id_of(r)
        try:
            got_series, got_seed = parse_run_id(rid)
        except RunIdError as e:
            excluded.append((rid, str(e)))
            continue
        seed = r["manifest"].get("seed")
        if seed is not None:
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                excluded.append((rid, f"manifest seed {seed!r} is not an integer"))
                continue
            if seed_int != got_seed:
                excluded.append((rid, f"run id encodes s{got_seed} but manifest seed is {seed_int}"))
                continue
        if got_series != series:
            continue
        kept.append(r)
    for rid, why in excluded:
        print(f"SKIP: {rid!r} - {why}; excluded from --series selection", file=sys.stderr)
    return kept, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--scan", default="", help="scan ROOT/*/manifest.json")
    ap.add_argument("--series", default="",
                    help="only keep runs of this exact experiment series, parsed from the full "
                         "run id <series>_s<seed>_<UTC timestamp> (e.g. 'ea', not a prefix)")
    ap.add_argument("--out", default="", help="write aggregate.json here (default: first dir)")
    ap.add_argument("--allow-duplicate-seeds", action="store_true",
                    help="aggregate repeated runs of the same seed (run variance, not seed variance)")
    args = ap.parse_args()

    if args.series and not SERIES_RE.match(args.series):
        print(f"--series {args.series!r} is not a valid series name (must match [A-Za-z0-9._-]+)",
              file=sys.stderr)
        return 2

    dirs = [Path(d) for d in args.dirs]
    if args.scan:
        dirs += sorted(Path(args.scan).glob("*/manifest.json"))
        dirs = [p.parent for p in dirs]
    if not dirs:
        print("no run directories given", file=sys.stderr)
        return 2

    runs = [load_run(d) for d in dirs]
    excluded: list[tuple[str, str]] = []
    if args.series:
        # Exact series match. A prefix test would let --series ea pull in ea_verify_s0_...
        # and silently mix two experiment families into one mean.
        runs, excluded = select_series(runs, args.series)
        if not runs:
            print(f"no run matches --series {args.series!r} under the given directories "
                  f"(series matching uses the full run id <series>_s<seed>_<UTC timestamp>)",
                  file=sys.stderr)
            return 2

    # consistency gates: never mix different versions/configs/models/protocols
    keys = CONSISTENCY_KEYS
    ref = runs[0]["manifest"]
    for r in runs[1:]:
        for k in keys:
            got, want = r["manifest"].get(k), ref.get(k)
            if got != want:
                print(f"REFUSE: {r['dir']} has {k}={got!r}, "
                      f"expected {want!r} - do not mix results across versions/configs/protocols",
                      file=sys.stderr)
                if k in ("config_fingerprint", "eval_protocol_fingerprint") and want is None:
                    print("HINT: this manifest predates the fingerprint fields; regenerate it "
                          "with tools/finalize_run.py (./run.sh report) before aggregating",
                          file=sys.stderr)
                return 3

    # a run without a protocol fingerprint is not comparable evidence
    missing_protocol = [r["dir"] for r in runs
                        if not r["manifest"].get("eval_protocol_fingerprint")]
    if len(runs) > 1 and missing_protocol and len(missing_protocol) != len(runs):
        print(f"REFUSE: {len(missing_protocol)} run(s) lack eval_protocol_fingerprint - regenerate "
              f"their manifests with tools/finalize_run.py", file=sys.stderr)
        return 3

    # duplicate seeds are not independent samples: two runs of seed 0 tell you about run-to-run
    # variance, not about seed variance, and the std would be silently wrong
    seeds = [r["manifest"].get("seed") for r in runs]
    dupes = sorted({s for s in seeds if s is not None and seeds.count(s) > 1})
    if dupes and not args.allow_duplicate_seeds:
        print(f"REFUSE: seeds {dupes} appear more than once - these are repeated runs of the "
              f"same seed, not independent samples. Use --allow-duplicate-seeds to aggregate "
              f"them anyway (reported as run variance, not seed variance)", file=sys.stderr)
        return 4

    rows = []
    for r in sorted(runs, key=lambda x: x["manifest"].get("seed") if x["manifest"].get("seed") is not None else -1):
        m = r["manifest"]
        rows.append({
            "run_id": m.get("experiment_id"),
            "seed": m.get("seed"),
            "status": m.get("status"),
            "normalized_return": m.get("normalized_return"),
            "failed_clips": m.get("failed_clips"),
            "training_time_min": m.get("training_time_min"),
            "updates": m.get("updates"),
            "dir": r["dir"],
        })

    rets = [x["normalized_return"] for x in rows
            if x["status"] == "PASS" and isinstance(x["normalized_return"], (int, float))]
    agg = {
        "series": args.series or None,
        "consistency": {k: ref.get(k) for k in keys},
        "per_seed": rows,
        "n_pass": len(rets),
        "unique_seeds": len({x["seed"] for x in rows if x["seed"] is not None}),
        "duplicate_seeds": dupes,
        "excluded_run_ids": [{"run_id": rid, "reason": why} for rid, why in excluded],
        "normalized_return_mean": round(statistics.mean(rets), 4) if rets else None,
        "normalized_return_std": round(statistics.stdev(rets), 4) if len(rets) > 1 else (0.0 if len(rets) == 1 else None),
    }

    print(f"{'seed':>5} {'status':<10} {'return':>8} {'failed':>8} {'train_min':>10} {'updates':>8}  run_id")
    for x in rows:
        print(f"{str(x['seed']):>5} {x['status']:<10} {str(x['normalized_return']):>8} "
              f"{str(x['failed_clips']):>8} {str(x['training_time_min']):>10} {str(x['updates']):>8}"
              f"  {x['run_id']}")
    print(f"\nPASS seeds: {agg['n_pass']}  mean={agg['normalized_return_mean']}  std={agg['normalized_return_std']}")

    out = Path(args.out) if args.out else Path(rows[0]["dir"]) / "aggregate.json"
    out.write_text(json.dumps(agg, indent=1, ensure_ascii=False))
    print(f"aggregate.json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
