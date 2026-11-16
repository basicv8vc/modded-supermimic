#!/usr/bin/env python3
"""Build the A100 controlled-baseline comparison deliverables.

Consumes two V2 benchmark records produced by ``scripts/v2/run_eval.py`` and
emits, atomically:

* ``reports/v2/a100_comparison_seed0.json`` -- per-clip + aggregates + integrity
* ``reports/v2/a100_comparison_seed0.csv``  -- per-clip table (task §10)
* ``docs/V2_A100_BASELINE_COMPARISON.md``   -- the human-readable report

The comparison never edits scores and never re-labels an A100 record as a
target-hardware (RTX PRO 6000 / sm_120) submission: both inputs must carry
``report_scope=controlled_baseline_non_target_hardware`` with
``submission_eligible=false`` and ``comparison_valid=true``.

    python scripts/v2/build_a100_comparison.py \
        reports/v2/a100_supermimic_seed0.json \
        reports/v2/a100_nablamimic_seed0.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.v2 import spec as S  # noqa: E402

SM_LABEL = "SuperMimic-vibed"
NM_LABEL = "NablaMimic"

# Historical context (task §10) -- fixed background values, not recomputed.
HISTORY = [
    ("SuperMimic-vibed", "约 0.84 / 0 failed", "0.2489 / 32 failed"),
    ("NablaMimic", "0.8345 / 0 failed", "0.4819 / 16 failed"),
]

CATEGORY_RULES = (
    ("walking", ("walk",)),
    ("running", ("run", "sprint")),
    ("dance", ("dance",)),
    ("fallAndGetUp", ("fallAndGetUp",)),
    ("fightAndSports", ("fightAndSports", "fight")),
    ("jumps", ("jumps",)),
)


def clip_category(clip: str) -> str:
    for name, prefixes in CATEGORY_RULES:
        for p in prefixes:
            if clip.startswith(p):
                return name
    return "other"


def clip_rows(record: dict) -> dict[str, dict]:
    """clip -> {normalized_return, failed, steps, alive_steps, ...}."""
    out: dict[str, dict] = {}
    diag = record["evaluation"].get("per_clip_diagnostics", {})
    for row in record["checkpoints"][0]["per_clip"]:
        d = diag.get(row["clip"], {})
        out[row["clip"]] = {
            "clip": row["clip"],
            "normalized_return": float(row["normalized_return"]),
            "failed": bool(row["failed"]),
            "steps": int(d.get("steps", 0)),
            "alive_steps": int(d.get("alive_steps", 0)),
            "fall_step": d.get("fall_step"),
            "fall_terms": list(d.get("fall_terms") or []),
        }
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def check_integrity(sm: dict, nm: dict, sm_rows: dict, nm_rows: dict) -> dict:
    problems = []

    def clips_of(rows):
        return sorted(rows)

    for label, rec, rows in ((SM_LABEL, sm, sm_rows), (NM_LABEL, nm, nm_rows)):
        ids = [r["clip"] for r in rec["checkpoints"][0]["per_clip"]]
        if len(ids) != 40:
            problems.append(f"{label}: {len(ids)} clips recorded, expected 40")
        if len(set(ids)) != len(ids):
            problems.append(f"{label}: duplicate clip ids present")
        proto = S.load_protocol_v2()
        missing = set(proto["clip_ids"]) - set(ids)
        extra = set(ids) - set(proto["clip_ids"])
        if missing:
            problems.append(f"{label}: missing clips {sorted(missing)}")
        if extra:
            problems.append(f"{label}: unexpected clips {sorted(extra)}")
        for row in rows.values():
            v = row["normalized_return"]
            if not math.isfinite(v):
                problems.append(f"{label}/{row['clip']}: non-finite return {v}")
        agg = rec["evaluation"]["aggregate_normalized_return"]
        recomputed = S.aggregate(r["normalized_return"] for r in rows.values())
        if abs(agg - recomputed) > 1e-9:
            problems.append(f"{label}: aggregation mismatch {agg} vs {recomputed}")
        n_failed = rec["evaluation"]["failed_clips"]
        if isinstance(n_failed, dict):
            n_failed = n_failed.get("count", sum(r["failed"] for r in rows.values()))
        if int(n_failed) != sum(r["failed"] for r in rows.values()):
            problems.append(f"{label}: failed_clips={n_failed} inconsistent with per-clip rows")

    def ok_scope(rec):
        return (rec.get("report_scope") == "controlled_baseline_non_target_hardware"
                and rec.get("submission_eligible") is False
                and rec.get("comparison_valid") is True
                and rec.get("target_hardware_met") is False)

    for label, rec in ((SM_LABEL, sm), (NM_LABEL, nm)):
        if not ok_scope(rec):
            problems.append(f"{label}: record is not marked as A100 controlled baseline")

    shared_keys = ("protocol_id",)
    for k in shared_keys:
        if sm.get(k) != nm.get(k):
            problems.append(f"protocol mismatch on {k}: {sm.get(k)} vs {nm.get(k)}")
    for k in ("protocol_sha256", "mjlab_commit", "mjlab_version",
              "robot_model_sha256", "dataset_manifest_sha256"):
        if sm["evaluator"].get(k) != nm["evaluator"].get(k):
            problems.append(f"runtime mismatch on evaluator.{k}")
    for k in ("gpu_name", "gpu_compute_capability", "actual_simulation_backend"):
        if sm["hardware"].get(k) != nm["hardware"].get(k):
            problems.append(f"runtime mismatch on hardware.{k}")
    if sm.get("seed") != nm.get("seed"):
        problems.append(f"seed mismatch: {sm.get('seed')} vs {nm.get('seed')}")

    return {
        "clips_exactly_40": not any("clips recorded" in p or "duplicate" in p or "missing" in p or "unexpected" in p for p in problems),
        "no_nonfinite_returns": not any("non-finite" in p for p in problems),
        "aggregation_consistent": not any("aggregation" in p or "failed_clips=" in p for p in problems),
        "same_protocol_and_runtime": not any(("protocol mismatch" in p) or ("runtime mismatch" in p) for p in problems),
        "a100_scope_markers": not any("controlled baseline" in p for p in problems),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }


def build(sm_path: Path, nm_path: Path) -> dict:
    sm = json.loads(sm_path.read_text())
    nm = json.loads(nm_path.read_text())
    sm_rows = clip_rows(sm)
    nm_rows = clip_rows(nm)

    clips = sorted(set(sm_rows) | set(nm_rows))
    per_clip = []
    for c in clips:
        s, n = sm_rows.get(c), nm_rows.get(c)
        per_clip.append({
            "clip": c,
            "category": clip_category(c),
            "supermimic": s["normalized_return"] if s else None,
            "nablamimic": n["normalized_return"] if n else None,
            "delta_return_nablamimic_minus_supermimic": (
                (n["normalized_return"] - s["normalized_return"]) if (s and n) else None),
            "failed_supermimic": s["failed"] if s else None,
            "failed_nablamimic": n["failed"] if n else None,
            "alive_fraction_supermimic": (s["alive_steps"] / s["steps"]) if (s and s["steps"]) else None,
            "alive_fraction_nablamimic": (n["alive_steps"] / n["steps"]) if (n and n["steps"]) else None,
            "termination_step_supermimic": s["fall_step"] if s else None,
            "termination_step_nablamimic": n["fall_step"] if n else None,
            "termination_terms_supermimic": s["fall_terms"] if s else [],
            "termination_terms_nablamimic": n["fall_terms"] if n else [],
        })

    sm_mean = mean(r["supermimic"] for r in per_clip)
    nm_mean = mean(r["nablamimic"] for r in per_clip)
    nm_wins = sum(1 for r in per_clip if r["nablamimic"] > r["supermimic"])
    sm_wins = sum(1 for r in per_clip if r["supermimic"] > r["nablamimic"])
    ties = sum(1 for r in per_clip if r["supermimic"] == r["nablamimic"])
    sm_failed = sum(1 for r in per_clip if r["failed_supermimic"])
    nm_failed = sum(1 for r in per_clip if r["failed_nablamimic"])

    by_cat: dict[str, list] = {}
    for r in per_clip:
        by_cat.setdefault(r["category"], []).append(r)
    categories = {
        cat: {
            "clips": len(rs),
            "supermimic_mean": mean(r["supermimic"] for r in rs),
            "nablamimic_mean": mean(r["nablamimic"] for r in rs),
            "supermimic_failed": sum(1 for r in rs if r["failed_supermimic"]),
            "nablamimic_failed": sum(1 for r in rs if r["failed_nablamimic"]),
        }
        for cat, rs in sorted(by_cat.items())
    }

    def fps_summary(rec):
        perf = rec["evaluation"].get("performance", {})
        vals = [p.get("steady_simulation_fps") for p in perf.values()
                if isinstance(p, dict) and p.get("steady_simulation_fps")]
        return mean(vals) if vals else None

    def per_term(rec):
        diag = rec["evaluation"].get("per_clip_diagnostics", {})
        acc: dict[str, float] = {}
        for d in diag.values():
            for k, v in (d.get("mean_term_score") or {}).items():
                acc[k] = acc.get(k, 0.0) + float(v)
        n = max(len(diag), 1)
        return {k: v / n for k, v in sorted(acc.items())}

    integrity = check_integrity(sm, nm, sm_rows, nm_rows)

    return {
        "record_type": "v2_a100_controlled_comparison",
        "protocol_id": sm.get("protocol_id"),
        "protocol_sha256": sm["evaluator"]["protocol_sha256"],
        "seed": sm.get("seed"),
        "hardware": {
            "gpu_name": sm["hardware"].get("gpu_name"),
            "gpu_compute_capability": sm["hardware"].get("gpu_compute_capability"),
        },
        "report_scope": "controlled_baseline_non_target_hardware",
        "target_hardware_met": False,
        "submission_eligible": False,
        "comparison_valid": True,
        "checkpoints": {
            SM_LABEL: {
                "checkpoint_sha256": sm["checkpoints"][0]["checkpoint_sha256"],
                "training_commit": sm.get("training_commit"),
                "training_wall_s": sm["checkpoints"][0]["training_wall_s"],
                "eval_wall_s": sm["checkpoints"][0].get("eval_wall_s"),
                "source_record": sm_path.name,
            },
            NM_LABEL: {
                "checkpoint_sha256": nm["checkpoints"][0]["checkpoint_sha256"],
                "training_commit": nm.get("training_commit"),
                "training_wall_s": nm["checkpoints"][0]["training_wall_s"],
                "eval_wall_s": nm["checkpoints"][0].get("eval_wall_s"),
                "source_record": nm_path.name,
            },
        },
        "aggregate": {
            "supermimic_mean_normalized_return": sm_mean,
            "nablamimic_mean_normalized_return": nm_mean,
            "delta_mean_return": nm_mean - sm_mean,
            "supermimic_failed_clips_of_40": sm_failed,
            "nablamimic_failed_clips_of_40": nm_failed,
            "failure_count_difference_supermimic_minus_nablamimic": sm_failed - nm_failed,
            "supermimic_mean_alive_fraction": mean(r["alive_fraction_supermimic"] for r in per_clip),
            "nablamimic_mean_alive_fraction": mean(r["alive_fraction_nablamimic"] for r in per_clip),
            "nablamimic_clips_won": nm_wins,
            "supermimic_clips_won": sm_wins,
            "ties": ties,
            "supermimic_evaluation_wall_s": sm["checkpoints"][0].get("eval_wall_s"),
            "nablamimic_evaluation_wall_s": nm["checkpoints"][0].get("eval_wall_s"),
            "supermimic_steady_simulation_fps": fps_summary(sm),
            "nablamimic_steady_simulation_fps": fps_summary(nm),
            "supermimic_mean_reward_term_contributions": per_term(sm),
            "nablamimic_mean_reward_term_contributions": per_term(nm),
        },
        "by_category": categories,
        "per_clip": per_clip,
        "historical_context": {
            "table": [
                {"model": m, "original_evaluation": o, "v1_historical": v, "v2_a100": None}
                for m, o, v in HISTORY
            ],
            "caveats": [
                "V1 NablaMimic historical score also suffered the pre-fix adapter time-index bug.",
                "V1 and V2 scores are not directly comparable (different protocol).",
                "Original evaluations used different physics than V2.",
                "Only the two V2 A100 scores on this page share one GPU + physics + reward + termination configuration.",
            ],
        },
        "integrity": integrity,
    }


def md_report(cmp: dict) -> str:
    a = cmp["aggregate"]
    lines = []
    w = lines.append
    w("# V2 A100 Controlled Baseline — SuperMimic-vibed vs NablaMimic")
    w("")
    w(f"- Protocol: `{cmp['protocol_id']}` / SHA `{cmp['protocol_sha256']}`")
    w(f"- Hardware: {cmp['hardware']['gpu_name']} ({cmp['hardware']['gpu_compute_capability']}), "
      "target GPU gate **not** met — controlled baseline, not a leaderboard submission")
    w(f"- Seed: {cmp['seed']} · 40 LAFAN1 G1 clips · no return clamping")
    w("")
    w("## Headline")
    w("")
    w("| Metric | SuperMimic-vibed | NablaMimic |")
    w("|---|---:|---:|")
    sm, nm = cmp["checkpoints"]["SuperMimic-vibed"], cmp["checkpoints"]["NablaMimic"]
    w(f"| Training wall time | {sm['training_wall_s']:.1f}s | {nm['training_wall_s']:.1f}s |")
    w(f"| Mean normalized return | {a['supermimic_mean_normalized_return']:.4f} | {a['nablamimic_mean_normalized_return']:.4f} |")
    w(f"| Failed clips | {a['supermimic_failed_clips_of_40']} / 40 | {a['nablamimic_failed_clips_of_40']} / 40 |")
    w(f"| Mean alive fraction | {a['supermimic_mean_alive_fraction']:.4f} | {a['nablamimic_mean_alive_fraction']:.4f} |")
    w(f"| Clips won | {a['supermimic_clips_won']} | {a['nablamimic_clips_won']} |")
    w(f"| Evaluation wall time | {a['supermimic_evaluation_wall_s']:.1f}s | {a['nablamimic_evaluation_wall_s']:.1f}s |")
    w(f"| Steady-state FPS | {a['supermimic_steady_simulation_fps']:.1f} | {a['nablamimic_steady_simulation_fps']:.1f} |")
    w("")
    w(f"Δ mean return (NablaMimic − SuperMimic) = **{a['delta_mean_return']:+.4f}** · "
      f"ties = {a['ties']}")
    w("")
    w("## Per-clip")
    w("")
    w("| Clip | SuperMimic | NablaMimic | Δ Return | Failed SM | Failed NM |")
    w("|---|---:|---:|---:|---|---|")
    for r in cmp["per_clip"]:
        w("| {clip} | {supermimic:.4f} | {nablamimic:.4f} | {d:+.4f} | {fsm} | {fnm} |".format(
            clip=r["clip"], supermimic=r["supermimic"], nablamimic=r["nablamimic"],
            d=r["delta_return_nablamimic_minus_supermimic"],
            fsm="FAILED" if r["failed_supermimic"] else "ok",
            fnm="FAILED" if r["failed_nablamimic"] else "ok"))
    w("")
    w("## By motion category")
    w("")
    w("| Category | Clips | SM mean | NM mean | SM failed | NM failed |")
    w("|---|---:|---:|---:|---:|---:|")
    for cat, d in cmp["by_category"].items():
        w(f"| {cat} | {d['clips']} | {d['supermimic_mean']:.4f} | {d['nablamimic_mean']:.4f} "
          f"| {d['supermimic_failed']} | {d['nablamimic_failed']} |")
    w("")
    w("## Historical context")
    w("")
    w("| Model | Original Evaluation | V1 Historical | V2 A100 |")
    w("|---|---|---|---|")
    hist = cmp["historical_context"]["table"]
    v2_vals = {
        "SuperMimic-vibed": f"{a['supermimic_mean_normalized_return']:.4f} / {a['supermimic_failed_clips_of_40']} failed",
        "NablaMimic": f"{a['nablamimic_mean_normalized_return']:.4f} / {a['nablamimic_failed_clips_of_40']} failed",
    }
    for row in hist:
        w(f"| {row['model']} | {row['original_evaluation']} | {row['v1_historical']} | {v2_vals[row['model']]} |")
    w("")
    for c in cmp["historical_context"]["caveats"]:
        w(f"- {c}")
    w("")
    w("## Integrity")
    w("")
    it = cmp["integrity"]
    w(f"- Status: **{it['status']}**")
    w(f"- Exactly 40 clips, no duplicates/missing: {it['clips_exactly_40']}")
    w(f"- No NaN/Inf returns: {it['no_nonfinite_returns']}")
    w(f"- Aggregation matches per-clip rows: {it['aggregation_consistent']}")
    w(f"- Identical protocol + runtime across models: {it['same_protocol_and_runtime']}")
    w(f"- A100 scope markers (submission_eligible=false): {it['a100_scope_markers']}")
    if it["problems"]:
        w("")
        w("Problems:")
        for p in it["problems"]:
            w(f"- {p}")
    w("")
    w("## Checkpoint provenance")
    w("")
    w(f"- SuperMimic: `{sm['checkpoint_sha256']}` (training commit `{sm['training_commit']}`)")
    w(f"- NablaMimic: `{nm['checkpoint_sha256']}` (training commit `{nm['training_commit']}`)")
    w("")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    sm_path, nm_path = Path(argv[0]), Path(argv[1])
    cmp = build(sm_path, nm_path)

    out_json = REPO / "reports/v2/a100_comparison_seed0.json"
    out_csv = REPO / "reports/v2/a100_comparison_seed0.csv"
    out_md = REPO / "docs/V2_A100_BASELINE_COMPARISON.md"

    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cmp, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_json)

    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Clip", "SuperMimic", "NablaMimic", "DeltaReturn",
                    "FailedSM", "FailedNM", "Category"])
        for r in cmp["per_clip"]:
            w.writerow([r["clip"], f"{r['supermimic']:.6f}", f"{r['nablamimic']:.6f}",
                        f"{r['delta_return_nablamimic_minus_supermimic']:+.6f}",
                        "FAILED" if r["failed_supermimic"] else "ok",
                        "FAILED" if r["failed_nablamimic"] else "ok",
                        r["category"]])
    tmp.replace(out_csv)

    tmp = out_md.with_suffix(".md.tmp")
    tmp.write_text(md_report(cmp), encoding="utf-8")
    tmp.replace(out_md)

    print(json.dumps(cmp["aggregate"], indent=2))
    print("integrity:", cmp["integrity"]["status"])
    print("wrote:", out_json, out_csv, out_md)
    return 0 if cmp["integrity"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
