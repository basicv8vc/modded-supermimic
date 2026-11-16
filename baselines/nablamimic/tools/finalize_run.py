#!/usr/bin/env python3
"""Finalize a NablaMimic training run into the standard workflow layout.

Pure stdlib except the optional --live provenance (needs the nablamimic stack to build the
model). Reads a run directory as produced by `python -m nablamimic.train` +
`bash scripts/evaluate_run.sh` and writes:

  manifest.json     standard report record (section 8 of the workflow README)
  provenance.json   model / engine / environment provenance
  summary.json      evaluation summary across checkpoints
  curve.csv         checkpoint time curve (iter, wall_min, full_norm_return, failed_clips)
  report.md         human-readable report incl. the standard table
  evaluation/per_clip.csv    per-clip rows for every evaluated checkpoint
  evaluation/summary.json    copy of summary.json (kept beside the raw eval JSON)
  evaluation/eval_nominal.json  copy of the raw upstream eval output
  checkpoints/      upstream ckpt/ renamed to the standard name

Usage:
  python3 finalize_run.py RUN_DIR [--live] [--git-sha SHA] [--drive-path P] [--run-id ID]

Exit codes: 0 = PASS or INCOMPLETE artifacts written; 2 = FAIL (no usable training log).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

MANIFEST_FIELDS = [
    "experiment_id", "algorithm", "git_sha", "engine", "seed", "gpu", "model",
    "training_time_min", "updates", "normalized_return", "failed_clips",
    "drive_path", "status",
]


def load_jsonl(path: Path):
    recs = []
    if not path.exists():
        return recs
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def ckpt_iters(run: Path):
    for sub in ("checkpoints", "ckpt"):
        d = run / sub
        if d.is_dir():
            its = sorted(int(p.stem.split("_")[1]) for p in d.glob("it_*.pkl"))
            if its:
                return its, sub
    return [], ""


def eval_checkpoints(ev: dict):
    out = {}
    for k, v in ev.items():
        if k.startswith("it") and k.endswith("/nominal") and isinstance(v, dict):
            try:
                out[int(k[2:].split("/")[0])] = v
            except ValueError:
                continue
    return out


# Parameters that describe where/how a run was produced rather than what algorithm ran.
# Two seeds differ only in `seed`; the same config written with a different argument order
# or repeated flags must fingerprint identically.
NON_ALGORITHM_KEYS = {
    "seed", "out", "drive-path", "drive_path", "run-id", "run_id", "git-sha", "git_sha",
    "clock", "device", "log-level", "log_level", "verbose",
}

# These are provenance/serialization fields, not effective training options. The
# recorded config.json already contains every resolved optimizer/reward/model field;
# raw argv can omit defaults and can include superseded arguments.
CONFIG_METADATA_KEYS = {
    "argv", "config_args", "config_file", "segments", "git_commit", "gpu",
}

# The evaluation protocol is fingerprinted separately: a run can share the config and still
# be incomparable if it was scored with different clips/solver settings/actor.
EVAL_PROTOCOL = {
    "script": "scripts/evaluate_run.sh",
    "dataset": "lafan1_g1",
    "clips": 40,
    "actor": "actor_ema",
    "deterministic": True,
    "solver_iterations": 10,
    "fall_policy": "forfeit_rest_of_clip",
    "plan": "first_hour_every_250_updates_then_every_1000_plus_last",
}


def parse_argv(argv) -> dict:
    """Fallback for argv-only fixtures, preserving every value of multi-arg flags."""
    out: dict = {}
    i = 0
    args = [str(a) for a in argv]
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
                out[k.replace("-", "_")] = v
            else:
                k = a[2:].replace("-", "_")
                values = []
                while i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 1
                    values.append(args[i])
                out[k] = values if len(values) > 1 else (values[0] if values else True)
        i += 1
    return out


def canonical_config(cfg: dict) -> dict:
    """Fingerprint *resolved* training options, including arrays and default values.

    train.py's config.json has the complete effective argparse namespace. Prefer
    it to reconstructing options from raw argv: flags such as --term-penalty and
    --action-target consume THREE values, and defaults aren't present in argv.
    The argv-only fallback is for old/synthetic records without a resolved config.
    """
    structured = {k: v for k, v in cfg.items()
                  if k not in CONFIG_METADATA_KEYS and k not in NON_ALGORITHM_KEYS}
    if any(k in structured for k in ("num_envs", "term_penalty", "action_target")):
        return structured
    kv = parse_argv(cfg.get("argv") or cfg.get("config_args") or [])
    for k, v in structured.items():
        kv[k] = v
    kv = {k: v for k, v in kv.items() if k not in NON_ALGORITHM_KEYS}
    return kv


def config_fingerprint(cfg: dict) -> str:
    blob = json.dumps(canonical_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def eval_protocol_fingerprint() -> str:
    blob = json.dumps(EVAL_PROTOCOL, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def eval_integrity(ev_ckpts: dict, budget_iter, required_clips: int = 40):
    """Have we actually got a complete, protocol-conformant evaluation at the budget point?

    Returns (problems, facts). Empty problems means the run may be marked PASS.
    """
    problems = []
    facts = {}
    if not ev_ckpts:
        return ["no evaluated checkpoint"], facts
    if budget_iter is None or budget_iter not in ev_ckpts:
        problems.append(f"budget checkpoint it_{budget_iter} was not evaluated")

    head = ev_ckpts.get(budget_iter) or {}
    clips = head.get("per_clip") or []
    facts["budget_clips"] = len(clips)
    if len(clips) != required_clips:
        problems.append(f"budget checkpoint has {len(clips)}/{required_clips} clips")

    actor = head.get("actor_key")
    facts["actor_key"] = actor
    if actor != EVAL_PROTOCOL["actor"]:
        problems.append(f"actor_key={actor!r}, expected {EVAL_PROTOCOL['actor']!r}")

    ret = head.get("full_norm_return")
    facts["normalized_return"] = ret
    if not isinstance(ret, (int, float)) or isinstance(ret, bool) or not math.isfinite(float(ret)):
        problems.append(f"normalized_return is not a finite number ({ret!r})")
    elif not (0.0 <= float(ret) <= 1.0):
        problems.append(f"normalized_return {ret} outside [0, 1]")

    bad = [c.get("clip") for c in clips
           if not isinstance(c.get("return"), (int, float))
           or isinstance(c.get("return"), bool)
           or not math.isfinite(float(c.get("return")))]
    if bad:
        problems.append(f"{len(bad)} clip rows have non-finite returns ({bad[:3]})")

    missing = required_clips - len({c.get("clip") for c in clips})
    if len(clips) == required_clips and missing > 0:
        problems.append(f"{missing} distinct clips missing (duplicates in per_clip)")
    return problems, facts


def live_provenance(run: Path, git_sha: str) -> dict:
    """Best-effort live provenance: builds the model and counts bodies/joints/geoms/pairs."""
    try:
        from nablamimic.env import EnvConfig, build_model  # type: ignore
        m, kp = build_model(EnvConfig())
        counts = {k: int(getattr(m, k)) for k in ("nbody", "njnt", "ngeom", "npair", "nq", "nv", "nu")}
        import mujoco  # type: ignore
        import jax  # type: ignore
        import warp  # type: ignore
        import nablaengine  # type: ignore
        versions = {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "mujoco": mujoco.__version__,
            "warp": warp.__version__,
            "nablaengine": getattr(nablaengine, "__version__", "?"),
        }
        return {
            "mode": "live",
            "model_counts": counts,
            "test_assertion": "(nq, nv, nu, ngeom, npair) == (36, 35, 29, 69, 502)",
            "versions": versions,
            "git_sha": git_sha,
            "model_source": "Menagerie unitree_g1/scene_mjx.xml @ f3b6013 "
                            "(scripts/setup_menagerie.sh, scripts/menagerie_unitree_g1.sha256)",
            "model_build": "src/nablamimic/robot.py build_spec/build/finish + env.build_model(); "
                           "explicit contact pairs (502), mjlab foot capsules",
        }
    except Exception as e:  # noqa: BLE001 - provenance is best-effort
        return {"mode": "live-fallback", "error": f"{type(e).__name__}: {e}", "git_sha": git_sha}


def config_provenance(cfg: dict, git_sha: str) -> dict:
    return {
        "mode": "config-derived",
        "git_sha": git_sha or cfg.get("git_commit", ""),
        "git_dirty": cfg.get("git_dirty", ""),
        "gpu": cfg.get("gpu", ""),
        "argv": cfg.get("argv", []),
        "seed": cfg.get("seed"),
        "physics": cfg.get("physics"),
        "residual_action": cfg.get("residual_action"),
        "actor_ema": cfg.get("actor_ema"),
        "actor_ema_warmup": cfg.get("actor_ema_warmup"),
        "num_envs": cfg.get("num_envs"),
        "horizon": cfg.get("horizon"),
        "ckpt_every": cfg.get("ckpt_every"),
        "hours_effective": cfg.get("hours"),
        "model_source": "Menagerie unitree_g1/scene_mjx.xml @ f3b6013 "
                        "(scripts/setup_menagerie.sh, scripts/menagerie_unitree_g1.sha256)",
        "evaluation_protocol": "scripts/evaluate_run.sh: eval_plan checkpoints (first hour every "
                               "250 updates, then every 1000, plus last), 40 LAFAN1 clips once "
                               "each from frame 0, deterministic EMA actor, MuJoCo Warp 10 "
                               "solver iterations, fall forfeits the rest of the clip",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--live", action="store_true", help="build the model for provenance")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--drive-path", default="")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    if not run.is_dir():
        print(f"FAIL: {run} is not a directory", file=sys.stderr)
        return 2
    run_id = args.run_id or run.name

    recs = load_jsonl(run / "log.jsonl")
    cfg = {}
    cfg_path = run / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(errors="replace"))
    ev = {}
    # `collect` writes the raw eval output to evaluation/; a run directory produced by the
    # pipeline keeps it at the top level. Accept both so a collected archive can be rebuilt.
    ev_src = run / "eval_nominal.json"
    if not ev_src.exists() and (run / "evaluation" / "eval_nominal.json").exists():
        ev_src = run / "evaluation" / "eval_nominal.json"
    if ev_src.exists():
        ev = json.loads(ev_src.read_text(errors="replace"))
    ev_ckpts = eval_checkpoints(ev)
    its, ckpt_sub = ckpt_iters(run)

    wall = {r["iter"]: r.get("wall_s", 0.0) for r in recs}
    last = recs[-1] if recs else {}
    final_iter = int(max([last.get("iter", 0)] + its + list(ev_ckpts)))
    training_time_min = round(float(last.get("wall_s", 0.0)) / 60.0, 2)

    # final evaluated checkpoint = the evaluated one with the largest iteration
    final_eval_iter = max(ev_ckpts) if ev_ckpts else None
    final_eval = ev_ckpts.get(final_eval_iter, {}) if final_eval_iter is not None else {}

    # headline checkpoint = the last evaluated checkpoint at or before the effective
    # training budget ("return at N minutes" reads the within-budget point; an overflow
    # checkpoint saved after the budget is not the headline). Falls back to the earliest
    # evaluated checkpoint when everything is after the budget, else the last one.
    budget_iter = final_eval_iter
    budget_ckpt_iter = None
    hours_eff = cfg.get("hours")
    if isinstance(hours_eff, (int, float)) and hours_eff > 0:
        budget_s = float(hours_eff) * 3600.0
        # The budget point is the last checkpoint SAVED within the training budget. It must
        # not slide backwards to an earlier evaluated checkpoint when the real budget
        # checkpoint failed to evaluate - that would report an easier number as "8 minutes".
        log_within = [int(r.get("iter", 0)) for r in recs
                      if float(r.get("wall_s", 0.0)) <= budget_s]
        max_iter_in_budget = max(log_within) if log_within else None
        if its and max_iter_in_budget is not None:
            cand = [i for i in its if i <= max_iter_in_budget]
            budget_ckpt_iter = max(cand) if cand else min(its)
        if budget_ckpt_iter is not None:
            budget_iter = budget_ckpt_iter
        elif ev_ckpts:
            within = [it for it in ev_ckpts
                      if float(ev_ckpts[it].get("wall_s", wall.get(it, 0.0))) <= budget_s]
            if within:
                budget_iter = max(within)
            else:
                budget_iter = min(ev_ckpts,
                                  key=lambda it: float(ev_ckpts[it].get("wall_s", wall.get(it, 0.0))))
    head_eval = ev_ckpts.get(budget_iter, {}) if budget_iter is not None else {}
    normalized_return = head_eval.get("full_norm_return")
    failed_frac = head_eval.get("failed_frac")
    failed_clips = None
    if failed_frac is not None:
        n_clips = len(head_eval.get("per_clip") or []) or 40
        failed_clips = f"{int(round(failed_frac * n_clips))}/{n_clips}"

    # PASS is a claim about the evidence, so it needs the budget point to be evaluated with
    # the full clip set, the EMA actor and finite numbers - a partial eval must not average
    # itself into a multi-seed result.
    integrity_problems, integrity_facts = eval_integrity(ev_ckpts, budget_iter)
    if not recs:
        status = "FAIL"
    elif integrity_problems:
        status = "INCOMPLETE"
    else:
        status = "PASS"

    # ---- curve.csv ----
    curve_rows = []
    for it in sorted(set(list(ev_ckpts) + its)):
        e = ev_ckpts.get(it, {})
        row_iter = next((r for r in recs if r.get("iter") == it), {})
        wsec = wall.get(it) or e.get("wall_s") or row_iter.get("wall_s", 0.0)
        curve_rows.append({
            "iter": it,
            "wall_min": round(float(wsec) / 60.0, 2),
            "full_norm_return": e.get("full_norm_return", ""),
            "failed_frac": e.get("failed_frac", ""),
            "rew_mean": row_iter.get("rew_mean", ""),
            "term_rate": row_iter.get("term_rate", ""),
        })

    # ---- evaluation/per_clip.csv ----
    eval_dir = run / "evaluation"
    try:
        eval_dir.mkdir(exist_ok=True)
    except FileExistsError:  # some pathlib shims ignore exist_ok
        pass
    with (eval_dir / "per_clip.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "wall_min", "clip", "return", "alive_frac", "fall_s", "cause"])
        for it in sorted(ev_ckpts):
            e = ev_ckpts[it]
            wl = round(float(wall.get(it, e.get("wall_s", 0.0))) / 60.0, 2)
            for c in e.get("per_clip") or []:
                w.writerow([f"it_{it}", wl, c.get("clip"), c.get("return"), c.get("alive_frac"),
                            "" if c.get("fall_s") is None else c.get("fall_s"),
                            "" if c.get("cause") is None else c.get("cause")])

    # ---- summary.json / evaluation/summary.json ----
    summary = {
        "run_id": run_id,
        "seed": cfg.get("seed"),
        "status": status,
        "integrity_problems": integrity_problems,
        "training": {
            "updates": final_iter,
            "training_time_min": training_time_min,
            "iter_s_last": last.get("iter_s"),
            "rew_mean_last": last.get("rew_mean"),
            "term_rate_last": last.get("term_rate"),
        },
        "evaluation": {
            f"it{it}": {
                "wall_min": round(float(wall.get(it, ev_ckpts[it].get("wall_s", 0.0))) / 60.0, 2),
                "full_norm_return": ev_ckpts[it].get("full_norm_return"),
                "failed_frac": ev_ckpts[it].get("failed_frac"),
                "alive_frac": ev_ckpts[it].get("alive_frac"),
                "fall_cause_counts": ev_ckpts[it].get("fall_cause_counts"),
                "actor_key": ev_ckpts[it].get("actor_key"),
            }
            for it in sorted(ev_ckpts)
        },
        "final": {
            "budget_checkpoint": f"it_{budget_iter}" if budget_iter is not None else None,
            "normalized_return": normalized_return,
            "failed_clips": failed_clips,
            "last_evaluated_checkpoint": f"it_{final_eval_iter}" if final_eval_iter is not None else None,
            "last_evaluated_return": final_eval.get("full_norm_return"),
        },
        "protocol": "official nominal LAFAN1 protocol (scripts/evaluate_run.sh; deterministic "
                    "EMA actor; MuJoCo Warp, 10 solver iterations; fall forfeits rest of clip)",
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if ev_src.exists():
        shutil.copyfile(ev_src, eval_dir / "eval_nominal.json")

    # ---- provenance.json ----
    git_sha = args.git_sha or str(cfg.get("git_commit", ""))
    if args.live:
        prov = live_provenance(run, git_sha)
        prov.setdefault("config", config_provenance(cfg, git_sha))
    else:
        prov = config_provenance(cfg, git_sha)
    (run / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str))

    # ---- layout: ckpt/ -> checkpoints/ ----
    if ckpt_sub == "ckpt" and not (run / "checkpoints").exists():
        (run / "ckpt").rename(run / "checkpoints")

    # ---- manifest.json ----
    manifest = {
        "experiment_id": run_id,
        "algorithm": "NablaMimic",
        "git_sha": git_sha,
        "engine": "NablaEngine (nablaengine[jax] @ c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c)",
        "seed": cfg.get("seed"),
        "gpu": cfg.get("gpu") or prov.get("gpu", ""),
        "model": "G1 29-DoF (Menagerie unitree_g1/scene_mjx.xml @ f3b6013, 31 bodies / 502 contact pairs)",
        "training_time_min": training_time_min,
        "updates": final_iter,
        "normalized_return": normalized_return,
        "failed_clips": failed_clips,
        "budget_checkpoint": f"it_{budget_iter}" if budget_iter is not None else None,
        "last_checkpoint": f"it_{final_eval_iter}" if final_eval_iter is not None else None,
        "drive_path": args.drive_path or "",
        "status": status,
        "config_fingerprint": config_fingerprint(cfg),
        "eval_protocol_fingerprint": eval_protocol_fingerprint(),
        "integrity_problems": integrity_problems,
        "integrity_facts": integrity_facts,
        "physics": cfg.get("physics"),
        "residual_action": cfg.get("residual_action"),
        "actor_ema": cfg.get("actor_ema"),
        "hours_effective": cfg.get("hours"),
        "checkpoint_curve": curve_rows,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))

    # ---- curve.csv ----
    with (run / "curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "wall_min", "full_norm_return",
                                          "failed_frac", "rew_mean", "term_rate"])
        w.writeheader()
        w.writerows(curve_rows)

    # ---- report.md ----
    lines = [
        f"# NablaMimic run report — {run_id}",
        "",
        "| 字段 | 内容 |",
        "|---|---|",
    ]
    disp = {
        "experiment_id": run_id, "algorithm": "NablaMimic", "git_sha": git_sha,
        "seed": cfg.get("seed"), "gpu": manifest["gpu"], "engine": manifest["engine"],
        "model": manifest["model"], "training_time_min": training_time_min,
        "updates": final_iter, "normalized_return": normalized_return,
        "failed_clips": failed_clips, "drive_path": manifest["drive_path"] or "(none)",
        "status": status,
    }
    for k in MANIFEST_FIELDS:
        lines.append(f"| {k} | {disp[k]} |")
    lines += ["", "## Checkpoint curve", "",
              "| iter | wall_min | full_norm_return | failed_frac | rew_mean |", "|---:|---:|---:|---:|---:|"]
    for r in curve_rows:
        lines.append(f"| {r['iter']} | {r['wall_min']} | {r['full_norm_return']} | "
                     f"{r['failed_frac']} | {r['rew_mean']} |")
    if ev_ckpts and head_eval.get("per_clip"):
        lines += ["", f"## Per-clip results ({'it_' + str(budget_iter) if budget_iter is not None else 'n/a'}, "
                  f"budget point)", "",
                  "| clip | return | alive_frac | fall_s | cause |", "|---|---:|---:|---:|---|"]
        for c in sorted(head_eval["per_clip"], key=lambda x: x.get("clip", "")):
            fall = "" if c.get("fall_s") is None else f"{c['fall_s']:.1f}"
            lines.append(f"| {c.get('clip')} | {c.get('return')} | {c.get('alive_frac')} | "
                         f"{fall} | {c.get('cause') or ''} |")
    elif not ev_ckpts:
        lines += ["", "> 评估未运行或未产出 eval_nominal.json — 状态 INCOMPLETE。"]
    if integrity_problems:
        lines += ["", "## Integrity problems", ""]
        lines += [f"- {p}" for p in integrity_problems]
        lines += ["", "> 这些问题使该 run 不能标为 PASS，也不会进入多 seed 汇总。"]
    lines += ["", "## Notes", "",
              "- Effective training budget is set by `--hours` only; the algorithm, reward, "
              "model and optimizer parameters are the official `configs/ours.args` values.",
              "- Wall-clock in `log.jsonl` (`wall_s`) counts training only, excluding JIT "
              "compilation and evaluation (matches the paper's timing).",
              "- Evaluation scores the Polyak-averaged (EMA) actor, decay 0.999 with warmup.",
              ""]
    (run / "report.md").write_text("\n".join(lines))

    print(json.dumps({k: disp[k] for k in MANIFEST_FIELDS}, indent=1, ensure_ascii=False, default=str))
    print(f"status={status} artifacts -> {run}")
    return 0 if status != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
