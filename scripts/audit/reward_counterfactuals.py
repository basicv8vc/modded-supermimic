"""Reward / termination decomposition of an existing canonical evaluation record.

This is a *read-only diagnostic*. It never re-runs the simulator and never
touches the official record: it reads the per-term decomposition the runner
already stored (``evaluation.per_clip_diagnostics[*].mean_term_score``) and
answers "what would this clip have scored if term X were different?".

Why that is exact for some counterfactuals and not others
--------------------------------------------------------
The evaluator's loop is::

    action = policy.act(state)        # depends on the robot state, not the reward
    result = env.step(action)         # physics; independent of reward
    total += result.reward            # only *after* the step
    if result.terminated: stop

The reward never feeds back into the trajectory. So a counterfactual that only
changes **reward weights** leaves every ``qpos``/``qvel``/action and every
termination instant bit-identical, and its effect is an exact linear
re-weighting of the stored per-term sums. Those are marked ``exact``.

A counterfactual that changes a **termination condition** (e.g. dropping
``ee_body_pos``) changes *when the rollout stops*, hence which steps are scored
at all. That cannot be recovered from a record: it needs a replay. Those are
marked ``requires_resim`` and are reported as a projection under a stated
assumption, never as a measurement.

Definitions (``runner.py``)
---------------------------
    normalized_return = sum_alive_steps(sum_terms) / ((T - 1) * 0.1)
    mean_term_score   = sum_alive_steps(term)        / alive_steps

so ``sum_alive_steps(term) = mean_term_score * alive_steps`` and the identity
``raw == sum(mean_term_score) * alive_steps / ((T-1) * 0.1)`` is verifiable.
This script asserts it on every clip before computing anything.

Usage
-----
    python -m scripts.audit.reward_counterfactuals \
        --record reports/nablamimic_canonical_eval_mjlab.json \
        --out reports/audit/reward_counterfactuals.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DENOM_PER_STEP_ENV = "MODDED_SUPERMIMIC_MAX_STEP_REWARD"  # 0.1, from constants.MAX_STEP_REWARD

# mjlab term -> (role, canonical weight, original NablaMimic weight or None)
TERMS = {
    "motion_global_root_pos": ("positive", 0.5, 0.5),
    "motion_global_root_ori": ("positive", 0.5, 0.5),
    "motion_body_pos": ("positive", 1.0, 1.0),
    "motion_body_ori": ("positive", 1.0, 1.0),
    "motion_body_lin_vel": ("positive", 1.0, 1.0),
    "motion_body_ang_vel": ("positive", 1.0, 1.0),
    "action_rate_l2": ("penalty", -0.1, -0.1),
    "joint_limit": ("penalty", -10.0, -2.0),
    "self_collisions": ("penalty", -10.0, None),  # not in the original reward at all
}

POSITIVE = [n for n, (r, _, _) in TERMS.items() if r == "positive"]


def _clamp(x: float) -> float:
    """``runner.evaluate_clip``: normalized_return = min(max(raw, 0), 1)."""
    return float(min(max(x, 0.0), 1.0))


def _sums(diag: dict, denom: float) -> dict[str, float]:
    """Per-clip per-term sums over alive steps, plus the reconstructed raw value."""
    alive = int(diag["alive_steps"])
    terms = {n: float(v) * alive for n, v in diag["mean_term_score"].items()}
    steps = int(diag["steps"])
    total = float(sum(terms.values()))
    raw = total / (steps * denom)
    claimed = float(diag["raw_normalized_return"])
    err = abs(raw - claimed)
    # The per-term values are float32-derived on the torch side (mjlab's reward
    # manager), so re-summing them in float64 reconstructs ``raw`` only to
    # ~1e-8. That residual is itself evidence about the record, so it is
    # returned rather than swallowed.
    if err > 1e-6:
        raise AssertionError(
            f"term-sum identity failed ({err:.3e}): the record's mean_term_score "
            "is not consistent with raw_normalized_return"
        )
    return {
        "terms": terms,
        "total": total,
        "raw": raw,
        "alive": alive,
        "steps": steps,
        "identity_residual": err,
    }


def analyse(record: dict) -> dict:
    evaluation = record["evaluation"]
    denom = float(evaluation.get("normalized_return_denominator", 0.1))
    diagnostics = evaluation["per_clip_diagnostics"]
    official = {
        row["clip"]: row for row in record["checkpoints"][0]["per_clip"]
    }

    clips = []
    residuals = []
    for clip, diag in sorted(diagnostics.items()):
        s = _sums(diag, denom)
        residuals.append(s["identity_residual"])
        terms, total, raw, alive, steps = s["terms"], s["total"], s["raw"], s["alive"], s["steps"]
        per_step = {n: v / alive for n, v in terms.items()} if alive else {}

        positive_sum = sum(v for n, v in terms.items() if TERMS[n][0] == "positive")
        self_coll = terms.get("self_collisions", 0.0)
        joint_limit = terms.get("joint_limit", 0.0)
        action_rate = terms.get("action_rate_l2", 0.0)

        # --- exact counterfactuals: reward weights only, trajectory unchanged ---
        raw_no_self = (total - self_coll) / (steps * denom)
        raw_jl_original = (total - joint_limit + joint_limit * 0.2) / (steps * denom)
        raw_original_penalties = (
            total - self_coll - joint_limit + joint_limit * 0.2
        ) / (steps * denom)

        # --- projection (NOT a measurement): assumes the per-alive-step reward
        # would have continued unchanged had the clip not terminated early ---
        projected_full = None
        if alive < steps and alive > 0:
            projected_full = (total / alive) * steps / (steps * denom)

        clips.append(
            {
                "clip": clip,
                "steps": steps,
                "alive_steps": alive,
                "alive_fraction": alive / steps,
                "fall_step": diag.get("fall_step"),
                "fall_terms": list(diag.get("fall_terms") or ()),
                "official": {
                    "normalized_return": official[clip]["normalized_return"],
                    "failed": bool(official[clip]["failed"]),
                },
                "reconstructed_raw": raw,
                "sums": {n: terms.get(n, 0.0) for n in TERMS},
                "per_alive_step": {n: per_step.get(n, 0.0) for n in TERMS},
                "positive_fraction_of_max": positive_sum / alive / denom if alive else 0.0,
                "counterfactual_exact": {
                    "raw": raw,
                    "raw_no_self_collision_penalty": raw_no_self,
                    "raw_joint_limit_original_weight": raw_jl_original,
                    "raw_original_penalties": raw_original_penalties,
                },
                "projection_requires_resim": {
                    "raw_if_no_termination_mean_reward_held": projected_full
                },
            }
        )

    def mean_clamped(rows, key_path):
        vals = []
        for r in rows:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(_clamp(v))
        return float(np.mean(vals))

    summary = {}
    for label, path in (
        ("official", ("counterfactual_exact", "raw")),
        ("no_self_collision_penalty", ("counterfactual_exact", "raw_no_self_collision_penalty")),
        ("joint_limit_original_weight", ("counterfactual_exact", "raw_joint_limit_original_weight")),
        ("original_penalties", ("counterfactual_exact", "raw_original_penalties")),
    ):
        summary[label] = {
            "mean_clamped_return": mean_clamped(clips, path),
            "mean_raw_return": float(np.mean([_get(r, path) for r in clips])),
            "n_clips_below_zero_raw": int(sum(_get(r, path) < 0.0 for r in clips)),
            "n_clips_above_one_raw": int(sum(_get(r, path) > 1.0 for r in clips)),
        }

    # Termination histogram, and which clips are reward-limited rather than fall-limited.
    hist: dict[str, int] = {}
    for r in clips:
        for term in r["fall_terms"] or ("<none>",):
            hist[term] = hist.get(term, 0) + 1

    failed = [r for r in clips if r["official"]["failed"]]
    zero_but_alive = [
        r["clip"]
        for r in clips
        if not r["official"]["failed"] and r["official"]["normalized_return"] <= 1e-9
    ]

    return {
        "provenance": {
            "source_record": None,  # filled by caller
            "protocol_id": record.get("protocol_id"),
            "algorithm": record.get("algorithm"),
            "evaluator": record.get("evaluator", {}).get("implementation"),
            "source_git_sha": record.get("evaluator", {}).get("source_git_sha"),
            "note": (
                "Read-only decomposition of the frozen official record. No re-simulation. "
                "counterfactual_exact entries are exact linear re-weightings (the reward does "
                "not feed back into the trajectory); projection_requires_resim entries are "
                "extrapolations under a stated assumption and must not be reported as scores."
            ),
        },
        "denominator_per_step": denom,
        "identity_residual_max": float(max(residuals)) if residuals else 0.0,
        "summary": summary,
        "termination_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "failed_clips": [r["clip"] for r in failed],
        "failed_clip_alive_fraction_mean": (
            float(np.mean([r["alive_fraction"] for r in failed])) if failed else None
        ),
        "alive_no_fall_but_zero_score": zero_but_alive,
        "clips": clips,
    }


def _get(row, path):
    value = row
    for key in path:
        value = value[key]
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reward_counterfactuals", description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    record = json.loads(args.record.read_text())
    payload = analyse(record)
    payload["provenance"]["source_record"] = str(args.record)
    payload["provenance"]["source_record_sha256"] = _sha256(args.record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["summary"]
    print(f"{'variant':34s} {'mean(clamped)':>13s} {'mean(raw)':>10s} {'raw<0':>6s} {'raw>1':>6s}")
    for name, row in s.items():
        print(
            f"{name:34s} {row['mean_clamped_return']:13.6f} {row['mean_raw_return']:10.6f} "
            f"{row['n_clips_below_zero_raw']:6d} {row['n_clips_above_one_raw']:6d}"
        )
    print(f"\ntermination histogram: {payload['termination_histogram']}")
    print(f"failed clips ({len(payload['failed_clips'])}): {payload['failed_clips']}")
    print(f"alive but scored 0: {payload['alive_no_fall_but_zero_score']}")
    print(f"wrote {args.out}")
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
