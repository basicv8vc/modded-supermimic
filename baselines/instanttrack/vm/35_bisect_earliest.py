"""Binary-search the earliest 0/40-failure (>=0.82) checkpoint on V2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

UPSTREAM = os.environ.get("UPSTREAM", "instanttrack").strip().lower()
SEED = int(os.environ.get("SEED", "0"))
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"
RETURN_THRESHOLD = float(os.environ.get("RETURN_THRESHOLD", "0.82"))


def run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _passes(rec: dict) -> bool:
    return rec["failed_clips"] == 0 and rec["aggregate_return"] >= RETURN_THRESHOLD


def _ckpt_meta(ckpt_path: Path, repo_dir: Path) -> tuple[int, float]:
    code = (
        "import json, sys, torch; "
        "ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "print(json.dumps({'it': int(ck.get('iteration', 0)), 'wall': float(ck.get('wall_time', 0.0))}))"
    )
    out = subprocess.check_output(
        [str(repo_dir / ".venv/bin/python"), "-c", code, str(ckpt_path)],
        text=True,
    ).strip()
    data = json.loads(out)
    return int(data["it"]), float(data["wall"])


def main() -> None:
    run_id = ds.resolve_run_id()
    if REQUIRE_DRIVE:
        ds.require_drive()
    drive_run = ds.drive_run_dir(run_id) if ds.drive_available() else None

    root = Path(os.environ.get("BENCHMARK_ROOT", "/content/modded-supermimic"))
    bm = (root / os.environ.get("BENCHMARK_SUBDIR", ".")).resolve()
    repo_dir = Path("/content/instanttrack_ws/repo")
    log_dir = Path("/content/instanttrack_runs") / run_id
    eval_dir = log_dir / "v2_evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if drive_run and (drive_run / "checkpoints").is_dir():
        ds.sync_tree(drive_run / "checkpoints", log_dir)
    if drive_run and (drive_run / "evals").is_dir():
        ds.sync_tree(drive_run / "evals", eval_dir)

    ckpts = []
    for ck in sorted(log_dir.glob("ckpt_*.pt")):
        it, wall_s = _ckpt_meta(ck, repo_dir)
        ckpts.append({"sp": ck, "iter": it, "wall_s": wall_s})
    ckpts.sort(key=lambda x: x["iter"])
    if not ckpts:
        raise FileNotFoundError(f"No ckpt_*.pt found in {log_dir}")

    sha_upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    os.environ["V2_EVALUATOR_GIT_SHA"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    algo_name = "BlinkMimic-FULL_COLLISION" if UPSTREAM == "blinkmimic" else "InstantTrack-v1.4-FULL_COLLISION"

    sys.path.insert(0, str(bm))
    from benchmark.v2 import spec as S  # noqa: E402

    cache: dict[int, dict] = {}
    for entry in ckpts:
        out_sp = eval_dir / f"{entry['sp'].stem}_v2.json"
        if out_sp.is_file():
            ds_rec = json.loads(out_sp.read_text())
            ev = ds_rec["evaluation"]
            cache[entry["iter"]] = {
                "label": entry["sp"].stem,
                "ckpt": str(entry["sp"]),
                "iter": entry["iter"],
                "wall_s": entry["wall_s"],
                "json": str(out_sp),
                "aggregate_return": ev["aggregate_normalized_return"],
                "failed_clips": ev["failed_clips"],
                "failed_names": [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")],
                "eligible": S.is_submission_eligible(ds_rec),
            }

    def eval_idx(idx: int) -> dict:
        entry = ckpts[idx]
        if entry["iter"] in cache:
            return cache[entry["iter"]]
        out_sp = eval_dir / f"{entry['sp'].stem}_v2.json"
        run(
            [
                sys.executable,
                "scripts/v2/fast_eval.py",
                "--backend", "batched",
                "--batch-envs", "40",
                "--policy-file", "baselines/instanttrack/policy.py:make_policy",
                "--checkpoint", str(entry["sp"]),
                "--algorithm", algo_name,
                "--training-commit", sha_upstream,
                "--training-wall-s", f"{entry['wall_s']:.2f}",
                "--seed", str(SEED),
                "--cache-references",
                "--gpu-name", "RTX PRO 6000 Blackwell",
                "--gpu-count", "1",
                "--out", str(out_sp),
            ],
            cwd=bm,
        )
        if drive_run:
            ds.sync_file_atomic(out_sp, drive_run / "evals" / out_sp.name)
        ds_rec = json.loads(out_sp.read_text())
        ev = ds_rec["evaluation"]
        rec = {
            "label": entry["sp"].stem,
            "ckpt": str(entry["sp"]),
            "iter": entry["iter"],
            "wall_s": entry["wall_s"],
            "json": str(out_sp),
            "aggregate_return": ev["aggregate_normalized_return"],
            "failed_clips": ev["failed_clips"],
            "failed_names": [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")],
            "eligible": S.is_submission_eligible(ds_rec),
        }
        cache[entry["iter"]] = rec
        print(
            f"[BISECT] iter={rec['iter']} wall={rec['wall_s']:.2f}s ret={rec['aggregate_return']:.5f} "
            f"failed={rec['failed_clips']} {rec['failed_names']}",
            flush=True,
        )
        return rec

    lo, hi = 0, len(ckpts) - 1
    if not _passes(eval_idx(hi)):
        print("Highest checkpoint did not pass 0/40 & >=0.82; skipping binary search.", flush=True)
    else:
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _passes(eval_idx(mid)):
                hi = mid
            else:
                lo = mid
        # Check whether lo passes too (when lo == 0)
        if lo == 0 and _passes(eval_idx(0)):
            hi = 0
        # Verify stability at hi + 1 if available
        if hi + 1 < len(ckpts):
            eval_idx(hi + 1)

    ordered = sorted(cache.values(), key=lambda r: r["iter"])
    passing = [r for r in ordered if _passes(r)]
    earliest = min(passing, key=lambda r: r["iter"]) if passing else None
    report = {
        "run_id": run_id,
        "return_threshold": RETURN_THRESHOLD,
        "earliest_passing": earliest,
        "evaluated": ordered,
    }
    report_p = eval_dir / "bisect_report.json"
    report_p.write_text(json.dumps(report, indent=2))
    if drive_run:
        ds.sync_file_atomic(report_p, drive_run / "evals" / "bisect_report.json")
    print(f"=== BISECT DONE === earliest={earliest['label'] if earliest else None}", flush=True)


if __name__ == "__main__":
    main()
