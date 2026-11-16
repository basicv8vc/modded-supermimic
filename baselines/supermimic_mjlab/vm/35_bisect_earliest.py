"""Binary-search the earliest 0/40-failure checkpoint from Drive-backed snapshots."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", "500"))
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"
RETURN_THRESHOLD = float(os.environ.get("RETURN_THRESHOLD", "0.82"))


def run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _passes(rec: dict) -> bool:
    return rec["failed_clips"] == 0 and rec["aggregate_return"] >= RETURN_THRESHOLD


def _parse_snap_meta(sp: Path) -> tuple[int, float]:
    w_path = sp.with_suffix(".wall")
    w_s = float(w_path.read_text().strip()) if w_path.exists() else 0.0
    meta_path = sp.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    est_iter = meta.get("est_iter")
    if est_iter is None:
        m = re.search(r"_iter(\d+)_", sp.name)
        if m:
            est_iter = int(m.group(1))
        else:
            m2 = re.search(r"actor_snap(\d+)_", sp.name)
            est_iter = (int(m2.group(1)) - 1) * SAVE_INTERVAL if m2 else 0
    return int(est_iter), w_s


def main() -> None:
    run_id = ds.resolve_run_id()
    if REQUIRE_DRIVE:
        ds.require_drive()
    drive_run = ds.drive_run_dir(run_id) if ds.drive_available() else None

    root = Path(os.environ.get("BENCHMARK_ROOT", "/content/modded-supermimic"))
    bm = (root / os.environ.get("BENCHMARK_SUBDIR", ".")).resolve()
    sm_dir = Path("/content/supermimic_ws/supermimic")
    snap_dir = Path("/content/snapshots")
    snap_eval_dir = Path("/content/snap_evals")
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_eval_dir.mkdir(parents=True, exist_ok=True)

    # Restore snapshots & existing evals from Google Drive
    if drive_run and (drive_run / "snapshots").is_dir():
        print(f"Syncing snapshots from Drive: {drive_run / 'snapshots'}", flush=True)
        ds.sync_tree(drive_run / "snapshots", snap_dir)
    if drive_run and (drive_run / "evals").is_dir():
        print(f"Syncing prior evals from Drive: {drive_run / 'evals'}", flush=True)
        ds.sync_tree(drive_run / "evals", snap_eval_dir)

    sha_sm = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sm_dir, text=True).strip()
    os.environ["V2_EVALUATOR_GIT_SHA"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    sys.path.insert(0, str(bm))
    from benchmark.v2 import spec as S  # noqa: E402

    # Index all snapshots sorted by est_iter (deduplicated per est_iter)
    by_iter: dict[int, dict] = {}
    for sp in sorted(snap_dir.glob("actor_snap*.pt")):
        est_iter, wall_s = _parse_snap_meta(sp)
        by_iter[est_iter] = {"sp": sp, "est_iter": est_iter, "wall_s": wall_s}
    snaps = [by_iter[k] for k in sorted(by_iter)]
    if not snaps:
        raise FileNotFoundError(f"No actor_snap*.pt found in {snap_dir} or {drive_run}")
    print(f"Indexed {len(snaps)} snapshots: iter {snaps[0]['est_iter']}..{snaps[-1]['est_iter']}", flush=True)

    # Load any already-evaluated records from snap_evals
    records_by_iter: dict[int, dict] = {}
    summary_p = snap_eval_dir / "sweep_summary.json"
    if summary_p.exists():
        for rec in json.loads(summary_p.read_text()):
            records_by_iter[int(rec["est_iter"])] = rec
    for entry in snaps:
        out_sp = snap_eval_dir / f"{entry['sp'].stem}_v2.json"
        if out_sp.is_file() and entry["est_iter"] not in records_by_iter:
            ds_rec = json.loads(out_sp.read_text())
            ev = ds_rec["evaluation"]
            records_by_iter[entry["est_iter"]] = {
                "label": entry["sp"].stem,
                "ckpt": str(entry["sp"]),
                "est_iter": entry["est_iter"],
                "wall_s": entry["wall_s"],
                "json": str(out_sp),
                "aggregate_return": ev["aggregate_normalized_return"],
                "failed_clips": ev["failed_clips"],
                "failed_names": [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")],
                "eligible": S.is_submission_eligible(ds_rec),
            }

    print(f"Pre-loaded {len(records_by_iter)} already-evaluated checkpoints: {sorted(records_by_iter)}", flush=True)

    def sync_eval_artifact(local_file: Path) -> None:
        if drive_run and local_file.is_file():
            ds.sync_file_atomic(local_file, drive_run / "evals" / local_file.name)

    def eval_entry(entry: dict, tag: str) -> dict:
        it = entry["est_iter"]
        if it in records_by_iter:
            rec = records_by_iter[it]
            print(
                f"[{tag}-CACHED] iter={it} wall={rec['wall_s']:.1f}s "
                f"ret={rec['aggregate_return']:.5f} failed={rec['failed_clips']} {rec['failed_names']}",
                flush=True,
            )
            return rec

        sp = entry["sp"]
        out_sp = snap_eval_dir / f"{sp.stem}_v2.json"
        run(
            [
                sys.executable,
                "scripts/v2/fast_eval.py",
                "--backend", "batched",
                "--batch-envs", "40",
                "--policy-file", "benchmark/policies/official_supermimic.py:make_policy",
                "--checkpoint", str(sp),
                "--algorithm", "SuperMimic-official-gencoord-proprio",
                "--training-commit", sha_sm,
                "--training-wall-s", f"{entry['wall_s']:.2f}",
                "--seed", "0",
                "--cache-references",
                "--gpu-name", "RTX PRO 6000 Blackwell",
                "--gpu-count", "1",
                "--out", str(out_sp),
            ],
            cwd=bm,
        )
        sync_eval_artifact(out_sp)
        ds_rec = json.loads(out_sp.read_text())
        ev = ds_rec["evaluation"]
        rec = {
            "label": sp.stem,
            "ckpt": str(sp),
            "est_iter": it,
            "wall_s": entry["wall_s"],
            "json": str(out_sp),
            "aggregate_return": ev["aggregate_normalized_return"],
            "failed_clips": ev["failed_clips"],
            "failed_names": [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")],
            "eligible": S.is_submission_eligible(ds_rec),
        }
        records_by_iter[it] = rec
        print(
            f"[{tag}] iter={it} wall={rec['wall_s']:.1f}s "
            f"ret={rec['aggregate_return']:.5f} failed={rec['failed_clips']} {rec['failed_names']}",
            flush=True,
        )
        if drive_run:
            ds.write_status(
                drive_run,
                [
                    f"run_id: {run_id}",
                    "phase: bisecting",
                    f"evaluated_count: {len(records_by_iter)}",
                    f"last_eval: iter={it} ret={rec['aggregate_return']:.5f} failed={rec['failed_clips']}",
                ],
            )
        return rec

    # Set upper bound to lowest known passing snapshot (if any)
    passing_indices = [
        i for i, e in enumerate(snaps)
        if e["est_iter"] in records_by_iter and _passes(records_by_iter[e["est_iter"]])
    ]
    lo = 0
    hi = min(passing_indices) if passing_indices else len(snaps) - 1
    if not passing_indices:
        top_rec = eval_entry(snaps[hi], "BISECT-TOP")
        if not _passes(top_rec):
            print(f"[WARN] Top snapshot iter={snaps[hi]['est_iter']} did not reach 0/40 >= {RETURN_THRESHOLD}", flush=True)

    print(f"Starting binary search over indices [{lo}..{hi}] (iter {snaps[lo]['est_iter']}..{snaps[hi]['est_iter']})", flush=True)
    while lo < hi:
        mid = (lo + hi) // 2
        rec = eval_entry(snaps[mid], "BISECT")
        if _passes(rec):
            hi = mid
        else:
            lo = mid + 1

    boundary_idx = hi
    print(f"Binary search converged at idx={boundary_idx} (iter={snaps[boundary_idx]['est_iter']})", flush=True)

    # Guard against non-monotonic RL edge: step below boundary until 2 consecutive failures (or idx 0)
    consec_fails = 0
    probe = boundary_idx - 1
    while probe >= 0 and consec_fails < 2:
        rec = eval_entry(snaps[probe], "EDGE-CHECK")
        if _passes(rec):
            consec_fails = 0
        else:
            consec_fails += 1
        probe -= 1

    # Also check one step above boundary if not yet evaluated
    if boundary_idx + 1 < len(snaps):
        eval_entry(snaps[boundary_idx + 1], "EDGE-ABOVE")

    records_sorted = sorted(records_by_iter.values(), key=lambda r: (r["est_iter"], r["wall_s"]))
    summary_p.write_text(json.dumps(records_sorted, indent=2))
    sync_eval_artifact(summary_p)

    passing_all = [r for r in records_sorted if _passes(r)]
    zero_fail_any_ret = [r for r in records_sorted if r["failed_clips"] == 0]
    if passing_all:
        best_rec = max(passing_all, key=lambda r: r["aggregate_return"])
        earliest_rec = min(passing_all, key=lambda r: r["est_iter"])
    else:
        best_rec = min(records_sorted, key=lambda r: (r["failed_clips"], -r["aggregate_return"]))
        earliest_rec = best_rec

    selected_p = snap_eval_dir / "selected.json"
    selected_p.write_text(json.dumps({"best": best_rec, "earliest": earliest_rec, "sweep": records_sorted}, indent=2))
    sync_eval_artifact(selected_p)

    bisect_p = snap_eval_dir / "bisect_report.json"
    bisect_p.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "return_threshold": RETURN_THRESHOLD,
                "earliest_passing": earliest_rec if passing_all else None,
                "earliest_zero_fail_any_ret": min(zero_fail_any_ret, key=lambda r: r["est_iter"]) if zero_fail_any_ret else None,
                "best_passing": best_rec if passing_all else None,
                "evaluated": records_sorted,
            },
            indent=2,
        )
    )
    sync_eval_artifact(bisect_p)

    if drive_run:
        log_path = Path("/content/pipeline.log")
        if log_path.exists():
            ds.sync_file_atomic(log_path, drive_run / "pipeline.log")
        ds.write_status(
            drive_run,
            [
                f"run_id: {run_id}",
                "phase: bisect_done",
                f"evaluated_count: {len(records_sorted)}",
                f"earliest_iter: {earliest_rec['est_iter']}",
                f"earliest_wall_s: {earliest_rec['wall_s']:.2f}",
                f"earliest_return: {earliest_rec['aggregate_return']:.5f}",
                f"earliest_failed: {earliest_rec['failed_clips']}",
                f"best_iter: {best_rec['est_iter']}",
                f"best_return: {best_rec['aggregate_return']:.5f}",
            ],
        )

    print("=== BISECT DONE ===", flush=True)


if __name__ == "__main__":
    main()
