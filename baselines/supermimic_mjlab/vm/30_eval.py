"""Evaluate final + sweep official SuperMimic checkpoints on V2 and sync results to Drive."""

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
MAX_ITERS = int(os.environ.get("ITERS", "35001"))
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"


def run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


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

    # Restore from Drive if running evaluation on a fresh VM
    if drive_run and not list(snap_dir.glob("actor_snap*.pt")) and (drive_run / "snapshots").is_dir():
        print(f"Restoring snapshots from Drive: {drive_run / 'snapshots'}", flush=True)
        ds.sync_tree(drive_run / "snapshots", snap_dir)
    if drive_run and not list(sm_dir.glob("logs/SuperMimic-GenCoord-G1/*/actor.pt")) and (drive_run / "logs").is_dir():
        print(f"Restoring logs from Drive: {drive_run / 'logs'}", flush=True)
        ds.sync_tree(drive_run / "logs", sm_dir / "logs/SuperMimic-GenCoord-G1")

    sha_sm = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sm_dir, text=True).strip()
    os.environ["V2_EVALUATOR_GIT_SHA"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    sys.path.insert(0, str(bm))
    from benchmark.v2 import spec as S  # noqa: E402

    actors = sorted(sm_dir.glob("logs/SuperMimic-GenCoord-G1/*/actor.pt"), key=lambda p: p.stat().st_mtime)
    if not actors:
        raise FileNotFoundError("No actor.pt found locally or in Drive run logs.")
    actor_path = actors[-1]
    wall_file = actor_path.parent / "train_wall_s.txt"
    train_wall_s = wall_file.read_text().strip() if wall_file.exists() else "0.0"
    final_w = float(train_wall_s)

    def sync_eval_artifact(local_file: Path) -> None:
        if drive_run and local_file.is_file():
            ds.sync_file_atomic(local_file, drive_run / "evals" / local_file.name)

    def eval_40(ckpt_path: Path, wall_s: str, out_json: Path) -> dict:
        run(
            [
                sys.executable,
                "scripts/v2/fast_eval.py",
                "--backend", "batched",
                "--batch-envs", "40",
                "--policy-file", "benchmark/policies/official_supermimic.py:make_policy",
                "--checkpoint", str(ckpt_path),
                "--algorithm", "SuperMimic-official-gencoord-proprio",
                "--training-commit", sha_sm,
                "--training-wall-s", str(wall_s),
                "--seed", "0",
                "--cache-references",
                "--gpu-name", "RTX PRO 6000 Blackwell",
                "--gpu-count", "1",
                "--out", str(out_json),
            ],
            cwd=bm,
        )
        sync_eval_artifact(out_json)
        return json.loads(out_json.read_text())

    # 1. Final checkpoint
    out_final = bm / f"reports/v2/g4_official_{run_id}_final.json"
    d_final = eval_40(actor_path, train_wall_s, out_final)
    audit_final = S.audit_v2_record(d_final)
    ret_final = d_final["evaluation"]["aggregate_normalized_return"]
    fail_final = d_final["evaluation"]["failed_clips"]
    fail_names_final = [c["clip_id"] for c in d_final["evaluation"].get("per_clip", []) if c.get("failed")]
    print(f"\n=== FINAL AUDIT === status={audit_final['status']} ret={ret_final:.5f} failed={fail_final} {fail_names_final}", flush=True)

    final_iter = ((MAX_ITERS - 1) // SAVE_INTERVAL) * SAVE_INTERVAL
    records = [
        {
            "label": "final",
            "ckpt": str(actor_path),
            "est_iter": final_iter,
            "wall_s": final_w,
            "json": str(out_final),
            "aggregate_return": ret_final,
            "failed_clips": fail_final,
            "failed_names": fail_names_final,
            "eligible": S.is_submission_eligible(d_final),
        }
    ]

    # 2. Index saved snapshots
    snaps_info = []
    for sp in sorted(snap_dir.glob("actor_snap*.pt"), key=lambda p: p.stat().st_mtime):
        w_path = sp.with_suffix(".wall")
        w_s = float(w_path.read_text().strip()) if w_path.exists() else final_w
        meta_path = sp.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        est_iter = meta.get("est_iter")
        if est_iter is None:
            m = re.search(r"actor_snap(\d+)_", sp.name)
            est_iter = (int(m.group(1)) - 1) * SAVE_INTERVAL if m else 0
        if abs(final_w - w_s) < 15.0 and int(est_iter) >= final_iter - 200:
            continue
        snaps_info.append({"sp": sp, "est_iter": int(est_iter), "wall_s": w_s})

    evaluated_ckpts: set[str] = set()

    def eval_snapshot_entry(entry: dict, phase_tag: str) -> dict | None:
        sp = entry["sp"]
        if str(sp) in evaluated_ckpts:
            return None
        evaluated_ckpts.add(str(sp))
        out_sp = snap_eval_dir / f"{sp.stem}_v2.json"
        ds_rec = eval_40(sp, f"{entry['wall_s']:.2f}", out_sp)
        ev = ds_rec["evaluation"]
        rec = {
            "label": sp.stem,
            "ckpt": str(sp),
            "est_iter": entry["est_iter"],
            "wall_s": entry["wall_s"],
            "json": str(out_sp),
            "aggregate_return": ev["aggregate_normalized_return"],
            "failed_clips": ev["failed_clips"],
            "failed_names": [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")],
            "eligible": S.is_submission_eligible(ds_rec),
        }
        records.append(rec)
        print(
            f"[{phase_tag}] iter={rec['est_iter']} wall={rec['wall_s']:.1f}s "
            f"ret={rec['aggregate_return']:.5f} failed={rec['failed_clips']} {rec['failed_names']}",
            flush=True,
        )
        return rec

    # Phase 1: 1,000-iter sweep over [25k .. final-1k]
    p1_entries = []
    for target_it in range(final_iter - 1000, max(24000, final_iter - 11000), -1000):
        cands = [e for e in snaps_info if abs(e["est_iter"] - target_it) <= 300]
        if cands:
            p1_entries.append(min(cands, key=lambda e: abs(e["est_iter"] - target_it)))

    for e in p1_entries:
        eval_snapshot_entry(e, "GRID-1K")

    # Phase 2: +-500 densification around best / earliest passing or top-3 closest
    passing_p1 = [r for r in records if r["failed_clips"] == 0 and r["aggregate_return"] >= 0.82]
    dense_targets: set[int] = set()
    if passing_p1:
        earliest_p1 = min(passing_p1, key=lambda r: r["wall_s"])
        best_p1 = max(passing_p1, key=lambda r: r["aggregate_return"])
        for base_iter in (earliest_p1["est_iter"] - 500, best_p1["est_iter"] - 500, best_p1["est_iter"] + 500):
            if 24500 <= base_iter <= final_iter - 500:
                dense_targets.add(base_iter)
    else:
        for r in sorted(records, key=lambda r: (r["failed_clips"], -r["aggregate_return"]))[:3]:
            for delta in (-500, 500):
                base_iter = r["est_iter"] + delta
                if 24500 <= base_iter <= final_iter - 500:
                    dense_targets.add(base_iter)

    for tgt in sorted(dense_targets, reverse=True):
        cands = [e for e in snaps_info if str(e["sp"]) not in evaluated_ckpts and abs(e["est_iter"] - tgt) <= 300]
        if cands:
            eval_snapshot_entry(min(cands, key=lambda e: abs(e["est_iter"] - tgt)), "DENSE-500")

    records_sorted = sorted(records, key=lambda r: (r["est_iter"], r["wall_s"]))
    summary_p = snap_eval_dir / "sweep_summary.json"
    summary_p.write_text(json.dumps(records_sorted, indent=2))
    sync_eval_artifact(summary_p)

    passing = [r for r in records_sorted if r["failed_clips"] == 0 and r["aggregate_return"] >= 0.82]
    if passing:
        best_rec = max(passing, key=lambda r: r["aggregate_return"])
        earliest_rec = min(passing, key=lambda r: r["wall_s"])
    else:
        best_rec = min(records_sorted, key=lambda r: (r["failed_clips"], -r["aggregate_return"]))
        earliest_rec = best_rec

    selected_p = snap_eval_dir / "selected.json"
    selected_p.write_text(json.dumps({"best": best_rec, "earliest": earliest_rec, "sweep": records_sorted}, indent=2))
    sync_eval_artifact(selected_p)

    # 2-clip serial vs batched parity check on selected checkpoint
    parity_out = bm / f"reports/v2/g4_parity_{run_id}_best_2clips.json"
    ser_out = bm / f"reports/v2/_parity_{run_id}_serial_2clips.json"
    bat_out = bm / f"reports/v2/_parity_{run_id}_batched_2clips.json"
    for backend, b_args, out_f in (
        ("workers", ["--workers", "2"], ser_out),
        ("batched", ["--batch-envs", "2"], bat_out),
    ):
        run(
            [
                sys.executable,
                "scripts/v2/fast_eval.py",
                "--backend", backend,
                *b_args,
                "--clips", "walk1_subject1", "dance1_subject1",
                "--policy-file", "benchmark/policies/official_supermimic.py:make_policy",
                "--checkpoint", best_rec["ckpt"],
                "--algorithm", "SuperMimic-official-gencoord-proprio",
                "--training-commit", sha_sm,
                "--training-wall-s", f"{best_rec['wall_s']:.2f}",
                "--seed", "0",
                "--cache-references",
                "--out", str(out_f),
            ],
            cwd=bm,
        )
    run(
        [
            sys.executable,
            "scripts/v2/compare_evals.py",
            "--a", str(ser_out),
            "--b", str(bat_out),
            "--tol-return", "0.01",
            "--tol-terms", "0.01",
            "--out", str(parity_out),
        ],
        cwd=bm,
    )
    sync_eval_artifact(parity_out)

    if drive_run:
        log_path = Path("/content/pipeline.log")
        if log_path.exists():
            ds.sync_file_atomic(log_path, drive_run / "pipeline.log")
        ds.write_status(
            drive_run,
            [
                f"run_id: {run_id}",
                "phase: eval_done",
                f"best_label: {best_rec['label']}",
                f"best_iter: {best_rec['est_iter']}",
                f"best_wall_s: {best_rec['wall_s']:.2f}",
                f"best_return: {best_rec['aggregate_return']:.5f}",
                f"best_failed: {best_rec['failed_clips']} {best_rec['failed_names']}",
                f"earliest_label: {earliest_rec['label']}",
                f"earliest_return: {earliest_rec['aggregate_return']:.5f}",
                f"earliest_failed: {earliest_rec['failed_clips']}",
            ],
        )

    print("=== EVAL DONE ===", flush=True)


if __name__ == "__main__":
    main()
