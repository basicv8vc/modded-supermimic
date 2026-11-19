"""Evaluate InstantTrack / BlinkMimic checkpoints on the frozen V2 referee."""

from __future__ import annotations

import csv
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
FAST_SCREEN = os.environ.get("FAST_SCREEN", "1") == "1"


def run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _ckpt_meta(ckpt_path: Path, repo_dir: Path) -> tuple[int, float]:
    """Read (iteration, wall_time_s) stored inside an InstantTrack/BlinkMimic checkpoint."""
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

    if drive_run and not list(log_dir.glob("ckpt_*.pt")) and (drive_run / "checkpoints").is_dir():
        print(f"Restoring checkpoints from Drive: {drive_run / 'checkpoints'}", flush=True)
        ds.sync_tree(drive_run / "checkpoints", log_dir)
    if drive_run and (drive_run / "logs").is_dir():
        ds.sync_tree(drive_run / "logs", log_dir)

    ckpts = sorted(log_dir.glob("ckpt_*.pt"))
    if not ckpts:
        last_pt = log_dir / "last.pt"
        if last_pt.is_file():
            ckpts = [last_pt]
        else:
            raise FileNotFoundError(f"No ckpt_*.pt or last.pt found in {log_dir}")

    sha_upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    os.environ["V2_EVALUATOR_GIT_SHA"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    algo_name = "BlinkMimic-FULL_COLLISION" if UPSTREAM == "blinkmimic" else "InstantTrack-v1.4-FULL_COLLISION"

    sys.path.insert(0, str(bm))
    from benchmark.v2 import spec as S  # noqa: E402

    # Optional fast upstream screening under the patched FULL_COLLISION env to narrow V2 bisect window.
    screen_passing: list[tuple[int, float]] = []
    eval_csv = log_dir / "eval.csv"
    if FAST_SCREEN and (repo_dir / "tools" / "eval_batch.py").is_file() and len(ckpts) > 3:
        already_screened = set()
        if eval_csv.is_file():
            with eval_csv.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    already_screened.add(int(float(row["iteration"])))
        if len(already_screened) < len(ckpts):
            print("=== Running upstream FULL_COLLISION fast screen (tools/eval_batch.py) ===", flush=True)
            run(
                [str(repo_dir / ".venv/bin/python"), "tools/eval_batch.py", "--run_dir", str(log_dir), "--group", "16"],
                cwd=repo_dir,
            )
        if eval_csv.is_file():
            if drive_run:
                ds.sync_file_atomic(eval_csv, drive_run / "logs" / "eval.csv")
            with eval_csv.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    it = int(float(row["iteration"]))
                    ret = float(row["norm_return"])
                    n_fail = int(float(row["n_failed"]))
                    if ret >= 0.82 and n_fail == 0:
                        screen_passing.append((it, ret))
            print("Fast screen passing iters:", [it for it, _ in screen_passing], flush=True)

    # Select target checkpoints for official V2 referee evaluation:
    # earliest screen-passing +/- 1..2 neighbors, peak screen-passing, and final checkpoint.
    ckpt_by_it: list[tuple[int, float, Path]] = []
    for ck in ckpts:
        it, wall_s = _ckpt_meta(ck, repo_dir)
        ckpt_by_it.append((it, wall_s, ck))
    ckpt_by_it.sort(key=lambda x: x[0])

    targets: list[tuple[int, float, Path]] = [ckpt_by_it[-1]]
    if screen_passing:
        first_pass_it = min(it for it, _ in screen_passing)
        peak_pass_it = max(screen_passing, key=lambda x: x[1])[0]
        idx = min(range(len(ckpt_by_it)), key=lambda i: abs(ckpt_by_it[i][0] - first_pass_it))
        for j in range(max(0, idx - 1), min(len(ckpt_by_it), idx + 3)):
            targets.append(ckpt_by_it[j])
        peak_idx = min(range(len(ckpt_by_it)), key=lambda i: abs(ckpt_by_it[i][0] - peak_pass_it))
        targets.append(ckpt_by_it[peak_idx])
    else:
        # Evaluate latest + midpoint + 1/3 checkpoints if no screen or few ckpts
        if len(ckpt_by_it) >= 2:
            targets.append(ckpt_by_it[len(ckpt_by_it) // 2])
        if len(ckpt_by_it) >= 3:
            targets.append(ckpt_by_it[(2 * len(ckpt_by_it)) // 3])

    dedup: dict[str, tuple[int, float, Path]] = {str(p): (it, w, p) for it, w, p in targets}
    targets_sorted = sorted(dedup.values(), key=lambda x: x[0])

    def sync_eval_file(path: Path) -> None:
        if drive_run and path.is_file():
            ds.sync_file_atomic(path, drive_run / "evals" / path.name)

    records: list[dict] = []
    for it, wall_s, ck in targets_sorted:
        out_json = eval_dir / f"{ck.stem}_v2.json"
        run(
            [
                sys.executable,
                "scripts/v2/fast_eval.py",
                "--backend", "batched",
                "--batch-envs", "40",
                "--policy-file", "baselines/instanttrack/policy.py:make_policy",
                "--checkpoint", str(ck),
                "--algorithm", algo_name,
                "--training-commit", sha_upstream,
                "--training-wall-s", f"{wall_s:.2f}",
                "--seed", str(SEED),
                "--cache-references",
                "--gpu-name", "RTX PRO 6000 Blackwell",
                "--gpu-count", "1",
                "--out", str(out_json),
            ],
            cwd=bm,
        )
        sync_eval_file(out_json)
        ds_rec = json.loads(out_json.read_text())
        audit = S.audit_v2_record(ds_rec)
        ev = ds_rec["evaluation"]
        failed_names = [c["clip_id"] for c in ev.get("per_clip", []) if c.get("failed")]
        rec = {
            "label": ck.stem,
            "ckpt": str(ck),
            "iter": it,
            "wall_s": wall_s,
            "json": str(out_json),
            "audit_status": audit["status"],
            "aggregate_return": ev["aggregate_normalized_return"],
            "failed_clips": ev["failed_clips"],
            "failed_names": failed_names,
            "eligible": S.is_submission_eligible(ds_rec),
        }
        records.append(rec)
        print(
            f"[V2] iter={it} wall={wall_s:.2f}s ret={rec['aggregate_return']:.5f} "
            f"failed={rec['failed_clips']} audit={audit['status']} {failed_names}",
            flush=True,
        )

    passing = [r for r in records if r["failed_clips"] == 0 and r["aggregate_return"] >= 0.82]
    selected = min(passing, key=lambda r: r["wall_s"]) if passing else min(
        records, key=lambda r: (r["failed_clips"], -r["aggregate_return"])
    )
    summary_p = eval_dir / "selected.json"
    summary_p.write_text(json.dumps({"selected": selected, "evaluated": records}, indent=2))
    sync_eval_file(summary_p)

    # 2-clip Option B vs serial parity check on the selected checkpoint.
    parity_out = eval_dir / f"parity_{selected['label']}_2clips.json"
    ser_out = eval_dir / "_parity_serial_2clips.json"
    bat_out = eval_dir / "_parity_batched_2clips.json"
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
                "--policy-file", "baselines/instanttrack/policy.py:make_policy",
                "--checkpoint", selected["ckpt"],
                "--algorithm", algo_name,
                "--training-commit", sha_upstream,
                "--training-wall-s", f"{selected['wall_s']:.2f}",
                "--seed", str(SEED),
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
    sync_eval_file(parity_out)

    if drive_run:
        ds.write_status(
            drive_run,
            [
                f"run_id: {run_id}",
                "phase: eval_done",
                f"selected_label: {selected['label']}",
                f"selected_iter: {selected['iter']}",
                f"selected_wall_s: {selected['wall_s']:.2f}",
                f"selected_return: {selected['aggregate_return']:.5f}",
                f"selected_failed: {selected['failed_clips']} {selected['failed_names']}",
            ],
        )

    print("=== EVAL DONE ===", flush=True)


if __name__ == "__main__":
    main()
