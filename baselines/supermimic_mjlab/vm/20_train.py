"""Train official HybridRobotics/supermimic with immediate Google Drive snapshot sync."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

MAX_ITERS = int(os.environ.get("ITERS", "35001"))
SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", "500"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "250"))
SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "60"))
STEEPNESS = os.environ.get("RETURN_STEEPNESS", "").strip()
EXPLORE = os.environ.get("RETURN_EXPLORE", "").strip()
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"


def _parse_latest_iter(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            f.seek(max(0, sz - 65536))
            tail = f.read().decode("utf-8", errors="replace")
        clean = re.sub(r"\x1b\[[0-9;]*m", "", tail)
        matches = re.findall(r"Learning iteration\s+(\d+)\s*/\s*\d+", clean)
        return int(matches[-1]) if matches else None
    except OSError:
        return None


def main() -> None:
    run_id = ds.resolve_run_id()
    if REQUIRE_DRIVE:
        ds.require_drive()
    drive_run = ds.drive_run_dir(run_id) if ds.drive_available() else None

    sm_dir = Path("/content/supermimic_ws/supermimic")
    snap_dir = Path("/content/snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("/content/pipeline.log")

    motion_files = sorted(glob.glob("/content/lafan1_official_npz/*.npz"))
    if len(motion_files) != 40:
        raise RuntimeError(f"Expected 40 NPZs in /content/lafan1_official_npz, found {len(motion_files)}")

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        ".venv/bin/python",
        "src/supermimic/scripts/train.py",
        "SuperMimic-GenCoord-G1",
        "--motion-file",
        ",".join(motion_files),
        "--runner.actor-obs",
        "proprio",
        "--runner.history-length",
        "10",
        "--runner.max-iterations",
        str(MAX_ITERS),
        "--runner.log-interval",
        str(LOG_INTERVAL),
        "--runner.save-interval",
        str(SAVE_INTERVAL),
        "--runner.run-name",
        run_id,
    ]
    if STEEPNESS:
        cmd.extend(["--runner.return-sampler.steepness", STEEPNESS])
    if EXPLORE:
        cmd.extend(["--runner.return-sampler.explore", EXPLORE])
    print(f">>> RUN_ID={run_id} | drive_run={drive_run}", flush=True)
    print(">>> Launching:", " ".join(cmd), flush=True)

    t0 = time.perf_counter()
    stop_event = threading.Event()
    snap_lock = threading.Lock()
    state = {"snaps": 0, "last_snap": "(none)"}

    def watch_snapshots() -> None:
        seen_mtime = 0.0
        idx = 0
        while not stop_event.is_set():
            actors = sorted(
                sm_dir.glob("logs/SuperMimic-GenCoord-G1/*/actor.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if actors:
                p = actors[-1]
                try:
                    st1 = p.stat()
                    if st1.st_mtime > seen_mtime + 0.5 and st1.st_size > 1_000_000:
                        time.sleep(2.0)
                        st2 = p.stat()
                        if st2.st_mtime == st1.st_mtime and st2.st_size == st1.st_size:
                            elapsed_now = time.perf_counter() - t0
                            seen_mtime = st2.st_mtime
                            idx += 1
                            est_iter = (idx - 1) * SAVE_INTERVAL
                            raw_iter = _parse_latest_iter(log_path)
                            stem = f"actor_snap{idx:03d}_iter{est_iter}_{elapsed_now:.1f}s"
                            dst = snap_dir / f"{stem}.pt"
                            wall_f = snap_dir / f"{stem}.wall"
                            meta_f = snap_dir / f"{stem}.meta.json"
                            shutil.copy2(p, dst)
                            wall_f.write_text(f"{elapsed_now:.2f}\n")
                            meta_f.write_text(
                                json.dumps(
                                    {
                                        "run_id": run_id,
                                        "snap_idx": idx,
                                        "est_iter": est_iter,
                                        "raw_iter": raw_iter,
                                        "wall_s": round(elapsed_now, 2),
                                        "size_bytes": st2.st_size,
                                        "mtime": st2.st_mtime,
                                    },
                                    indent=2,
                                )
                            )
                            if drive_run:
                                ds.sync_file_atomic(dst, drive_run / "snapshots" / dst.name)
                                ds.sync_file_atomic(wall_f, drive_run / "snapshots" / wall_f.name)
                                ds.sync_file_atomic(meta_f, drive_run / "snapshots" / meta_f.name)
                            with snap_lock:
                                state["snaps"] = idx
                                state["last_snap"] = dst.name
                            print(
                                f">>> Snapshot synced: {dst.name} (est_iter={est_iter}, raw_iter={raw_iter}, wall={elapsed_now:.1f}s)",
                                flush=True,
                            )
                except FileNotFoundError:
                    pass
            stop_event.wait(2.0)

    def periodic_sync() -> None:
        if not drive_run:
            return
        while not stop_event.is_set():
            stop_event.wait(SYNC_INTERVAL)
            ds.sync_tree(sm_dir / "logs/SuperMimic-GenCoord-G1", drive_run / "logs")
            if log_path.exists():
                ds.sync_file_atomic(log_path, drive_run / "pipeline.log")
            with snap_lock:
                snaps_n = state["snaps"]
                last_snap = state["last_snap"]
            ds.write_status(
                drive_run,
                [
                    f"run_id: {run_id}",
                    "phase: training",
                    f"elapsed_s: {time.perf_counter() - t0:.1f}",
                    f"latest_parsed_iter: {_parse_latest_iter(log_path)}",
                    f"snapshots_saved: {snaps_n}",
                    f"latest_snapshot: {last_snap}",
                ],
            )

    t_snap = threading.Thread(target=watch_snapshots, daemon=True)
    t_sync = threading.Thread(target=periodic_sync, daemon=True)
    t_snap.start()
    t_sync.start()

    proc = subprocess.run(cmd, cwd=sm_dir, env=env)
    elapsed = time.perf_counter() - t0
    # Give snapshot watcher one short cycle to catch the final save
    time.sleep(3.5)
    stop_event.set()
    t_snap.join(timeout=5.0)
    t_sync.join(timeout=5.0)

    actors = sorted(sm_dir.glob("logs/SuperMimic-GenCoord-G1/*/actor.pt"), key=lambda p: p.stat().st_mtime)
    if actors:
        latest = actors[-1]
        (latest.parent / "train_wall_s.txt").write_text(f"{elapsed:.2f}\n")
        print("FINAL actor.pt:", latest, "size:", latest.stat().st_size, "wall_s:", f"{elapsed:.2f}", flush=True)

    if drive_run:
        ds.sync_tree(snap_dir, drive_run / "snapshots")
        ds.sync_tree(sm_dir / "logs/SuperMimic-GenCoord-G1", drive_run / "logs")
        if log_path.exists():
            ds.sync_file_atomic(log_path, drive_run / "pipeline.log")
        ds.write_status(
            drive_run,
            [
                f"run_id: {run_id}",
                f"phase: train_finished (rc={proc.returncode})",
                f"train_wall_s: {elapsed:.2f}",
                f"snapshots_saved: {state['snaps']}",
                f"latest_snapshot: {state['last_snap']}",
            ],
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed rc={proc.returncode}")
    print("=== TRAIN DONE ===", flush=True)


if __name__ == "__main__":
    main()
