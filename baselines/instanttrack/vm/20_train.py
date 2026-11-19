"""Train InstantTrack v1.4 / BlinkMimic with FULL_COLLISION and atomic Drive sync."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

UPSTREAM = os.environ.get("UPSTREAM", "instanttrack").strip().lower()
PKG_NAME = "blinkmimic" if UPSTREAM == "blinkmimic" else "instanttrack"
PRESET_NAME = os.environ.get("PRESET", "main" if UPSTREAM == "blinkmimic" else "instanttrack").strip()
SEED = int(os.environ.get("SEED", "0"))
BUDGET_MINUTES = float(os.environ.get("BUDGET_MINUTES", "15"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "100"))
CKPT_MINUTES = float(os.environ.get("CKPT_MINUTES", "1.0"))
SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "30"))
EXTRA_ARGS = os.environ.get("EXTRA_ARGS", "").strip()
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"


def _load_preset_flags(repo_dir: Path, preset_name: str) -> list[str]:
    if not preset_name or preset_name in {"none", "supermimic"}:
        return []
    presets_sh = repo_dir / "scripts" / "presets.sh"
    if not presets_sh.is_file():
        raise FileNotFoundError(f"Missing {presets_sh}")
    cmd = f"source {shlex.quote(str(presets_sh))} && printf '%s' \"${{PRESET[{shlex.quote(preset_name)}]:-}}\""
    raw = subprocess.check_output(["bash", "-lc", cmd], cwd=str(repo_dir), text=True).strip()
    if not raw:
        raise ValueError(f"Preset {preset_name!r} resolved empty in {presets_sh}")
    return shlex.split(raw)


def _tail_progress(progress_csv: Path) -> str:
    if not progress_csv.is_file():
        return "(no progress.csv yet)"
    lines = progress_csv.read_text(errors="replace").strip().splitlines()
    return lines[-1] if lines else "(empty)"


def main() -> None:
    run_id = ds.resolve_run_id()
    if REQUIRE_DRIVE:
        ds.require_drive()
    drive_run = ds.drive_run_dir(run_id) if ds.drive_available() else None

    repo_dir = Path("/content/instanttrack_ws/repo")
    log_dir = Path("/content/instanttrack_runs") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = Path("/content/pipeline_instanttrack.log")

    preset_flags = _load_preset_flags(repo_dir, PRESET_NAME)
    max_hours = BUDGET_MINUTES / 60.0
    cmd = [
        ".venv/bin/python",
        "-u",
        "-m",
        f"{PKG_NAME}.train",
        "--log_dir",
        str(log_dir),
        *preset_flags,
        "--seed",
        str(SEED),
        "--max_hours",
        f"{max_hours:.6f}",
        "--ckpt_every",
        str(CKPT_EVERY),
        "--ckpt_minutes",
        str(CKPT_MINUTES),
    ]
    if EXTRA_ARGS:
        cmd.extend(shlex.split(EXTRA_ARGS))

    (log_dir / "launcher_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "upstream": UPSTREAM,
                "pkg": PKG_NAME,
                "preset": PRESET_NAME,
                "seed": SEED,
                "budget_minutes": BUDGET_MINUTES,
                "ckpt_every": CKPT_EVERY,
                "ckpt_minutes": CKPT_MINUTES,
                "extra_args": EXTRA_ARGS,
                "cmd": cmd,
            },
            indent=2,
        )
    )

    print(f">>> RUN_ID={run_id} | upstream={UPSTREAM} | preset={PRESET_NAME} | seed={SEED} | budget={BUDGET_MINUTES}m", flush=True)
    print(">>> Launching:", " ".join(cmd), flush=True)

    t0 = time.perf_counter()
    stop_event = threading.Event()
    synced_ckpts: set[str] = set()
    state = {"ckpts": 0, "last_ckpt": "(none)"}

    def sync_loop() -> None:
        while not stop_event.is_set():
            ckpts = sorted(log_dir.glob("ckpt_*.pt"))
            for ck in ckpts:
                if ck.name not in synced_ckpts:
                    if drive_run:
                        if ds.sync_file_atomic(ck, drive_run / "checkpoints" / ck.name):
                            synced_ckpts.add(ck.name)
                    else:
                        synced_ckpts.add(ck.name)
            state["ckpts"] = len(ckpts)
            if ckpts:
                state["last_ckpt"] = ckpts[-1].name
            if drive_run:
                for fname in ("args.json", "progress.csv", "launcher_meta.json", "stdout.log"):
                    fpath = log_dir / fname
                    if fpath.is_file():
                        ds.sync_file_atomic(fpath, drive_run / "logs" / fname)
                if pipeline_log.is_file():
                    ds.sync_file_atomic(pipeline_log, drive_run / "pipeline.log")
                ds.write_status(
                    drive_run,
                    [
                        f"run_id: {run_id}",
                        f"upstream: {UPSTREAM} ({PRESET_NAME})",
                        "phase: training",
                        f"elapsed_s: {time.perf_counter() - t0:.1f}",
                        f"checkpoints: {state['ckpts']}",
                        f"latest_ckpt: {state['last_ckpt']}",
                        f"progress_tail: {_tail_progress(log_dir / 'progress.csv')}",
                    ],
                )
            stop_event.wait(SYNC_INTERVAL)

    t_sync = threading.Thread(target=sync_loop, daemon=True)
    t_sync.start()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with (log_dir / "stdout.log").open("w") as out_f:
        proc = subprocess.Popen(cmd, cwd=repo_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            out_f.flush()
        rc = proc.wait()

    elapsed = time.perf_counter() - t0
    stop_event.set()
    t_sync.join(timeout=10.0)

    if drive_run:
        ds.sync_tree(log_dir, drive_run / "logs")
        for ck in sorted(log_dir.glob("ckpt_*.pt")):
            ds.sync_file_atomic(ck, drive_run / "checkpoints" / ck.name)
        last_pt = log_dir / "last.pt"
        if last_pt.is_file():
            ds.sync_file_atomic(last_pt, drive_run / "checkpoints" / "last.pt")
        if pipeline_log.is_file():
            ds.sync_file_atomic(pipeline_log, drive_run / "pipeline.log")
        ds.write_status(
            drive_run,
            [
                f"run_id: {run_id}",
                f"upstream: {UPSTREAM} ({PRESET_NAME})",
                f"phase: train_finished (rc={rc})",
                f"train_wall_s: {elapsed:.2f}",
                f"checkpoints: {len(list(log_dir.glob('ckpt_*.pt')))}",
                f"progress_tail: {_tail_progress(log_dir / 'progress.csv')}",
            ],
        )

    if rc != 0:
        raise RuntimeError(f"Training failed rc={rc}")
    print(f"=== TRAIN DONE ({elapsed:.1f}s) ===", flush=True)


if __name__ == "__main__":
    main()
