"""Print live Colab VM + Google Drive sync status for the active InstantTrack run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402


def main() -> None:
    run_id = ds.resolve_run_id() if ds.RUN_STATE_FILE.exists() or os.environ.get("RUN_ID") else "(none)"
    print("--- tmux ---")
    print(subprocess.run("tmux ls 2>/dev/null || echo '(no sessions)'", shell=True, text=True, capture_output=True).stdout.strip())

    print("--- gpu ---")
    print(
        subprocess.run(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo '(no gpu)'",
            shell=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )

    log_dir = Path("/content/instanttrack_runs") / run_id if run_id != "(none)" else None
    ckpts = sorted(log_dir.glob("ckpt_*.pt")) if log_dir and log_dir.is_dir() else []
    evals = sorted((log_dir / "v2_evals").glob("*_v2.json")) if log_dir and (log_dir / "v2_evals").is_dir() else []
    print(f"--- local --- run_id={run_id} checkpoints={len(ckpts)} v2_evals={len(evals)}")
    if ckpts:
        print("latest local checkpoint:", ckpts[-1].name)

    pipe_log = Path("/content/pipeline_instanttrack.log")
    if pipe_log.is_file():
        lines = pipe_log.read_text(errors="replace").splitlines()
        print("--- pipeline_instanttrack.log tail ---")
        for line in lines[-12:]:
            print(line)

    print("--- google drive ---")
    if not ds.drive_available():
        print("Drive NOT mounted at /content/drive/MyDrive")
        return
    print(f"Drive base: {ds.DRIVE_BASE}")
    if run_id != "(none)":
        d_run = ds.DRIVE_BASE / "runs" / run_id
        d_ckpts = sorted((d_run / "checkpoints").glob("ckpt_*.pt")) if (d_run / "checkpoints").is_dir() else []
        d_evals = sorted((d_run / "evals").glob("*.json")) if (d_run / "evals").is_dir() else []
        print(f"Drive run: {d_run} | checkpoints={len(d_ckpts)} eval_jsons={len(d_evals)}")
        status_f = d_run / "STATUS.txt"
        if status_f.is_file():
            print("--- Drive STATUS.txt ---")
            print(status_f.read_text().strip())


if __name__ == "__main__":
    main()
