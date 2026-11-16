"""Print live Colab VM + Google Drive sync status for the active run."""

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

    snaps = sorted(Path("/content/snapshots").glob("actor_snap*.pt")) if Path("/content/snapshots").exists() else []
    evals = sorted(Path("/content/snap_evals").glob("*_v2.json")) if Path("/content/snap_evals").exists() else []
    print(f"--- local --- run_id={run_id} snapshots={len(snaps)} evals={len(evals)}")
    if snaps:
        print("latest local snapshot:", snaps[-1].name)

    log_path = Path("/content/pipeline.log")
    if log_path.exists():
        lines = [
            l for l in log_path.read_text(errors="replace").splitlines()
            if "linesearch iterations" not in l and "solver iterations" not in l
        ]
        print("--- pipeline.log tail ---")
        for l in lines[-12:]:
            print(l)

    print("--- google drive ---")
    if not ds.drive_available():
        print("Drive NOT mounted at /content/drive/MyDrive")
        return
    print(f"Drive base: {ds.DRIVE_BASE}")
    if run_id != "(none)":
        d_run = ds.DRIVE_BASE / "runs" / run_id
        d_snaps = sorted((d_run / "snapshots").glob("actor_snap*.pt")) if (d_run / "snapshots").is_dir() else []
        d_evals = sorted((d_run / "evals").glob("*.json")) if (d_run / "evals").is_dir() else []
        print(f"Drive run: {d_run} | snapshots={len(d_snaps)} eval_jsons={len(d_evals)}")
        status_f = d_run / "STATUS.txt"
        if status_f.exists():
            print("--- Drive STATUS.txt ---")
            print(status_f.read_text().strip())


if __name__ == "__main__":
    main()
