"""Shared Google Drive persistence helpers for Colab VM scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
DRIVE_BASE = Path(os.environ.get("DRIVE_BASE", str(DRIVE_MYDRIVE / "supermimic_v2")))
RUN_STATE_FILE = Path("/content/.current_run_id")


def require_drive() -> Path:
    """Return DRIVE_BASE if Google Drive is mounted, else raise."""
    if not DRIVE_MYDRIVE.is_dir():
        raise RuntimeError(
            f"Google Drive is not mounted at {DRIVE_MYDRIVE}. "
            "Run `colab drivemount -s <session>` before launching VM steps."
        )
    DRIVE_BASE.mkdir(parents=True, exist_ok=True)
    return DRIVE_BASE


def drive_available() -> bool:
    return DRIVE_MYDRIVE.is_dir()


def resolve_run_id(explicit: str | None = None) -> str:
    """Resolve active RUN_ID (env/arg -> persisted file -> new timestamped id)."""
    if explicit:
        RUN_STATE_FILE.write_text(explicit.strip() + "\n")
        return explicit.strip()
    env_id = os.environ.get("RUN_ID", "").strip()
    if env_id:
        RUN_STATE_FILE.write_text(env_id + "\n")
        return env_id
    if RUN_STATE_FILE.exists():
        saved = RUN_STATE_FILE.read_text().strip()
        if saved:
            return saved
    run_name = os.environ.get("RUN", "official_35k").strip() or "official_35k"
    run_id = f"{run_name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    RUN_STATE_FILE.write_text(run_id + "\n")
    return run_id


def drive_run_dir(run_id: str) -> Path:
    d = DRIVE_BASE / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def sync_tree(src: Path, dst: Path, timeout_s: int = 300) -> bool:
    """Mirror directory ``src`` to ``dst`` tolerating FUSE permission quirks."""
    if not src.exists():
        return True
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        cmd = [
            "rsync",
            "-a",
            "--partial",
            "--no-perms",
            "--no-owner",
            "--no-group",
            f"{src}/",
            f"{dst}/",
        ]
    else:
        cmd = ["cp", "-ru", f"{src}/.", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if res.returncode != 0:
        print(f"[WARN] sync_tree {src} -> {dst} rc={res.returncode}: {res.stderr.strip()}", flush=True)
        return False
    return True


def sync_file_atomic(src: Path, dst: Path) -> bool:
    """Copy a single file onto Drive FUSE via a hidden temp file + rename."""
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return True
    except OSError as exc:
        print(f"[WARN] sync_file_atomic {src} -> {dst} failed: {exc}", flush=True)
        tmp.unlink(missing_ok=True)
        return False


def write_status(run_dir: Path, lines: list[str]) -> None:
    """Atomically write human-readable STATUS.txt on Drive."""
    dst = run_dir / "STATUS.txt"
    tmp = run_dir / ".STATUS.txt.tmp"
    payload = "\n".join([f"updated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", *lines]) + "\n"
    try:
        tmp.write_text(payload)
        os.replace(tmp, dst)
    except OSError as exc:
        print(f"[WARN] write_status failed: {exc}", flush=True)
        tmp.unlink(missing_ok=True)
