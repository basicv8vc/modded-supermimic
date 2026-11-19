"""Shared Google Drive persistence helpers for InstantTrack Colab VM scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_MYDRIVE = Path("/content/drive/MyDrive")
DRIVE_BASE = Path(os.environ.get("DRIVE_BASE", str(DRIVE_MYDRIVE / "supermimic_v2")))
RUN_STATE_FILE = Path("/content/.current_instanttrack_run_id")


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
    run_name = os.environ.get("RUN", "instanttrack_v14").strip() or "instanttrack_v14"
    seed = os.environ.get("SEED", "0").strip() or "0"
    run_id = f"{run_name}_s{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
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
    cmd = ["rsync", "-rlt", "--no-perms", "--no-owner", "--no-group", f"{src}/", f"{dst}/"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if res.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        for root_dir, _, files in os.walk(src):
            rel = Path(root_dir).relative_to(src)
            out_dir = dst / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                s_f = Path(root_dir) / fname
                d_f = out_dir / fname
                if not d_f.exists() or s_f.stat().st_mtime > d_f.stat().st_mtime + 0.5 or s_f.stat().st_size != d_f.stat().st_size:
                    shutil.copy2(s_f, d_f)
        return True
    except OSError as exc:
        print(f"[drive_sync WARN] sync_tree({src} -> {dst}) failed: {exc}", flush=True)
        return False


def sync_file_atomic(src: Path, dst: Path) -> bool:
    """Copy a single file to Drive atomically via a temporary sibling file."""
    if not src.is_file():
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return True
    except OSError as exc:
        print(f"[drive_sync WARN] sync_file_atomic({src} -> {dst}) failed: {exc}", flush=True)
        return False


def write_status(drive_run: Path | None, lines: list[str]) -> None:
    """Write STATUS.txt locally and atomically to Drive."""
    txt = "\n".join(lines) + "\n"
    local_status = Path("/content/STATUS_instanttrack.txt")
    local_status.write_text(txt)
    if drive_run is not None:
        sync_file_atomic(local_status, drive_run / "STATUS.txt")
