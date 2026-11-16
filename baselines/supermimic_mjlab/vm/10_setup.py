"""Colab VM setup for official HybridRobotics/supermimic + V2 evaluator + Drive dataset cache."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

OFFICIAL_SM_COMMIT = os.environ.get("OFFICIAL_SM_COMMIT", "abc124c1bd5976341034668af3d12c087f2f1f55")
MJLAB_RESEARCH_COMMIT = os.environ.get("MJLAB_RESEARCH_COMMIT", "dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd")
BENCHMARK_BRANCH = os.environ.get("BENCHMARK_BRANCH", "main")
BENCHMARK_REPO = os.environ.get("BENCHMARK_REPO", "basicv8vc/modded-supermimic")
BENCHMARK_SUBDIR = os.environ.get("BENCHMARK_SUBDIR", ".")
REQUIRE_G4 = os.environ.get("REQUIRE_G4", "1") == "1"
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"


def run(cmd: str, cwd: str | None = None, redact: str | None = None) -> None:
    shown = cmd.replace(redact, "***") if redact else cmd
    print(f"\n>>> {shown}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {shown}")


def _dir_has_40(path: Path) -> bool:
    return path.is_dir() and len(list(path.iterdir())) >= 40


def main() -> None:
    print("=== 1. Drive & Hardware Check ===", flush=True)
    if REQUIRE_DRIVE:
        drive_base = ds.require_drive()
    elif ds.drive_available():
        drive_base = ds.require_drive()
    else:
        drive_base = None
        print("[WARN] Drive not mounted (REQUIRE_DRIVE=0); dataset will not persist.", flush=True)

    run("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader")
    run("df -h /content && nproc")

    import torch  # noqa: E402

    cap = tuple(torch.cuda.get_device_capability(0))
    print("System torch:", torch.__version__, "cuda:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0), "cap:", cap, flush=True)
    if REQUIRE_G4 and cap != (12, 0):
        raise RuntimeError(f"Expected sm_120 (Colab G4 RTX PRO 6000), got {cap}. Set REQUIRE_G4=0 to override.")

    print("=== 2. Clone Benchmark Repo & Install V2 Stack ===", flush=True)
    gh_token = os.environ.get("GH_TOKEN", "")
    root = Path("/content/modded-supermimic")
    if not (root / ".git").exists():
        url = (
            f"https://x-access-token:{gh_token}@github.com/{BENCHMARK_REPO}.git"
            if gh_token
            else f"https://github.com/{BENCHMARK_REPO}.git"
        )
        run(f"git clone --depth 1 --branch {BENCHMARK_BRANCH} {url} {root}", redact=gh_token or None)

    bm = (root / BENCHMARK_SUBDIR).resolve()
    run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux rsync")
    run('pip -q install "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0" uv pytest')
    run('pip -q install "mjlab @ git+https://github.com/mujocolab/mjlab.git@0fb8a681136be94ffc636a3dd423cabb97d91f10"')
    run('python3 -m pip uninstall -y -q mujoco-core mujoco && python3 -m pip install -q --no-cache-dir --no-deps "mujoco==3.11.0"')

    print("=== 3. Clone Official HybridRobotics Repos & uv sync ===", flush=True)
    ws = Path("/content/supermimic_ws")
    ws.mkdir(parents=True, exist_ok=True)
    sm_dir = ws / "supermimic"
    mj_dir = ws / "mjlab_research"
    if not (sm_dir / ".git").exists():
        run("git clone https://github.com/HybridRobotics/supermimic.git", cwd=str(ws))
    run(f"git checkout -q {OFFICIAL_SM_COMMIT}", cwd=str(sm_dir))
    if not (mj_dir / ".git").exists():
        run("git clone https://github.com/HybridRobotics/mjlab_research.git", cwd=str(ws))
    run(f"git checkout -q {MJLAB_RESEARCH_COMMIT}", cwd=str(mj_dir))
    if not (sm_dir / ".venv/bin/python").exists():
        run("uv sync", cwd=str(sm_dir))
    run(
        '.venv/bin/python -c "import torch, mjlab, supermimic; x=torch.randn(8,8,device=\'cuda\'); _=x@x; '
        'print(\'official env ok:\', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"',
        cwd=str(sm_dir),
    )

    print("=== 4. Restore or Build 40-Clip Datasets (Drive-Backed) ===", flush=True)
    local_csv = bm / "data/lafan1_g1"
    local_v2_npz = bm / "data/lafan1_g1_npz"
    local_off_npz = Path("/content/lafan1_official_npz")
    drive_data = (drive_base / "data") if drive_base else None
    drive_ready_flag = (drive_data / "DATA_READY") if drive_data else None

    restored = False
    if drive_data and drive_ready_flag and drive_ready_flag.exists():
        d_csv = drive_data / "lafan1_g1"
        d_v2 = drive_data / "lafan1_g1_npz"
        d_off = drive_data / "lafan1_official_npz"
        if _dir_has_40(d_csv) and _dir_has_40(d_v2) and _dir_has_40(d_off):
            print(f"Restoring 40-clip datasets from Drive: {drive_data}", flush=True)
            ds.sync_tree(d_csv, local_csv)
            ds.sync_tree(d_v2, local_v2_npz)
            ds.sync_tree(d_off, local_off_npz)
            restored = True

    if not restored:
        run("python3 scripts/fetch_lafan1.py", cwd=str(bm))
        warm_script = Path("/tmp/warm_v2_cache.py")
        warm_script.write_text(
            "import sys, time\n"
            "sys.path.insert(0, '.')\n"
            "from benchmark.evaluator import constants as C\n"
            "from benchmark.v2 import batched as B, spec as S\n\n"
            "def main() -> None:\n"
            "    clips = list(S.load_protocol_v2()['clip_ids'])\n"
            "    t0 = time.perf_counter()\n"
            "    refs = B._load_references(clips, C.DATA_DIR, use_cache=True)\n"
            "    print(f'Warmed {len(refs)} V2 NPZs in {time.perf_counter()-t0:.1f}s')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        run(f"python3 {warm_script}", cwd=str(bm))

        # Build official MotionLoader NPZs directly from V2 NPZs (zero mismatch)
        build_off = Path("/tmp/build_official_npz.py")
        build_off.write_text(
            "import json, pathlib, sys\n"
            "import numpy as np\n"
            f"sys.path.insert(0, {str(bm)!r})\n"
            "import numpy as np\n"
            "from benchmark.evaluator import constants as C\n"
            f"proto = json.loads(pathlib.Path({str(bm / 'benchmark/v2/protocol_v2.json')!r}).read_text())\n"
            "out_dir = pathlib.Path('/content/lafan1_official_npz')\n"
            "out_dir.mkdir(parents=True, exist_ok=True)\n"
            f"src_dir = pathlib.Path({str(bm / 'data/lafan1_g1_npz')!r})\n"
            "joint_names = np.array(C.JOINT_ORDER)\n"
            "for cid in proto['clip_ids']:\n"
            "    with np.load(src_dir / f'{cid}.npz') as d:\n"
            "        np.savez(\n"
            "            out_dir / f'{cid}.npz',\n"
            "            root_pos_w=d['body_pos_w'][:, 0].astype(np.float32),\n"
            "            root_quat_w=d['body_quat_w'][:, 0].astype(np.float32),\n"
            "            root_lin_vel_w=d['body_lin_vel_w'][:, 0].astype(np.float32),\n"
            "            root_ang_vel_w=d['body_ang_vel_w'][:, 0].astype(np.float32),\n"
            "            joint_pos=d['joint_pos'].astype(np.float32),\n"
            "            joint_vel=d['joint_vel'].astype(np.float32),\n"
            "            joint_names=joint_names,\n"
            "        )\n"
            "print(f'Wrote {len(proto[\"clip_ids\"])} official NPZs')\n"
        )
        run(f"python3 {build_off}")

        if drive_data:
            print(f"Uploading dataset cache to Google Drive: {drive_data}", flush=True)
            ok1 = ds.sync_tree(local_csv, drive_data / "lafan1_g1")
            ok2 = ds.sync_tree(local_v2_npz, drive_data / "lafan1_g1_npz")
            ok3 = ds.sync_tree(local_off_npz, drive_data / "lafan1_official_npz")
            if ok1 and ok2 and ok3:
                (drive_data / "DATA_READY").write_text("40 clips csv + v2_npz + official_npz\n")
            else:
                print("[WARN] Partial Drive upload; DATA_READY not marked", flush=True)

    assert _dir_has_40(local_csv) and _dir_has_40(local_v2_npz) and _dir_has_40(local_off_npz)
    run("pytest -q tests/test_official_supermimic_policy.py tests/test_v2_batched.py", cwd=str(bm))
    print("=== SETUP DONE ===", flush=True)


if __name__ == "__main__":
    main()
