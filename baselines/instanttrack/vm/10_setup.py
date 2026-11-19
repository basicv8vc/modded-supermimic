"""Colab G4 VM setup for InstantTrack v1.4 / BlinkMimic + V2 evaluator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_sync as ds  # noqa: E402

UPSTREAM = os.environ.get("UPSTREAM", "instanttrack").strip().lower()
UPSTREAM_DEFAULTS = {
    "instanttrack": (
        "https://github.com/Huy-Quang-Dao/InstantTrack.git",
        "b9512a96d4e82e6ddea9a6de3b03276888bbcb2c",  # v1.4.0
        "instanttrack",
    ),
    "blinkmimic": (
        "https://github.com/fuyichen0725/blinkmimic.git",
        "d264658d6c9ce61fb52f9f0f8985a369f3ccc409",  # v1.0.0
        "blinkmimic",
    ),
}
if UPSTREAM not in UPSTREAM_DEFAULTS:
    raise SystemExit(f"Unsupported UPSTREAM={UPSTREAM!r}; choose instanttrack or blinkmimic")

DEFAULT_URL, DEFAULT_COMMIT, PKG_NAME = UPSTREAM_DEFAULTS[UPSTREAM]
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", DEFAULT_URL)
UPSTREAM_COMMIT = os.environ.get("UPSTREAM_COMMIT", DEFAULT_COMMIT)
BENCHMARK_BRANCH = os.environ.get("BENCHMARK_BRANCH", "main")
BENCHMARK_REPO = os.environ.get("BENCHMARK_REPO", "basicv8vc/modded-supermimic")
BENCHMARK_SUBDIR = os.environ.get("BENCHMARK_SUBDIR", ".")
REQUIRE_G4 = os.environ.get("REQUIRE_G4", "1") == "1"
REQUIRE_DRIVE = os.environ.get("REQUIRE_DRIVE", "1") == "1"
NCONMAX = int(os.environ.get("NCONMAX", "256"))
NJMAX = int(os.environ.get("NJMAX", "1024"))


def run(cmd: str, cwd: str | None = None, redact: str | None = None) -> None:
    shown = cmd.replace(redact, "***") if redact else cmd
    print(f"\n>>> {shown}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {shown}")


def _dir_has_40(path: Path) -> bool:
    return path.is_dir() and len(list(path.glob("*"))) >= 40


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

    print("=== 2. Clone Benchmark Repo & Install Frozen V2 Referee ===", flush=True)
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

    print(f"=== 3. Clone {UPSTREAM} @ {UPSTREAM_COMMIT[:12]} & uv sync ===", flush=True)
    ws = Path("/content/instanttrack_ws")
    ws.mkdir(parents=True, exist_ok=True)
    repo_dir = ws / "repo"
    if not (repo_dir / ".git").exists():
        run(f"git clone {UPSTREAM_URL} {repo_dir}")
    run("git checkout -q -- .", cwd=str(repo_dir))
    run(f"git fetch -q origin && git checkout -q {UPSTREAM_COMMIT}", cwd=str(repo_dir))
    if not (repo_dir / ".venv/bin/python").exists():
        run("uv sync", cwd=str(repo_dir))
    run(
        f'.venv/bin/python -c "import torch, mjlab, {PKG_NAME}; '
        'x=torch.randn(8,8,device=\\"cuda\\"); _=x@x; '
        'print(\\"upstream env ok:\\", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"',
        cwd=str(repo_dir),
    )

    print("=== 4. Restore or Fetch Pinned 40-Clip LAFAN1 Dataset ===", flush=True)
    local_csv = bm / "data/lafan1_g1"
    local_v2_npz = bm / "data/lafan1_g1_npz"
    drive_data = (drive_base / "data") if drive_base else None
    restored = False
    if drive_data and _dir_has_40(drive_data / "lafan1_g1") and _dir_has_40(drive_data / "lafan1_g1_npz"):
        print(f"Restoring 40-clip dataset + V2 NPZ cache from Drive: {drive_data}", flush=True)
        ds.sync_tree(drive_data / "lafan1_g1", local_csv)
        ds.sync_tree(drive_data / "lafan1_g1_npz", local_v2_npz)
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
        if drive_data:
            ds.sync_tree(local_csv, drive_data / "lafan1_g1")
            ds.sync_tree(local_v2_npz, drive_data / "lafan1_g1_npz")

    assert _dir_has_40(local_csv) and _dir_has_40(local_v2_npz)
    run("python3 scripts/freeze_manifest.py --check", cwd=str(bm))

    print("=== 5. Patch G1 Constants to V2 FULL_COLLISION & Smoke Check ===", flush=True)
    patcher = Path("/content/vm/patch_full_collision.py")
    if not patcher.is_file():
        patcher = bm / "baselines/instanttrack/vm/patch_full_collision.py"
    constants_file = repo_dir / "src" / PKG_NAME / "g1" / "g1_constants.py"
    run(
        f"python3 {patcher} --constants {constants_file} "
        f"--mjcf {bm / 'assets/g1/g1.xml'} --motion-dir {local_csv} "
        f"--nconmax {NCONMAX} --njmax {NJMAX}"
    )
    run(
        f'.venv/bin/python -c "'
        f"from {PKG_NAME}.g1.g1_constants import SPEC, get_spec; "
        f"s = get_spec(); "
        f"coll = [g for g in s.geoms if g.contype == 1 and g.conaffinity == 1]; "
        f"assert len(coll) > 0, 'no FULL_COLLISION geoms'; "
        f"print('FULL_COLLISION geoms:', len(coll), 'nconmax:', SPEC.nconmax, 'njmax:', SPEC.njmax)"
        f'"',
        cwd=str(repo_dir),
    )
    run(".venv/bin/python tools/test_env.py --num_envs 64", cwd=str(repo_dir))
    run("python3 -m pytest -q tests/test_instanttrack_policy.py tests/test_v2_batched.py", cwd=str(bm))
    print("=== SETUP DONE ===", flush=True)


if __name__ == "__main__":
    main()
