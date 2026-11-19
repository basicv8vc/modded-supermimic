"""Align upstream InstantTrack / BlinkMimic G1 constants to V2 FULL_COLLISION.

Replaces the upstream URDF-compiled MJCF and ``contype=0, conaffinity=1``
(self-collision disabled) setup with:
1. Frozen V2 ``assets/g1/g1.xml``;
2. Official ``FULL_COLLISION`` rules matching ``benchmark/evaluator/spec.py``;
3. Pinned 40-clip LAFAN1 directory from ``scripts/fetch_lafan1.py``;
4. ``nconmax`` / ``njmax`` capacity headroom for self-collision contacts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def patch_g1_constants(
    constants_path: Path,
    mjcf_path: Path,
    motion_dir: Path,
    *,
    nconmax: int = 256,
    njmax: int = 1024,
) -> None:
    text = constants_path.read_text(encoding="utf-8")

    # Point MJCF_PATH and MOTION_DIR to the frozen V2 assets/data.
    text, n1 = re.subn(
        r'^MJCF_PATH\s*=.*$',
        f'MJCF_PATH = {str(mjcf_path)!r}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, n2 = re.subn(
        r'^MOTION_DIR\s*=.*$',
        f'MOTION_DIR = {str(motion_dir)!r}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n1 != 1 or n2 != 1:
        raise RuntimeError(f"Failed to patch MJCF_PATH/MOTION_DIR in {constants_path} ({n1}, {n2})")

    # Replace get_spec() to enforce V2 FULL_COLLISION matching benchmark/evaluator/spec.py.
    new_get_spec = (
        "def get_spec() -> mujoco.MjSpec:\n"
        "    import re as _re\n"
        "    spec = mujoco.MjSpec.from_file(MJCF_PATH)\n"
        "    foot_re = _re.compile(r\"^(left|right)_foot[1-7]_collision$\")\n"
        "    coll_re = _re.compile(r\".*_collision$\")\n"
        "    for geom in spec.geoms:\n"
        "        name = geom.name or \"\"\n"
        "        if coll_re.fullmatch(name):\n"
        "            geom.contype = 1\n"
        "            geom.conaffinity = 1\n"
        "            if foot_re.fullmatch(name):\n"
        "                geom.condim = 3\n"
        "                geom.priority = 1\n"
        "                geom.friction[0] = 0.6\n"
        "            else:\n"
        "                geom.condim = 1\n"
        "                geom.priority = 0\n"
        "        else:\n"
        "            geom.contype = 0\n"
        "            geom.conaffinity = 0\n"
        "    return spec\n"
    )
    text, n3 = re.subn(
        r"def get_spec\(\) -> mujoco\.MjSpec:.*?(?=\ndef _actuators\(\))",
        new_get_spec + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n3 != 1:
        raise RuntimeError(f"Failed to patch get_spec() in {constants_path}")

    # Inject nconmax / njmax into SPEC = RobotSpec(...) cleanly (idempotent).
    text = re.sub(r"^\s*nconmax=.*,\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*njmax=.*,\n", "", text, flags=re.MULTILINE)
    text, n4 = re.subn(
        r"(load_motion=load_lafan1_csv,\n)\)",
        rf"\1    njmax={int(njmax)},\n    nconmax={int(nconmax)},\n)",
        text,
        count=1,
    )
    if n4 != 1:
        raise RuntimeError(f"Failed to patch SPEC RobotSpec(...) in {constants_path}")

    constants_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--nconmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=1024)
    args = parser.parse_args()
    patch_g1_constants(
        args.constants,
        args.mjcf,
        args.motion_dir,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    print(
        f"Patched {args.constants} -> mjcf={args.mjcf} motion_dir={args.motion_dir} "
        f"(FULL_COLLISION, njmax={args.njmax}, nconmax={args.nconmax})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
