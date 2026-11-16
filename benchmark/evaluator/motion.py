"""Reference motions: official mjlab conversion + loading.

The LAFAN1 CSV -> 50 Hz npz conversion and the npz layout are mjlab v1.6.0
(commit ``0fb8a681136be94ffc636a3dd423cabb97d91f10``) code:

* resampling / SLERP / velocity gradients:
  ``mjlab.scripts.csv_to_npz.MotionLoader``;
* forward kinematics: the same teleport + ``sim.forward()`` loop as
  ``csv_to_npz.run_sim`` -- root/joint state written through the official
  ``Scene``/``Entity`` API, ``robot.data.body_link_*`` read back;
* npz layout: the six float32 arrays that
  ``mjlab.tasks.tracking.mdp.commands.MotionLoader`` loads at runtime.

Nothing here re-implements interpolation, quaternion or kinematics math. This
module only wires the official pieces to the benchmark's cache layout
(``data/lafan1_g1_npz/<clip>.npz``, with the same provenance keys the cache
has always carried: ``clip_id`` / ``fps`` / ``source_sha256``, plus a
``converter_version`` tag -- see :data:`CONVERTER_VERSION` -- so caches from
a different conversion pipeline are rebuilt instead of silently loaded) and
exposes :class:`MotionReference`, the read-only arrays the policy contract
consumes.

The official ``csv_to_npz.py`` CLI uploads its npz to Weights & Biases and
hardcodes ``/tmp/motion.npz``; those are workflow artifacts of its CLI, not
algorithm code, so the glue writes to the benchmark cache instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import constants as C

NPZ_CACHE_DIR = C.REPO_ROOT / "data" / "lafan1_g1_npz"

_INPUT_FPS = 30.0
_OUTPUT_FPS = 50.0

# Identity of the CSV -> npz conversion pipeline. Bump whenever the
# conversion semantics change (resampling rule, FK readout, velocity
# convention, ...). The tag is stored inside every npz; a cache whose tag
# is missing or different is rebuilt instead of loaded, so stale caches
# (e.g. the pre-refactor numpy FK, whose body_lin_vel_w convention differs
# from the official body_link_* readout) can never silently poison a run.
CONVERTER_VERSION = "official-csv-to-npz-mjlab1.6.0-1"


@dataclass
class MotionReference:
    """Reference motion at 50 Hz (``T`` frames, first frame == frame 0)."""

    clip_id: str
    joint_pos: np.ndarray  # (T, 29) robot joint order == LAFAN1 column order
    joint_vel: np.ndarray  # (T, 29)
    body_pos_w: np.ndarray  # (T, nbody-1, 3) indexed by model body id - 1
    body_quat_w: np.ndarray  # (T, nbody-1, 4) wxyz
    body_lin_vel_w: np.ndarray  # (T, nbody-1, 3)
    body_ang_vel_w: np.ndarray  # (T, nbody-1, 3)
    source_sha256: str = ""
    fps: float = _OUTPUT_FPS
    npz_path: Path | None = field(default=None, compare=False)

    @property
    def num_frames(self) -> int:
        """``MotionLoader.time_step_total``: number of 50 Hz frames."""
        return self.joint_pos.shape[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialConverter:
    """CSV -> npz through mjlab's own resampler and forward kinematics.

    One ``Scene`` + ``Simulation`` is built and reused across clips; each frame
    writes the reference root/joint state into the sim and reads the body
    poses/velocities back, exactly like ``csv_to_npz.run_sim``.
    """

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from mjlab.scene import Scene
        from mjlab.sim.sim import Simulation, SimulationCfg
        from mjlab.tasks.tracking.config.g1.env_cfgs import (
            unitree_g1_flat_tracking_env_cfg,
        )

        self.device = device
        self._torch = torch
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = 1.0 / _OUTPUT_FPS
        self.scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
        model = self.scene.compile()
        self.sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]
        self._joint_indexes = self.robot.find_joints(
            list(C.JOINT_ORDER), preserve_order=True
        )[0]

    def convert(self, csv_path: Path, npz_path: Path) -> MotionReference:
        from mjlab.scripts.csv_to_npz import MotionLoader

        torch = self._torch
        motion = MotionLoader(
            motion_file=str(csv_path),
            input_fps=int(_INPUT_FPS),
            output_fps=int(_OUTPUT_FPS),
            device=self.device,
        )

        keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        log: dict[str, list] = {k: [] for k in keys}
        self.scene.reset()

        while True:
            (
                (base_pos, base_rot, base_lin, base_ang, dof_pos, dof_vel),
                reset_flag,
            ) = motion.get_next_state()

            root_states = self.robot.data.default_root_state.clone()
            root_states[:, 0:3] = base_pos
            root_states[:, :2] += self.scene.env_origins[:, :2]
            root_states[:, 3:7] = base_rot
            root_states[:, 7:10] = base_lin
            root_states[:, 10:] = base_ang
            self.robot.write_root_state_to_sim(root_states)

            joint_pos = self.robot.data.default_joint_pos.clone()
            joint_vel = self.robot.data.default_joint_vel.clone()
            joint_pos[:, self._joint_indexes] = dof_pos
            joint_vel[:, self._joint_indexes] = dof_vel
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

            self.sim.forward()
            self.scene.update(self.sim.mj_model.opt.timestep)

            log["joint_pos"].append(self.robot.data.joint_pos[0].cpu().numpy())
            log["joint_vel"].append(self.robot.data.joint_vel[0].cpu().numpy())
            log["body_pos_w"].append(self.robot.data.body_link_pos_w[0].cpu().numpy())
            log["body_quat_w"].append(self.robot.data.body_link_quat_w[0].cpu().numpy())
            log["body_lin_vel_w"].append(
                self.robot.data.body_link_lin_vel_w[0].cpu().numpy()
            )
            log["body_ang_vel_w"].append(
                self.robot.data.body_link_ang_vel_w[0].cpu().numpy()
            )

            # Same sanity check csv_to_npz.run_sim makes: the root's link
            # velocity must equal the state we just wrote. The tolerance is
            # looser than torch's float32 defaults because the CPU warp
            # round-trip accumulates ~1e-5 on fast roots; the check still
            # catches unit/frame/sign errors, which are O(1).
            torch.testing.assert_close(
                self.robot.data.body_link_lin_vel_w[0, 0],
                base_lin[0],
                rtol=1e-5,
                atol=5e-5,
            )
            torch.testing.assert_close(
                self.robot.data.body_link_ang_vel_w[0, 0],
                base_ang[0],
                rtol=1e-5,
                atol=5e-5,
            )

            if reset_flag:
                break

        arrays = {k: np.stack(v, axis=0).astype(np.float32) for k, v in log.items()}
        reference = MotionReference(
            clip_id=csv_path.stem,
            joint_pos=arrays["joint_pos"],
            joint_vel=arrays["joint_vel"],
            body_pos_w=arrays["body_pos_w"],
            body_quat_w=arrays["body_quat_w"],
            body_lin_vel_w=arrays["body_lin_vel_w"],
            body_ang_vel_w=arrays["body_ang_vel_w"],
            source_sha256=sha256_file(csv_path),
            fps=_OUTPUT_FPS,
            npz_path=npz_path,
        )
        save_reference(reference, npz_path)
        return reference


_converter: OfficialConverter | None = None


def _get_converter() -> OfficialConverter:
    global _converter
    if _converter is None:
        _converter = OfficialConverter(device="cpu")
    return _converter


def save_reference(ref: MotionReference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        clip_id=np.array(ref.clip_id),
        fps=np.array(ref.fps),
        source_sha256=np.array(ref.source_sha256),
        converter_version=np.array(CONVERTER_VERSION),
        joint_pos=ref.joint_pos,
        joint_vel=ref.joint_vel,
        body_pos_w=ref.body_pos_w,
        body_quat_w=ref.body_quat_w,
        body_lin_vel_w=ref.body_lin_vel_w,
        body_ang_vel_w=ref.body_ang_vel_w,
    )


def load_reference_npz(path: Path) -> MotionReference:
    with np.load(path) as data:
        return MotionReference(
            clip_id=str(data["clip_id"]),
            fps=float(data["fps"]),
            source_sha256=str(data["source_sha256"]),
            joint_pos=data["joint_pos"],
            joint_vel=data["joint_vel"],
            body_pos_w=data["body_pos_w"],
            body_quat_w=data["body_quat_w"],
            body_lin_vel_w=data["body_lin_vel_w"],
            body_ang_vel_w=data["body_ang_vel_w"],
            npz_path=path,
        )


def _cache_is_current(ref: MotionReference, csv_path: Path) -> bool:
    """A cache is usable only if it matches BOTH the source CSV and the
    conversion pipeline that produced it.

    The converter tag is read straight from the file so a cache written by
    an older pipeline (no tag / different tag) is rejected even when the
    CSV is unchanged.
    """
    if ref.source_sha256 != sha256_file(csv_path):
        return False
    try:
        with np.load(ref.npz_path) as data:
            tag = str(data["converter_version"]) if "converter_version" in data else ""
    except OSError:
        return False
    return tag == CONVERTER_VERSION


def get_reference(
    csv_path: Path, cache_dir: Path | None = None, use_cache: bool = True
) -> MotionReference:
    """Build (or read) the 50 Hz reference for one clip.

    The npz file is not optional: the official runtime
    (``MotionCommand``/``MotionLoader``) loads the reference from it. The
    cache doubles as that file. ``use_cache=False`` forces reconversion from
    the CSV through :class:`OfficialConverter`. A cache is also rebuilt
    automatically when its embedded ``converter_version`` tag does not
    match :data:`CONVERTER_VERSION` (see :func:`_cache_is_current`).
    """
    clip_id = csv_path.stem
    cache = (cache_dir or NPZ_CACHE_DIR) / f"{clip_id}.npz"
    if use_cache and cache.exists():
        ref = load_reference_npz(cache)
        if _cache_is_current(ref, csv_path):
            return ref
    return _get_converter().convert(csv_path, cache)
