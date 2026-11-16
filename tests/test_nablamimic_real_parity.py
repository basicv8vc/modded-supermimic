"""Observation parity against the original environment, on real data.

This is the strongest check in the suite and it is opt-in by availability: it
needs a checkout of ``sailab-ai/NablaMimic``, its ``mujoco``/``jax``
dependencies, and at least one LAFAN1 G1 CSV. Where any of that is absent the
tests skip rather than pass silently, so it cannot be mistaken for coverage.

It exists because the pure-algebra tests in
``test_nablamimic_temporal_alignment.py`` can only check the frame *rule*; they
cannot catch a wrong rotation composition, a wrong quaternion convention or a
wrong velocity frame. Here the ground truth is the original code itself, running
on real MuJoCo forward kinematics over a real clip.

Environment:
    NABLAMIMIC_ROOT     path to the upstream checkout (default ~/Downloads/NablaMimic)
    LAFAN1_G1_DIR       directory of the 40 CSVs (default $NABLAMIMIC_ROOT/data/lafan1_g1)
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

UPSTREAM = Path(os.environ.get("NABLAMIMIC_ROOT", Path.home() / "Downloads/NablaMimic"))
DATA = Path(os.environ.get("LAFAN1_G1_DIR", UPSTREAM / "data/lafan1_g1"))
CLIP = "walk3_subject4"


def _upstream_ready() -> bool:
    if not (UPSTREAM / "src/nablamimic/env.py").exists():
        return False
    if importlib.util.find_spec("mujoco") is None or importlib.util.find_spec("jax") is None:
        return False
    return (DATA / f"{CLIP}.csv").exists()


@unittest.skipUnless(
    _upstream_ready(),
    f"needs the upstream checkout at {UPSTREAM}, mujoco+jax, and {CLIP}.csv in {DATA}",
)
class RealClipParity(unittest.TestCase):
    """One clip, three frames, both adapters, against the original env."""

    @classmethod
    def setUpClass(cls):
        from scripts.audit.parity_original_env import UpstreamRig

        cls.rig = UpstreamRig(CLIP, seed=7)
        last = int(cls.rig.lib.last[0])
        cls.frames = [1, last // 2, max(last - 2, 1)]

    def _observe(self, frame: int, prefix: bool):
        rig = self.rig
        qpos, qvel = rig.drifted_state(frame)
        dx = rig.forward(qpos, qvel)
        prev_a = np.linspace(-0.3, 0.3, rig.nu)
        p_true, g_true, anchor = rig.truth(dx, frame, prev_a)
        step = rig.adapter_step(qpos, qvel, frame, prev_a, anchor)
        proprio, goal = rig.adapter_obs(step, frame, prefix=prefix)
        return proprio, goal, p_true, g_true

    def test_gravity_matches_the_original(self):
        for frame in self.frames:
            with self.subTest(frame=frame):
                proprio, _, truth, _ = self._observe(frame, prefix=False)
                err = float(np.abs(proprio[61:64] - truth[61:64]).max())
                self.assertLess(err, 1e-5, f"gravity channel differs by {err:.3e}")

    def test_goal_matches_the_original(self):
        for frame in self.frames:
            with self.subTest(frame=frame):
                _, goal, _, truth = self._observe(frame, prefix=False)
                self.assertEqual(goal.shape, truth.shape)
                err = float(np.abs(goal - truth).max())
                self.assertLess(err, 2e-4, f"686-dim goal differs by {err:.3e}")

    def test_whole_proprioception_matches_the_original(self):
        for frame in self.frames:
            with self.subTest(frame=frame):
                proprio, _, truth, _ = self._observe(frame, prefix=False)
                err = float(np.abs(proprio - truth).max())
                self.assertLess(err, 2e-4, f"93-dim proprioception differs by {err:.3e}")

    def test_the_check_has_power(self):
        """The pre-fix adapter must fail these tolerances on the same data.

        Not at every frame: the error is ``omega_torso * dt``, so the initial
        frame (from rest) and a slow near-final frame cannot expose it. What the
        test pins is that (a) at the clip's fastest torso frame the old adapter
        is off by more than 1e-2, and (b) at frame 1 it is *not* — which is
        precisely why a short test starting from rest saw only ~1e-3.
        """
        ang = np.abs(np.asarray(self.rig.lib.qvel[:, 3:6])).sum(1)
        end = int(self.rig.lib.last[0])
        fast = int(np.argmax(np.where(np.arange(len(ang)) < end, ang, -1.0)))

        proprio, goal, truth_p, truth_g = self._observe(fast, prefix=True)
        gravity_err = float(np.abs(proprio[61:64] - truth_p[61:64]).max())
        goal_err = float(np.abs(goal - truth_g).max())
        self.assertGreater(gravity_err, 1e-2, f"frame {fast}: {gravity_err:.3e}")
        self.assertGreater(goal_err, 1e-2, f"frame {fast}: {goal_err:.3e}")

        proprio, _, truth_p, _ = self._observe(1, prefix=True)
        at_rest = float(np.abs(proprio[61:64] - truth_p[61:64]).max())
        self.assertLess(at_rest, 5e-3, f"frame 1: {at_rest:.3e}")


if __name__ == "__main__":
    unittest.main()
