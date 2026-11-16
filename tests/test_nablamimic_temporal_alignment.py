"""Temporal alignment of the NablaMimic adapter's observations.

What this file is for
---------------------
The adapter used to build one observation out of two different reference
frames: goal features and the residual base from frame ``f = reference_index-1``,
but the anchor orientation from ``reference_index = f+1`` (because mjlab's
``motion_anchor_ori_b`` always reads ``time_steps``). The gravity channel
composes ``A(q) M A(q_ref)^T g_ref``, so a wrong-frame ``M`` leaks the
reference torso's one-frame rotation straight into ``proprio[61:64]``, and the
last six goal entries condition the decoder on the wrong orientation.

These tests pin the frame algebra *without* a simulator, because the claim is
algebraic: with the frame-``f`` anchor the composition reproduces
``R_pelvis^T g`` **exactly**, and with the frame-``f+1`` anchor it does not.
Both halves are asserted -- a test that only checked the first would also pass
if the anchor were ignored entirely.

The synthetic clip is built from explicit rotations rather than sampled from
LAFAN1, so the tests run on any machine with numpy (no MuJoCo, no mjlab, no
checkpoint). They verify *semantics and frame handling*, not the numbers of the
evaluated policy; ``tests/test_nablamimic_policy.py`` holds the
simulator-backed checks.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import mathx  # noqa: E402
from benchmark.evaluator.policy_api import (  # noqa: E402
    ClipContext,
    ReferenceClip,
    StepInput,
)
from benchmark.policies import nablamimic as nm  # noqa: E402

DT = C.STEP_DT
T = 40
G_WORLD = np.array([0.0, 0.0, -1.0])
TOL = 1e-12


def _rot(axis, angle: float) -> np.ndarray:
    return nm.axis_rotation(tuple(float(a) for a in axis), float(angle))


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) of a rotation matrix, largest-component branch."""
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


class SyntheticClip:
    """A T-frame clip whose torso turns at a controlled rate."""

    def __init__(self, torso_rate: float = 2.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.torso_rate = float(torso_rate)
        t = np.arange(T) * DT
        # Yaw sweep + the controlled roll sweep: the torso genuinely rotates
        # between consecutive frames, which is what the frame test needs.
        self.root_rot = np.stack(
            [
                _rot((0, 0, 1), 0.4 * t[i])
                @ _rot((1, 0, 0), self.torso_rate * t[i])
                @ _rot((0, 1, 0), 0.05 * np.sin(6.0 * t[i]))
                for i in range(T)
            ]
        )
        self.root_pos = np.stack(
            [0.3 * t, 0.02 * np.sin(3 * t), 0.75 + 0.01 * np.sin(9 * t)], axis=1
        )
        self.joint = 0.25 * np.sin(
            np.outer(t, np.linspace(1.0, 3.0, nm.NUM_J)) + rng.uniform(0, 1, nm.NUM_J)
        )
        self.joint_vel = np.gradient(self.joint, DT, axis=0)
        self.ref = ReferenceClip(
            num_frames=T,
            control_dt=DT,
            joint_pos=self.joint,
            joint_vel=self.joint_vel,
            root_pos_w=self.root_pos,
            root_quat_w=np.stack([_quat_from_mat(m) for m in self.root_rot]),
        )

    def A(self, frame: int) -> np.ndarray:
        return nm.waist_rotation(self.joint[max(frame, 0)])

    def ref_torso_rot(self, frame: int) -> np.ndarray:
        """``R_pelvis(frame) @ A(q_ref)``: the reference torso orientation."""
        return self.root_rot[frame] @ self.A(frame)

    def torso_step_rad(self) -> float:
        """Rotation angle of the reference torso between two frames, radians."""
        return float(mathx.quat_error_magnitude(
            _quat_from_mat(self.ref_torso_rot(5)), _quat_from_mat(self.ref_torso_rot(6))
        ))


def _anchor6(clip: SyntheticClip, robot_rot: np.ndarray, q: np.ndarray, frame: int) -> np.ndarray:
    """``R_robot_torso^T R_ref_torso(frame)`` in mjlab's interleaved 6-D layout."""
    R_robot_torso = robot_rot @ nm.waist_rotation(q)
    m = R_robot_torso.T @ clip.ref_torso_rot(frame)
    return m[:, :2].reshape(-1)


def _tiny_policy() -> nm.NablaMimicPolicy:
    """A structurally valid actor: 680 -> tanh -> concat(history, anchor) -> 29."""
    rng = np.random.default_rng(7)
    latent, hidden = 8, 16
    hist = nm.HIST * nm.PROPRIO_DIM
    enc = [
        (rng.normal(0, 0.05, (nm.GOAL_DIM - nm.ANCHOR_DIM, hidden)), np.zeros(hidden)),
        (rng.normal(0, 0.05, (hidden, latent)), np.zeros(latent)),
    ]
    dec = [
        (rng.normal(0, 0.05, (latent + hist + nm.ANCHOR_DIM, hidden)), np.zeros(hidden)),
        (rng.normal(0, 0.05, (hidden, nm.NUM_J)), np.zeros(nm.NUM_J)),
    ]
    return nm.NablaMimicPolicy({"enc": enc, "dec": dec, "anchor_dec": np.zeros(1)})


class Rig:
    """A policy plus the physical quantities a ``StepInput`` would carry."""

    def __init__(self, clip: SyntheticClip, robot_offset_rad: float = 0.25):
        self.clip = clip
        self.policy = _tiny_policy()
        self.q0 = nm.default_joint_pos_reference()
        self.scale = nm.action_scale_reference()
        # The robot is near the reference but not on it, which is the normal
        # closed-loop situation and makes a wrong-frame anchor detectable.
        self.robot_rot = np.stack(
            [_rot((0, 1, 0), robot_offset_rad) @ R for R in clip.root_rot]
        )
        self.policy.reset(
            ClipContext(
                clip="synthetic",
                joint_order=C.JOINT_ORDER,
                default_joint_pos=self.q0,
                action_scale=self.scale,
                control_dt=DT,
                reference=clip.ref,
            )
        )

    def true_gravity(self, frame: int) -> np.ndarray:
        """``R_pelvis^T g``: what ``env.proprio`` reads from the robot's own root quat."""
        return self.robot_rot[frame].T @ G_WORLD

    def step(self, frame: int, *, q: np.ndarray | None = None, previous=None) -> StepInput:
        clip = self.clip
        q = clip.joint[frame] if q is None else q
        return StepInput(
            step=frame,
            reference_index=frame + 1,
            time_s=frame * DT,
            joint_pos=q,
            joint_vel=clip.joint_vel[frame],
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.array([0.1, -0.2, 0.3]),
            anchor_pos_b=np.zeros(3),
            # mjlab's field: the anchor at reference_index (frame + 1).
            anchor_ori_b=_anchor6(clip, self.robot_rot[frame], q, frame + 1),
            previous_action=np.zeros(nm.NUM_J) if previous is None else previous,
            reference=clip.ref,
            anchor_pos_b_prev=np.zeros(3),
            # The field this adapter must read: the anchor at the state's frame.
            anchor_ori_b_prev=_anchor6(clip, self.robot_rot[frame], q, frame),
        )


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


class AnchorFrameTest(unittest.TestCase):
    """The adapter must anchor at the frame its state belongs to."""

    def setUp(self):
        self.clip = SyntheticClip(torso_rate=6.0)
        self.rig = Rig(self.clip)

    def test_proprioception_uses_frame_f_not_f_plus_1(self):
        for frame in (0, 1, 7, 20, 35):
            step = self.rig.step(frame)
            obs = self.rig.policy.proprioception(step, frame)
            np.testing.assert_allclose(
                obs[61:64], self.rig.true_gravity(frame), atol=TOL, err_msg=f"frame {frame}"
            )

    def test_swapping_the_anchor_fields_breaks_gravity(self):
        """Power check: the test above is not vacuous."""
        frame = 10
        step = self.rig.step(frame)
        swapped = StepInput(
            **{
                **step.__dict__,
                "anchor_ori_b_prev": step.anchor_ori_b,
                "anchor_ori_b": step.anchor_ori_b_prev,
            }
        )
        obs = self.rig.policy.proprioception(swapped, frame)
        error = float(np.abs(obs[61:64] - self.rig.true_gravity(frame)).max())
        self.assertGreater(error, 0.1)

    def test_goal_anchor_block_is_frame_f(self):
        frame = 12
        step = self.rig.step(frame)
        goal = self.rig.policy.goal(frame, step)
        expected = nm.anchor_ori_stacked(
            _anchor6(self.clip, self.rig.robot_rot[frame], self.clip.joint[frame], frame)
        )
        np.testing.assert_allclose(goal[-nm.ANCHOR_DIM :], expected, atol=TOL)

        wrong = nm.anchor_ori_stacked(
            _anchor6(self.clip, self.rig.robot_rot[frame], self.clip.joint[frame], frame + 1)
        )
        self.assertGreater(float(np.abs(goal[-nm.ANCHOR_DIM :] - wrong).max()), 1e-2)

    def test_missing_field_fails_closed(self):
        frame = 3
        step = self.rig.step(frame)
        stripped = types.SimpleNamespace(
            **{k: v for k, v in step.__dict__.items() if k != "anchor_ori_b_prev"}
        )
        with self.assertRaises(nm.NablaMimicContractError):
            self.rig.policy.act(stripped)


# ---------------------------------------------------------------------------
# Gravity reconstruction across angular velocities
# ---------------------------------------------------------------------------


class GravityReconstructionTest(unittest.TestCase):
    """``A(q) M(f) A(q_ref)^T g_ref`` is exact; with ``M(f+1)`` it is not."""

    def test_exact_at_frame_f(self):
        for rate in (0.5, 2.0, 6.0, 12.0):
            rig = Rig(SyntheticClip(torso_rate=rate))
            for frame in (0, 5, 19, 33):
                obs = rig.policy.proprioception(rig.step(frame), frame)
                np.testing.assert_allclose(
                    obs[61:64], rig.true_gravity(frame), atol=1e-10,
                    err_msg=f"rate {rate} frame {frame}",
                )

    def test_mixed_frame_error_tracks_the_one_frame_rotation(self):
        """Quantifies the bug: the injected error is the torso's one-frame turn."""
        rows = []
        for rate in (0.5, 2.0, 6.0, 12.0):
            clip = SyntheticClip(torso_rate=rate)
            rig = Rig(clip)
            frame = 10
            step = rig.step(frame)
            # Feed the frame-(f+1) anchor where the frame-f one belongs: the
            # exact mistake the adapter used to make.
            buggy = StepInput(
                **{**step.__dict__, "anchor_ori_b_prev": step.anchor_ori_b}
            )
            obs = rig.policy.proprioception(buggy, frame)
            error = float(np.abs(obs[61:64] - rig.true_gravity(frame)).max())
            theta = clip.torso_step_rad()
            rows.append((rate, theta, error))
            # A rotation of theta can move a unit vector by at most 2 sin(theta/2).
            self.assertLessEqual(error, 2.0 * np.sin(theta / 2.0) + 1e-9)
            self.assertGreater(error, 0.0)

        # Monotone in angular velocity: more rotation per frame, more error.
        errors = [r[2] for r in rows]
        self.assertEqual(errors, sorted(errors), f"not monotone: {rows}")
        # At 12 rad/s the one-frame torso turn is large enough that a 1e-3
        # gravity tolerance could never have caught it.
        self.assertGreater(rows[-1][2], 0.05)


# ---------------------------------------------------------------------------
# Full observation vectors
# ---------------------------------------------------------------------------


class ObservationVectorTest(unittest.TestCase):
    def setUp(self):
        self.clip = SyntheticClip(torso_rate=3.0)
        self.rig = Rig(self.clip)

    def test_proprioception_all_93(self):
        frame = 9
        q = self.clip.joint[frame] + 0.01  # robot is not exactly on the reference
        step = self.rig.step(frame, q=q)
        obs = self.rig.policy.proprioception(step, frame)
        self.assertEqual(obs.shape, (nm.PROPRIO_DIM,))

        np.testing.assert_allclose(obs[0:29], q - self.rig.q0, atol=TOL)
        np.testing.assert_allclose(obs[29:58], step.joint_vel * 0.2, atol=TOL)
        np.testing.assert_allclose(obs[58:61], step.base_ang_vel * 0.5, atol=TOL)
        np.testing.assert_allclose(obs[61:64], self.rig.true_gravity(frame), atol=1e-10)
        # 64:93 is the previous *raw* action; zero before the first step.
        np.testing.assert_allclose(obs[64:93], np.zeros(nm.NUM_J), atol=TOL)

    def test_goal_all_680_features(self):
        """Every goal feature, checked against the original's construction."""
        clip, rig = self.clip, self.rig
        frame = 8
        goal = rig.policy.goal(frame, rig.step(frame))
        self.assertEqual(goal.shape, (nm.GOAL_DIM,))

        q0 = rig.q0
        pos, quat = clip.root_pos, np.stack([_quat_from_mat(m) for m in clip.root_rot])
        grav = np.stack([mathx.projected_gravity(q) for q in quat])
        lin_w, ang = nm._root_velocities(pos, quat, DT)  # noqa: SLF001
        yaw = np.arctan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        c, s = np.cos(yaw), np.sin(yaw)
        lin_h = np.stack(
            [c * lin_w[:, 0] + s * lin_w[:, 1], -s * lin_w[:, 0] + c * lin_w[:, 1], lin_w[:, 2]],
            axis=1,
        )
        feats = np.concatenate(
            [clip.joint - q0, clip.joint_vel * 0.2, grav, ang * 0.5, lin_h, pos[:, 2:3]], axis=1
        )
        idx = np.minimum(frame + np.arange(nm.FUTURE), T - 1)
        np.testing.assert_allclose(goal[: nm.FUTURE * nm.GOAL_FRAME_DIM], feats[idx].reshape(-1), atol=TOL)

    def test_goal_window_clamps_at_clip_end(self):
        last = T - 1
        goal = self.rig.policy.goal(last - 2, self.rig.step(last - 2))
        block = goal[: nm.FUTURE * nm.GOAL_FRAME_DIM].reshape(nm.FUTURE, nm.GOAL_FRAME_DIM)
        # Frames last-2, last-1, last, last, last, ...
        np.testing.assert_allclose(block[2:], np.repeat(block[2:3], nm.FUTURE - 2, axis=0), atol=TOL)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class HistoryTest(unittest.TestCase):
    """``evaluate.py`` tiles the first observation, then rolls a 10-frame window."""

    def _run(self, frames):
        clip = SyntheticClip(torso_rate=4.0)
        rig = Rig(clip)
        seen = []
        prev = np.zeros(nm.NUM_J)
        for frame in frames:
            step = rig.step(frame, previous=prev)
            obs = rig.policy.proprioception(step, frame)
            seen.append(obs)
            prev = rig.policy.act(step)
        return rig, seen

    def test_first_observation_is_tiled_then_rolls(self):
        frames = [0, 1, 2]
        rig, seen = self._run(frames)
        # After three acts the buffer holds obs(0), obs(0), ..., obs(0), obs(1), obs(2).
        expected = np.stack([seen[0]] * (nm.HIST - 2) + [seen[1], seen[2]])
        np.testing.assert_allclose(rig.policy._hist, expected, atol=TOL)  # noqa: SLF001

    def test_history_is_newest_last(self):
        rig, seen = self._run([0, 1])
        np.testing.assert_allclose(rig.policy._hist[-1], seen[1], atol=TOL)  # noqa: SLF001
        np.testing.assert_allclose(rig.policy._hist[-2], seen[0], atol=TOL)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class ActionParityTest(unittest.TestCase):
    """Residual conversion and action-rate semantics."""

    def setUp(self):
        self.clip = SyntheticClip(torso_rate=2.0)
        self.rig = Rig(self.clip)

    def test_residual_rewrite_commands_the_same_setpoint(self):
        rig, clip = self.rig, self.clip
        prev = np.zeros(nm.NUM_J)
        for frame in (0, 1, 5, 30):
            step = rig.step(frame, previous=prev)
            action = rig.policy.act(step)
            raw = rig.policy._prev_action_raw  # noqa: SLF001
            k = min(frame + 1, T - 1)
            # NablaMimic: target = ref[min(f+1, last)] + scale * raw.
            np.testing.assert_allclose(
                rig.q0 + rig.scale * action, clip.joint[k] + rig.scale * raw, atol=1e-12
            )
            prev = action

    def test_action_rate_matches_the_original_setpoint_rule(self):
        """``u_t - u_{t-1}`` over scale equals the adapter's action difference.

        This is why rewriting the residual around ``q0`` does not perturb the
        canonical action-rate term: the reference base advances with the frame,
        so its contribution is exactly the original's ``ref[f+1] - ref[f]``.
        """
        rig, clip = self.rig, self.clip
        scale = rig.scale
        prev_action = np.zeros(nm.NUM_J)
        prev_raw = np.zeros(nm.NUM_J)
        prev_frame = None
        for frame in (0, 1, 2, 3, 20, 21):
            step = rig.step(frame, previous=prev_action)
            action = rig.policy.act(step)
            raw = rig.policy._prev_action_raw  # noqa: SLF001

            if prev_frame is None:
                # Reset case: the original treats the previous setpoint as q0.
                expected = raw + (clip.joint[min(frame + 1, T - 1)] - rig.q0) / scale
                np.testing.assert_allclose(action, expected, atol=1e-12)
            else:
                k, k_prev = min(frame + 1, T - 1), min(prev_frame + 1, T - 1)
                original_change = (raw - prev_raw) + (clip.joint[k] - clip.joint[k_prev]) / scale
                adapter_change = action - prev_action
                np.testing.assert_allclose(adapter_change, original_change, atol=1e-12)

            prev_action, prev_raw, prev_frame = action, raw, frame

    def test_previous_action_is_the_raw_output_not_the_benchmark_action(self):
        rig = self.rig
        step = rig.step(0)
        action = rig.policy.act(step)
        raw = rig.policy._prev_action_raw  # noqa: SLF001
        self.assertGreater(float(np.abs(action - raw).max()), 1e-3)
        obs = rig.policy.proprioception(rig.step(1, previous=action), 1)
        np.testing.assert_allclose(obs[64:93], raw, atol=TOL)

    def test_history_desync_is_detected(self):
        rig = self.rig
        rig.policy.act(rig.step(0))
        with self.assertRaises(nm.NablaMimicContractError):
            rig.policy.act(rig.step(1, previous=np.zeros(nm.NUM_J) + 0.5))


# ---------------------------------------------------------------------------
# Waist rotation, independently
# ---------------------------------------------------------------------------


class WaistRotationTest(unittest.TestCase):
    def test_matches_quaternion_composition_in_documented_order(self):
        """``Rz(yaw) Rx(roll) Ry(pitch)``, cross-checked with quaternions."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            yaw, roll, pitch = rng.uniform(-1.2, 1.2, 3)
            q = np.zeros(nm.NUM_J)
            q[nm._WAIST_IDX[0]] = yaw  # noqa: SLF001
            q[nm._WAIST_IDX[1]] = roll  # noqa: SLF001
            q[nm._WAIST_IDX[2]] = pitch  # noqa: SLF001
            mat = nm.waist_rotation(q)

            quat = mathx.quat_mul(
                mathx.quat_mul(
                    _quat_from_mat(_rot((0, 0, 1), yaw)),
                    _quat_from_mat(_rot((1, 0, 0), roll)),
                ),
                _quat_from_mat(_rot((0, 1, 0), pitch)),
            )
            np.testing.assert_allclose(mat, mathx.matrix_from_quat(quat), atol=1e-12)
            np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
