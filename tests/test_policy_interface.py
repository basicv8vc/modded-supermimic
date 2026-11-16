"""The evaluator must not care *how* a policy is built.

The benchmark owns the task, the robot, the control interface and the score. An
algorithm owns its observation encoding, history, network and checkpoint. These
tests pin that boundary:

* a policy whose observation is 7 numbers wide drives a real rollout;
* a checkpoint that contains no actor MLP is accepted (the evaluator only hashes
  the file, the algorithm loads it);
* a policy that keeps its own history buffer works;
* illegal actions (wrong width/rank, non-numeric, NaN/inf) never reach MuJoCo;
* the policy interface is exactly ``reset(context)`` / ``act(step)``.

The mjlab-native MLP stays available as *one* implementation; nothing here
requires it.
"""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import motion, policy, policy_api, runner, spec  # noqa: E402

SHORT_CLIP = "fallAndGetUp3_subject1"
SHORT_CSV = C.DATA_DIR / f"{SHORT_CLIP}.csv"


def _have_dataset() -> bool:
    return SHORT_CSV.exists()


def _scratch(suffix: str) -> Path:
    return REPO_ROOT / f"_test_policy_{uuid.uuid4().hex[:12]}{suffix}"


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass  # sandboxed deletes are best-effort; the pattern is gitignored


class ActionValidationTest(unittest.TestCase):
    """Whatever the algorithm returns, it must be a legal 29-DoF command."""

    def test_accepts_a_plain_29_vector(self):
        action = policy_api.check_action(np.zeros(29), clip="c", step=0)
        self.assertEqual(action.shape, (29,))
        self.assertEqual(action.dtype, np.float64)

    def test_accepts_lists_and_float32(self):
        self.assertEqual(policy_api.check_action([0.0] * 29).shape, (29,))
        self.assertEqual(
            policy_api.check_action(np.zeros(29, dtype=np.float32)).dtype, np.float64
        )

    def test_rejects_wrong_width(self):
        for bad in (np.zeros(28), np.zeros(30), np.zeros(1)):
            with self.assertRaises(policy_api.ActionFormatError):
                policy_api.check_action(bad, clip="c", step=3)

    def test_rejects_wrong_rank(self):
        for bad in (np.zeros((1, 29)), np.zeros((29, 1)), np.zeros((29, 29))):
            with self.assertRaises(policy_api.ActionFormatError):
                policy_api.check_action(bad, clip="c", step=0)

    def test_rejects_non_finite(self):
        for value in (np.nan, np.inf, -np.inf):
            bad = np.zeros(29)
            bad[7] = value
            with self.assertRaises(policy_api.ActionFormatError):
                policy_api.check_action(bad, clip="c", step=0)

    def test_rejects_non_numeric_payloads(self):
        for bad in (None, "action", np.zeros(29, dtype=bool), {"a": 1}):
            with self.assertRaises(policy_api.ActionFormatError):
                policy_api.check_action(bad, clip="c", step=0)

    def test_error_names_the_clip_and_step(self):
        with self.assertRaises(policy_api.ActionFormatError) as ctx:
            policy_api.check_action(np.zeros(3), clip="dance1_subject1", step=17)
        message = str(ctx.exception)
        self.assertIn("dance1_subject1", message)
        self.assertIn("17", message)


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class DecouplingTest(unittest.TestCase):
    """End-to-end: policies with arbitrary internals drive the real robot."""

    @classmethod
    def setUpClass(cls):
        cls.info = spec.build_model()
        cls.ref = motion.get_reference(SHORT_CSV)

    def test_seven_dim_observation_is_enough(self):
        """A 7-number observation must be as legal as mjlab's 160-number one."""

        class NarrowObservation:
            kind = "test-narrow"

            def __init__(self):
                self.widths = []
                self.context = None

            def reset(self, context):
                self.context = context

            def act(self, step):
                obs = np.concatenate([step.joint_pos[:4], step.joint_vel[:3]])
                self.widths.append(obs.shape[0])
                target = self.context.reference.joint_pos[step.reference_index]
                return (target - self.context.default_joint_pos) / self.context.action_scale

        policy_object = NarrowObservation()
        result = runner.evaluate_clip(self.info, self.ref, policy_object, SHORT_CLIP)

        self.assertTrue(policy_object.widths)
        self.assertEqual(set(policy_object.widths), {7})  # never 160
        self.assertEqual(result.steps, self.ref.num_frames - 1)
        self.assertTrue(np.isfinite(result.normalized_return))

    def test_own_history_buffer(self):
        """History is the policy's business; the evaluator stores none for it."""

        class HistoryPolicy:
            kind = "test-history"

            def __init__(self):
                self.buffer = []
                self.context = None

            def reset(self, context):
                self.context = context
                self.buffer = []  # reset must clear per-clip state

            def act(self, step):
                self.buffer.append(step.joint_pos.copy())
                self.buffer = self.buffer[-3:]  # the policy owns its window
                mean = np.mean(self.buffer, axis=0)
                return (mean - self.context.default_joint_pos) / self.context.action_scale

        policy_object = HistoryPolicy()
        runner.evaluate_clip(self.info, self.ref, policy_object, SHORT_CLIP)
        self.assertTrue(policy_object.buffer)
        self.assertLessEqual(len(policy_object.buffer), 3)  # pruned by the policy

    def test_checkpoint_without_actor_mlp_is_fine(self):
        """The evaluator hashes the file; only the algorithm parses it."""
        path = _scratch(".npz")
        self.addCleanup(_remove, path)
        gain = np.linspace(0.0, 1.0, C.ACTION_DIM, dtype=np.float32)
        np.savez(path, **{"policy/gain": gain, "policy/naive": np.zeros(3)})

        # The bundled mjlab loader refuses it -- that is *its* contract, not the
        # benchmark's, and it is exactly the assumption being removed here.
        with self.assertRaises(policy.PolicyFormatError):
            policy.load_policy(path, self.info)

        class CustomFormatPolicy:
            kind = "test-custom-format"

            def __init__(self, checkpoint):
                with np.load(checkpoint) as data:
                    self.gain = np.asarray(data["policy/gain"], dtype=np.float64)
                self.context = None

            def reset(self, context):
                self.context = context

            def act(self, step):
                target = self.context.reference.joint_pos[step.reference_index]
                return (target - self.context.default_joint_pos) / self.context.action_scale

        result = runner.evaluate_clip(
            self.info, self.ref, CustomFormatPolicy(path), SHORT_CLIP
        )
        self.assertEqual(result.steps, self.ref.num_frames - 1)

    def test_interface_is_exactly_reset_and_act(self):
        class OnlyAct:
            def act(self, step):
                return np.zeros(29)

        with self.assertRaises(policy_api.PolicyInterfaceError) as ctx:
            runner.evaluate_clip(self.info, self.ref, OnlyAct(), SHORT_CLIP)
        self.assertIn("reset", str(ctx.exception))

    def test_bad_action_stops_the_rollout(self):
        class BrokenPolicy:
            def reset(self, context):
                self.context = context

            def act(self, step):
                return np.full(28, 0.0)

        with self.assertRaises(policy_api.ActionFormatError):
            runner.evaluate_clip(self.info, self.ref, BrokenPolicy(), SHORT_CLIP)

    def test_policy_cannot_reach_the_simulator(self):
        """The payload carries physics state only -- no MjModel/MjData, no reward."""
        seen = {}

        class Spy:
            def reset(self, context):
                seen["context"] = set(vars(context))

            def act(self, step):
                seen["step"] = set(vars(step))
                return np.zeros(29)

        runner.evaluate_clip(self.info, self.ref, Spy(), SHORT_CLIP)
        for forbidden in ("model", "data", "reward", "terminations"):
            self.assertNotIn(forbidden, seen["step"])
            self.assertNotIn(forbidden, seen["context"])
        self.assertIn("joint_pos", seen["step"])
        self.assertIn("reference", seen["context"])

    def test_reference_is_read_only(self):
        class Mutator:
            def reset(self, context):
                self.context = context

            def act(self, step):
                step.reference.joint_pos[0, 0] = 99.0  # must not be possible
                return np.zeros(29)

        with self.assertRaises(ValueError):
            runner.evaluate_clip(self.info, self.ref, Mutator(), SHORT_CLIP)

    def test_cli_plugin_entry_point(self):
        """The documented ``path.py:factory`` entry point actually loads."""
        import run_canonical_eval as cli  # noqa: PLC0415

        path = _scratch(".py")
        self.addCleanup(_remove, path)
        path.write_text(
            "import numpy as np\n"
            "\n"
            "class Zero:\n"
            "    kind = 'test-plugin'\n"
            "    def __init__(self, checkpoint=None):\n"
            "        self.checkpoint = checkpoint\n"
            "    def reset(self, context):\n"
            "        self.context = context\n"
            "    def act(self, step):\n"
            "        return np.zeros(29)\n"
            "\n"
            "def make_policy(checkpoint=None):\n"
            "    return Zero(checkpoint)\n",
            encoding="utf-8",
        )
        factory = cli.load_policy_factory(f"{path}:make_policy")
        policy_object = factory()
        self.assertEqual(policy_object.kind, "test-plugin")

        result = runner.evaluate_clip(self.info, self.ref, policy_object, SHORT_CLIP)
        self.assertEqual(result.steps, self.ref.num_frames - 1)

    def test_record_does_not_mention_observation_width(self):
        """The record describes the policy kind, never an assumed encoding."""
        class Minimal:
            kind = "test-minimal"

            def reset(self, context):
                self.context = context

            def act(self, step):
                return np.zeros(29)

        result = runner.evaluate_clip(self.info, self.ref, Minimal(), SHORT_CLIP)
        record = runner.build_record(
            algorithm="unit-test",
            training_commit="0" * 40,
            seed=0,
            gpu_name="cpu-only",
            gpu_count=1,
            checkpoint_sha256="0" * 64,
            training_wall_s=1.0,
            setup_wall_s=1.0,
            results=[result],
            mode="checkpoint",
            extra={"policy_kind": "test-minimal"},
        )
        evaluation = json.dumps(record["evaluation"])
        self.assertNotIn("obs_dim", evaluation)
        self.assertNotIn("160", evaluation)
        # A non-smoke kind passes the extra gate (the clip subset check is the
        # CLI's job, not build_record's).
        self.assertEqual(runner.submission_errors(record), [])

    def test_smoke_kind_is_blocked_from_submission(self):
        class Scripted:
            kind = "smoke_scripted"

            def reset(self, context):
                self.context = context

            def act(self, step):
                return np.zeros(29)

        result = runner.evaluate_clip(self.info, self.ref, Scripted(), SHORT_CLIP)
        record = runner.build_record(
            algorithm="unit-test",
            training_commit="0" * 40,
            seed=0,
            gpu_name="cpu-only",
            gpu_count=1,
            checkpoint_sha256="0" * 64,
            training_wall_s=1.0,
            setup_wall_s=1.0,
            results=[result],
            mode="smoke",
            extra={"policy_kind": "smoke_scripted"},
        )
        errors = runner.submission_errors(record)
        self.assertTrue(any("mode" in e for e in errors))
        self.assertTrue(any("smoke" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
