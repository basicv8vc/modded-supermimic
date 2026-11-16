"""Example algorithm: PD the reference joint angles and ignore the robot state.

This is the smallest complete implementation of the evaluator's contract, and it
is not meant as a submission -- it never looks at what the robot is actually
doing, so it falls over quickly. It exists to show that an algorithm only needs
two methods and has to know nothing about the evaluator's internals:

    reset(context)   once per clip: joint order, default pose, action scale, the
                     reference motion
    act(step)        once per 50 Hz control step: return 29 raw actions, which
                     the evaluator turns into ``action * action_scale + default``

The action is the only thing the evaluator reads. Observation encoding, history,
network and checkpoint format are the algorithm's own business.

Run it::

    python scripts/run_canonical_eval.py \
        --policy-file examples/scripted_replayer.py:make_policy \
        --out reports/example.json

Without ``--checkpoint`` the record is marked non-submittable, because the
protocol requires a checkpoint hash.
"""

from __future__ import annotations

import numpy as np


class ScriptedReplayer:
    """Replays the reference joint trajectory through the position interface."""

    kind = "example-scripted-replayer"

    def __init__(self, checkpoint=None):
        # A real algorithm would load its weights here; this one has none.
        self.checkpoint = checkpoint
        self._default_joint_pos = None
        self._action_scale = None
        self._reference = None

    def reset(self, context):
        self._default_joint_pos = context.default_joint_pos
        self._action_scale = context.action_scale
        self._reference = context.reference

    def act(self, step):
        target = self._reference.joint_pos[step.reference_index]
        return (target - self._default_joint_pos) / self._action_scale


def make_policy(checkpoint=None):
    """Factory used by ``--policy-file examples/scripted_replayer.py:make_policy``."""
    return ScriptedReplayer(checkpoint)
