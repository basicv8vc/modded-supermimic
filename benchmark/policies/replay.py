"""Scripted reference-replay smoke policy for throughput probes and CI.

NOT a learned policy.  ``act`` returns the reference pose expressed in
action space, which exercises the same observation/extraction path as a
real policy without requiring a checkpoint.  Used by
``scripts/v2/fast_eval.py --policy-file benchmark/policies/replay.py``
in probe mode only; never part of a scored V2 record.
"""

from __future__ import annotations


class ReplayPolicy:
    kind = "v2_smoke_scripted"

    def __init__(self):
        self._context = None

    def reset(self, context) -> None:
        self._context = context

    def act(self, step):
        target = self._context.reference.joint_pos[step.reference_index]
        return (target - self._context.default_joint_pos) / self._context.action_scale


def make_policy(checkpoint=None):
    """Factory matching the policy interface; ``checkpoint`` is ignored."""
    return ReplayPolicy()
