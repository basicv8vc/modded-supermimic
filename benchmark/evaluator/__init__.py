"""Canonical evaluator for protocol ``modded-supermimic-g1-lafan1-v1``.

``canonical-mjlab-g1-v1`` reproduces the physics / observation / action /
reward / termination semantics of mjlab v1.6.0
``Mjlab-Tracking-Flat-Unitree-G1`` while dropping the RL-training machinery
(no randomization, no curriculum, no auto-reset). The frozen definitions live in
``docs/CANONICAL_EVALUATOR.md``.

Modules:
  constants   -- every frozen number (joins the mjlab source values verbatim)
  mathx       -- wxyz quaternion helpers (``mjlab.utils.lab_api.math`` semantics)
  spec        -- builds the compiled G1 tracking model from the official MJCF
  motion      -- LAFAN1 CSV -> 50 Hz reference arrays (mjlab csv_to_npz replay)
  env         -- deterministic rollout: 50 Hz control, full clip, no sampling
  policy_api  -- the only policy contract: reset(context) / act(step) -> (29,)
  policy      -- one implementation: the mjlab-native 160-dim rsl_rl actor
  runner      -- per-clip results -> ``benchmark.py`` compatible record

The benchmark fixes the robot, physics, motions and score; the algorithm owns its
observation encoding, network and checkpoint format. See
``docs/POLICY_INTERFACE.md``.
"""

from __future__ import annotations

__all__ = ["EVALUATOR_IMPLEMENTATION"]

EVALUATOR_IMPLEMENTATION = "canonical-mjlab-g1-v1"
