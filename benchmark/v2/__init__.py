"""Benchmark V2: the original-evaluation-aligned protocol.

V1 (``modded-supermimic-g1-lafan1-v1``) lives in ``benchmark/evaluator/`` and
is intentionally untouched.  V2 is additive: a new protocol ID, a new protocol
JSON + SHA256, a new evaluator implementation identity, its own records and its
own documentation.

Modules:
  spec     -- GPU-free executable Reward / Termination / Normalization spec
  config   -- official mjlab cfg overrides + minimal pelvis-termination glue
  runner   -- V2 rollout driver and record assembly (requires MuJoCo Warp)
"""

from . import config, spec  # noqa: F401

__all__ = ["config", "spec", "PROTOCOL_ID", "EVALUATOR_IMPLEMENTATION"]

PROTOCOL_ID = spec.PROTOCOL_ID
EVALUATOR_IMPLEMENTATION = spec.EVALUATOR_IMPLEMENTATION
