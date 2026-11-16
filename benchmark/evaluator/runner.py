"""Rollout driver + record assembly.

Per clip::

    normalized_return = sum_t r_t / ((T - 1) * MAX_STEP_REWARD)
    failed            = any termination other than time_out fired

``T`` is the number of 50 Hz reference frames, so the episode has ``T - 1``
control steps and the reward of the terminating step still counts (subsequent
steps score 0). This is exactly the definition the published baselines used
(``NablaMimic/src/nablamimic/evaluate.py``: ``full_len = lengths - 1``,
``tsum += where(alive, reward, 0)``, ``tsum.sum() / (full_len * max_reward)``),
and ``MAX_STEP_REWARD == 0.1`` reproduces its denominator.

No new metric is introduced: the record only carries ``per_clip``
``{clip, normalized_return, failed}`` plus a separate diagnostics block.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import constants as C
from . import policy_api
from .env import TrackingEnv
from .manifest import load_manifest
from .motion import CONVERTER_VERSION, MotionReference, get_reference
from .policy_api import ClipContext
from .protocol import load_protocol, protocol_clip_ids, protocol_sha256
from .spec import ModelInfo


def source_git_sha() -> str:
    """Git SHA of the evaluator source tree at run time ("unknown" without git).

    Records the exact code that produced a result so the ``implementation``
    label never has to carry that meaning alone.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=C.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        sha = out.stdout.strip()
        return sha if len(sha) == 40 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class ClipResult:
    clip: str
    normalized_return: float
    failed: bool
    steps: int
    alive_steps: int
    fall_step: int | None
    fall_terms: tuple[str, ...]
    raw_normalized_return: float
    mean_term_score: dict[str, float] = field(default_factory=dict)
    reference_wall_s: float = 0.0
    rollout_wall_s: float = 0.0

    @property
    def wall_s(self) -> float:
        return self.reference_wall_s + self.rollout_wall_s

    def record_row(self) -> dict:
        """The only three fields the protocol validator reads."""
        return {
            "clip": self.clip,
            "normalized_return": self.normalized_return,
            "failed": self.failed,
        }

    def diagnostics(self) -> dict:
        payload = asdict(self)
        payload.pop("clip")
        payload.pop("normalized_return")
        payload.pop("failed")
        return payload


def evaluate_clip(
    info: ModelInfo, ref: MotionReference, policy, clip_id: str
) -> ClipResult:
    started = time.perf_counter()
    policy_api.require_interface(policy)  # fail before paying for a model build
    env = TrackingEnv(ref)
    # ``spec.py`` is the frozen conformance checker: the live mjlab action
    # mapping must agree with the fingerprinted spec. The env quantizes the
    # offset/scale tensors to float32 (~3e-8), hence the 1e-6 tolerance.
    scale_err = float(np.abs(env.action_scale - info.action_scale).max())
    pose_err = float(np.abs(env.default_joint_pos - info.default_joint_pos).max())
    if scale_err > 1e-6 or pose_err > 1e-6:
        raise RuntimeError(
            f"{clip_id}: live mjlab action mapping drifted from the frozen spec "
            f"(scale_err={scale_err:.3e}, pose_err={pose_err:.3e})"
        )
    # The policy contract carries the frozen spec's float64 values: the
    # SuperMimic checkpoint asserts agreement to 1e-8 (scale) / 1e-12 (pose),
    # which the env's float32 tensors cannot meet. The residual target error
    # this leaves in the env (~3e-8 rad) is recorded in the alignment report.
    policy.reset(
        ClipContext(
            clip=clip_id,
            joint_order=C.JOINT_ORDER,
            default_joint_pos=np.array(info.default_joint_pos, dtype=np.float64),
            action_scale=np.array(info.action_scale, dtype=np.float64),
            control_dt=C.STEP_DT,
            reference=env.reference_view,
        )
    )

    n_steps = ref.num_frames - 1
    if n_steps < 1:
        raise ValueError(f"{clip_id}: clip is shorter than two frames")

    total = 0.0
    alive = True
    fall_step: int | None = None
    fall_terms: tuple[str, ...] = ()
    alive_steps = 0
    term_sums: dict[str, float] = {}

    for step in range(n_steps):
        # The evaluator understands nothing about the policy's inputs; it only
        # checks that what comes back is a legal 29-DoF command.
        action = policy_api.check_action(
            policy.act(env.step_input()), clip=clip_id, step=step
        )
        result = env.step(action)
        if alive:
            total += result.reward
            alive_steps += 1
            for name, value in result.terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value
        if alive and result.terminated:
            alive = False
            fall_step = step + 1
            fall_terms = result.terminated_terms
            # Remaining steps are masked to zero by the metric definition, so the
            # rollout can stop here; ``steps`` (the denominator) is unchanged.
            break

    denominator = n_steps * C.MAX_STEP_REWARD
    raw = total / denominator
    mean_terms = {
        name: value / max(alive_steps, 1) for name, value in sorted(term_sums.items())
    }
    return ClipResult(
        clip=clip_id,
        normalized_return=float(min(max(raw, 0.0), 1.0)),
        failed=fall_step is not None,
        steps=n_steps,
        alive_steps=alive_steps,
        fall_step=fall_step,
        fall_terms=fall_terms,
        raw_normalized_return=float(raw),
        mean_term_score=mean_terms,
        rollout_wall_s=time.perf_counter() - started,
    )


def evaluate_all(
    info: ModelInfo,
    policy,
    clip_ids: Sequence[str] | None = None,
    data_dir: Path | None = None,
    progress: Callable[[ClipResult], None] | None = None,
    use_cache: bool = False,
) -> list[ClipResult]:
    ids = list(clip_ids or protocol_clip_ids())
    root = data_dir or C.DATA_DIR
    results = []
    for clip in ids:
        csv_path = root / f"{clip}.csv"
        reference_started = time.perf_counter()
        ref = get_reference(csv_path, use_cache=use_cache)
        reference_wall_s = time.perf_counter() - reference_started
        result = evaluate_clip(info, ref, policy, clip)
        result.reference_wall_s = reference_wall_s
        results.append(result)
        if progress is not None:
            progress(result)
    return results


def dependency_versions() -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    return {
        "mujoco": version("mujoco"),
        "mujoco_warp": version("mujoco-warp"),
        "warp_lang": version("warp-lang"),
    }


def build_record(
    *,
    algorithm: str,
    training_commit: str,
    seed: int,
    gpu_name: str,
    gpu_count: int,
    checkpoint_sha256: str,
    training_wall_s: float,
    setup_wall_s: float,
    results: Sequence[ClipResult],
    mode: str,
    extra: dict | None = None,
) -> dict:
    manifest = load_manifest()
    versions = dependency_versions()
    eval_wall_s = float(sum(r.wall_s for r in results))

    record = {
        "protocol_id": load_protocol()["protocol_id"],
        "algorithm": algorithm,
        "training_commit": training_commit,
        "seed": seed,
        "hardware": {"gpu_name": gpu_name, "gpu_count": gpu_count},
        "evaluator": {
            "implementation": C.EVALUATOR_IMPLEMENTATION,
            "mujoco": versions["mujoco"],
            "mujoco_warp": versions["mujoco_warp"],
            "warp_lang": versions["warp_lang"],
            "robot_dof": C.ROBOT_DOF,
            "self_collision": True,
            "robot_model_sha256": manifest["robot_model"]["sha256"],
            "dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
            "mjlab_commit": C.MJLAB_COMMIT,
            "mjlab_version": C.MJLAB_VERSION,
            # Identity fields (not validated by benchmark/core.py): which
            # source tree and which CSV->npz pipeline produced this record.
            # Together they disambiguate results across evaluator refactors;
            # the frozen ``implementation`` label alone does not.
            "source_git_sha": source_git_sha(),
            "motion_converter_version": CONVERTER_VERSION,
        },
        "checkpoints": [
            {
                "checkpoint_sha256": checkpoint_sha256,
                "training_wall_s": training_wall_s,
                "setup_wall_s": setup_wall_s,
                "eval_wall_s": eval_wall_s,
                "per_clip": [r.record_row() for r in results],
            }
        ],
        # Non-protocol metadata (ignored by benchmark/core.py):
        "evaluation": {
            "mode": mode,
            "protocol_sha256": protocol_sha256(),
            "control_hz": 1.0 / C.STEP_DT,
            "clip_steps": {r.clip: r.steps for r in results},
            "normalized_return_denominator": C.MAX_STEP_REWARD,
            "aggregate_normalized_return": float(
                np.mean([r.normalized_return for r in results])
            ),
            "failed_clips": int(sum(r.failed for r in results)),
            "per_clip_diagnostics": {r.clip: r.diagnostics() for r in results},
            **(extra or {}),
        },
    }
    return record


def submission_errors(record: dict) -> list[str]:
    """Extra gate on top of ``benchmark/core.py`` (which is left untouched).

    A record may only be submitted when it was produced from a real policy
    checkpoint, not by a scripted smoke policy, over the full 40-clip protocol.
    It says nothing about how that policy works -- only that it exists.
    """
    errors = []
    evaluation = record.get("evaluation") or {}
    if evaluation.get("mode") != "checkpoint":
        errors.append(
            "evaluation.mode is not 'checkpoint' (scripted smoke records are not "
            "submittable)"
        )
    kind = str(evaluation.get("policy_kind") or "")
    if not kind:
        errors.append("evaluation.policy_kind is missing (declare what was run)")
    elif kind.startswith("smoke"):
        errors.append(f"evaluation.policy_kind {kind!r} marks a scripted smoke policy")
    if evaluation.get("clip_subset") is not None:
        errors.append("record covers a clip subset, not the full protocol of 40")
    return errors
