"""Strict, dependency-free validation of 40-clip evaluation records.

This is deliberately *not* a physics evaluator. A score computed here never
certifies that the independent MuJoCo-Warp evaluator actually ran.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
HEX40 = re.compile(r"^[a-fA-F0-9]{40}$")
HEX64 = re.compile(r"^[a-fA-F0-9]{64}$")


def load_protocol() -> dict[str, Any]:
    cfg = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert len(cfg["clip_ids"]) == 40
    assert len(set(cfg["clip_ids"])) == 40
    return cfg


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _is_sha(value: Any, length: int) -> bool:
    return isinstance(value, str) and bool((HEX40 if length == 40 else HEX64).fullmatch(value))


def audit_record(record: Any) -> dict[str, Any]:
    """Check format + self-declared provenance; calculate, never trust aggregates.

    All accepted scores remain *provisional* until a separately audited,
    immutable physical evaluator and reference asset hashes are integrated.
    """
    protocol = load_protocol()
    errors: list[str] = []
    result: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "status": "INVALID",
        "physics_e2e_verified": False,
        "errors": errors,
        "checkpoints": [],
        "first_qualifying_checkpoint": None,
    }

    if not isinstance(record, dict):
        errors.append("record must be a JSON object")
        return result
    if record.get("protocol_id") != protocol["protocol_id"]:
        errors.append("protocol_id mismatch")

    algorithm = record.get("algorithm")
    if not isinstance(algorithm, str) or not (1 <= len(algorithm.strip()) <= 128):
        errors.append("algorithm must be 1..128 nonblank characters")
    else:
        result["algorithm"] = algorithm.strip()
    if not _is_sha(record.get("training_commit"), 40):
        errors.append("training_commit must be a full 40-hex git SHA")
    else:
        result["training_commit"] = record["training_commit"]
    seed = record.get("seed")
    if type(seed) is not int or seed < 0:
        errors.append("seed must be a nonnegative integer")
    else:
        result["seed"] = seed

    hardware = record.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be an object")
    else:
        gpu_name = hardware.get("gpu_name")
        if not isinstance(gpu_name, str) or protocol["gpu_name_contains"] not in gpu_name:
            errors.append("GPU model is not Colab G4 RTX PRO 6000 Blackwell")
        if type(hardware.get("gpu_count")) is not int or hardware["gpu_count"] != 1:
            errors.append("GPU count must be exactly 1")

    evaluator = record.get("evaluator")
    if not isinstance(evaluator, dict):
        errors.append("evaluator must be an object")
    else:
        if evaluator.get("implementation") != protocol["evaluator_implementation"]:
            errors.append("evaluator implementation mismatch")
        for key, expected in protocol["required_versions"].items():
            if evaluator.get(key) != expected:
                errors.append(f"evaluator version {key} must equal {expected}")
        if type(evaluator.get("robot_dof")) is not int or evaluator["robot_dof"] != protocol["robot_dof"]:
            errors.append("robot must have 29 DoF")
        if evaluator.get("self_collision") is not True:
            errors.append("full self-collision must be enabled")
        for key in ("robot_model_sha256", "dataset_manifest_sha256"):
            if not _is_sha(evaluator.get(key), 64):
                errors.append(f"evaluator.{key} must be SHA256")
        if not _is_sha(evaluator.get("mjlab_commit"), 40):
            errors.append("evaluator.mjlab_commit must be a full git SHA")

    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        errors.append("checkpoints must be a nonempty list")
        return result

    required_ids = set(protocol["clip_ids"])
    sha_seen: set[str] = set()
    last_time = -1.0
    for index, checkpoint in enumerate(checkpoints):
        name = f"checkpoints[{index}]"
        if not isinstance(checkpoint, dict):
            errors.append(f"{name} must be an object")
            continue
        sha = checkpoint.get("checkpoint_sha256")
        if not _is_sha(sha, 64):
            errors.append(f"{name}.checkpoint_sha256 must be SHA256")
        elif sha in sha_seen:
            errors.append(f"{name} duplicate checkpoint SHA256")
        else:
            sha_seen.add(sha)
        times_valid = True
        for key in ("training_wall_s", "setup_wall_s", "eval_wall_s"):
            if not _finite_nonnegative(checkpoint.get(key)):
                errors.append(f"{name}.{key} must be finite and nonnegative")
                times_valid = False
        training_seconds = checkpoint.get("training_wall_s")
        if times_valid:
            training_seconds = float(training_seconds)
            if training_seconds <= last_time:
                errors.append(f"{name}.training_wall_s must be strictly increasing")
            last_time = training_seconds
        else:
            continue

        clips = checkpoint.get("per_clip")
        if not isinstance(clips, list) or len(clips) != 40:
            errors.append(f"{name}.per_clip must have exactly 40 entries")
            continue

        ids: list[str] = []
        returns: list[float] = []
        failed = 0
        bad_row = False
        for row_idx, clip in enumerate(clips):
            if not isinstance(clip, dict):
                errors.append(f"{name}.per_clip[{row_idx}] must be an object")
                bad_row = True
                continue
            cid = clip.get("clip")
            if not isinstance(cid, str):
                errors.append(f"{name}.per_clip[{row_idx}].clip must be a string")
                bad_row = True
            else:
                ids.append(cid)
            ret = clip.get("normalized_return")
            if not _finite_nonnegative(ret) or float(ret) > 1:
                errors.append(f"{name}.per_clip[{row_idx}].normalized_return must be in [0,1]")
                bad_row = True
            else:
                returns.append(float(ret))
            if type(clip.get("failed")) is not bool:
                errors.append(f"{name}.per_clip[{row_idx}].failed must be boolean")
                bad_row = True
            elif clip["failed"]:
                failed += 1

        if len(set(ids)) != 40 or set(ids) != required_ids:
            errors.append(f"{name} has duplicate, missing, or unexpected clip IDs")
            bad_row = True
        if bad_row or len(returns) != 40:
            continue
        normalized_return = sum(returns) / len(returns)
        qualifies = (
            normalized_return >= protocol["target_normalized_return"]
            and failed == protocol["target_failed_clips"]
        )
        entry = {
            "checkpoint_sha256": sha,
            "training_wall_s": training_seconds,
            "setup_wall_s": checkpoint.get("setup_wall_s"),
            "eval_wall_s": checkpoint.get("eval_wall_s"),
            "normalized_return": normalized_return,
            "failed_clips": failed,
            "qualifies": qualifies,
        }
        result["checkpoints"].append(entry)

    if errors:
        return result
    first = next((c for c in result["checkpoints"] if c["qualifies"]), None)
    result["first_qualifying_checkpoint"] = first
    result["status"] = "QUALIFIED_PENDING_PHYSICS_AUDIT" if first else "NOT_QUALIFIED"
    return result


def load_and_audit(path: Path) -> dict[str, Any]:
    try:
        return audit_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, UnicodeError) as exc:
        return {"status": "INVALID", "errors": [f"unreadable JSON: {exc}"], "physics_e2e_verified": False}
