#!/usr/bin/env python3
"""Offline stdlib-only audit for reports/v2/registry.json and records/v2/.

Verifies:
1. Protocol SHA256 matches benchmark/v2/protocol_v2.json.
2. Every entry in registry.json records[] resolves, matches protocol_v2 versions/physics,
   has all 40 clips with unweighted aggregate mean matching exactly, and is submission-eligible.
3. Every follow_up_sweeps[] file reference resolves inside the repository.
4. No diagnostic or retired artifact leaks into records[].
5. Every records/v2/*/ directory has a README.md card.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _audit_v2_record_stdlib(record: dict, proto: dict) -> dict:
    errors: list[str] = []
    if record.get("protocol_id") != proto["protocol_id"]:
        errors.append("protocol_id mismatch")
    ev = record.get("evaluator") or {}
    if ev.get("implementation") != proto["evaluator_implementation"]:
        errors.append("evaluator.implementation mismatch")
    if ev.get("self_collision") is not True:
        errors.append("full self-collision physics must be enabled")
    if ev.get("self_collision_reward_penalty") is not False:
        errors.append("V2 must declare self_collision_reward_penalty=false")
    for key, expected in proto["required_versions"].items():
        if ev.get(key) != expected:
            errors.append(f"evaluator version {key} must equal {expected}")
    clip_ids = set(proto["clip_ids"])
    cps = record.get("checkpoints")
    if not isinstance(cps, list) or not cps:
        errors.append("checkpoints must be a nonempty list")
        return {"errors": errors, "clipping_applied": False}
    returns: list[float] = []
    failed_total = 0
    for i, cp in enumerate(cps):
        rows = cp.get("per_clip") if isinstance(cp, dict) else None
        if not isinstance(rows, list) or len(rows) != 40:
            errors.append(f"checkpoints[{i}].per_clip must have 40 entries")
            continue
        ids: list[str] = []
        for j, row in enumerate(rows):
            cid = row.get("clip")
            if isinstance(cid, str):
                ids.append(cid)
            ret = row.get("normalized_return")
            if not isinstance(ret, (int, float)) or isinstance(ret, bool) or not math.isfinite(float(ret)):
                errors.append(f"checkpoints[{i}].per_clip[{j}].normalized_return must be finite")
            else:
                returns.append(float(ret))
            fl = row.get("failed")
            if type(fl) is not bool:
                errors.append(f"checkpoints[{i}].per_clip[{j}].failed must be boolean")
            elif fl:
                failed_total += 1
        if len(set(ids)) != 40 or set(ids) != clip_ids:
            errors.append(f"checkpoints[{i}] clip id set mismatch")
    return {
        "errors": errors,
        "clipping_applied": False,
        "aggregate_normalized_return": (sum(returns) / len(returns)) if returns else 0.0,
        "failed_clips": failed_total,
    }


def main() -> int:
    v2_dir = REPO / "reports" / "v2"
    proto_path = REPO / "benchmark" / "v2" / "protocol_v2.json"
    reg_path = v2_dir / "registry.json"

    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    reg = json.loads(reg_path.read_text(encoding="utf-8"))

    expected_sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    assert reg["protocol_id"] == proto["protocol_id"], "protocol_id mismatch"
    assert reg["protocol_sha256"] == expected_sha, (
        f"protocol_sha256 mismatch: {reg['protocol_sha256']} != {expected_sha}"
    )

    spec_mod = None
    try:
        sys.path.insert(0, str(REPO))
        from benchmark.v2 import spec as spec_mod  # noqa: PLC0415
    except ImportError:
        spec_mod = None

    acc_path = v2_dir / reg["acceptance_artifact"]["path"]
    assert acc_path.is_file(), f"Missing acceptance artifact: {acc_path}"
    acc = json.loads(acc_path.read_text(encoding="utf-8"))
    assert acc["acceptance_status"] == "PASS", acc["acceptance_status"]

    clip_ids = set(proto["clip_ids"])
    assert len(clip_ids) == 40

    seen_paths: set[str] = set()
    for entry in reg["records"]:
        rel = entry["path"]
        assert rel not in seen_paths, f"Duplicate record path: {rel}"
        seen_paths.add(rel)
        rec_path = (v2_dir / rel).resolve()
        assert rec_path.is_file(), f"Missing record file: {rec_path}"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        audit = spec_mod.audit_v2_record(rec) if spec_mod is not None else _audit_v2_record_stdlib(rec, proto)
        assert not audit["errors"], f"{rel} audit errors: {audit['errors']}"
        assert not audit["clipping_applied"], f"{rel} applied clipping"
        assert abs(audit["aggregate_normalized_return"] - entry["aggregate_normalized_return"]) < 1e-12, (
            f"{rel} mean mismatch: {audit['aggregate_normalized_return']} != {entry['aggregate_normalized_return']}"
        )
        assert audit["failed_clips"] == entry["failed_clips"], f"{rel} failed_clips mismatch"
        assert rec.get("submission_eligible") is True, f"{rel} submission_eligible != True"
        assert rec.get("target_hardware_met") is True, f"{rel} target_hardware_met != True"
        assert rec.get("report_scope") == "acceptance", f"{rel} report_scope != acceptance"
        assert rec["checkpoints"][0]["checkpoint_sha256"] == entry["checkpoint_sha256"], (
            f"{rel} checkpoint_sha256 mismatch"
        )

    for diag in reg.get("diagnostic_artifacts", []):
        assert diag not in seen_paths, f"Diagnostic artifact leaked into records[]: {diag}"
        assert (v2_dir / diag).is_file(), f"Missing diagnostic artifact: {diag}"

    for retired in reg.get("retired_baselines", []):
        assert retired["path"] not in seen_paths, f"Retired baseline leaked into records[]: {retired['path']}"
        assert (v2_dir / retired["path"]).is_file(), f"Missing retired baseline: {retired['path']}"

    for sw in reg.get("follow_up_sweeps", []):
        summary_p = (v2_dir / sw["summary_path"]).resolve()
        assert summary_p.is_file(), f"Missing sweep summary: {sw['summary_path']}"
        if "training_summary_path" in sw:
            ts_p = (v2_dir / sw["training_summary_path"]).resolve()
            assert ts_p.is_file(), f"Missing training summary: {sw['training_summary_path']}"
        for milestone_key in ("earliest_zero_fail", "stable_zero_fail_onset", "best_passing", "final"):
            m = sw.get(milestone_key, {})
            if "record_path" in m:
                assert (v2_dir / m["record_path"]).resolve().is_file(), m["record_path"]
            if "parity_2clips_summary" in m:
                assert (v2_dir / m["parity_2clips_summary"]).resolve().is_file(), m["parity_2clips_summary"]

    records_v2_dir = REPO / "records" / "v2"
    assert records_v2_dir.is_dir(), "Missing records/v2/"
    cards = sorted(p for p in records_v2_dir.iterdir() if p.is_dir())
    assert cards, "Expected at least one record directory under records/v2/"
    for card in cards:
        assert (card / "README.md").is_file(), f"Missing README.md in {card}"

    print(
        f"OK: protocol_sha256={expected_sha[:12]}..., "
        f"records={len(reg['records'])}, sweeps={len(reg.get('follow_up_sweeps', []))}, "
        f"record_cards={len(cards)}, backend={'spec' if spec_mod is not None else 'stdlib'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
