from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.core import audit_record, load_protocol


ROOT = Path(__file__).resolve().parents[1]
GIT_SHA = "a" * 40
FILE_SHA = "b" * 64


def valid_record(ret: float = 0.83, failed: int = 0) -> dict:
    return {
        "protocol_id": "modded-supermimic-g1-lafan1-v1",
        "algorithm": "test-algorithm",
        "training_commit": GIT_SHA,
        "seed": 0,
        "hardware": {"gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_count": 1},
        "evaluator": {
            "implementation": "canonical-mjlab-g1-v1",
            "mujoco": "3.11.0",
            "mujoco_warp": "3.11.0",
            "warp_lang": "1.16.0",
            "robot_dof": 29,
            "self_collision": True,
            "mjlab_commit": GIT_SHA,
            "robot_model_sha256": FILE_SHA,
            "dataset_manifest_sha256": FILE_SHA,
        },
        "checkpoints": [
            {
                "checkpoint_sha256": FILE_SHA,
                "training_wall_s": 600,
                "setup_wall_s": 100,
                "eval_wall_s": 120,
                "per_clip": [
                    {"clip": cid, "normalized_return": ret, "failed": i < failed}
                    for i, cid in enumerate(load_protocol()["clip_ids"])
                ],
            }
        ],
    }


class BenchmarkTests(unittest.TestCase):
    def test_protocol_has_exactly_40_unique_clips(self):
        self.assertEqual(len(set(load_protocol()["clip_ids"])), 40)

    def test_target_qualified_same_checkpoint(self):
        result = audit_record(valid_record())
        self.assertEqual(result["status"], "QUALIFIED_PENDING_PHYSICS_AUDIT")
        self.assertEqual(result["first_qualifying_checkpoint"]["failed_clips"], 0)
        self.assertEqual(result["first_qualifying_checkpoint"]["training_wall_s"], 600)
        self.assertFalse(result["physics_e2e_verified"])

    def test_exact_threshold_qualifies(self):
        self.assertEqual(audit_record(valid_record(0.82))["status"], "QUALIFIED_PENDING_PHYSICS_AUDIT")

    def test_one_failure_disqualifies_even_above_threshold(self):
        self.assertEqual(audit_record(valid_record(0.93, 1))["status"], "NOT_QUALIFIED")

    def test_low_return_disqualifies_zero_failures(self):
        self.assertEqual(audit_record(valid_record(0.81, 0))["status"], "NOT_QUALIFIED")

    def test_cannot_mix_score_and_failures_across_checkpoints(self):
        record = valid_record(0.83, 1)
        other = copy.deepcopy(valid_record(0.81)["checkpoints"][0])
        other["checkpoint_sha256"] = "c" * 64
        other["training_wall_s"] = 620
        record["checkpoints"].append(other)
        self.assertEqual(audit_record(record)["status"], "NOT_QUALIFIED")

    def test_picks_first_checkpoint_reaching_both_targets(self):
        record = valid_record(0.81)
        other = copy.deepcopy(valid_record(0.85)["checkpoints"][0])
        other["checkpoint_sha256"] = "c" * 64
        other["training_wall_s"] = 900
        record["checkpoints"].append(other)
        result = audit_record(record)
        self.assertEqual(result["first_qualifying_checkpoint"]["training_wall_s"], 900)

    def test_duplicate_clip_invalid(self):
        record = valid_record()
        record["checkpoints"][0]["per_clip"][0]["clip"] = record["checkpoints"][0]["per_clip"][1]["clip"]
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_missing_clip_invalid(self):
        record = valid_record()
        record["checkpoints"][0]["per_clip"].pop()
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_wrong_evaluator_version_invalid(self):
        record = valid_record()
        record["evaluator"]["warp_lang"] = "1.18.0"
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_disabled_self_collision_invalid(self):
        record = valid_record()
        record["evaluator"]["self_collision"] = False
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_wrong_gpu_invalid(self):
        record = valid_record()
        record["hardware"]["gpu_name"] = "NVIDIA H100"
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_untrusted_aggregate_cannot_replace_per_clip(self):
        record = valid_record()
        record["checkpoints"][0].pop("per_clip")
        record["checkpoints"][0]["normalized_return"] = 0.99
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_nan_invalid(self):
        record = valid_record()
        record["checkpoints"][0]["per_clip"][2]["normalized_return"] = float("nan")
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_boolean_return_invalid(self):
        record = valid_record()
        record["checkpoints"][0]["per_clip"][2]["normalized_return"] = True
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_missing_times_invalid_not_exception(self):
        record = valid_record()
        record["checkpoints"][0].pop("setup_wall_s")
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_nonfinite_time_invalid_without_json_leakage(self):
        record = valid_record()
        record["checkpoints"][0]["eval_wall_s"] = float("inf")
        checked = audit_record(record)
        self.assertEqual(checked["status"], "INVALID")
        json.dumps(checked, allow_nan=False)

    def test_duplicate_checkpoint_sha_invalid(self):
        record = valid_record()
        record["checkpoints"].append(copy.deepcopy(record["checkpoints"][0]))
        record["checkpoints"][1]["training_wall_s"] = 900
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_non_monotonic_time_invalid(self):
        record = valid_record()
        record["checkpoints"].append(copy.deepcopy(record["checkpoints"][0]))
        record["checkpoints"][1]["checkpoint_sha256"] = "c" * 64
        self.assertEqual(audit_record(record)["status"], "INVALID")

    def test_record_must_be_object(self):
        self.assertEqual(audit_record([])["status"], "INVALID")

    def test_cli_validate_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            submissions = root / "submissions"
            submissions.mkdir()
            rec = valid_record()
            (submissions / "sample.json").write_text(json.dumps(rec))
            checked = subprocess.run(
                [sys.executable, str(ROOT / "benchmark.py"), "validate", str(submissions / "sample.json")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("QUALIFIED_PENDING_PHYSICS_AUDIT", checked.stdout)
            board = subprocess.run(
                [sys.executable, str(ROOT / "benchmark.py"), "leaderboard", str(submissions)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(board.returncode, 0, board.stderr)
            self.assertIn("PROVISIONAL / PHYSICS NOT AUDITED", board.stdout)


if __name__ == "__main__":
    unittest.main()
