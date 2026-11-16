"""Thin accessors over the frozen ``benchmark/protocol.json``.

The evaluator must never redefine the protocol: clip list, versions, DoF count
and the quality gate are read from the existing file so that the evaluator and
``benchmark/core.py`` can never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "protocol.json"


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def protocol_clip_ids() -> list[str]:
    return list(load_protocol()["clip_ids"])


def protocol_sha256() -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
