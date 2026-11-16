"""Frozen asset / dataset hashes for the canonical evaluator.

``manifest.json`` is the record the evaluator reports into
``record["evaluator"]["robot_model_sha256"]`` and ``["dataset_manifest_sha256"]``.

* ``robot_model_sha256`` hashes every vendored mjlab G1 file (``g1.xml`` plus the
  35 STL assets) as a sorted ``<relative path>  <sha256>`` listing.
* ``dataset_manifest_sha256`` hashes the canonical JSON of the 40
  ``clip_id -> csv sha256`` pairs.
* ``compiled_fingerprint`` is informational: it proves the model that actually
  ran is the one that was hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import constants as C
from . import spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robot_model_files() -> list[Path]:
    root = C.ASSET_DIR
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".xml", ".stl"}),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def robot_model_sha256() -> str:
    lines = []
    for path in robot_model_files():
        rel = path.relative_to(C.ASSET_DIR).as_posix()
        lines.append(f"{rel}  {_sha256(path)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dataset_hashes() -> dict[str, str]:
    """Recompute the 40 clip hashes from the on-disk CSVs."""
    from .protocol import protocol_clip_ids  # noqa: PLC0415

    out = {}
    for clip in protocol_clip_ids():
        path = C.DATA_DIR / f"{clip}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing dataset clip {path}; run scripts/fetch_lafan1.py first"
            )
        out[clip] = _sha256(path)
    return out


def dataset_manifest_sha256(hashes: dict[str, str] | None = None) -> str:
    if hashes is None:
        frozen = C.DATA_DIR.parent / "lafan1_g1.sha256.json"
        if frozen.exists():
            hashes = json.loads(frozen.read_text(encoding="utf-8"))
        else:
            hashes = dataset_hashes()
    canonical = json.dumps(dict(sorted(hashes.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    info = spec.build_model()
    return {
        "evaluator_implementation": C.EVALUATOR_IMPLEMENTATION,
        "mjlab": {
            "repo": C.MJLAB_REPO,
            "version": C.MJLAB_VERSION,
            "commit": C.MJLAB_COMMIT,
        },
        "robot_model": {
            "source": "mjlab asset_zoo/robots/unitree_g1/xmls",
            "files": len(robot_model_files()),
            "sha256": robot_model_sha256(),
        },
        "dataset": {
            "source": f"{C.LAFAN1_HF_REPO}@g1 (revision {C.LAFAN1_HF_REVISION})",
            "clips": len(json.loads((C.DATA_DIR.parent / "lafan1_g1.sha256.json").read_text())),
            "manifest_sha256": dataset_manifest_sha256(),
        },
        "compiled_fingerprint": spec.fingerprint(info),
    }


def write_manifest(path: Path | None = None) -> dict:
    manifest = build_manifest()
    target = path or C.MANIFEST_PATH
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest() -> dict:
    if not C.MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{C.MANIFEST_PATH} missing; run `python scripts/freeze_manifest.py`"
        )
    return json.loads(C.MANIFEST_PATH.read_text(encoding="utf-8"))
