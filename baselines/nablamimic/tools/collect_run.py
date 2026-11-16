#!/usr/bin/env python3
"""Collect one NablaMimic run from the Colab VM without trusting the transfer.

Why this exists (review P1-2): `colab download` per file with no retry and no checksum
means a dropped proxy connection can leave a truncated checkpoint behind, and the old
collector then wrote a `collect_manifest.txt` that made the archive look回收成功. The
persistence gate in `down` would only catch it later, against a VM that may already be gone.

Contract implemented here:

  * the source is asked for a *source-of-truth snapshot* first: size + sha256 of every file
    we intend to collect, framed by COLLECT_SNAPSHOT_BEGIN/END, plus the run identity read
    from its own manifest.json. No snapshot, no collection (fail-closed) - the bytes are
    always checked against source-side facts, never against a manifest the remote (or a
    previous local run) produced;
  * each file is downloaded into `<name>.part.<pid>` with up to 3 attempts; a wrong size
    or a wrong sha256 counts as a failed attempt and is retried;
  * only a file that matches the snapshot byte-for-byte is atomically renamed onto its
    final path; a verified file is never overwritten by an unverified one, and a failed
    retry never touches an already verified file;
  * `collect_manifest.txt` (the "collection succeeded" receipt that `down` reads) is
    written atomically and **only** when every required file is verified. On failure it is
    removed and `collect_manifest.txt.failed` records what went wrong instead. Before a
    re-collect can overwrite *any* byte of an existing archive, that receipt is renamed
    aside to `collect_manifest.txt.superseded` - an untrusted name no gate reads - so an
    abrupt death (SIGINT/SIGTERM/SIGKILL, a lost VM) cannot leave a stale `verify=ok`
    vouching for a half-replaced archive. The new receipt is written only at the end;
  * `COLLECT_CKPT=none|latest|all` keeps its meaning: none = logs/metadata only, latest =
    the numerically newest checkpoint (it_1000 > it_500; lexicographic order is wrong),
    all = every checkpoint on the VM. `latest`/`all` fail when a requested checkpoint
    cannot be verified, and `all` never reports success for a partial set.

Resumable collection (C-1). A second `collect` into the same directory does not re-download
bytes that are already there, but it never *assumes* they are good either:

  * the trusted source list is fetched fresh from the source every time, so a changed
    source hash invalidates any local copy;
  * every file already on disk is re-measured (size + sha256) against that list; only an
    exact match of both is skipped, anything else is fetched again;
  * the per-file progress of the current run is recorded in `.collect_partial.json`, written
    atomically after each verified file. That file is explicitly `verify=partial` and is
    never read as proof of a complete archive - only `collect_manifest.txt` with `verify=ok`
    is, and it is only ever written when the whole set verified. It is removed again once
    the run is fully verified, so a successful archive carries no stale progress marker.
    Reading it back is best-effort diagnostics: every field is type-checked and a malformed
    document of any shape is ignored, never allowed to abort the collection;

Remote source (C-2). `--source` points the collector at the real run directory instead of the
default `/content/nabla/runs/<run_id>` - typically a mounted Drive copy used to recover a run
whose VM is gone, without a hand-made symlink. The path is validated and normalised (absolute,
no `.`/`..` segments, no shell metacharacters, canonical form), its basename must equal the
run id, and the run identity is confirmed against the source's own `manifest.json`. For an
explicit `--source` the confirmation is strict: a successful archive is built only when
`experiment_id` is present and equals `--run-id`; an absent, unparseable or foreign value
is rejected with exit code 2, because a directory name alone cannot tie a Drive folder to
the run. The default VM source keeps its historical leniency (older runs predate
`experiment_id`) - there the run id comes from the VM, not from a human typing a path.

Provenance (C-2, the part that is easy to get wrong). Byte equality inside a Drive copy only
proves *the copy is internally intact*; it says nothing about whether that copy is what the
training VM wrote. So the two facts are recorded separately:

  * `verify=ok`                     the collected bytes match the source snapshot (integrity);
  * `provenance=...`+`degraded=`    whether the source itself is attributable to the training
                                    VM (attribution), which needs evidence: the `.sync_receipt`
                                    written on the training VM by vm/90_sync.sh after it
                                    compared source-vs-Drive file by file. Each collected file
                                    must appear there as `OK <rel> <size> <sha>` with the same
                                    size and sha256, and the receipt must name this directory.
                                    Without that evidence the collector still verifies what it
                                    was given, but labels the result `drive-copy-only` with
                                    `provenance_degraded=1` instead of pretending the bytes are
                                    known to be the training VM's.

Exit codes: 0 ok, 2 usage/identity, 3 remote query/snapshot failed, 4 some file failed to
verify, 5 local filesystem error (the archive could not be written or read to completion).

Local I/O is part of the same contract: a disk error while re-collecting into an existing
directory must not leave the previous run's success receipt standing, so every OSError is
routed through the same `fail()` that removes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_BEGIN = "COLLECT_SNAPSHOT_BEGIN"
SNAPSHOT_END = "COLLECT_SNAPSHOT_END"
RECEIPT_BEGIN = "COLLECT_RECEIPT_BEGIN"
RECEIPT_END = "COLLECT_RECEIPT_END"
ATTEMPTS = 3
CKPT_RE = re.compile(r"^checkpoints/it_(?P<it>\d+)\.pkl$")
SHA_RE = re.compile(r"[0-9a-f]{64}")

DEFAULT_SOURCE_ROOT = "/content/nabla/runs"
PARTIAL_NAME = ".collect_partial.json"
DIAG_NAME = "collect_diagnostics.txt"
# The previous success receipt is moved to this name before the first byte of a re-collect
# is written. No gate, script or report reads `collect_manifest.txt.superseded`.
SUPERSEDED_SUFFIX = ".superseded"
STATUS_PROBE_TIMEOUT = 45

# Files at or above this size must use streaming download only.  The `colab download`
# fallback loads the entire file as base64 JSON in memory (~4x file size), which OOMs
# on large checkpoints.  A streaming failure for a file this size is retried as streaming,
# never routed to the old path.
STREAMING_ONLY_THRESHOLD = 64 * 1024 * 1024  # 64 MiB

# --source must be a plain absolute VM path: no quoting, globbing, redirection or
# whitespace that could be re-read as something else by a later shell.
SOURCE_CHARS_RE = re.compile(r"^/[A-Za-z0-9._/@+=,-]+$")

DATA_SOURCE_VM = "training-vm"
DATA_SOURCE_DRIVE = "drive-recovery"
PROV_VM = "training-vm"
PROV_DRIVE_EVIDENCED = "drive-recovery-verified"
PROV_DRIVE_DEGRADED = "drive-copy-only"

# Always required: without these the archive cannot be interpreted at all.
CORE_FILES = ("manifest.json", "config.json", "log.jsonl")
# Downloaded and verified when the VM has them (older runs may predate train.log/ea_report.json).
OPTIONAL_FILES = (
    "provenance.json", "summary.json", "curve.csv", "report.md", "train.log", "ea_report.json",
    "evaluation/eval_nominal.json", "evaluation/per_clip.csv", "evaluation/summary.json",
)
# The down-gate requires exactly these for a PERSISTED/COMPLETE run; a missing one here means
# the archive would fail that gate later, so collect refuses instead of pretending.
REQUIRED_WHEN_PERSISTED = (
    "summary.json", "curve.csv", "report.md",
    "evaluation/eval_nominal.json", "evaluation/per_clip.csv",
)

SNAPSHOT_PROGRAM = '''\
import glob, hashlib, json, os
root = {root}
files = {files}
print("{begin}")
print("ROOT", root)
identity = ""
try:
    with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8", errors="replace") as f:
        identity = str(json.load(f).get("experiment_id") or "")
except Exception:
    identity = ""
print("IDENTITY", identity or "-")
def emit(rel, p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print("FILE", rel, os.path.getsize(p), h.hexdigest(), p)
for rel in files:
    p = os.path.join(root, rel)
    if not os.path.isfile(p) and rel.startswith("evaluation/"):
        p = os.path.join(root, os.path.basename(rel))
    if not os.path.isfile(p):
        print("MISSING", rel)
        continue
    emit(rel, p)
raw = []
for pattern in ("checkpoints/it_*.pkl", "ckpt/it_*.pkl"):
    raw.extend(glob.glob(os.path.join(root, pattern)))
uniq = {{}}
for p in sorted(raw):
    rel = "checkpoints/" + os.path.basename(p)
    if rel in uniq:
        continue
    uniq[rel] = p
    emit(rel, p)
rp = os.path.join(root, ".sync_receipt")
if os.path.isfile(rp):
    print("{rcpt_begin}")
    try:
        with open(rp, "r", encoding="utf-8", errors="replace") as f:
            for line in f.read().splitlines()[:400]:
                print("RCPT", line)
    except OSError as exc:
        print("RCPT", "error=unreadable", exc)
    print("{rcpt_end}")
print("{end}", len(raw), len(uniq))
'''


class RemoteError(RuntimeError):
    """The source could not be queried, or answered with something we cannot trust."""


class UsageError(RuntimeError):
    """The arguments cannot denote a single, well-defined run directory."""


class ReceiptError(RuntimeError):
    """The previous verdict cannot be revoked, so the archive must not be touched."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_session_endpoint(session_name: str):
    """(base_url, token) for a colab-cli session, or None if unknown.

    Reads the colab-cli session store to resolve a session name to the Jupyter
    server URL and auth token needed for the /files/ streaming endpoint.
    """
    config = Path.home() / ".config" / "colab-cli" / "sessions.json"
    try:
        data = json.loads(config.read_text())
    except (OSError, ValueError):
        return None
    sess = data.get(session_name) or {}
    url = (sess.get("url") or "").rstrip("/")
    token = sess.get("token") or ""
    return (url, token) if url and token else None


def stream_fetch(remote: str, part: Path, session: str, timeout: int) -> str:
    """Stream a remote file to *part* via the Jupyter /files/ endpoint.

    Unlike ``colab download`` (which materialises the whole file as base64 JSON
    in memory via /api/contents?content=1 — causing OOM for large checkpoints),
    this reads raw bytes in 1 MiB chunks so peak memory stays constant regardless
    of file size.  Returns '' on success or a reason string on failure.

    Authentication uses the ``X-Colab-Runtime-Proxy-Token`` request header
    (recommended by Google's Colab API docs to avoid leaking tokens in URLs);
    the query parameter is also included for compatibility with older proxies.
    """
    ep = get_session_endpoint(session)
    if ep is None:
        return f"session {session!r} not in colab-cli sessions.json"
    base_url, token = ep
    quoted = urllib.parse.quote(remote.strip("/"), safe="/")
    url = f"{base_url}/files/{quoted}?colab-runtime-proxy-token={token}"
    req = urllib.request.Request(url)
    req.add_header("X-Colab-Runtime-Proxy-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(part, "wb") as f:
                for chunk in iter(lambda: resp.read(1 << 20), b""):
                    f.write(chunk)
        return ""
    except Exception as e:
        return f"stream error: {e}"


def colab_exec(session: str, code: str, timeout: int) -> str:
    try:
        p = subprocess.run(
            ["colab", "exec", "-s", session, "--timeout", str(timeout)],
            input=code, capture_output=True, text=True, timeout=timeout + 120,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RemoteError(f"colab exec failed: {e}") from e
    if p.returncode != 0:
        detail = (p.stderr or p.stdout or "").strip().splitlines()
        raise RemoteError(f"colab exec exited {p.returncode}"
                          + (f": {detail[-1]}" if detail else ""))
    return p.stdout


# --- source path + run identity -------------------------------------------

def normalize_source(raw: str, run_id: str) -> str:
    """Validate `--source` and return the canonical absolute remote path.

    The path is interpolated into a remote Python program and passed to `colab download`,
    so it is validated as a *path*, not as text: absolute, canonical (a path that changes
    under normpath is ambiguous - `/a//b`, `/a/./b`, `/a/b/`), no parent segments, no
    quoting/globbing/redirection/whitespace, and a basename equal to the run id so a typo
    cannot silently point the collector at a different run.
    """
    path = (raw or "").strip()
    if not path:
        raise UsageError("--source must not be empty")
    if not path.startswith("/"):
        raise UsageError(f"--source must be an absolute path on the VM (got {raw!r})")
    if any(ch in path for ch in "\n\r\t '`\"\\$;&|<>()*?!#"):
        raise UsageError(f"--source contains a character that is not allowed in a run "
                         f"directory (got {raw!r})")
    if not SOURCE_CHARS_RE.match(path):
        raise UsageError(f"--source must match /[A-Za-z0-9._/@+=,-]+ (got {raw!r})")
    segments = path.split("/")
    if any(part in ("..",) for part in segments):
        raise UsageError(f"--source must not contain '..' (got {raw!r})")
    canonical = posixpath.normpath(path)
    if canonical != path:
        raise UsageError(f"--source is not canonical: {raw!r} normalises to {canonical!r}; "
                         f"pass the normalised form")
    if path == "/":
        raise UsageError("--source must be a run directory, not /")
    base = posixpath.basename(path)
    if base != run_id:
        raise UsageError(f"--source must end in the run id {run_id!r} (got {base!r}); "
                         f"the directory and the run must be named alike")
    return canonical


def parse_snapshot(text: str):
    """Parse the framed source snapshot. Anything unexpected is a hard error."""
    files: dict[str, tuple[int, str, str]] = {}
    missing: list[str] = []
    receipt: list[str] = []
    identity = ""
    raw_ck = uniq_ck = None
    state = "before"
    in_receipt = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if state == "before":
            if line == SNAPSHOT_BEGIN:
                state = "inside"
            continue
        if state == "after":
            continue
        if in_receipt:
            if line == RECEIPT_END:
                in_receipt = False
            elif line.startswith("RCPT "):
                receipt.append(line[5:])
            elif line:
                raise RemoteError(f"unexpected line inside the source receipt: {line!r}")
            continue
        if line == RECEIPT_BEGIN:
            in_receipt = True
            continue
        if line.startswith(SNAPSHOT_END):
            parts = line.split()
            if len(parts) != 3 or not all(x.isdigit() for x in parts[1:]):
                raise RemoteError(f"malformed snapshot footer: {line!r}")
            raw_ck, uniq_ck = int(parts[1]), int(parts[2])
            state = "after"
            continue
        # A source that has no readable experiment_id may emit either `IDENTITY -` (what
        # SNAPSHOT_PROGRAM does) or a bare `IDENTITY` (empty value). Both mean "no identity",
        # never an unexpected line: the strict --source check below needs to see that state
        # so it can refuse with exit 2 rather than a misleading snapshot-parse failure.
        if line == "IDENTITY" or line.startswith("IDENTITY "):
            identity = line[len("IDENTITY"):].strip()
            continue
        if line.startswith("ROOT "):
            continue
        if line.startswith("MISSING "):
            missing.append(line.split(None, 1)[1])
            continue
        if line.startswith("FILE "):
            parts = line.split()
            if len(parts) != 5:
                raise RemoteError(f"malformed snapshot line: {line!r}")
            _, rel, size, digest, remote = parts
            if not size.isdigit():
                raise RemoteError(f"snapshot size for {rel} is not a number: {size!r}")
            if not SHA_RE.fullmatch(digest):
                raise RemoteError(f"snapshot sha256 for {rel} is malformed: {digest!r}")
            if rel in files:
                raise RemoteError(f"snapshot lists {rel} twice")
            files[rel] = (int(size), digest, remote)
            continue
        raise RemoteError(f"unexpected line in snapshot: {line!r}")
    if state != "after":
        raise RemoteError("snapshot is unterminated (no COLLECT_SNAPSHOT_END)")
    if in_receipt:
        raise RemoteError("the source receipt block is unterminated")
    return files, missing, raw_ck, uniq_ck, identity, receipt


def read_source(session: str, source: str, timeout: int):
    """Ask the source for size+sha256 of every candidate file. Never guesses."""
    code = SNAPSHOT_PROGRAM.format(
        root=repr(source), files=list(CORE_FILES + OPTIONAL_FILES),
        begin=SNAPSHOT_BEGIN, end=SNAPSHOT_END,
        rcpt_begin=RECEIPT_BEGIN, rcpt_end=RECEIPT_END,
    )
    return parse_snapshot(colab_exec(session, code, timeout))


def checkpoints(files: dict[str, tuple[int, str, str]]):
    """[(iteration, rel, size, sha, remote)] sorted numerically, not lexicographically."""
    found = []
    for rel, (size, digest, remote) in files.items():
        m = CKPT_RE.match(rel)
        if m:
            found.append((int(m.group("it")), rel, size, digest, remote))
    return sorted(found)


# --- provenance ------------------------------------------------------------

def parse_sync_receipt(lines: list[str]):
    """(state, {rel: (size, sha)}, meta) from the training VM's `.sync_receipt`.

    vm/90_sync.sh writes one `OK <rel> <size> <sha>` line per file it compared
    source-vs-Drive on the training VM, then a `verified=` verdict. Only those OK lines
    bind a Drive byte-string to the original run, so only those can be used as evidence.
    """
    state = "none"
    evidence: dict[str, tuple[int, str]] = {}
    meta: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("OK "):
            parts = line.split()
            if len(parts) == 4 and parts[2].isdigit() and SHA_RE.fullmatch(parts[3]):
                evidence[parts[1]] = (int(parts[2]), parts[3])
            continue
        key, sep, value = line.partition("=")
        if sep:
            meta.setdefault(key.strip(), value.strip())
    if "verified" in meta:
        state = meta["verified"] or "none"
    return state, evidence, meta


def assess_provenance(data_source: str, source: str, planned: list[str],
                      files: dict[str, tuple[int, str, str]], receipt_lines: list[str]):
    """Separate 'these bytes match the source' from 'the source is the training VM's'.

    Returns (provenance, evidence_kind, degraded, notes, receipt_meta).
    """
    if data_source == DATA_SOURCE_VM:
        return (PROV_VM, "source-sha256-snapshot", False,
                ["the source is the live training VM path, so integrity and attribution "
                 "share one source of truth"], {})

    state, evidence, meta = parse_sync_receipt(receipt_lines)
    notes: list[str] = []
    if not receipt_lines:
        notes.append(f"{source}/.sync_receipt is absent: nothing on this copy says it was "
                     f"compared against the training VM")
        return PROV_DRIVE_DEGRADED, "none", True, notes, meta
    if state not in ("ok", "partial"):
        notes.append(f".sync_receipt reports verified={state}; it does not certify the copy")
    dst = meta.get("dst", "")
    if dst and dst.rstrip("/") != source.rstrip("/"):
        notes.append(f".sync_receipt was written for dst={dst}, not {source}")
    elif not dst:
        notes.append(".sync_receipt does not name the directory it verified")

    unevidenced = []
    for rel in planned:
        size, digest, _remote = files[rel]
        got = evidence.get(rel)
        if got is None:
            unevidenced.append(f"{rel} (not listed as OK)")
        elif got != (size, digest):
            unevidenced.append(f"{rel} (receipt has a different size/sha256)")
    if unevidenced:
        shown = "; ".join(unevidenced[:4]) + ("; ..." if len(unevidenced) > 4 else "")
        notes.append(f"{len(unevidenced)}/{len(planned)} collected file(s) are not bound to the "
                     f"training VM by the receipt: {shown}")

    if notes:
        return PROV_DRIVE_DEGRADED, "none", True, notes, meta
    return (PROV_DRIVE_EVIDENCED, "vm-sync-receipt", False,
            [f".sync_receipt (verified_at={meta.get('verified_at', '?')}, "
             f"verified={state}, {meta.get('checkpoints_verified', '?')}/"
             f"{meta.get('checkpoints_total', '?')} checkpoints) covers every collected file"],
            meta)


# --- pre-flight diagnostics (C-3) -----------------------------------------

class Diagnostics:
    """Best-effort record of *how* this collection was started.

    Written before any bytes move and rewritten at the end. It must never make the run
    fail: an archive that is completely verified is a good archive even if the session
    cannot be interrogated any more (that is exactly the Drive-recovery situation).
    """

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[tuple[str, str]] = [("started_at_utc", utcnow())]

    def add(self, key: str, value: str) -> None:
        self.lines.append((key, str(value)))

    def flush(self, outcome: str, extra: list[tuple[str, str]] | None = None) -> None:
        rows = list(self.lines) + list(extra or []) + [("outcome", outcome),
                                                       ("written_at_utc", utcnow())]
        body = "".join(f"{k}={v}\n" for k, v in rows)
        try:
            write_atomic(self.path, body)
        except OSError as e:
            # Diagnostics are not evidence of anything; losing them must not lose the archive.
            print(f"WARN: could not write {self.path}: {e}", file=sys.stderr)


def probe_colab_status(session: str) -> list[tuple[str, str]]:
    """`colab status` for the record - its failure is reported as a failure, never as ok."""
    if not session:
        return [("colab_status", "skipped (no session given)"), ("colab_status_rc", "-")]
    try:
        p = subprocess.run(["colab", "status", "-s", session],
                           capture_output=True, text=True, timeout=STATUS_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [("colab_status", f"timeout after {STATUS_PROBE_TIMEOUT}s"),
                ("colab_status_rc", "timeout")]
    except OSError as e:
        return [("colab_status", f"could not run `colab status`: {e}"), ("colab_status_rc", "-")]
    out_lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    err_lines = [ln for ln in (p.stderr or "").splitlines() if ln.strip()]
    rows: list[tuple[str, str]] = [("colab_status_rc", str(p.returncode))]
    if p.returncode != 0:
        rows.append(("colab_status", "FAILED"))
        for ln in (err_lines or out_lines)[:10]:
            rows.append(("colab_status_stderr", ln))
        return rows
    rows.append(("colab_status", "ok" if out_lines else "empty (rc=0 but no output)"))
    for ln in out_lines[:10]:
        rows.append(("colab_status_line", ln))
    rows.append(("colab_session_id", parse_session_id(out_lines)))
    return rows


def parse_session_id(lines: list[str]) -> str:
    """Pull a session identifier out of `colab status` output; say so when there is none."""
    for line in lines:
        m = re.search(r"(?i)\bsession(?:\s*id)?\s*[:=]\s*(\S+)", line)
        if m:
            return m.group(1)
    for line in lines:
        m = re.search(r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", line)
        if m:
            return m.group(1)
    return "(not reported by `colab status`)"


# --- transfer --------------------------------------------------------------

def local_state(path: Path, size: int, digest: str):
    """(matches, why) for a file already on disk versus the trusted source entry.

    Only an exact size *and* sha256 match counts: this is what licenses skipping a
    download, so a partially written or silently corrupted local file must never pass.
    """
    try:
        if not path.exists():
            return False, "missing"
        if not path.is_file():
            return False, "not a regular file"
        if path.stat().st_size != size:
            return False, "local size differs from the source"
        if sha256_of(path) != digest:
            return False, "local sha256 differs from the source"
        return True, "matches the source snapshot"
    except OSError as e:
        return False, f"unreadable locally ({e.strerror or e})"


def download_verified(session: str, remote: str, dest: Path, size: int, digest: str,
                      attempts: int = ATTEMPTS, timeout: int = 1800):
    """Download one file and prove it matches the source snapshot before publishing it.

    Returns (ok, reason). The final path is only ever written by an atomic rename of a
    fully verified temp file, so an existing verified copy survives a failed retry.

    A local I/O error (no space left, permission denied, destination is not a file) is a
    failed attempt like any other: it is retried and finally reported, never propagated as
    a traceback, because an uncaught exception would skip the receipt revocation in fail().
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.part.{os.getpid()}")
    streaming_only = size >= STREAMING_ONLY_THRESHOLD
    reason = "not attempted"
    for attempt in range(1, attempts + 1):
        try:
            if part.exists():
                part.unlink()
            # Transfer: Jupyter /files/ streaming first (constant memory for any
            # file size).  Small files (< STREAMING_ONLY_THRESHOLD) may fall back
            # to `colab download`; large files must never use the old path because
            # it materialises the whole file in memory (~4x) and OOMs on checkpoints.
            reason = stream_fetch(remote, part, session, timeout)
            if reason and not streaming_only:
                try:
                    p = subprocess.run(["colab", "download", "-s", session, remote, str(part)],
                                       capture_output=True, text=True, timeout=timeout)
                except (subprocess.TimeoutExpired, OSError) as e:
                    reason = f"download error: {e}"
                else:
                    if p.returncode != 0:
                        detail = (p.stderr or p.stdout or "").strip().splitlines()
                        reason = f"download exited {p.returncode}" + (f": {detail[-1]}" if detail else "")
                    else:
                        reason = ""
            if not reason:
                if not part.exists():
                    reason = "no file was written"
                else:
                    got_size = part.stat().st_size
                    if got_size != size:
                        reason = f"size {got_size} != source {size} (truncated transfer)"
                    else:
                        got = sha256_of(part)
                        if got != digest:
                            reason = f"sha256 {got} != source {digest}"
                        else:
                            os.replace(part, dest)   # atomic publish, verified bytes only
                            return True, ""
        except OSError as e:
            # Local side of the transfer: unreadable part file, unwritable destination,
            # rename refused. Retry, then let the caller fail the collection cleanly.
            reason = f"local i/o error: {e.strerror or e}"
        finally:
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
        if attempt < attempts:
            print(f"  retry {attempt}/{attempts - 1} for {dest.name}: {reason}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 5))
    return False, reason


def write_atomic(path: Path, text: str) -> None:
    """Write via rename so a reader never sees half a receipt. Leaves no temp on failure."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        # An interrupt in the middle of writing a receipt/progress file must not leave a
        # half-written temp behind either. The published path is still the old version (or
        # nothing) in that case; no reader can mistake the temp for a verdict.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


class PartialManifest:
    """Per-file progress of the current collection, written atomically after every file.

    This file is *not* a receipt and cannot become one: it always says `verify=partial`,
    it is only ever consulted for diagnostics, and a resume re-measures the files on disk
    against a fresh source snapshot rather than believing what is written here. It is
    deleted once the collection is fully verified.
    """

    def __init__(self, out: Path, run_id: str, source: str, data_source: str, planned: list[str]):
        self.path = out / PARTIAL_NAME
        self.data = {
            "schema": 1,
            "kind": "collect-partial",
            "verify": "partial",
            "archive_state": "collecting",
            "receipt_superseded": None,
            "note": ("Per-file progress of an INCOMPLETE collect. This document is never a "
                     "success receipt and must never be read as one: only collect_manifest.txt "
                     "carries a passing verdict. While archive_state=collecting the previous "
                     "success receipt has been revoked and no verdict about this archive is "
                     "valid. Re-runs re-measure the local files against a fresh source "
                     "snapshot and ignore what is written here."),
            "run_id": run_id,
            "source": source,
            "data_source": data_source,
            "started_at": utcnow(),
            "updated_at": utcnow(),
            "planned": list(planned),
            "files": {},
        }

    @staticmethod
    def schema_error(old: object) -> str | None:
        """Validate *every* field we read back. Returns why the document must be ignored.

        This file lives next to the archive and can be truncated, hand-edited or forged, so no
        operation on it may assume a type. In particular `len(old["files"])` used to raise
        TypeError for `"files": 42` / `"files": true`, which escaped the caller's OSError/
        ValueError handler and killed the run with a traceback - taking any stale receipt
        with it. Strict validation makes that class of bug impossible.
        """
        if not isinstance(old, dict):
            return f"it is a {type(old).__name__}, not a collect-partial document"
        if old.get("kind") != "collect-partial":
            return "it is not a collect-partial document"
        if old.get("verify") != "partial":
            return (f"it claims verify={old.get('verify')!r}; only an explicit "
                    f"verify=partial progress file is ever written here")
        files = old.get("files")
        if not isinstance(files, dict):
            return f"its 'files' member is a {type(files).__name__}, not a mapping"
        for rel, entry in files.items():
            if not isinstance(rel, str) or not isinstance(entry, dict):
                return f"its 'files' member has a non-object entry for {rel!r}"
            size, digest = entry.get("size"), entry.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                return f"the recorded size for {rel!r} is not a byte count"
            if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
                return f"the recorded sha256 for {rel!r} is not a sha256"
        if not isinstance(old.get("run_id"), str):
            return "its run_id is not a string"
        if not isinstance(old.get("updated_at"), str):
            return "its updated_at is not a string"
        return None

    def load_existing(self) -> str:
        """Note what a previous run left behind. Returns a human-readable status line.

        Pure diagnostics: no exception from reading, parsing or *describing* this file may
        abort a collection, because it is never evidence of anything.
        """
        if not self.path.exists():
            return "no partial manifest from a previous run"
        try:
            old = json.loads(self.path.read_text())
            why = self.schema_error(old)
            if why is not None:
                return f"ignoring the partial manifest of a previous run: {why}"
            if old["run_id"] != self.data["run_id"]:
                return (f"ignoring the partial manifest of a previous run: it belongs to "
                        f"{old['run_id']!r}")
            return (f"a previous attempt recorded {len(old['files'])} verified file(s) at "
                    f"{old['updated_at']} (not trusted; re-measuring)")
        except (OSError, ValueError) as e:
            return (f"ignoring an unreadable partial manifest ({type(e).__name__}): it is never "
                    f"trusted, local files are re-measured against the source")
        except Exception as e:  # noqa: BLE001 - best-effort diagnostics, never evidence
            return (f"ignoring a partial manifest this collector cannot read "
                    f"({type(e).__name__}): it is never trusted, local files are re-measured "
                    f"against the source")

    def mark(self, rel: str, size: int, digest: str, origin: str) -> None:
        self.data["files"][rel] = {
            "size": size, "sha256": digest, "origin": origin, "verified_at": utcnow(),
        }
        self.data["updated_at"] = utcnow()

    def save(self) -> None:
        self.data["updated_at"] = utcnow()
        write_atomic(self.path, json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def record_failure(self, failures: list[str]) -> None:
        self.data["verify"] = "partial"
        self.data["archive_state"] = "incomplete"
        self.data["failed_at"] = utcnow()
        self.data["failed"] = list(failures)
        try:
            self.save()
        except OSError as e:
            print(f"WARN: could not update {self.path}: {e}", file=sys.stderr)

    def discard(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"WARN: could not remove {self.path}: {e}", file=sys.stderr)


# --- revoking the previous verdict before the first write ------------------

def supersede_receipt(receipt: Path, partial: PartialManifest) -> str:
    """Move a previous `verify=ok` out of the trusted path BEFORE any archive byte moves.

    Without this, a `FORCE=1` re-collect keeps the old receipt alive while it replaces files.
    A normal failure routes through fail() and revokes it, but SIGINT/SIGTERM/SIGKILL (or a
    VM that simply vanishes) never reaches fail(): the archive would then advertise a verdict
    about bytes that no longer exist. os.replace() is atomic, so from this instant on there
    is no `collect_manifest.txt` for any gate to trust; the old text survives under a name
    nothing reads, purely as a forensic record of what the archive used to claim.

    If the rename fails we must not touch the archive at all: the caller turns that into a
    fail-closed error. The receipt is still in place, but no byte has been modified yet.
    """
    aside = receipt.with_name(receipt.name + SUPERSEDED_SUFFIX)
    if not receipt.exists():
        if aside.exists():
            # A previous attempt was killed after revoking but before finishing. Carry the
            # fact forward so a resume that fails again still says which verdict is void.
            try:
                before = aside.read_text()
            except OSError:
                before = ""
            partial.data["receipt_superseded"] = describe_receipt(aside.name, before)
            partial.data["receipt_superseded"]["revoked_by"] = "an earlier attempt"
            return f"none (an earlier attempt already revoked it -> {aside.name})"
        return "none"
    try:
        before = receipt.read_text()
    except OSError:
        before = ""            # the rename is what matters; the copy is only forensics
    try:
        os.replace(receipt, aside)
    except OSError as e:
        raise ReceiptError(
            f"cannot revoke the previous success receipt ({e.strerror or e}): refusing to "
            f"re-collect while a verdict about the old bytes is still in place") from e
    partial.data["receipt_superseded"] = describe_receipt(aside.name, before)
    partial.data["receipt_superseded"]["revoked_by"] = "this attempt"
    return f"{partial.data['receipt_superseded']['verdict']} revoked -> {aside.name}"


def describe_receipt(name: str, text: str) -> dict:
    """Forensic description of a verdict that is no longer at the trusted path."""
    return {
        "name": name,
        "verdict": next((ln for ln in text.splitlines() if ln.startswith("verify=")), "unknown"),
        "collected_at": next((ln.split("=", 1)[1] for ln in text.splitlines()
                              if ln.startswith("collected_at=")), ""),
        "sha256": hashlib.sha256(text.encode()).hexdigest() if text else "",
    }


def retire_superseded(receipt: Path) -> None:
    """The new verdict is authoritative; drop the forensic copy of the old one."""
    aside = receipt.with_name(receipt.name + SUPERSEDED_SUFFIX)
    try:
        aside.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"WARN: could not remove {aside}: {e}", file=sys.stderr)


# --- entry point -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="",
                    help="remote run directory (default %s/<run-id>); "
                         "e.g. a mounted Drive copy for recovery" % DEFAULT_SOURCE_ROOT)
    ap.add_argument("--ckpt-mode", default="latest", choices=("none", "latest", "all"))
    ap.add_argument("--state", default="", help="pipeline state reported by the VM (PERSISTED...)")
    ap.add_argument("--run", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--budget-minutes", default="")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS)
    ap.add_argument("--timeout", type=int, default=600, help="remote snapshot timeout (s)")
    args = ap.parse_args()

    # The run id is interpolated into the remote program, so it must be a plain run id.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        print(f"collect: --run-id must match [A-Za-z0-9._-]+ (got {args.run_id!r})", file=sys.stderr)
        return 2
    try:
        args.source = normalize_source(args.source or f"{DEFAULT_SOURCE_ROOT}/{args.run_id}",
                                       args.run_id)
    except UsageError as e:
        print(f"collect: {e}", file=sys.stderr)
        return 2

    args.default_source = (args.source == f"{DEFAULT_SOURCE_ROOT}/{args.run_id}")
    out = Path(args.out)
    receipt = out / "collect_manifest.txt"
    try:
        return collect(args, out, receipt)
    except OSError as e:
        # Anything that escapes the per-file retry loop lands here (a directory that cannot be
        # created, a receipt that cannot be written). With FORCE=1 the archive already holds a
        # receipt from an earlier good collect, so it must be revoked here rather than survive
        # an uncaught traceback that claims nothing about the archive's validity.
        where = getattr(e, "filename", None) or "?"
        return fail(out, receipt,
                    [f"local filesystem error: {e.strerror or e} ({where})"],
                    "the local archive could not be written or read to completion",
                    code=5)


def collect(args, out: Path, receipt: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    data_source = DATA_SOURCE_VM if args.default_source else DATA_SOURCE_DRIVE

    diag = Diagnostics(out / DIAG_NAME)
    diag.add("run_id", args.run_id)
    diag.add("source_path", args.source)
    diag.add("data_source", data_source)
    diag.add("source_argument", args.source if not args.default_source else "(default)")
    diag.add("colab_session", args.session)
    for key, value in probe_colab_status(args.session):
        diag.add(key, value)
    diag.add("ckpt_mode", args.ckpt_mode)
    diag.flush("pending")

    print(f"snapshot of {args.run_id} from {args.source} "
          f"({data_source}, session {args.session}, ckpt-mode={args.ckpt_mode})")
    try:
        files, missing, raw_ck, uniq_ck, identity, receipt_lines = read_source(
            args.session, args.source, args.timeout)
    except RemoteError as e:
        return fail(out, receipt,
                    [f"cannot read the source snapshot from {args.source}: {e}"],
                    "the source could not be queried, so nothing can be verified",
                    code=3, diag=diag)
    diag.add("source_files_listed", str(len(files)))
    diag.add("run_identity", identity or "-")

    # The run identity must come from the source's own manifest.json, not from a directory a
    # human typed. For an explicit --source that is a hard requirement: a Drive folder whose
    # manifest.json lacks (or cannot parse) experiment_id cannot be tied to this run, and a
    # byte-perfect copy of the wrong experiment is still the wrong experiment. The default
    # VM path is not operator-supplied (run.sh discovered the run id on the VM), so it keeps
    # the historical behaviour for runs that predate the field.
    if identity and identity != "-" and identity != args.run_id:
        return fail(out, receipt,
                    [f"run identity mismatch: {args.source}/manifest.json says "
                     f"experiment_id={identity!r} but --run-id is {args.run_id!r}"],
                    "the source directory belongs to a different run", code=2, diag=diag)
    if (not identity or identity == "-") and data_source == DATA_SOURCE_DRIVE:
        return fail(out, receipt,
                    [f"cannot confirm run identity: {args.source}/manifest.json has no "
                     f"readable experiment_id (got {identity or '<empty>'!r}); a recovery "
                     f"source must name the run it holds, so it cannot be accepted as "
                     f"{args.run_id!r} on the strength of its directory name alone"],
                    "the source directory cannot be tied to this run", code=2, diag=diag)
    if not identity or identity == "-":
        print(f"WARN: {args.source}/manifest.json did not report an experiment_id; the default "
              f"VM path is trusted by run id only, and the receipt will say run_identity=-",
              file=sys.stderr)
        diag.add("run_identity_check", "unavailable (default VM source)")
    else:
        diag.add("run_identity_check", "experiment_id matches --run-id")

    if raw_ck != uniq_ck:
        print(f"WARN: the source holds {raw_ck} checkpoint files for {uniq_ck} distinct "
              f"iterations (duplicated ckpt/ and checkpoints/ trees); collecting the distinct set",
              file=sys.stderr)

    # --- what must be downloaded, and what merely can be --------------------
    state = args.state
    state_source = "source state query" if state else "unknown"
    receipt_state = parse_sync_receipt(receipt_lines)[0]
    diag.add("source_sync_receipt", receipt_state if receipt_lines else "absent")
    if not state and data_source == DATA_SOURCE_DRIVE:
        # The VM is normally gone here, so the copy's own sync verdict is the best statement
        # about which files the training run was supposed to have. It is *evidence*, not
        # licence: the bytes still have to match, and provenance is still assessed below.
        if receipt_state == "ok":
            state, state_source = "PERSISTED", "sync receipt"
        elif receipt_state == "partial":
            state, state_source = "INCOMPLETE", "sync receipt"
    diag.add("pipeline_state", state or "unknown")
    diag.add("state_source", state_source)

    required = list(CORE_FILES)
    if state in ("PERSISTED", "COMPLETE"):
        required += [r for r in REQUIRED_WHEN_PERSISTED if r not in required]
    wanted: list[str] = []
    for rel in required:
        if rel in files:
            wanted.append(rel)
        else:
            failures.append(f"{rel}: required file is missing at {args.source}")
    for rel in OPTIONAL_FILES:
        if rel in files and rel not in wanted:
            wanted.append(rel)

    ckpt_all = checkpoints(files)
    if args.ckpt_mode == "none":
        ckpt_plan = []
    elif args.ckpt_mode == "latest":
        ckpt_plan = ckpt_all[-1:]
    else:
        ckpt_plan = ckpt_all
    if args.ckpt_mode != "none" and not ckpt_all:
        failures.append(f"no checkpoints at {args.source}: COLLECT_CKPT={args.ckpt_mode} needs at "
                        f"least one (use COLLECT_CKPT=none to collect logs only)")

    planned = wanted + [rel for _it, rel, _s, _d, _r in ckpt_plan]
    print(f"  {len(files)} file(s) present ({len(missing)} reported missing), {len(wanted)} to fetch, "
          f"{len(ckpt_plan)}/{len(ckpt_all)} checkpoint(s), pipeline state={state or 'unknown'}")

    prov, prov_evidence, prov_degraded, prov_notes, prov_meta = assess_provenance(
        data_source, args.source, planned, files, receipt_lines)
    diag.add("provenance", prov)
    diag.add("provenance_degraded", "1" if prov_degraded else "0")
    for note in prov_notes:
        diag.add("provenance_note", note)
    if prov_degraded:
        for note in prov_notes:
            print(f"WARN provenance: {note}", file=sys.stderr)
        print(f"WARN provenance: recording {prov} (degraded) - the bytes are verified against "
              f"{args.source}, but this is NOT evidence that {args.source} is what the training "
              f"VM produced", file=sys.stderr)

    # A previous attempt's per-file progress: reported, never believed.
    partial = PartialManifest(out, args.run_id, args.source, data_source, planned)
    previous = partial.load_existing()
    print(f"  {previous}")
    diag.add("previous_partial", previous)

    # Revoke a previous success verdict BEFORE the first write to the archive. Deleting the
    # receipt in fail() is too late: a signal or a dead VM never gets there. From here until
    # the final atomic write there is no collect_manifest.txt, so an interrupted re-collect
    # can only ever leave an unverified archive, never a falsely verified one.
    try:
        superseded = supersede_receipt(receipt, partial)
    except ReceiptError as e:
        # No archive file was modified, so fail() only has to invalidate the old verdict.
        return fail(out, receipt, [str(e)],
                    "the previous success receipt could not be revoked, so no archive file "
                    "was touched", code=5, diag=diag)
    diag.add("previous_receipt", superseded)
    if superseded != "none":
        print(f"  previous verdict revoked before the re-collect: {superseded}", file=sys.stderr)

    # --- transfer ----------------------------------------------------------
    verified: list[str] = []
    downloaded: list[str] = []
    resumed: list[str] = []
    try:
        partial.save()
        for rel in planned:
            size, digest, remote = files[rel]
            dest = out / rel
            ok, why = local_state(dest, size, digest)
            if ok:
                resumed.append(rel)
                partial.mark(rel, size, digest, "resumed")
                partial.save()
                print(f"  verified locally, skipped download: {rel}")
                verified.append(rel)
                continue
            if dest.exists() or why != "missing":
                print(f"  fetching {rel} ({why})", file=sys.stderr)
            ok, reason = download_verified(args.session, remote, dest, size, digest, args.attempts)
            if ok:
                downloaded.append(rel)
                partial.mark(rel, size, digest, "downloaded")
                partial.save()
                verified.append(rel)
            else:
                failures.append(f"{rel}: {reason} after {args.attempts} attempts")
    except OSError:
        # Partial progress is worth keeping for the next attempt; the success receipt must not
        # be produced, and fail() (via main) revokes any stale one.
        try:
            partial.record_failure(failures + ["local filesystem error during transfer"])
        except OSError:
            pass
        raise

    ckpt_local = sum(1 for rel in verified if CKPT_RE.match(rel))

    if failures:
        partial.record_failure(failures)
        return fail(out, receipt, failures,
                    f"{len(failures)} file(s) could not be verified "
                    f"({ckpt_local}/{len(ckpt_all)} checkpoint(s) is a partial archive, not a success)",
                    diag=diag,
                    extra=[("verified_files", str(len(verified))),
                           ("downloaded_files", str(len(downloaded))),
                           ("resumed_files", str(len(resumed)))])

    # --- success receipt ---------------------------------------------------
    drive_run = ""
    try:
        drive_run = json.loads((out / "manifest.json").read_text()).get("drive_path", "") or ""
    except (OSError, ValueError):
        drive_run = ""
    receipt_sha = hashlib.sha256("\n".join(receipt_lines).encode()).hexdigest() if receipt_lines else ""
    lines = [
        f"session={args.session} gpu={args.gpu} run={args.run} seed={args.seed} "
        f"run_id={args.run_id} budget_minutes={args.budget_minutes}",
        f"collected_at={utcnow()}",
        f"drive_run={drive_run or '(none)'}",
        f"pipeline_state={state or 'unknown'}",
        f"state_source={state_source}",
        f"ckpt_mode={args.ckpt_mode}",
        f"ckpt_expected={len(ckpt_all)}",
        f"ckpt_local={ckpt_local}",
        f"files_verified={len(verified)}",
        f"files_downloaded={len(downloaded)}",
        f"files_resumed={len(resumed)}",
        f"attempts_per_file={args.attempts}",
        "integrity=source-sha256-snapshot",
        f"data_source={data_source}",
        f"source_path={args.source}",
        f"run_identity={identity or '-'}",
        f"provenance={prov}",
        f"provenance_evidence={prov_evidence}",
        f"provenance_degraded={'1' if prov_degraded else '0'}",
        f"provenance_receipt_verified={prov_meta.get('verified', 'n/a') if prov_meta else 'n/a'}",
        f"provenance_receipt_sha256={receipt_sha or 'n/a'}",
        f"diagnostics={DIAG_NAME}",
        "verify=ok",
    ]
    write_atomic(receipt, "\n".join(lines) + "\n")
    retire_superseded(receipt)
    stale = out / "collect_manifest.txt.failed"
    try:
        stale.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        # The archive is verified; a leftover marker is cosmetic, so warn instead of failing.
        print(f"WARN: could not remove {stale}: {e}", file=sys.stderr)
    partial.discard()
    diag.flush("verified", [("ok", "1")])
    print(f"verified {len(verified)} file(s): "
          f"{len(verified) - ckpt_local} artifact(s) + {ckpt_local}/{len(ckpt_all)} checkpoint(s) "
          f"({len(downloaded)} downloaded, {len(resumed)} already on disk)")
    print(f"provenance: {prov}" + (" (DEGRADED)" if prov_degraded else ""))
    print(f"collect_manifest.txt -> {receipt}")
    if ckpt_local != len(ckpt_all):
        print(f"WARN: local archive holds {ckpt_local}/{len(ckpt_all)} checkpoints "
              f"(COLLECT_CKPT={args.ckpt_mode}); it is a complete transfer but not a complete "
              f"weight archive - use COLLECT_CKPT=all for that", file=sys.stderr)
    return 0


def fail(out: Path, receipt: Path, failures: list[str], headline: str, code: int = 4,
         diag: "Diagnostics | None" = None,
         extra: "list[tuple[str, str]] | None" = None) -> int:
    """No success receipt may survive a failed collection; leave a diagnostic instead.

    Called from the happy-path failure branches *and* from the OSError handler, so it must
    never raise: if the receipt cannot be removed, invalidate its content, and if even that
    fails, say so loudly. A non-zero exit is guaranteed either way.
    """
    if receipt.exists():
        try:
            receipt.unlink()
        except OSError as e:
            print(f"ERROR: could not remove the stale receipt {receipt}: {e}", file=sys.stderr)
            try:
                receipt.write_text("verify=FAILED\n"
                                   f"error=could not revoke the previous receipt: {e}\n")
                print("ERROR: the receipt was overwritten with verify=FAILED instead",
                      file=sys.stderr)
            except OSError as e2:
                print(f"ERROR: could not invalidate the receipt either ({e2}); "
                      f"treat {receipt} as suspect and delete it by hand", file=sys.stderr)
    detail = ["verify=FAILED", f"collected_at={utcnow()}"]
    detail += [f"error={f}" for f in failures]
    try:
        write_atomic(out / "collect_manifest.txt.failed", "\n".join(detail) + "\n")
    except OSError as e:
        print(f"ERROR: could not write the failure diagnostics next to {out}: {e}", file=sys.stderr)
    if diag is not None:
        diag.flush("failed", list(extra or []) + [("error", f) for f in failures[:5]])
    for f in failures:
        print(f"ERROR: {f}", file=sys.stderr)
    print(f"collect FAILED: {headline}", file=sys.stderr)
    print("no collect_manifest.txt was written (the archive is NOT verified)", file=sys.stderr)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RemoteError as exc:
        print(f"collect FAILED: {exc}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        # SIGINT (or Ctrl-C). SIGTERM/SIGKILL never reach Python at all - which is exactly
        # why the receipt is revoked *before* the transfer instead of in a signal handler.
        # The archive carries no verdict at this point; .collect_partial.json records the
        # interrupted state and the next run re-measures and resumes.
        print("collect interrupted: no success receipt is left behind; the verified files on "
              "disk are kept and the next collect resumes from them", file=sys.stderr)
        sys.exit(130)
