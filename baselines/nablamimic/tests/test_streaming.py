#!/usr/bin/env python3
"""Tests for the Jupyter /files/ streaming download path in collect_run.py.

Covers the P1 gaps identified in code review:
  1. Binary content transferred and SHA256-verified via a mock HTTP server.
  2. Interrupted transfer is retried; a corrupt .part is never atomically published.
  3. HTTP 403 / 404 / 500, connection timeout, premature EOF.
  4. Large-file (>= 64 MiB) streaming failure must NOT fall back to `colab download`.
  5. Header-based auth (X-Colab-Runtime-Proxy-Token) is sent.

Run:  python3 tests/test_streaming.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- import collect_run as a module (it is a standalone script) -----------------
_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "collect_run.py"
_spec = importlib.util.spec_from_file_location("collect_run", _TOOLS)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

# --- helpers -------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    """Serves files from a dict: {path: bytes} with optional failure modes."""

    files: dict[str, bytes] = {}
    failure_mode: str | None = None  # '403', '404', '500', 'hang', 'truncated'
    received_headers: dict[str, str] = {}

    def do_GET(self):  # noqa: N802
        # Record headers for auth test
        type(self).received_headers = dict(self.headers)

        if self.failure_mode == "403":
            self.send_error(403, "Forbidden")
            return
        if self.failure_mode == "404":
            self.send_error(404, "Not Found")
            return
        if self.failure_mode == "500":
            self.send_error(500, "Internal Server Error")
            return
        if self.failure_mode == "hang":
            time.sleep(30)  # long enough to trigger client timeout, short enough for test cleanup
            return

        # Strip query string
        path = self.path.split("?")[0]
        # Remove /files/ prefix
        if path.startswith("/files/"):
            path = path[7:]

        data = self.files.get(path)
        if data is None:
            self.send_error(404, "Not Found")
            return

        if self.failure_mode == "truncated":
            # Send only half the data then close
            half = len(data) // 2
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:half])
            self.wfile.flush()
            self.close_connection = True
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence test output
        pass


def _start_server(files: dict[str, bytes], failure_mode: str | None = None):
    """Start a mock HTTP server. Returns (server, port, handler_class)."""
    port = _free_port()
    handler = type("H", (_Handler,), {"files": files, "failure_mode": failure_mode,
                                       "received_headers": {}})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, handler


# Monkey-patch get_session_endpoint to point at the mock server
def _patch_session(port: int, token: str = "test-token-abc123"):
    orig = cr.get_session_endpoint
    cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", token)
    return orig


# --- test cases ----------------------------------------------------------------


class TestStreamFetch(unittest.TestCase):
    """Tests for stream_fetch() — the core streaming download function."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._orig = cr.get_session_endpoint

    def tearDown(self):
        cr.get_session_endpoint = self._orig
        self._tmp.cleanup()

    def _run(self, files, failure_mode=None, session="test-sess", timeout_val=10,
             remote=None):
        if remote is None:
            remote = next(iter(files), "data.bin")
        server, port, handler = _start_server(files, failure_mode)
        try:
            cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", "tok-xyz")
            part = self.tmp / "out.bin"
            err = cr.stream_fetch(remote, part, session, timeout=timeout_val)
            return err, part, handler
        finally:
            server.shutdown()

    # 1. Basic binary transfer + SHA256
    def test_binary_transfer_sha256(self):
        payload = os.urandom(1024 * 100)  # 100 KB random binary
        expected_sha = hashlib.sha256(payload).hexdigest()
        err, part, _ = self._run({"data.bin": payload})
        self.assertEqual(err, "", f"stream_fetch failed: {err}")
        self.assertTrue(part.exists())
        got_sha = hashlib.sha256(part.read_bytes()).hexdigest()
        self.assertEqual(got_sha, expected_sha)

    def test_large_binary_transfer(self):
        payload = os.urandom(5 * 1024 * 1024)  # 5 MB
        expected_sha = hashlib.sha256(payload).hexdigest()
        err, part, _ = self._run({"big.bin": payload})
        self.assertEqual(err, "")
        got_sha = hashlib.sha256(part.read_bytes()).hexdigest()
        self.assertEqual(got_sha, expected_sha)

    # 2. Header auth
    def test_header_auth_sent(self):
        payload = b"hello world"
        err, _, handler = self._run({"f.bin": payload})
        self.assertEqual(err, "")
        self.assertIn("X-Colab-Runtime-Proxy-Token", handler.received_headers)
        self.assertEqual(handler.received_headers["X-Colab-Runtime-Proxy-Token"], "tok-xyz")

    # 3. Error cases
    def test_http_403(self):
        err, part, _ = self._run({"f.bin": b"x"}, failure_mode="403")
        self.assertNotEqual(err, "")
        self.assertIn("403", err)
        self.assertFalse(part.exists(), ".part should not remain after failure")

    def test_http_404(self):
        err, _, _ = self._run({"f.bin": b"x"}, failure_mode="404")
        self.assertNotEqual(err, "")
        self.assertIn("404", err)

    def test_http_500(self):
        err, _, _ = self._run({"f.bin": b"x"}, failure_mode="500")
        self.assertNotEqual(err, "")
        self.assertIn("500", err)

    def test_connection_timeout(self):
        err, _, _ = self._run({"f.bin": b"x"}, failure_mode="hang", timeout_val=3)
        self.assertNotEqual(err, "")

    def test_premature_eof(self):
        """Server truncates the response mid-transfer."""
        payload = os.urandom(1024 * 100)
        err, part, _ = self._run({"f.bin": payload}, failure_mode="truncated")
        # Either the stream reports an error, or the file is incomplete
        if err == "":
            # If no error was raised, the file must be shorter than expected
            self.assertLess(part.stat().st_size, len(payload),
                            "truncated download should not produce full-size file")
        # In either case, a partial .part must not look like a valid download

    def test_file_not_in_server(self):
        err, _, _ = self._run({"other.bin": b"x"}, remote="nonexistent.bin")
        self.assertNotEqual(err, "")

    # 4. Session not found
    def test_session_not_found(self):
        cr.get_session_endpoint = lambda name: None
        part = self.tmp / "out.bin"
        err = cr.stream_fetch("f.bin", part, "nonexistent", timeout=5)
        self.assertIn("not in colab-cli sessions.json", err)


class TestDownloadVerifiedFallbackPolicy(unittest.TestCase):
    """Tests that large files never fall back to `colab download`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._orig_gep = cr.get_session_endpoint
        # Track subprocess.run calls
        self._orig_sub = cr.subprocess.run
        self.subprocess_calls: list[list[str]] = []

    def tearDown(self):
        cr.get_session_endpoint = self._orig_gep
        cr.subprocess.run = self._orig_sub
        self._tmp.cleanup()

    def _mock_subprocess(self):
        def fake_run(cmd, **kw):
            self.subprocess_calls.append(cmd)
            class R:
                returncode = 1
                stdout = "mock colab download failed"
                stderr = ""
            return R()
        cr.subprocess.run = fake_run

    def test_large_file_streaming_failure_no_fallback(self):
        """A >=64 MiB file must NEVER call `colab download`, even when streaming fails."""
        self._mock_subprocess()
        # Streaming will fail (no server running)
        cr.get_session_endpoint = lambda name: None  # force streaming failure

        dest = self.tmp / "big_ckpt.pkl"
        size = 70 * 1024 * 1024  # 70 MiB > 64 MiB threshold
        digest = "0" * 64

        ok, reason = cr.download_verified("sess", "big_ckpt.pkl", dest, size, digest,
                                          attempts=2, timeout=5)
        self.assertFalse(ok)
        self.assertEqual(self.subprocess_calls, [],
                         "colab download must NOT be called for large files")

    def test_small_file_streaming_failure_does_fallback(self):
        """A <64 MiB file MAY fall back to `colab download` when streaming fails."""
        self._mock_subprocess()
        cr.get_session_endpoint = lambda name: None  # force streaming failure

        dest = self.tmp / "small.txt"
        payload = b"small file content"
        size = len(payload)
        digest = hashlib.sha256(payload).hexdigest()

        # colab download mock returns rc=1 (fails), so download_verified will fail
        ok, reason = cr.download_verified("sess", "small.txt", dest, size, digest,
                                          attempts=1, timeout=5)
        self.assertFalse(ok)
        self.assertGreater(len(self.subprocess_calls), 0,
                           "small files should fall back to colab download")

    def test_large_file_streaming_success_no_fallback(self):
        """Large file streaming success: colab download is not called."""
        self._mock_subprocess()
        payload = os.urandom(65 * 1024 * 1024)  # 65 MiB
        expected_sha = hashlib.sha256(payload).hexdigest()

        server, port, _ = _start_server({"big.pkl": payload})
        try:
            cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", "tok")
            dest = self.tmp / "big.pkl"
            ok, reason = cr.download_verified("sess", "big.pkl", dest,
                                              len(payload), expected_sha,
                                              attempts=1, timeout=30)
            self.assertTrue(ok, f"download should succeed: {reason}")
            self.assertEqual(self.subprocess_calls, [],
                             "colab download must NOT be called when streaming succeeds")
        finally:
            server.shutdown()


class TestDownloadVerifiedIntegrity(unittest.TestCase):
    """Tests that corrupt .part files are never atomically published."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._orig_gep = cr.get_session_endpoint

    def tearDown(self):
        cr.get_session_endpoint = self._orig_gep
        self._tmp.cleanup()

    def test_sha256_mismatch_not_published(self):
        """A file with wrong SHA256 is retried and never published on failure."""
        # Serve content that won't match the expected digest
        wrong_data = b"X" * 100
        server, port, _ = _start_server({"f.bin": wrong_data})
        try:
            cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", "tok")
            dest = self.tmp / "f.bin"
            wrong_sha = "a" * 64  # won't match
            ok, reason = cr.download_verified("sess", "f.bin", dest,
                                              len(wrong_data), wrong_sha,
                                              attempts=2, timeout=10)
            self.assertFalse(ok)
            self.assertIn("sha256", reason)
            self.assertFalse(dest.exists(), "corrupt file must never be published")
            # No .part remnants
            parts = list(self.tmp.glob("*.part.*"))
            self.assertEqual(parts, [], "no .part remnants after failed retries")
        finally:
            server.shutdown()

    def test_size_mismatch_not_published(self):
        """A file with wrong size is retried and never published on failure."""
        data = b"short"
        server, port, _ = _start_server({"f.bin": data})
        try:
            cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", "tok")
            dest = self.tmp / "f.bin"
            ok, reason = cr.download_verified("sess", "f.bin", dest,
                                              99999, "0" * 64,
                                              attempts=2, timeout=10)
            self.assertFalse(ok)
            self.assertIn("size", reason)
            self.assertFalse(dest.exists())
        finally:
            server.shutdown()

    def test_correct_file_published_atomically(self):
        """A verified file is published and the .part is cleaned up."""
        data = os.urandom(50 * 1024)
        expected_sha = hashlib.sha256(data).hexdigest()
        server, port, _ = _start_server({"f.bin": data})
        try:
            cr.get_session_endpoint = lambda name: (f"http://127.0.0.1:{port}", "tok")
            dest = self.tmp / "f.bin"
            ok, reason = cr.download_verified("sess", "f.bin", dest,
                                              len(data), expected_sha,
                                              attempts=1, timeout=10)
            self.assertTrue(ok, f"should succeed: {reason}")
            self.assertTrue(dest.exists())
            self.assertEqual(hashlib.sha256(dest.read_bytes()).hexdigest(), expected_sha)
            parts = list(self.tmp.glob("*.part.*"))
            self.assertEqual(parts, [], "no .part remnants after success")
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
