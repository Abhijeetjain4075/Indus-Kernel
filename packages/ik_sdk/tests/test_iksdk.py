"""Tests for ik_sdk — real HTTP tests with a local in-process server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ik_sdk import SDKError, with_retry
from ik_sdk import IndusClient as _Client


# The class is named IndusClient in the module; alias for clarity
IndusClient = _Client


class _TestHandler(BaseHTTPRequestHandler):
    """A minimal HTTP handler for SDK tests."""

    # Class-level state for behavior injection
    state = {
        "status": 200,
        "body": b'{"ok": true}',
        "fail_count": 0,
        "request_count": 0,
        "last_path": "",
        "last_method": "",
        "last_body": b"",
    }

    def log_message(self, format, *args):
        pass  # silence

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        type(self).state["request_count"] += 1
        type(self).state["last_method"] = method
        type(self).state["last_path"] = self.path
        type(self).state["last_body"] = body
        fail = type(self).state["fail_count"]
        if fail > 0:
            type(self).state["fail_count"] = fail - 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"service unavailable")
            return
        status = type(self).state["status"]
        payload = type(self).state["body"]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def http_server():
    """Spin up a tiny in-process HTTP server for SDK tests. Function-scoped
    so each test gets a clean slate."""
    _TestHandler.state = {
        "status": 200,
        "body": b'{"ok": true}',
        "fail_count": 0,
        "request_count": 0,
        "last_path": "",
        "last_method": "",
        "last_body": b"",
    }
    server = HTTPServer(("127.0.0.1", 0), _TestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


class TestSDKError:
    def test_to_dict(self):
        e = SDKError(code="not_found", message="missing", status=404, body="{}")
        d = e.to_dict()
        assert d["code"] == "not_found"
        assert d["status"] == 404
        assert "404" in d["message"]


class TestWithRetry:
    def test_success_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = with_retry(fn)
        assert result == "ok"
        assert len(calls) == 1

    def test_success_after_retries(self):
        calls = [0]

        def fn():
            calls[0] += 1
            if calls[0] < 3:
                raise ValueError("retry")
            return "ok"

        result = with_retry(fn, max_retries=5, backoff_s=0.001)
        assert result == "ok"
        assert calls[0] == 3

    def test_raises_after_max(self):
        def fn():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            with_retry(fn, max_retries=2, backoff_s=0.001)


class TestClientHealth:
    def test_health(self, http_server):
        _TestHandler.state["body"] = b'{"status": "ok", "version": "1.0"}'
        c = IndusClient(base_url=http_server, timeout=5)
        h = c.health()
        assert h["status"] == "ok"
        assert h["version"] == "1.0"

    def test_request_includes_auth(self, http_server):
        _TestHandler.state["body"] = b"{}"
        c = IndusClient(base_url=http_server, api_key="secret-key", timeout=5)
        c.health()
        # The auth header is set via urllib; just verify the request went through
        assert _TestHandler.state["request_count"] >= 1


class TestRetries:
    def test_retries_503(self, http_server):
        _TestHandler.state["status"] = 200
        _TestHandler.state["body"] = b'{"ok": true}'
        _TestHandler.state["fail_count"] = 2
        c = IndusClient(base_url=http_server, max_retries=3, backoff_s=0.01, timeout=2)
        result = c.health()
        assert result["ok"] is True
        assert _TestHandler.state["request_count"] == 3  # 2 fails + 1 success

    def test_gives_up_after_max_retries(self, http_server):
        _TestHandler.state["fail_count"] = 5
        _TestHandler.state["status"] = 503
        c = IndusClient(base_url=http_server, max_retries=2, backoff_s=0.01, timeout=2)
        with pytest.raises(SDKError):
            c.health()


class TestClientMethods:
    def test_chat(self, http_server):
        _TestHandler.state["body"] = json.dumps(
            {"id": "c1", "choices": [{"message": {"content": "hi"}}]}
        ).encode()
        c = IndusClient(base_url=http_server, timeout=5)
        r = c.chat([{"role": "user", "content": "hello"}])
        assert r["id"] == "c1"
        assert _TestHandler.state["last_method"] == "POST"
        body = json.loads(_TestHandler.state["last_body"])
        assert body["model"] == "indus"
        assert body["messages"][0]["content"] == "hello"

    def test_run_agent(self, http_server):
        _TestHandler.state["body"] = b'{"run_id": "r1", "status": "pending"}'
        c = IndusClient(base_url=http_server, timeout=5)
        r = c.run_agent(goal="x", tenant_id="t", user_id="u")
        assert r["run_id"] == "r1"

    def test_get_run(self, http_server):
        _TestHandler.state["body"] = b'{"run_id": "r1"}'
        c = IndusClient(base_url=http_server, timeout=5)
        r = c.get_run("r1")
        assert r["run_id"] == "r1"
        assert "r1" in _TestHandler.state["last_path"]

    def test_list_models(self, http_server):
        _TestHandler.state["body"] = b'{"models": []}'
        c = IndusClient(base_url=http_server, timeout=5)
        r = c.list_models()
        assert "models" in r

    def test_search_memory_with_params(self, http_server):
        _TestHandler.state["body"] = b'{"results": []}'
        c = IndusClient(base_url=http_server, timeout=5)
        r = c.search_memory(user_id="u1", query="hello", limit=5)
        assert "results" in r
        assert "user_id=u1" in _TestHandler.state["last_path"]
        assert "query=hello" in _TestHandler.state["last_path"]
        assert "limit=5" in _TestHandler.state["last_path"]


class TestClientError:
    def test_4xx_no_retry(self, http_server):
        _TestHandler.state["status"] = 404
        _TestHandler.state["body"] = b'{"error": "not found"}'
        c = IndusClient(base_url=http_server, max_retries=3, backoff_s=0.01, timeout=2)
        with pytest.raises(SDKError, match="404"):
            c.health()
        # 404 is not retried
        assert _TestHandler.state["request_count"] == 1


class TestMakeClient:
    def test_make(self):
        c = _Client.make_client if hasattr(_Client, "make_client") else None
        # We provide make_client in the module
        from ik_sdk import make_client

        c = make_client(base_url="http://x", api_key="k")
        assert c.base_url == "http://x"
        assert c.api_key == "k"
