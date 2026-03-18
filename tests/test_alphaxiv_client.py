from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_agent.alphaxiv_client import AlphaXivClient, AlphaXivClientError, FileTokenStorage
from mcp_agent.config import Settings


class _FakeModel:
    def __init__(self, payload: dict):
        self.payload = payload

    @classmethod
    def model_validate(cls, payload: dict) -> "_FakeModel":
        return cls(payload)

    def model_dump(self, **_: object) -> dict:
        return dict(self.payload)


def _fake_model_loader() -> tuple[type[_FakeModel], type[_FakeModel]]:
    return _FakeModel, _FakeModel


def test_token_cache_round_trip(tmp_path: Path) -> None:
    storage = FileTokenStorage(tmp_path / "tokens.json", model_loader=_fake_model_loader)

    asyncio.run(storage.set_tokens(_FakeModel({"access_token": "abc"})))
    asyncio.run(storage.set_client_info(_FakeModel({"client_id": "client-1"})))

    tokens = asyncio.run(storage.get_tokens())
    client_info = asyncio.run(storage.get_client_info())

    assert tokens is not None
    assert client_info is not None
    assert tokens.payload["access_token"] == "abc"
    assert client_info.payload["client_id"] == "client-1"


def test_invalid_mcp_url_reports_specific_error() -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="not-a-url",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-invalid-url.json",
        )
    )

    with pytest.raises(AlphaXivClientError) as exc_info:
        asyncio.run(client.ensure_authenticated())

    assert "valid http(s) URL" in str(exc_info.value)
    assert exc_info.value.metadata["auth_status"] == "invalid_mcp_url"


def test_oauth_bootstrap_required_path_reports_clear_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-oauth-required.json",
        )
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(AlphaXivClientError) as exc_info:
        asyncio.run(client._handle_oauth_callback())

    assert exc_info.value.metadata["auth_status"] == "interactive_bootstrap_required"


def test_nested_exception_group_unwraps_leaf_errors() -> None:
    client = AlphaXivClient(Settings(MCP_MODE="mock"))

    try:
        try:
            raise RuntimeError("leaf failure")
        except RuntimeError as inner:
            raise ExceptionGroup("outer", [inner, ValueError("bad redirect")])
    except Exception as exc:
        details = client._exception_details(exc)

    assert any("RuntimeError: leaf failure" in detail for detail in details)
    assert any("ValueError: bad redirect" in detail for detail in details)


def test_ping_events_are_treated_as_heartbeat() -> None:
    client = AlphaXivClient(Settings(MCP_MODE="mock"))

    assert client._is_heartbeat_event(event_name="ping", data="{}") is True
    assert client._is_heartbeat_event(event_name="heartbeat", data="") is True
    assert client._is_heartbeat_event(event_name="message", data="") is True
    assert client._is_heartbeat_event(event_name="message", data='{"jsonrpc":"2.0"}') is False


@dataclass
class _FakeTool:
    name: str


class _FakeListToolsResult:
    def __init__(self, names: list[str]) -> None:
        self.tools = [_FakeTool(name=name) for name in names]


class _FakeCallToolResult:
    def __init__(self, payload: dict) -> None:
        self.structuredContent = payload


class _FakeSession:
    def __init__(self, read_stream: object, write_stream: object) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(["embedding_similarity_search"])

    async def call_tool(self, name: str, arguments: dict[str, str]) -> _FakeCallToolResult:
        assert name == "embedding_similarity_search"
        assert "query" in arguments
        return _FakeCallToolResult({"papers": [{"title": "Paper One"}]})


class _FakeSSETransport:
    async def __aenter__(self) -> tuple[object, object]:
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _fake_sse_client(**_: object) -> _FakeSSETransport:
    return _FakeSSETransport()


def _patch_live_oauth(monkeypatch: pytest.MonkeyPatch, client: AlphaXivClient) -> None:
    async def fake_build_live_auth_context():
        return SimpleNamespace(
            auth=None,
            headers=None,
            auth_mode="oauth_discovery",
            auth_status="cached_token",
            token_cache_used=True,
            oauth_bootstrap_performed=False,
        )

    monkeypatch.setattr(client, "_build_live_auth_context", fake_build_live_auth_context)
    monkeypatch.setattr(client, "_load_mcp_client_deps", lambda: (_FakeSession, _fake_sse_client))


def test_initialize_and_tool_listing_with_mocked_mcp_session(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-tools.json",
        ),
        debug_mcp=True,
    )
    _patch_live_oauth(monkeypatch, client)

    metadata = asyncio.run(client.ensure_authenticated())

    assert metadata["auth_status"] == "sse_connected"
    assert metadata["tool_list"] == ["embedding_similarity_search"]


def test_auth_only_marks_oauth_callback_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-callback-complete.json",
        )
    )
    _patch_live_oauth(monkeypatch, client)
    client._oauth_bootstrap_performed = True
    client._oauth_callback_completed = True
    client._token_cache_used = False

    metadata = asyncio.run(client.ensure_authenticated())

    assert metadata["oauth_bootstrap_performed"] is True
    assert metadata["token_cache_used"] is False
    assert metadata["auth_status"] == "oauth_callback_completed"


def test_embedding_similarity_search_parses_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-search.json",
        )
    )
    _patch_live_oauth(monkeypatch, client)

    payload = asyncio.run(client.embedding_similarity_search("expanded query"))

    assert payload["papers"][0]["title"] == "Paper One"


def test_auth_failure_path_classifies_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-unauthorized.json",
        )
    )
    _patch_live_oauth(monkeypatch, client)

    class UnauthorizedSession(_FakeSession):
        async def initialize(self) -> None:
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(client, "_load_mcp_client_deps", lambda: (UnauthorizedSession, _fake_sse_client))

    with pytest.raises(AlphaXivClientError) as exc_info:
        asyncio.run(client.ensure_authenticated())

    assert exc_info.value.metadata["auth_status"] == "unauthorized"


def test_tool_call_timeout_reports_specific_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-timeout.json",
            MCP_REQUEST_TIMEOUT_SECONDS=0.01,
        )
    )
    _patch_live_oauth(monkeypatch, client)

    class SlowSession(_FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, str]) -> _FakeCallToolResult:
            await asyncio.sleep(0.1)
            return await super().call_tool(name, arguments)

    monkeypatch.setattr(client, "_load_mcp_client_deps", lambda: (SlowSession, _fake_sse_client))

    with pytest.raises(AlphaXivClientError) as exc_info:
        asyncio.run(client.embedding_similarity_search("expanded query"))

    assert exc_info.value.metadata["auth_status"] == "tool_call_timed_out"
    assert "embedding_similarity_search" in str(exc_info.value)


def test_sse_connect_uses_request_timeout_window(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AlphaXivClient(
        Settings(
            MCP_MODE="live",
            ALPHAXIV_MCP_URL="https://api.alphaxiv.org/mcp/v1",
            MCP_TOKEN_STORAGE_PATH="/tmp/alphaxiv-connect-timeout.json",
            MCP_AUTH_TIMEOUT_SECONDS=5,
            MCP_REQUEST_TIMEOUT_SECONDS=90,
        )
    )
    _patch_live_oauth(monkeypatch, client)

    observed: list[tuple[str, float]] = []
    original_await_stage = client._await_stage

    async def recording_await_stage(awaitable, *, stage: str, timeout_seconds: float, metadata: dict):
        observed.append((stage, timeout_seconds))
        return await original_await_stage(
            awaitable,
            stage=stage,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )

    monkeypatch.setattr(client, "_await_stage", recording_await_stage)

    asyncio.run(client.ensure_authenticated())

    assert observed[0] == ("sse_connect", 90)
