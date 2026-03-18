from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from mcp_agent.alphaxiv_client import AlphaXivClientError
from mcp_agent.cli import _run
from mcp_agent.config import Settings


class _FakeAuthOnlyClient:
    def __init__(self, exc: AlphaXivClientError | None = None) -> None:
        self._exc = exc

    def clear_token_cache(self) -> bool:
        return False

    async def ensure_authenticated(self) -> dict:
        if self._exc is not None:
            raise self._exc
        return {"auth_status": "ok"}


def _auth_only_args() -> argparse.Namespace:
    return argparse.Namespace(
        floor_id=None,
        floor_title=None,
        floor_description=None,
        query=None,
        llm_model=None,
        json=False,
        auth_only=True,
        debug_mcp=False,
        clear_token_cache=False,
        use_legacy_bearer_token=False,
    )


def test_auth_only_failure_prints_structured_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exc = AlphaXivClientError(
        "alphaXiv rejected the OAuth token before opening the SSE MCP session.",
        metadata={
            "auth_status": "token_invalid",
            "auth_rejection_detail": 'Invalid JWT type "at+jwt". Expected "JWT".',
        },
    )
    fake_client = _FakeAuthOnlyClient(exc=exc)

    monkeypatch.setattr("mcp_agent.cli.get_settings", lambda: Settings(MCP_MODE="live"))
    monkeypatch.setattr("mcp_agent.cli.AlphaXivClient", lambda *args, **kwargs: fake_client)

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(_run(_auth_only_args()))

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == str(exc)
    assert payload["auth_status"] == "token_invalid"
