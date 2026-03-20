from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from mcp_agent.config import Settings

logger = logging.getLogger(__name__)


class AlphaXivClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        debug_details: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.metadata = metadata or {}
        self.debug_details = debug_details or []


@dataclass
class _LiveAuthContext:
    auth: Any | None
    headers: dict[str, str] | None
    auth_mode: str
    auth_status: str
    token_cache_used: bool
    oauth_bootstrap_performed: bool


@dataclass
class _LiveCallArtifacts:
    payload: dict[str, Any]
    metadata: dict[str, Any]


def _default_token_storage_path() -> str:
    return str(Path("~/.cache/mcp_agent/alphaxiv_oauth_tokens.json").expanduser())


class FileTokenStorage:
    """Persist MCP OAuth tokens and client registration data as JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_loader: Callable[[], tuple[type[Any], type[Any]]] | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self._model_loader = model_loader or self._load_models

    async def get_tokens(self) -> Any | None:
        raw = self._read()
        tokens = raw.get("tokens")
        if not tokens:
            return None
        token_model, _ = self._model_loader()
        return token_model.model_validate(tokens)

    async def set_tokens(self, tokens: Any) -> None:
        raw = self._read()
        raw["tokens"] = self._dump_model(tokens)
        self._write(raw)

    async def get_client_info(self) -> Any | None:
        raw = self._read()
        client_info = raw.get("client_info")
        if not client_info:
            return None
        _, client_info_model = self._model_loader()
        return client_info_model.model_validate(client_info)

    async def set_client_info(self, client_info: Any) -> None:
        raw = self._read()
        raw["client_info"] = self._dump_model(client_info)
        self._write(raw)

    def clear(self) -> bool:
        if not self.path.exists():
            return False
        self.path.unlink()
        return True

    def has_cache(self) -> bool:
        raw = self._read()
        return bool(raw.get("tokens"))

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid token cache file at %s", self.path)
            return {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _dump_model(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", by_alias=True)
        if isinstance(value, dict):
            return value
        raise TypeError(f"Cannot persist OAuth model of type {type(value)!r}")

    @staticmethod
    def _load_models() -> tuple[type[Any], type[Any]]:
        from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

        return OAuthToken, OAuthClientInformationFull


class AlphaXivClient:
    def __init__(
        self,
        settings: Settings,
        *,
        debug_mcp: bool = False,
        use_legacy_bearer_token: bool = False,
    ) -> None:
        self._settings = settings
        self._debug_mcp = debug_mcp
        self._use_legacy_bearer_token = use_legacy_bearer_token
        self._token_storage = FileTokenStorage(settings.mcp_token_storage_path)
        self._oauth_bootstrap_performed = False
        self._oauth_callback_completed = False
        self._token_cache_used = False
        self._last_metadata = self._base_metadata()

    @property
    def last_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def clear_token_cache(self) -> bool:
        return self._token_storage.clear()

    async def ensure_authenticated(self) -> dict[str, Any]:
        if self._settings.mcp_mode.lower() == "mock":
            self._last_metadata = {
                **self._base_metadata(),
                "auth_mode": "mock",
                "auth_status": "not_applicable",
            }
            return self.last_metadata

        artifacts = await self._execute_live_call(query=None, auth_only=True)
        return artifacts.metadata

    async def embedding_similarity_search(self, expanded_query: str) -> dict[str, Any]:
        if self._settings.mcp_mode.lower() == "mock":
            self._last_metadata = {
                **self._base_metadata(),
                "auth_mode": "mock",
                "auth_status": "not_applicable",
            }
            return self._build_mock_response(expanded_query)

        artifacts = await self._execute_live_call(query=expanded_query, auth_only=False)
        return artifacts.payload

    async def _execute_live_call(self, query: str | None, *, auth_only: bool) -> _LiveCallArtifacts:
        self._validate_mcp_url(self._settings.alphaxiv_mcp_url)
        result: Any = None
        stage = "auth_setup"
        metadata = self._base_metadata()
        transport_context: Any | None = None
        transport_opened = False

        try:
            client_session_cls = self._load_mcp_client_deps()
            auth_context = await self._build_live_auth_context()
            metadata.update(
                {
                    "auth_mode": auth_context.auth_mode,
                    "auth_status": auth_context.auth_status,
                    "token_cache_used": self._token_cache_used,
                    "oauth_bootstrap_performed": auth_context.oauth_bootstrap_performed,
                }
            )
            metadata["oauth_bootstrap_performed"] = self._oauth_bootstrap_performed

            stage = "sse_connect"
            transport_context = self._open_sse_transport(auth_context=auth_context)
            read_stream, write_stream = await self._await_stage(
                transport_context.__aenter__(),
                stage=stage,
                timeout_seconds=self._settings.mcp_request_timeout_seconds,
                metadata=metadata,
            )
            transport_opened = True
            metadata["token_cache_used"] = self._token_cache_used
            metadata["auth_status"] = "sse_connected"

            stage = "initialize"
            async with client_session_cls(read_stream, write_stream) as session:
                await self._await_stage(
                    session.initialize(),
                    stage=stage,
                    timeout_seconds=self._settings.mcp_request_timeout_seconds,
                    metadata=metadata,
                )

                stage = "tool_list"
                metadata["tool_list"] = await self._await_stage(
                    self._list_tool_names(session),
                    stage=stage,
                    timeout_seconds=self._settings.mcp_request_timeout_seconds,
                    metadata=metadata,
                )

                if auth_only:
                    metadata["oauth_bootstrap_performed"] = self._oauth_bootstrap_performed
                    metadata["token_cache_used"] = self._token_cache_used
                    metadata["auth_status"] = (
                        "oauth_callback_completed"
                        if self._oauth_callback_completed
                        else metadata["auth_status"]
                    )
                    self._last_metadata = metadata
                    return _LiveCallArtifacts(payload={"papers": []}, metadata=metadata)

                stage = "tool_call"
                result = await self._await_stage(
                    session.call_tool(
                        "embedding_similarity_search",
                        arguments={"query": query},
                    ),
                    stage=stage,
                    timeout_seconds=self._settings.mcp_request_timeout_seconds,
                    metadata=metadata,
                )

            stage = "response_parse"
            payload = self._coerce_tool_result(result)
            metadata["oauth_bootstrap_performed"] = self._oauth_bootstrap_performed
            metadata["token_cache_used"] = self._token_cache_used
            metadata["auth_status"] = self._resolve_auth_status(metadata["token_cache_used"])
            self._last_metadata = metadata
            return _LiveCallArtifacts(payload=payload, metadata=metadata)
        except AlphaXivClientError as exc:
            self._last_metadata = {**metadata, **exc.metadata}
            raise
        except Exception as exc:  # pragma: no cover - network boundary
            error = self._build_error(
                self._safe_error_message(stage=stage, exc=exc),
                stage=stage,
                exc=exc,
                metadata=metadata,
            )
            self._last_metadata = error.metadata
            raise error from exc
        finally:
            if transport_context is not None and transport_opened:
                await transport_context.__aexit__(None, None, None)

    async def _build_live_auth_context(self) -> _LiveAuthContext:
        clerk_session_token = self._settings.alphaxiv_clerk_session_token
        if clerk_session_token:
            self._oauth_bootstrap_performed = False
            self._oauth_callback_completed = False
            self._token_cache_used = False
            return _LiveAuthContext(
                auth=None,
                headers={"Authorization": f"Bearer {clerk_session_token}"},
                auth_mode="clerk_session_jwt",
                auth_status="clerk_session_token_supplied",
                token_cache_used=False,
                oauth_bootstrap_performed=False,
            )

        exchanged_auth_context = await self._build_token_exchange_auth_context()
        if exchanged_auth_context is not None:
            return exchanged_auth_context

        legacy_token = self._settings.get_legacy_bearer_token()
        if self._use_legacy_bearer_token:
            if not legacy_token:
                raise AlphaXivClientError(
                    "Legacy bearer-token mode was requested, but no legacy token is configured.",
                    metadata={
                        **self._base_metadata(),
                        "auth_mode": "legacy_bearer_token",
                        "auth_status": "missing_legacy_token",
                        "token_cache_used": False,
                        "oauth_bootstrap_performed": False,
                        "debug_hint": "Set ALPHAXIV_LEGACY_BEARER_TOKEN only for temporary debugging, or remove --use-legacy-bearer-token to use OAuth discovery.",
                    },
                )

            logger.warning("Using deprecated legacy bearer-token mode for alphaXiv MCP debugging.")
            return _LiveAuthContext(
                auth=None,
                headers={"Authorization": f"Bearer {legacy_token}"},
                auth_mode="legacy_bearer_token",
                auth_status="legacy_token_supplied",
                token_cache_used=False,
                oauth_bootstrap_performed=False,
            )

        self._oauth_bootstrap_performed = False
        self._oauth_callback_completed = False
        OAuthClientProvider, OAuthClientMetadata, AnyUrl = self._load_oauth_deps()
        try:
            cached_token = await self._token_storage.get_tokens()
        except ImportError:
            cached_token = None
        self._token_cache_used = bool(cached_token)

        oauth_auth = OAuthClientProvider(
            server_url=self._settings.alphaxiv_mcp_url,
            client_metadata=OAuthClientMetadata(
                client_name="mcp_agent alphaXiv client",
                redirect_uris=[AnyUrl("http://localhost:3000/callback")],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="user",
            ),
            storage=self._token_storage,
            redirect_handler=self._handle_oauth_redirect,
            callback_handler=self._handle_oauth_callback,
        )

        return _LiveAuthContext(
            auth=oauth_auth,
            headers=None,
            auth_mode="oauth_discovery",
            auth_status="cached_token" if cached_token else "oauth_discovery_pending",
            token_cache_used=bool(cached_token),
            oauth_bootstrap_performed=False,
        )

    async def _build_token_exchange_auth_context(self) -> _LiveAuthContext | None:
        try:
            cached_token = await self._token_storage.get_tokens()
        except ImportError:
            return None

        access_token = getattr(cached_token, "access_token", None)
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        exchanged_token = await self._exchange_oauth_access_token_for_clerk_jwt(access_token)
        self._oauth_bootstrap_performed = False
        self._oauth_callback_completed = False
        self._token_cache_used = True
        return _LiveAuthContext(
            auth=None,
            headers={"Authorization": f"Bearer {exchanged_token}"},
            auth_mode="oauth_token_exchange",
            auth_status="clerk_session_token_exchanged",
            token_cache_used=True,
            oauth_bootstrap_performed=False,
        )

    async def _exchange_oauth_access_token_for_clerk_jwt(self, access_token: str) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise AlphaXivClientError(
                "httpx is required to exchange the OAuth access token for an alphaXiv Clerk session JWT.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_token_exchange",
                    "auth_status": "missing_httpx",
                    "debug_hint": "Install project dependencies so the token-exchange request can run.",
                },
            ) from exc

        started_at = perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._settings.mcp_auth_timeout_seconds,
                    read=self._settings.mcp_auth_timeout_seconds,
                    write=self._settings.mcp_auth_timeout_seconds,
                    pool=self._settings.mcp_auth_timeout_seconds,
                )
            ) as client:
                response = await client.post(
                    self._settings.alphaxiv_auth_exchange_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                        "subject_token": access_token,
                        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                        "template": self._settings.alphaxiv_clerk_template,
                    },
                )
                response.raise_for_status()
        except Exception as exc:
            raise AlphaXivClientError(
                "Failed to exchange the cached OAuth access token for an alphaXiv Clerk session JWT.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_token_exchange",
                    "auth_status": "token_exchange_failed",
                    "token_cache_used": True,
                    "debug_hint": (
                        "The alphaXiv auth proxy token exchange failed. "
                        "Check ALPHAXIV_AUTH_EXCHANGE_URL, the cached OAuth token, and ALPHAXIV_CLERK_TEMPLATE."
                    ),
                },
            ) from exc

        payload = response.json()
        exchanged_token = payload.get("access_token")
        if not isinstance(exchanged_token, str) or not exchanged_token.strip():
            raise AlphaXivClientError(
                "alphaXiv auth proxy returned a token-exchange response without an access_token.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_token_exchange",
                    "auth_status": "token_exchange_failed",
                    "token_cache_used": True,
                    "debug_hint": "Inspect the alphaXiv auth proxy response body and ensure it returns access_token.",
                },
            )

        self._debug_log("token_exchange", started_at)
        return exchanged_token

    def _handle_oauth_redirect(self, auth_url: str) -> Awaitable[None]:
        async def _runner() -> None:
            started_at = perf_counter()
            self._oauth_bootstrap_performed = True
            self._token_cache_used = False
            self._debug_log("oauth_redirect_start", started_at, extra={"token_cache_used": self._token_storage.has_cache()})
            print("alphaXiv OAuth authorization required.")
            print(f"Open this URL in your browser and complete login:\n{auth_url}")
            print("After the browser redirects, paste the full callback URL here.")

        return _runner()

    def _handle_oauth_callback(self) -> Awaitable[tuple[str, str | None]]:
        async def _runner() -> tuple[str, str | None]:
            started_at = perf_counter()
            if not sys.stdin.isatty():
                raise AlphaXivClientError(
                    "Interactive OAuth bootstrap is required before live alphaXiv retrieval can continue.",
                    metadata={
                        **self._base_metadata(),
                        "auth_mode": "oauth_discovery",
                        "auth_status": "interactive_bootstrap_required",
                        "token_cache_used": self._token_storage.has_cache(),
                        "oauth_bootstrap_performed": self._oauth_bootstrap_performed,
                        "debug_hint": "Run the CLI with --auth-only to complete OAuth bootstrap and cache the token before running retrieval.",
                    },
                )

            callback_url = input("Paste callback URL: ").strip()
            if not callback_url or "code=" not in callback_url:
                raise AlphaXivClientError(
                    "The OAuth callback URL did not contain an authorization code.",
                    metadata={
                        **self._base_metadata(),
                        "auth_mode": "oauth_discovery",
                        "auth_status": "invalid_callback",
                        "token_cache_used": self._token_storage.has_cache(),
                        "oauth_bootstrap_performed": self._oauth_bootstrap_performed,
                        "debug_hint": "Retry --auth-only and paste the full callback URL returned by the browser.",
                    },
                )

            query = parse_qs(urlparse(callback_url).query)
            self._oauth_callback_completed = True
            self._debug_log("oauth_callback_completed", started_at)
            return query["code"][0], query.get("state", [None])[0]

        return _runner()

    @asynccontextmanager
    async def _open_sse_transport(self, *, auth_context: _LiveAuthContext) -> Any:
        started_at = perf_counter()
        if self._debug_mcp:
            logger.debug(
                "Connecting to alphaXiv MCP endpoint %s (token_cache_used=%s)",
                self._settings.alphaxiv_mcp_url,
                auth_context.token_cache_used,
            )

        async with self._alphaxiv_sse_client(
            url=self._settings.alphaxiv_mcp_url,
            headers=auth_context.headers,
            timeout=self._settings.mcp_auth_timeout_seconds,
            sse_read_timeout=self._settings.mcp_request_timeout_seconds,
            auth=auth_context.auth,
        ) as streams:
            self._debug_log("sse_connect", started_at)
            yield streams

    async def _list_tool_names(self, session: Any) -> list[str]:
        try:
            tools_result = await session.list_tools()
        except AttributeError:
            return []
        tools = getattr(tools_result, "tools", []) or []
        names = [getattr(tool, "name", "<unknown>") for tool in tools]
        if self._debug_mcp:
            logger.debug("alphaXiv MCP tools after initialize: %s", names)
        return names

    def _coerce_tool_result(self, result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured

        content = getattr(result, "content", None)
        text_blocks: list[str] = []
        if isinstance(content, Sequence):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text.strip())

        if text_blocks:
            joined = "\n\n".join(text_blocks)
            try:
                parsed = json.loads(joined)
            except json.JSONDecodeError:
                return {"papers": [], "raw_text": joined}
            if isinstance(parsed, dict):
                return parsed

        raise AlphaXivClientError(
            "alphaXiv MCP returned a malformed tool response.",
            metadata={
                **self._last_metadata,
                "auth_status": self._last_metadata.get("auth_status", "authorized"),
                "debug_hint": "Enable --debug-mcp to inspect the tool result shape returned by the server.",
            },
        )

    def _build_error(
        self,
        public_message: str,
        *,
        stage: str,
        exc: BaseException,
        metadata: dict[str, Any],
    ) -> AlphaXivClientError:
        details = self._exception_details(exc)
        classified_metadata = {
            **metadata,
            **self._classify_exception(stage=stage, exc=exc, details=details),
        }
        if self._debug_mcp and details:
            logger.debug("alphaXiv MCP failure at stage=%s", stage)
            for detail in details:
                logger.debug("  %s", detail)
        return AlphaXivClientError(public_message, metadata=classified_metadata, debug_details=details)

    def _safe_error_message(self, *, stage: str, exc: BaseException) -> str:
        classification = self._classify_exception(
            stage=stage,
            exc=exc,
            details=self._exception_details(exc),
        )
        auth_status = classification.get("auth_status")
        if auth_status == "interactive_bootstrap_required":
            return "Interactive OAuth bootstrap is required before live alphaXiv retrieval can continue."
        if auth_status == "auth_discovery_failed":
            return "alphaXiv OAuth discovery failed before the MCP session could be established."
        if auth_status == "unauthorized":
            return "alphaXiv rejected the authenticated MCP request with 401 Unauthorized."
        if auth_status == "token_invalid":
            return "alphaXiv rejected the supplied token before opening the SSE MCP session. Use a Clerk session JWT or the alphaXiv auth-proxy token exchange."
        if auth_status == "token_exchange_failed":
            return "alphaXiv OAuth token exchange failed before the SSE MCP session could start."
        if auth_status == "token_refresh_failed":
            return "alphaXiv OAuth token refresh failed during MCP authentication."
        if auth_status == "invalid_callback":
            return "The OAuth browser callback could not be processed."
        if auth_status == "sse_connect_failed":
            return "The alphaXiv SSE MCP transport could not be established."
        if auth_status == "initialize_failed":
            return "The MCP session connected but failed during initialization."
        if auth_status == "tool_call_timed_out":
            return "The embedding_similarity_search MCP tool call timed out after the session connected."
        if auth_status == "tool_call_failed":
            return "The embedding_similarity_search MCP tool call failed after session initialization."
        return "alphaXiv MCP retrieval failed."

    def _classify_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        details: list[str],
    ) -> dict[str, Any]:
        combined = " | ".join(details + [type(exc).__name__, str(exc)]).lower()
        metadata: dict[str, Any] = {
            "debug_hint": "Re-run with --debug-mcp for fully unwrapped transport and auth diagnostics.",
        }

        if isinstance(exc, AlphaXivClientError):
            return {**metadata, **exc.metadata}

        if "401" in combined or "unauthorized" in combined:
            return {**metadata, "auth_status": "unauthorized"}
        if "token-invalid" in combined or "invalid jwt type" in combined:
            return {**metadata, "auth_status": "token_invalid"}
        if "token exchange" in combined:
            return {**metadata, "auth_status": "token_exchange_failed"}
        if "refresh" in combined and "token" in combined:
            return {**metadata, "auth_status": "token_refresh_failed"}
        if "redirect" in combined or "callback" in combined or "state" in combined:
            return {**metadata, "auth_status": "invalid_callback"}
        if "protected resource" in combined or "authorization server" in combined or "metadata" in combined:
            return {**metadata, "auth_status": "auth_discovery_failed"}
        if stage == "sse_connect":
            return {**metadata, "auth_status": "sse_connect_failed"}
        if stage == "initialize":
            return {**metadata, "auth_status": "initialize_failed"}
        if stage == "tool_call":
            return {**metadata, "auth_status": "tool_call_failed"}
        if stage == "response_parse":
            return {**metadata, "auth_status": "malformed_tool_response"}
        return {**metadata, "auth_status": "retrieval_failed"}

    def _exception_details(self, exc: BaseException) -> list[str]:
        details: list[str] = []

        def visit(current: BaseException) -> None:
            label = f"{type(current).__name__}: {current}"
            if label not in details:
                details.append(label)

            nested = getattr(current, "exceptions", None)
            if nested:
                for item in nested:
                    if isinstance(item, BaseException):
                        visit(item)

            cause = getattr(current, "__cause__", None)
            if isinstance(cause, BaseException):
                visit(cause)

            context = getattr(current, "__context__", None)
            if isinstance(context, BaseException):
                visit(context)

        visit(exc)
        return details

    def _resolve_auth_status(self, token_cache_used: bool) -> str:
        if self._oauth_callback_completed:
            return "oauth_callback_completed"
        if token_cache_used:
            return "sse_connected"
        return "authorized"

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "source": "alphaxiv_mcp",
            "tool": "embedding_similarity_search",
            "mcp_mode": self._settings.mcp_mode,
            "mcp_endpoint": self._settings.alphaxiv_mcp_url,
            "token_cache_used": False,
            "oauth_bootstrap_performed": False,
            "auth_mode": "uninitialized",
            "auth_status": "not_started",
            "debug_hint": "Use --auth-only to bootstrap OAuth tokens and --debug-mcp for detailed transport logs.",
        }

    def _validate_mcp_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AlphaXivClientError(
                "ALPHAXIV_MCP_URL must be a valid http(s) URL.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_discovery",
                    "auth_status": "invalid_mcp_url",
                    "debug_hint": "Set ALPHAXIV_MCP_URL to the documented SSE endpoint, e.g. https://api.alphaxiv.org/mcp/v1.",
                },
            )

    def _load_mcp_client_deps(self) -> type[Any]:
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise AlphaXivClientError(
                "The mcp package is required for live alphaXiv calls.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_discovery",
                    "auth_status": "missing_mcp_sdk",
                    "debug_hint": "Run poetry install so the mcp SDK is available in the active environment.",
                },
            ) from exc

        return ClientSession

    def _load_oauth_deps(self) -> tuple[type[Any], type[Any], type[Any]]:
        try:
            from pydantic import AnyUrl
            from mcp.client.auth import OAuthClientProvider, TokenStorage
            from mcp.shared.auth import OAuthClientMetadata
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise AlphaXivClientError(
                "The installed mcp SDK does not expose the required OAuth client helpers.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_discovery",
                    "auth_status": "missing_oauth_helpers",
                    "debug_hint": "Upgrade the mcp SDK to a version that includes OAuthClientProvider and TokenStorage support.",
                },
            ) from exc

        _ = TokenStorage

        return OAuthClientProvider, OAuthClientMetadata, AnyUrl

    async def _await_stage(
        self,
        awaitable: Awaitable[Any],
        *,
        stage: str,
        timeout_seconds: float,
        metadata: dict[str, Any],
    ) -> Any:
        started_at = perf_counter()
        try:
            if self._debug_mcp:
                logger.debug("alphaXiv %s starting (timeout=%.1fs)", stage, timeout_seconds)
            result = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except TimeoutError as exc:
            status = "tool_call_timed_out" if stage == "tool_call" else self._classify_exception(
                stage=stage,
                exc=exc,
                details=self._exception_details(exc),
            ).get("auth_status", "retrieval_failed")
            error = self._build_error(
                self._timeout_message_for_stage(stage),
                stage=stage,
                exc=exc,
                metadata={
                    **metadata,
                    "auth_status": status,
                    "oauth_bootstrap_performed": self._oauth_bootstrap_performed,
                    "token_cache_used": self._token_cache_used,
                },
            )
            self._last_metadata = error.metadata
            raise error from exc

        self._debug_log(stage, started_at)
        return result

    def _timeout_message_for_stage(self, stage: str) -> str:
        if stage == "sse_connect":
            return "Timed out while establishing the alphaXiv SSE transport."
        if stage == "initialize":
            return "Timed out while initializing the alphaXiv MCP session."
        if stage == "tool_list":
            return "Timed out while listing alphaXiv MCP tools after initialization."
        if stage == "tool_call":
            return "Timed out while waiting for embedding_similarity_search to return."
        return "Timed out while communicating with alphaXiv MCP."

    def _debug_log(self, stage: str, started_at: float, *, extra: dict[str, Any] | None = None) -> None:
        if not self._debug_mcp:
            return

        elapsed = perf_counter() - started_at
        if extra:
            logger.debug("alphaXiv %s completed in %.3fs (%s)", stage, elapsed, extra)
            return
        logger.debug("alphaXiv %s completed in %.3fs", stage, elapsed)

    @asynccontextmanager
    async def _alphaxiv_sse_client(
        self,
        *,
        url: str,
        headers: dict[str, Any] | None,
        timeout: float,
        sse_read_timeout: float,
        auth: Any,
    ) -> Any:
        try:
            import anyio
            import httpx
            from anyio.abc import TaskStatus
            from httpx_sse import SSEError, aconnect_sse
            from mcp import types
            from mcp.shared._httpx_utils import create_mcp_http_client
            from mcp.shared.message import SessionMessage
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise AlphaXivClientError(
                "The installed MCP stack is missing SSE transport dependencies.",
                metadata={
                    **self._base_metadata(),
                    "auth_mode": "oauth_discovery",
                    "auth_status": "missing_mcp_sdk",
                    "debug_hint": "Ensure the mcp SDK and httpx-sse dependencies are installed in the active Poetry environment.",
                },
            ) from exc

        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        async with create_mcp_http_client(
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(
                connect=timeout,
                read=sse_read_timeout,
                write=timeout,
                pool=timeout,
            ),
        ) as client:
            async with aconnect_sse(client, "GET", url) as event_source:
                event_source.response.raise_for_status()
                self._raise_for_rejected_sse_response(event_source.response.headers)

                async def sse_reader(task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED) -> None:
                    try:
                        async for sse in event_source.aiter_sse():
                            event_name = sse.event or "message"
                            if self._is_heartbeat_event(event_name=event_name, data=sse.data):
                                if self._debug_mcp:
                                    logger.debug("Ignoring alphaXiv SSE heartbeat event: %s", event_name)
                                continue

                            match event_name:
                                case "endpoint":
                                    endpoint_url = urljoin(url, sse.data)
                                    task_status.started(endpoint_url)
                                case "message":
                                    if not sse.data:
                                        continue
                                    try:
                                        message = types.jsonrpc_message_adapter.validate_json(sse.data, by_name=False)
                                    except Exception as exc:
                                        logger.exception("Error parsing alphaXiv SSE message")
                                        await read_stream_writer.send(exc)
                                        continue
                                    await read_stream_writer.send(SessionMessage(message))
                                case _:
                                    logger.warning("Ignoring unsupported alphaXiv SSE event: %s", event_name)
                    except SSEError as sse_exc:
                        raise sse_exc
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                    finally:
                        await read_stream_writer.aclose()

                async def post_writer(endpoint_url: str) -> None:
                    try:
                        async with write_stream_reader, write_stream:
                            async for session_message in write_stream_reader:
                                response = await client.post(
                                    endpoint_url,
                                    json=session_message.message.model_dump(
                                        by_alias=True,
                                        mode="json",
                                        exclude_unset=True,
                                    ),
                                )
                                response.raise_for_status()
                    except Exception as exc:
                        await read_stream_writer.send(exc)

                async with read_stream_writer, read_stream, write_stream, write_stream_reader, anyio.create_task_group() as tg:
                    endpoint_url = await tg.start(sse_reader)
                    tg.start_soon(post_writer, endpoint_url)
                    yield read_stream, write_stream
                    tg.cancel_scope.cancel()

    def _is_heartbeat_event(self, *, event_name: str, data: str | None) -> bool:
        normalized = event_name.strip().lower()
        if normalized in {"ping", "keepalive", "heartbeat"}:
            return True
        return normalized == "message" and not (data or "").strip()

    def _raise_for_rejected_sse_response(self, headers: Any) -> None:
        clerk_status = headers.get("x-clerk-auth-status")
        clerk_reason = headers.get("x-clerk-auth-reason")
        clerk_message = headers.get("x-clerk-auth-message")
        if clerk_status != "signed-out" and clerk_reason != "token-invalid":
            return

        detail = clerk_message or "alphaXiv returned signed-out/token-invalid headers during SSE setup."
        raise AlphaXivClientError(
            "alphaXiv rejected the OAuth token before opening the SSE MCP session.",
            metadata={
                **self._base_metadata(),
                "auth_mode": "oauth_discovery",
                "auth_status": "token_invalid",
                "token_cache_used": self._token_cache_used,
                "oauth_bootstrap_performed": self._oauth_bootstrap_performed,
                "debug_hint": (
                    "alphaXiv returned x-clerk-auth-status/x-clerk-auth-reason headers. "
                    "Use ALPHAXIV_CLERK_SESSION_TOKEN or exchange the OAuth access token through ALPHAXIV_AUTH_EXCHANGE_URL before opening SSE."
                ),
                "auth_rejection_detail": detail,
            },
            debug_details=[detail],
        )

    def _build_mock_response(self, expanded_query: str) -> dict[str, Any]:
        return {
            "query": expanded_query,
            "papers": [
                {
                    "title": "Protein Language Models Improve Variant Effect Prediction",
                    "authors": ["Jane Doe", "Alex Kim"],
                    "organizations": ["Genome Lab", "AI Biology Institute"],
                    "publication_date": "2024-05-10",
                    "arxiv_id": "2405.12345",
                    "abstract_preview": "Protein language model embeddings are used to predict functional effects of missense variants across diverse benchmarks.",
                    "visit_count": 421,
                    "likes": 33,
                },
                {
                    "title": "Foundation Models for Functional Genomics and Protein Engineering",
                    "authors": ["Priya Shah", "M. Chen"],
                    "organizations": ["BioCompute Center"],
                    "publication_date": "2023-11-02",
                    "arxiv_id": "2311.98765",
                    "abstract_preview": "This work surveys and benchmarks large-scale sequence models for genomics, protein design, and biological property prediction.",
                    "visit_count": 318,
                    "likes": 21,
                },
                {
                    "title": "Zero-Shot Fitness Estimation with Evolutionary and Language Priors",
                    "authors": ["Luis Ortega"],
                    "organizations": ["Protein Systems Group"],
                    "publication_date": "2022-08-18",
                    "arxiv_id": "2208.45678",
                    "abstract_preview": "Combined evolutionary signals and protein language representations support zero-shot variant scoring and fitness ranking.",
                    "visit_count": 197,
                    "likes": 14,
                },
            ],
        }
