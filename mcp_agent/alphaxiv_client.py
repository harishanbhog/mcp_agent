from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

        try:
            client_session_cls, sse_client = self._load_mcp_client_deps()
            auth_context = await self._build_live_auth_context()
            metadata.update(
                {
                    "auth_mode": auth_context.auth_mode,
                    "auth_status": auth_context.auth_status,
                    "token_cache_used": auth_context.token_cache_used,
                    "oauth_bootstrap_performed": auth_context.oauth_bootstrap_performed,
                }
            )

            async with asyncio.timeout(self._settings.mcp_request_timeout_seconds):
                stage = "sse_connect"
                async with self._open_sse_transport(
                    sse_client=sse_client,
                    auth_context=auth_context,
                ) as (read_stream, write_stream):
                    stage = "session_initialize"
                    async with client_session_cls(read_stream, write_stream) as session:
                        await session.initialize()

                        stage = "tool_list"
                        metadata["tool_list"] = await self._list_tool_names(session)

                        if auth_only:
                            metadata["oauth_bootstrap_performed"] = self._oauth_bootstrap_performed
                            metadata["auth_status"] = (
                                "interactive_bootstrap_completed"
                                if self._oauth_bootstrap_performed
                                else metadata["auth_status"]
                            )
                            self._last_metadata = metadata
                            return _LiveCallArtifacts(payload={"papers": []}, metadata=metadata)

                        stage = "tool_call"
                        result = await session.call_tool(
                            "embedding_similarity_search",
                            arguments={"query": query},
                        )

            stage = "response_parse"
            payload = self._coerce_tool_result(result)
            metadata["oauth_bootstrap_performed"] = self._oauth_bootstrap_performed
            metadata["auth_status"] = self._resolve_auth_status(metadata["token_cache_used"])
            self._last_metadata = metadata
            return _LiveCallArtifacts(payload=payload, metadata=metadata)
        except TimeoutError as exc:
            error = self._build_error(
                "Timed out while communicating with alphaXiv MCP.",
                stage=stage,
                exc=exc,
                metadata=metadata,
            )
            self._last_metadata = error.metadata
            raise error from exc
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

    async def _build_live_auth_context(self) -> _LiveAuthContext:
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
        OAuthClientProvider, OAuthClientMetadata, AnyUrl = self._load_oauth_deps()
        try:
            cached_token = await self._token_storage.get_tokens()
        except ImportError:
            cached_token = None

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

    def _handle_oauth_redirect(self, auth_url: str) -> Awaitable[None]:
        async def _runner() -> None:
            self._oauth_bootstrap_performed = True
            print("alphaXiv OAuth authorization required.")
            print(f"Open this URL in your browser and complete login:\n{auth_url}")
            print("After the browser redirects, paste the full callback URL here.")

        return _runner()

    def _handle_oauth_callback(self) -> Awaitable[tuple[str, str | None]]:
        async def _runner() -> tuple[str, str | None]:
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

            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(callback_url).query)
            return query["code"][0], query.get("state", [None])[0]

        return _runner()

    @asynccontextmanager
    async def _open_sse_transport(self, *, sse_client: Callable[..., Any], auth_context: _LiveAuthContext) -> Any:
        kwargs: dict[str, Any] = {
            "url": self._settings.alphaxiv_mcp_url,
            "headers": auth_context.headers,
            "timeout": self._settings.mcp_auth_timeout_seconds,
            "sse_read_timeout": self._settings.mcp_request_timeout_seconds,
            "auth": auth_context.auth,
        }
        if self._debug_mcp:
            logger.debug("Connecting to alphaXiv MCP endpoint %s", self._settings.alphaxiv_mcp_url)

        async with sse_client(**kwargs) as streams:
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
        if auth_status == "token_refresh_failed":
            return "alphaXiv OAuth token refresh failed during MCP authentication."
        if auth_status == "invalid_callback":
            return "The OAuth browser callback could not be processed."
        if auth_status == "sse_connect_failed":
            return "The alphaXiv SSE MCP transport could not be established."
        if auth_status == "mcp_session_initialize_failed":
            return "The MCP session connected but failed during initialization."
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
        if "refresh" in combined and "token" in combined:
            return {**metadata, "auth_status": "token_refresh_failed"}
        if "redirect" in combined or "callback" in combined or "state" in combined:
            return {**metadata, "auth_status": "invalid_callback"}
        if "protected resource" in combined or "authorization server" in combined or "metadata" in combined:
            return {**metadata, "auth_status": "auth_discovery_failed"}
        if stage == "sse_connect":
            return {**metadata, "auth_status": "sse_connect_failed"}
        if stage == "session_initialize":
            return {**metadata, "auth_status": "mcp_session_initialize_failed"}
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
        if self._oauth_bootstrap_performed:
            return "interactive_bootstrap_completed"
        if token_cache_used:
            return "cached_token"
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

    def _load_mcp_client_deps(self) -> tuple[type[Any], Callable[..., Any]]:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
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

        return ClientSession, sse_client

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
