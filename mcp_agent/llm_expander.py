from __future__ import annotations

import asyncio

from mcp_agent.config import Settings
from mcp_agent.schemas import MCPAgentRequest

_EXPANSION_PROMPT = """You rewrite short retrieval queries into a stronger semantic search description for alphaXiv paper search.
Produce exactly 2 to 3 sentences.
Cover the core research area, important methods or techniques, practical applications or use cases, and adjacent concepts or related terms.
Be concise, specific, and optimized for semantic paper retrieval.
Do not answer the user's question.
Do not use bullet points.
Return only the rewritten query text."""


class QueryExpansionError(RuntimeError):
    pass


class LLMQueryExpander:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def expand_query(self, request: MCPAgentRequest) -> str:
        if not self._settings.openai_api_key:
            raise QueryExpansionError("OPENAI_API_KEY is required for live query expansion.")

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AsyncOpenAI,
                AuthenticationError,
                BadRequestError,
                NotFoundError,
            )
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise QueryExpansionError("The openai package is required for live query expansion.") from exc

        client = AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url or None,
            timeout=self._settings.request_timeout_seconds,
            max_retries=1,
        )

        user_prompt = (
            f"Floor title: {request.floor_title}\n"
            f"Floor description: {request.floor_description}\n"
            f"Original query: {request.query}\n"
            "Rewrite this into a semantic research retrieval description."
        )

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=request.llm_model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": _EXPANSION_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                ),
                timeout=self._settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise QueryExpansionError(
                f"Timed out while expanding the query with the LLM after {self._settings.request_timeout_seconds} seconds."
            ) from exc
        except APITimeoutError as exc:
            raise QueryExpansionError(
                f"The LLM provider timed out while expanding the query after {self._settings.request_timeout_seconds} seconds."
            ) from exc
        except APIConnectionError as exc:
            raise QueryExpansionError(
                "Could not reach the LLM provider while expanding the query. "
                f"Check network access, OPENAI_API_KEY, and OPENAI_BASE_URL (current value: {self._settings.openai_base_url or 'default OpenAI endpoint'})."
            ) from exc
        except AuthenticationError as exc:
            raise QueryExpansionError("LLM authentication failed. Check OPENAI_API_KEY.") from exc
        except NotFoundError as exc:
            raise QueryExpansionError(
                f"The configured LLM model '{request.llm_model}' was not found by the provider."
            ) from exc
        except BadRequestError as exc:
            raise QueryExpansionError(f"The LLM provider rejected the expansion request: {exc}") from exc
        except Exception as exc:  # pragma: no cover - exercised via integration boundaries
            raise QueryExpansionError(f"Failed to expand query with LLM: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        expanded = (content or "").strip()
        if not expanded:
            raise QueryExpansionError("LLM returned an empty expanded query.")
        return expanded


class StubQueryExpander:
    """Deterministic expander for tests and local offline runs."""

    async def expand_query(self, request: MCPAgentRequest) -> str:
        query = request.query.strip().rstrip(".")
        title = request.floor_title.strip()
        description = request.floor_description.strip()
        return (
            f"Research in {title} focused on {query}, grounded in the floor context: {description}. "
            f"Include related methods, evaluation settings, downstream applications, and adjacent terminology relevant to {query}."
        )
