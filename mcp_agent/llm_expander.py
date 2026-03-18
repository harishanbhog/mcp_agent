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
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise QueryExpansionError("The openai package is required for live query expansion.") from exc

        client = AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url or None,
            timeout=self._settings.request_timeout_seconds,
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
            raise QueryExpansionError("Timed out while expanding the query with the LLM.") from exc
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
