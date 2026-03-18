from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from mcp_agent.config import Settings


class AlphaXivClientError(RuntimeError):
    pass


class AlphaXivClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def embedding_similarity_search(self, expanded_query: str) -> dict[str, Any]:
        if self._settings.mcp_mode.lower() == "mock":
            return self._build_mock_response(expanded_query)
        return await self._call_live_mcp(expanded_query)

    async def _call_live_mcp(self, expanded_query: str) -> dict[str, Any]:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise AlphaXivClientError("The mcp package is required for live alphaXiv calls.") from exc

        if not self._settings.alphaxiv_oauth_access_token:
            raise AlphaXivClientError(
                "alphaXiv live mode requires OAuth 2.0 authentication. "
                "Set ALPHAXIV_OAUTH_ACCESS_TOKEN to a valid bearer token from your OAuth flow, "
                "or integrate the MCP OAuth flow in the calling runtime."
            )

        headers = {
            "Authorization": f"Bearer {self._settings.alphaxiv_oauth_access_token}",
        }

        try:
            async with asyncio.timeout(self._settings.request_timeout_seconds):
                async with sse_client(self._settings.alphaxiv_mcp_url, headers=headers) as streams:
                    read_stream, write_stream = streams
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "embedding_similarity_search",
                            arguments={"query": expanded_query},
                        )
        except TimeoutError as exc:
            raise AlphaXivClientError("Timed out while calling alphaXiv MCP.") from exc
        except Exception as exc:  # pragma: no cover - network boundary
            raise AlphaXivClientError(
                "alphaXiv MCP request failed while using the configured OAuth bearer token. "
                f"Check ALPHAXIV_MCP_URL, token validity, and MCP/SSE connectivity. Original error: {exc}"
            ) from exc

        return self._coerce_tool_result(result)

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

        return {"papers": [], "raw_text": "\n\n".join(text_blocks)}

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
