from __future__ import annotations

from mcp_agent.alphaxiv_client import AlphaXivClient, AlphaXivClientError
from mcp_agent.config import Settings, get_settings
from mcp_agent.formatter import format_response_text, normalize_papers, synthesize_summary
from mcp_agent.llm_expander import LLMQueryExpander, QueryExpansionError, StubQueryExpander
from mcp_agent.schemas import MCPAgentRequest, MCPAgentResponse


async def run_mcp_agent(
    request: MCPAgentRequest,
    *,
    settings: Settings | None = None,
    expander: LLMQueryExpander | StubQueryExpander | None = None,
    client: AlphaXivClient | None = None,
) -> MCPAgentResponse:
    runtime_settings = settings or get_settings()
    validated_request = MCPAgentRequest.model_validate(request)
    query_expander = expander or _build_expander(runtime_settings)
    alphaxiv_client = client or AlphaXivClient(runtime_settings)

    expansion_error: str | None = None
    retrieval_error: str | None = None

    try:
        expanded_query = await query_expander.expand_query(validated_request)
        expansion_mode = "llm"
    except QueryExpansionError as exc:
        expansion_error = str(exc)
        expanded_query = await StubQueryExpander().expand_query(validated_request)
        expansion_mode = "stub_fallback"

    try:
        raw_results = await alphaxiv_client.embedding_similarity_search(expanded_query)
        client_metadata = alphaxiv_client.last_metadata
    except AlphaXivClientError as exc:
        retrieval_error = str(exc)
        raw_results = {"papers": []}
        client_metadata = {**alphaxiv_client.last_metadata, **exc.metadata}

    papers = normalize_papers(raw_results)
    summary = synthesize_summary(papers)
    metadata = {
        "source": "alphaxiv_mcp",
        "tool": "embedding_similarity_search",
        "default_block": "Research Areas",
        "floor_type": "BioInformatics",
        "llm_model_used": validated_request.llm_model,
        "retrieval_count": len(papers),
        "mcp_mode": runtime_settings.mcp_mode,
        "query_expansion_mode": expansion_mode,
    }
    metadata.update(client_metadata)
    if expansion_error:
        metadata["query_expansion_error"] = expansion_error
    if retrieval_error:
        metadata["retrieval_error"] = retrieval_error

    response = MCPAgentResponse(
        floor_id=validated_request.floor_id,
        floor_title=validated_request.floor_title,
        original_query=validated_request.query,
        expanded_query=expanded_query,
        summary=summary,
        papers=papers,
        response_text="",
        metadata=metadata,
    )
    response.response_text = format_response_text(response)
    return response


def _build_expander(settings: Settings) -> LLMQueryExpander | StubQueryExpander:
    if settings.mcp_mode.lower() == "mock" and not settings.openai_api_key:
        return StubQueryExpander()
    return LLMQueryExpander(settings)
