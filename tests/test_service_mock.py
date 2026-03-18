import asyncio

from mcp_agent.config import Settings
from mcp_agent.schemas import MCPAgentRequest
from mcp_agent.service import run_mcp_agent


def test_run_mcp_agent_in_mock_mode_returns_contract() -> None:
    settings = Settings(MCP_MODE="mock", REQUEST_TIMEOUT_SECONDS=5)
    request = MCPAgentRequest(
        floor_id="bio_floor",
        query="protein language models for variant effect prediction",
        llm_model="gpt-4.1-mini",
        floor_title="BioInformatics Floor",
        floor_description="A research floor for bioinformatics, computational biology, genomics, protein modeling, drug discovery, and related AI/ML research.",
    )

    response = asyncio.run(run_mcp_agent(request, settings=settings))

    assert response.floor_id == "bio_floor"
    assert response.floor_title == "BioInformatics Floor"
    assert response.original_query == request.query
    assert response.expanded_query
    assert response.summary
    assert response.response_text
    assert response.metadata["source"] == "alphaxiv_mcp"
    assert response.metadata["tool"] == "embedding_similarity_search"
    assert response.metadata["retrieval_count"] == len(response.papers)
    assert len(response.papers) >= 1
