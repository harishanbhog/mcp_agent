import asyncio

from mcp_agent.llm_expander import StubQueryExpander
from mcp_agent.schemas import MCPAgentRequest


def test_stub_expander_uses_floor_context() -> None:
    request = MCPAgentRequest(
        floor_id="bio_floor",
        query="protein language models for variant effect prediction",
        llm_model="gpt-4.1-mini",
        floor_title="BioInformatics Floor",
        floor_description="Computational biology, genomics, and AI methods for protein modeling.",
    )

    expanded = asyncio.run(StubQueryExpander().expand_query(request))

    assert expanded
    assert "BioInformatics Floor" in expanded
    assert "protein language models for variant effect prediction" in expanded
    assert "Computational biology" in expanded
