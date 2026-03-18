from mcp_agent.formatter import format_response_text
from mcp_agent.schemas import MCPAgentResponse, PaperItem


def test_format_response_text_includes_ranked_papers() -> None:
    response = MCPAgentResponse(
        floor_id="bio_floor",
        floor_title="BioInformatics Floor",
        original_query="variant effect prediction",
        expanded_query="Expanded query text.",
        summary="Summary paragraph.",
        papers=[
            PaperItem(
                rank=1,
                title="Paper One",
                authors=["Jane Doe"],
                organizations=["Genome Lab"],
                publication_date="2024-01-01",
                arxiv_id="2401.12345",
                abstract_preview="A concise abstract preview.",
                visit_count=10,
                likes=2,
            )
        ],
        response_text="",
        metadata={"source": "alphaxiv_mcp"},
    )

    text = format_response_text(response)

    assert "I found 1 alphaXiv papers" in text
    assert "1. Paper One" in text
    assert "Authors: Jane Doe" in text
    assert "arXiv ID: 2401.12345" in text
    assert "Abstract preview: A concise abstract preview." in text
