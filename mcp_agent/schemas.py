from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPAgentRequest(BaseModel):
    floor_id: str
    query: str
    llm_model: str
    floor_title: str
    floor_description: str


class PaperItem(BaseModel):
    rank: int
    title: str
    authors: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    arxiv_id: str | None = None
    abstract_preview: str | None = None
    visit_count: int | None = None
    likes: int | None = None


class MCPAgentResponse(BaseModel):
    floor_id: str
    floor_title: str
    original_query: str
    expanded_query: str
    summary: str
    papers: list[PaperItem] = Field(default_factory=list)
    response_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
