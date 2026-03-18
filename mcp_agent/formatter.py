from __future__ import annotations

from collections import Counter
from typing import Iterable

from mcp_agent.schemas import MCPAgentResponse, PaperItem


def normalize_papers(payload: dict) -> list[PaperItem]:
    raw_papers = payload.get("papers") or []
    papers: list[PaperItem] = []
    for index, item in enumerate(raw_papers, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Untitled paper").strip()
        authors = _ensure_text_list(item.get("authors"))
        organizations = _ensure_text_list(item.get("organizations"))
        papers.append(
            PaperItem(
                rank=int(item.get("rank") or index),
                title=title,
                authors=authors,
                organizations=organizations,
                publication_date=_clean_optional(item.get("publication_date")),
                arxiv_id=_clean_optional(item.get("arxiv_id")),
                abstract_preview=_clean_optional(item.get("abstract_preview")),
                visit_count=_coerce_int(item.get("visit_count")),
                likes=_coerce_int(item.get("likes")),
            )
        )
    return papers


def synthesize_summary(papers: list[PaperItem]) -> str:
    if not papers:
        return "No papers were retrieved, so there are no grounded research themes to summarize yet."

    title_tokens: list[str] = []
    abstract_tokens: list[str] = []
    for paper in papers[:5]:
        title_tokens.extend(_keyword_tokens(paper.title))
        abstract_tokens.extend(_keyword_tokens(paper.abstract_preview or ""))

    common_terms = [term for term, _ in Counter(title_tokens + abstract_tokens).most_common(6)]
    dominant_terms = ", ".join(common_terms[:4]) if common_terms else "related retrieval themes"

    top_titles = "; ".join(paper.title for paper in papers[:3])
    return (
        f"The retrieved papers cluster around {dominant_terms}, with emphasis on the themes surfaced by {top_titles}. "
        "Across the results, the dominant focus is on methods, evaluation settings, and practical research applications that are directly aligned with the expanded retrieval query."
    )


def format_response_text(response: MCPAgentResponse) -> str:
    intro = f"I found {len(response.papers)} alphaXiv papers for {response.floor_title}."
    lines = [intro, "", response.summary, "", "Top papers:"]

    if not response.papers:
        lines.append("1. No papers returned.")
        return "\n".join(lines)

    for paper in response.papers:
        authors = ", ".join(paper.authors) if paper.authors else "Unknown authors"
        organizations = ", ".join(paper.organizations) if paper.organizations else "Unknown organizations"
        publication_date = paper.publication_date or "Unknown date"
        arxiv_id = paper.arxiv_id or "Unknown arXiv ID"
        visit_count = paper.visit_count if paper.visit_count is not None else "n/a"
        likes = paper.likes if paper.likes is not None else "n/a"
        abstract_preview = paper.abstract_preview or "No abstract preview available."
        lines.extend(
            [
                f"{paper.rank}. {paper.title}",
                f"   Authors: {authors}",
                f"   Organizations: {organizations}",
                f"   Publication date: {publication_date}",
                f"   arXiv ID: {arxiv_id}",
                f"   Visit count: {visit_count}",
                f"   Likes: {likes}",
                f"   Abstract preview: {abstract_preview}",
            ]
        )
    return "\n".join(lines)


def _ensure_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _keyword_tokens(text: str) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "into", "using", "used", "this", "that", "are", "about",
        "paper", "papers", "study", "models", "model", "research", "retrieval", "results", "across", "their",
    }
    tokens = []
    for token in text.lower().replace("-", " ").split():
        cleaned = "".join(char for char in token if char.isalnum())
        if len(cleaned) > 3 and cleaned not in stopwords:
            tokens.append(cleaned)
    return tokens
