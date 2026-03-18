from __future__ import annotations

import argparse
import asyncio
import json

from mcp_agent.schemas import MCPAgentRequest
from mcp_agent.service import run_mcp_agent


async def _run(args: argparse.Namespace) -> None:
    request = MCPAgentRequest(
        floor_id=args.floor_id,
        floor_title=args.floor_title,
        floor_description=args.floor_description,
        query=args.query,
        llm_model=args.llm_model,
    )
    response = await run_mcp_agent(request)
    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
        return

    print("Expanded query:\n")
    print(response.expanded_query)
    print("\nResponse:\n")
    print(response.response_text)
    print("\nMetadata:\n")
    print(json.dumps(response.metadata, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone alphaXiv MCP research retrieval agent")
    parser.add_argument("--floor-id", required=True)
    parser.add_argument("--floor-title", required=True)
    parser.add_argument("--floor-description", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--json", action="store_true", help="Print the full structured response as JSON")
    return parser


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv:
        load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
