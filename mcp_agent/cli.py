from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from mcp_agent.alphaxiv_client import AlphaXivClient, AlphaXivClientError
from mcp_agent.config import get_settings
from mcp_agent.schemas import MCPAgentRequest
from mcp_agent.service import run_mcp_agent


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = AlphaXivClient(
        settings,
        debug_mcp=args.debug_mcp,
        use_legacy_bearer_token=args.use_legacy_bearer_token,
    )

    if args.clear_token_cache:
        cleared = client.clear_token_cache()
        print("Cleared token cache." if cleared else "Token cache was already empty.")
        if args.auth_only and not _has_query_args(args):
            return

    if args.auth_only:
        try:
            metadata = await client.ensure_authenticated()
        except AlphaXivClientError as exc:
            print(json.dumps({"error": str(exc), **exc.metadata}, indent=2))
            raise SystemExit(1) from exc

        print(json.dumps(metadata, indent=2))
        return

    _validate_query_args(args)
    request = MCPAgentRequest(
        floor_id=args.floor_id,
        floor_title=args.floor_title,
        floor_description=args.floor_description,
        query=args.query,
        llm_model=args.llm_model,
    )
    response = await run_mcp_agent(request, settings=settings, client=client)
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
    parser.add_argument("--floor-id")
    parser.add_argument("--floor-title")
    parser.add_argument("--floor-description")
    parser.add_argument("--query")
    parser.add_argument("--llm-model")
    parser.add_argument("--json", action="store_true", help="Print the full structured response as JSON")
    parser.add_argument("--auth-only", action="store_true", help="Perform OAuth bootstrap/login and token caching without running retrieval")
    parser.add_argument("--debug-mcp", action="store_true", help="Enable verbose MCP/OAuth/SSE diagnostics")
    parser.add_argument("--clear-token-cache", action="store_true", help="Clear the cached OAuth token/client registration state before continuing")
    parser.add_argument(
        "--use-legacy-bearer-token",
        action="store_true",
        help="Use the deprecated legacy bearer-token fallback for debugging only",
    )
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
    get_settings.cache_clear()
    if args.debug_mcp:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    asyncio.run(_run(args))


def _has_query_args(args: argparse.Namespace) -> bool:
    return any([args.floor_id, args.floor_title, args.floor_description, args.query, args.llm_model])


def _validate_query_args(args: argparse.Namespace) -> None:
    required = {
        "--floor-id": args.floor_id,
        "--floor-title": args.floor_title,
        "--floor-description": args.floor_description,
        "--query": args.query,
        "--llm-model": args.llm_model,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required arguments for retrieval: {', '.join(missing)}")


if __name__ == "__main__":
    main()
