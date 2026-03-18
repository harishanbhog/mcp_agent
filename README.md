# mcp_agent

A standalone Python 3.11+ project for V1 research retrieval with alphaXiv MCP. It accepts xFloor-style context, expands the user query with an LLM, calls alphaXiv's `embedding_similarity_search`, normalizes the returned papers, and produces a stable structured response.

This package is designed so xFloor can later integrate it with:

```python
response = await run_mcp_agent(request)
```

## Features

- Async-first orchestration
- Pydantic request/response models
- Internal LLM-based query expansion using floor context
- alphaXiv MCP client with `mock` and `live` modes
- CLI harness for standalone testing
- Pytest coverage for formatter, expander, and service flow

## Setup

1. Install Poetry if needed.
2. Install dependencies:

```bash
poetry install
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Fill in credentials as needed.

## Environment

Important settings:

- `OPENAI_API_KEY` for query expansion in live mode
- `OPENAI_BASE_URL` optional for OpenAI-compatible providers
- `ALPHAXIV_MCP_URL` defaults to `https://api.alphaxiv.org/mcp/v1`
- `MCP_MODE=mock` for local testing or `MCP_MODE=live` for alphaXiv MCP
- `REQUEST_TIMEOUT_SECONDS` to control LLM and MCP timeouts

## alphaXiv authentication note

As of March 18, 2026, the alphaXiv MCP docs describe the MCP endpoint as SSE transport with OAuth 2.0 authentication, but they do not document a static bearer-token environment variable for direct client use. This project therefore keeps the live adapter transport isolated while assuming authentication will be handled by the MCP stack or a future official OAuth integration path.

## Run the CLI

```bash
poetry run python -m mcp_agent.cli \
  --floor-id bio_floor \
  --floor-title "BioInformatics Floor" \
  --floor-description "A research floor for bioinformatics, computational biology, genomics, protein modeling, drug discovery, and related AI/ML research." \
  --query "protein language models for variant effect prediction" \
  --llm-model "gpt-4.1-mini"
```

For JSON output:

```bash
poetry run python -m mcp_agent.cli --json ...
```

## Run tests

```bash
poetry run pytest
```

## Mock vs live MCP mode

- Use `MCP_MODE=mock` for local development and automated tests.
- Use `MCP_MODE=live` to connect to alphaXiv MCP over SSE.
- In live mode, this package does not invent a custom token setting; it follows the current docs and leaves OAuth handling to the MCP client/runtime boundary.

The alphaXiv adapter is isolated in `mcp_agent/alphaxiv_client.py`, so later router integration only needs to call `run_mcp_agent`.

## Future integration target

This project is intended to later be routed from xFloor's `router_node` when `call_tool == "mcp"`.
