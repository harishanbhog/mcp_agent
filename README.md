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
- `MCP_TOKEN_STORAGE_PATH` controls where OAuth tokens/client registration are cached
- `MCP_AUTH_TIMEOUT_SECONDS` controls OAuth discovery / auth handshake timeout
- `MCP_REQUEST_TIMEOUT_SECONDS` controls the SSE session / tool-call timeout
- `MCP_MODE=mock` for local testing or `MCP_MODE=live` for alphaXiv MCP
- `REQUEST_TIMEOUT_SECONDS` to control LLM timeout

## alphaXiv authentication note

alphaXiv documents:

- MCP endpoint: `https://api.alphaxiv.org/mcp/v1`
- transport: SSE
- auth: OAuth 2.0

The live client now uses the Python MCP SDK's OAuth discovery flow instead of assuming a manually pasted bearer token. On the first live run, the CLI can perform an interactive OAuth bootstrap and cache the resulting token/client registration locally. Subsequent runs reuse the cached token automatically.

`ALPHAXIV_LEGACY_BEARER_TOKEN` is retained only as an explicit fallback for temporary debugging when combined with `--use-legacy-bearer-token`.

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

## OAuth bootstrap and token cache

Bootstrap OAuth without running retrieval:

```bash
poetry run python -m mcp_agent.cli --auth-only
```

Bootstrap with verbose MCP/OAuth diagnostics:

```bash
poetry run python -m mcp_agent.cli --auth-only --debug-mcp
```

Clear the cached token/client registration:

```bash
poetry run python -m mcp_agent.cli --clear-token-cache --auth-only
```

## Run tests

```bash
poetry run pytest
```

## Mock vs live MCP mode

- Use `MCP_MODE=mock` for local development and automated tests.
- Use `MCP_MODE=live` to connect to alphaXiv MCP over SSE.
- In live mode, provide `OPENAI_API_KEY` for query expansion.
- For alphaXiv auth, run `--auth-only` once to complete OAuth bootstrap and cache the token, then run the normal CLI command.
- If debugging a pre-issued raw bearer token, pass `--use-legacy-bearer-token` and set `ALPHAXIV_LEGACY_BEARER_TOKEN`, but this is not the default path.

## Live debugging examples

Run a live query after OAuth bootstrap:

```bash
poetry run python -m mcp_agent.cli \
  --floor-id bio_floor \
  --floor-title "BioInformatics Floor" \
  --floor-description "A research floor for bioinformatics, computational biology, genomics, protein modeling, drug discovery, and related AI/ML research." \
  --query "protein language models for variant effect prediction" \
  --llm-model "gpt-4.1-mini"
```

Run the same query with verbose MCP diagnostics:

```bash
poetry run python -m mcp_agent.cli \
  --floor-id bio_floor \
  --floor-title "BioInformatics Floor" \
  --floor-description "A research floor for bioinformatics, computational biology, genomics, protein modeling, drug discovery, and related AI/ML research." \
  --query "protein language models for variant effect prediction" \
  --llm-model "gpt-4.1-mini" \
  --debug-mcp
```

The alphaXiv adapter is isolated in `mcp_agent/alphaxiv_client.py`, so later router integration only needs to call `run_mcp_agent`.

## Future integration target

This project is intended to later be routed from xFloor's `router_node` when `call_tool == "mcp"`.
