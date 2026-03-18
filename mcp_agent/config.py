import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    alphaxiv_mcp_url: str = Field(default="https://api.alphaxiv.org/mcp/v1", alias="ALPHAXIV_MCP_URL")
    alphaxiv_legacy_bearer_token: str | None = Field(default=None, alias="ALPHAXIV_LEGACY_BEARER_TOKEN")
    alphaxiv_oauth_access_token: str | None = Field(default=None, alias="ALPHAXIV_OAUTH_ACCESS_TOKEN")
    mcp_token_storage_path: str = Field(
        default_factory=lambda: str(Path("~/.cache/mcp_agent/alphaxiv_oauth_tokens.json").expanduser()),
        alias="MCP_TOKEN_STORAGE_PATH",
    )
    mcp_auth_timeout_seconds: float = Field(default=30.0, alias="MCP_AUTH_TIMEOUT_SECONDS")
    mcp_request_timeout_seconds: float = Field(default=60.0, alias="MCP_REQUEST_TIMEOUT_SECONDS")
    mcp_mode: str = Field(default="mock", alias="MCP_MODE")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def get_legacy_bearer_token(self) -> str | None:
        return resolve_legacy_bearer_token(self)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_legacy_bearer_token(settings: Settings) -> str | None:
    token = settings.alphaxiv_legacy_bearer_token or settings.alphaxiv_oauth_access_token
    if token:
        warnings.warn(
            "Legacy alphaXiv bearer tokens are deprecated. Use OAuth discovery/bootstrap instead of raw tokens where possible.",
            stacklevel=2,
        )
    return token
