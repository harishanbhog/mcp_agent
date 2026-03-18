from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    alphaxiv_mcp_url: str = Field(default="https://api.alphaxiv.org/mcp/v1", alias="ALPHAXIV_MCP_URL")
    alphaxiv_auth_token: str | None = Field(default=None, alias="ALPHAXIV_AUTH_TOKEN")
    mcp_mode: str = Field(default="mock", alias="MCP_MODE")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
