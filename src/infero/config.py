"""Runtime settings (env / ``.env``) and ``config.yaml`` loading.

:class:`Settings` holds process knobs and secrets, from the environment or a
``.env`` file — API keys and the gateway auth key live here and *never* in
``config.yaml``. :func:`load_config` reads the declarative
backend/routing config from YAML into an :class:`AppConfig`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from infero.models.config_models import AppConfig


class Settings(BaseSettings):
    """Runtime settings, read from ``INFERO_*`` env vars or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="INFERO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_path: str = "config.yaml"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # Where the CLI keeps the background server's log and record. None = ~/.infero.
    state_dir: str | None = None

    # Optional static client->gateway auth key. Off when unset.
    api_key: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.yaml`` into an :class:`AppConfig`.

    Falls back to built-in defaults if the file is absent, so the app can boot
    for a health check before any config exists.
    """

    resolved = Path(path or get_settings().config_path)
    if not resolved.exists():
        return AppConfig()

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
