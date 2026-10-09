"""Build concrete backends from config.

Dispatches on ``BackendConfig.type``. Four types, not one per vendor: local
llama.cpp, the test mock, Anthropic, and a generic OpenAI-compatible class
covering every other provider (SPEC #17R).

A ``ModelManager`` is passed through only to ``managed: true`` backends.
Construction is also where a hosted backend without its API key fails, so an
unkeyed provider never becomes a routing candidate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from infero.backends.anthropic import AnthropicBackend
from infero.backends.base import Backend
from infero.backends.llamacpp import LlamaCppBackend
from infero.backends.mock import MockBackend
from infero.backends.openai_compat import OpenAICompatBackend
from infero.models.config_models import BackendConfig
from infero.providers import pricing_warning

if TYPE_CHECKING:
    from infero.engine.manager import ModelManager

log = logging.getLogger("infero")

SUPPORTED_TYPES = frozenset({"llamacpp", "mock", "anthropic", "openai_compat"})


def build_backend(
    config: BackendConfig,
    client: httpx.AsyncClient,
    manager: ModelManager | None = None,
) -> Backend:
    if config.type == "llamacpp":
        backend: Backend = LlamaCppBackend(config, client, manager)
    elif config.type == "mock":
        backend = MockBackend(config)
    elif config.type == "anthropic":
        backend = AnthropicBackend(config, client)
    elif config.type == "openai_compat":
        backend = OpenAICompatBackend(config, client)
    else:
        raise NotImplementedError(f"backend type '{config.type}' is not implemented")

    # Once, at build: an unpriced backend still works, but every cost figure it
    # produces reads $0, and silence is the one way that must not happen.
    warning = pricing_warning(config, backend.pricing_source)
    if warning:
        log.warning("%s", warning)
    return backend
