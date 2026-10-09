"""Build a :class:`~infero.engine.manager.ModelManager` from config.

Mirrors ``backends/factory.py``: config models stay declarative, and the
translation into constructor arguments lives in one place.
"""

from __future__ import annotations

import httpx

from infero.engine.catalog import ModelCatalog
from infero.engine.fit import FitPolicy
from infero.engine.manager import ModelManager
from infero.models.config_models import AppConfig, EngineConfig


def build_fit_policy(config: EngineConfig) -> FitPolicy:
    return FitPolicy(
        enabled=config.fit.enabled,
        target_mib=config.fit.target_mib,
        min_ctx=config.fit.min_ctx,
        n_gpu_layers=config.fit.n_gpu_layers,
        ctx_size=config.fit.ctx_size,
    )


def build_manager(
    config: EngineConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> ModelManager:
    return ModelManager(
        ModelCatalog(config.models_dir),
        server_path=config.server_path,
        host=config.host,
        base_port=config.base_port,
        max_resident=config.max_resident,
        fit=build_fit_policy(config),
        ready_timeout_s=config.ready_timeout_s,
        extra_args=config.extra_args,
        log_dir=config.log_dir,
        client=client,
    )


def needs_manager(config: AppConfig) -> bool:
    """Whether any enabled backend asks infero to own a model process."""

    return any(b.enabled and b.managed and b.type == "llamacpp" for b in config.backends)
