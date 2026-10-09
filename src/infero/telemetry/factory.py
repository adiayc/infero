"""Build a :class:`TelemetryStore` from config.

Also owns the one policy decision: where the database lives when config doesn't
say. Next to the server log in the state directory, which already holds what
belongs to the machine rather than the checkout — request history inside the
repo would be a good way to commit it by accident.
"""

from __future__ import annotations

import logging
from pathlib import Path

from infero.config import Settings, get_settings
from infero.models.config_models import RoutingConfig, TelemetryConfig
from infero.telemetry.base import TelemetryStore
from infero.telemetry.memory import InMemoryTelemetryStore
from infero.telemetry.sqlite import SQLiteTelemetryStore, default_path

log = logging.getLogger("infero")


def telemetry_path(config: TelemetryConfig, settings: Settings | None = None) -> Path:
    if config.path:
        return Path(config.path)
    from infero.cli.daemon import state_dir

    return default_path(state_dir(settings or get_settings()))


def build_telemetry(
    config: TelemetryConfig,
    routing: RoutingConfig,
    *,
    settings: Settings | None = None,
) -> TelemetryStore:
    if config.backend == "memory":
        return InMemoryTelemetryStore(ewma_alpha=routing.ewma_alpha)

    path = telemetry_path(config, settings)
    try:
        return SQLiteTelemetryStore(
            path,
            ewma_alpha=routing.ewma_alpha,
            store_prompts=config.store_prompts,
        )
    except OSError as exc:
        # A read-only home or a locked file should mean a working server with
        # amnesia, not a server that will not start.
        log.error("telemetry: cannot open %s (%s) — falling back to in-memory", path, exc)
        return InMemoryTelemetryStore(ewma_alpha=routing.ewma_alpha)
