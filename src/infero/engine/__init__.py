"""Local model engine: catalog, GPU fit policy, and process supervision.

Owns ``llama-server`` processes rather than embedding an inference engine, so
model lifecycle is a product feature without this project acquiring
token-generation code.
"""

from infero.engine.catalog import CatalogError, LocalModel, ModelCatalog
from infero.engine.fit import FitPolicy, GpuInfo, detect_gpus, estimate_placement
from infero.engine.gguf import GGUFError, GGUFInfo, read_gguf_info
from infero.engine.manager import (
    ManagedInstance,
    ModelLoadError,
    ModelManager,
    RuntimeProps,
    find_llama_server,
    instance_table,
)

__all__ = [
    "CatalogError",
    "FitPolicy",
    "GGUFError",
    "GGUFInfo",
    "GpuInfo",
    "LocalModel",
    "ManagedInstance",
    "ModelCatalog",
    "ModelLoadError",
    "ModelManager",
    "RuntimeProps",
    "detect_gpus",
    "estimate_placement",
    "find_llama_server",
    "instance_table",
    "read_gguf_info",
]
