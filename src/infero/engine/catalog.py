"""The local model catalog: what models this machine has, and where.

Two sources of truth about a file. The filename is cheap and occasionally wrong,
giving quantization and parameter-count labels; the GGUF header is exact and
costs one small read, giving architecture, layer count and context length.
:meth:`ModelCatalog.list` uses the first so listing stays instant over a
directory of multi-GB files; :meth:`ModelCatalog.inspect` adds the second and is
what the manager calls before a load.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from infero.engine.gguf import GGUFError, read_gguf_info

# Downloads land here first so an interrupted pull never looks like a real model.
PARTIAL_SUFFIX = ".part"

# Quantization label: Q4_K_M, IQ3_XXS, F16, BF16…
_QUANT_RE = re.compile(
    r"(?:^|[.\-_])(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)(?=$|[.\-_])",
    re.IGNORECASE,
)

# Parameter-count label: 7B, 3.8B, 500M.
_PARAMS_RE = re.compile(r"(?:^|[.\-_])(\d+(?:\.\d+)?[BM])(?=$|[.\-_])", re.IGNORECASE)

# Split-model shard suffix, e.g. "-00001-of-00003".
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")

ProgressFn = Callable[[int, int | None], None]
"""Called with ``(bytes_downloaded, total_bytes_or_None)`` during a pull."""


class CatalogError(Exception):
    """Raised when a model reference cannot be resolved or a pull fails."""


@dataclass(frozen=True)
class LocalModel:
    """A GGUF file on disk, plus whatever we know about it."""

    id: str
    path: Path
    size_bytes: int

    # From the filename.
    quant: str | None = None
    params: str | None = None
    shards: int = 1

    # From the GGUF header; populated by ``ModelCatalog.inspect``.
    architecture: str | None = None
    display_name: str | None = None
    block_count: int | None = None
    context_length: int | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    @property
    def inspected(self) -> bool:
        """Whether GGUF metadata has been read for this entry."""
        return self.architecture is not None or self.block_count is not None

    def label(self) -> str:
        """A one-line human description, e.g. ``qwen2 7B Q4_K_M · 4.4 GB``."""

        parts = [p for p in (self.architecture, self.params, self.quant) if p]
        head = " ".join(parts) if parts else self.id
        return f"{head} · {self.size_gb:.1f} GB"


class ModelCatalog:
    """Discovers GGUF models under a directory and fetches new ones."""

    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self._inspected: dict[tuple[Path, int, int], LocalModel] = {}

    # -- discovery ---------------------------------------------------------

    def list(self) -> list[LocalModel]:
        """All models in the directory, sorted by id.

        Multi-part models appear as their first shard only; llama.cpp opens the
        rest itself.
        """

        if not self.models_dir.is_dir():
            return []

        found: list[LocalModel] = []
        for path in self.models_dir.rglob("*.gguf"):
            if not path.is_file() or path.name.endswith(PARTIAL_SUFFIX):
                continue
            model = self._from_path(path)
            if model is not None:
                found.append(model)
        return sorted(found, key=lambda m: m.id)

    def get(self, model_id: str) -> LocalModel | None:
        """Look up by exact id (the filename stem, shard suffix removed)."""

        return next((m for m in self.list() if m.id == model_id), None)

    def resolve(self, ref: str) -> LocalModel:
        """Resolve an id, filename, path, or unique case-insensitive substring.

        So ``infero run qwen`` works with one Qwen installed, and says so
        explicitly when more than one is.
        """

        if not ref:
            raise CatalogError("no model specified")

        # A direct path wins over anything in the catalog.
        candidate = Path(ref)
        if candidate.suffix == ".gguf" and candidate.is_file():
            model = self._from_path(candidate)
            if model is not None:
                return model

        models = self.list()
        if not models:
            raise CatalogError(
                f"no models found in {self.models_dir}/ — fetch one with 'infero models pull'"
            )

        for model in models:
            if ref == model.id or ref == model.path.name:
                return model

        needle = ref.casefold()
        partial = [m for m in models if needle in m.id.casefold()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = ", ".join(m.id for m in partial)
            raise CatalogError(f"'{ref}' is ambiguous — matches {names}")

        known = ", ".join(m.id for m in models)
        raise CatalogError(f"no model matching '{ref}' (have: {known})")

    def inspect(self, model: LocalModel) -> LocalModel:
        """Return ``model`` enriched with GGUF header metadata.

        Cached on (path, size, mtime). An unparseable file comes back unchanged:
        a bad header means fall back to filename metadata, not refuse to run.
        """

        if model.inspected:
            return model

        try:
            stat = model.path.stat()
        except OSError as exc:
            raise CatalogError(f"cannot stat {model.path}: {exc}") from exc

        # Nanosecond mtime: a same-sized file swapped in within one second
        # would otherwise serve stale metadata.
        key = (model.path, stat.st_size, stat.st_mtime_ns)
        cached = self._inspected.get(key)
        if cached is not None:
            return cached

        try:
            info = read_gguf_info(model.path)
        except GGUFError:
            return model

        enriched = replace(
            model,
            architecture=info.architecture,
            display_name=info.name,
            block_count=info.block_count,
            context_length=info.context_length,
        )
        self._inspected[key] = enriched
        return enriched

    # -- fetching ----------------------------------------------------------

    async def pull(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        client: httpx.AsyncClient | None = None,
        progress: ProgressFn | None = None,
        overwrite: bool = False,
    ) -> LocalModel:
        """Download a GGUF from Hugging Face into the models directory.

        Streams to a ``.part`` file and renames on success, so an interrupted
        pull is never mistaken for a usable model. Hits the plain resolve URL
        rather than adding ``huggingface_hub`` for one HTTP GET.
        """

        if not filename.endswith(".gguf"):
            raise CatalogError(f"'{filename}' is not a .gguf file")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        target = self.models_dir / Path(filename).name
        if target.exists() and not overwrite:
            existing = self._from_path(target)
            if existing is not None:
                return existing

        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        partial = target.with_name(target.name + PARTIAL_SUFFIX)

        owned = client is None
        http = client or httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0))
        try:
            await self._stream_to(http, url, partial, progress)
        except httpx.HTTPError as exc:
            partial.unlink(missing_ok=True)
            raise CatalogError(f"failed to download {url}: {exc}") from exc
        finally:
            if owned:
                await http.aclose()

        partial.replace(target)
        model = self._from_path(target)
        if model is None:  # pragma: no cover - target was just written
            raise CatalogError(f"downloaded {target} but it is not readable")
        return model

    async def _stream_to(
        self,
        client: httpx.AsyncClient,
        url: str,
        destination: Path,
        progress: ProgressFn | None,
    ) -> None:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            raw_total = response.headers.get("content-length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None

            self._check_space(total)
            downloaded = 0
            with destination.open("wb") as fh:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)

    def _check_space(self, needed: int | None) -> None:
        """Fail before writing gigabytes we cannot finish writing."""

        if needed is None:
            return
        try:
            free = shutil.disk_usage(self.models_dir).free
        except OSError:
            return
        if free < needed:
            raise CatalogError(
                f"not enough disk space: need {needed / 1024**3:.1f} GB, "
                f"{free / 1024**3:.1f} GB free"
            )

    # -- internals ---------------------------------------------------------

    def _from_path(self, path: Path) -> LocalModel | None:
        """Build an entry from a file, or ``None`` if it is a non-first shard."""

        stem = path.stem
        shards = 1
        shard_match = _SHARD_RE.search(stem)
        if shard_match is not None:
            index, count = int(shard_match.group(1)), int(shard_match.group(2))
            if index != 1:
                return None  # llama.cpp opens the remaining shards itself
            stem = stem[: shard_match.start()]
            shards = count

        try:
            size = path.stat().st_size
        except OSError:
            return None

        quant = _QUANT_RE.search(stem)
        params = _PARAMS_RE.search(stem)
        return LocalModel(
            id=stem,
            path=path,
            size_bytes=size,
            quant=quant.group(1).upper() if quant else None,
            params=params.group(1).upper() if params else None,
            shards=shards,
        )
