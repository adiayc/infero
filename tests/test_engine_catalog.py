"""Model discovery, reference resolution, and pulling."""

from __future__ import annotations

import httpx
import pytest

from infero.engine.catalog import CatalogError, ModelCatalog
from tests.test_engine_gguf import STRING, UINT32, build_gguf


def make_catalog(tmp_path, *names: str) -> ModelCatalog:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for name in names:
        (models / name).write_bytes(b"")
    return ModelCatalog(models)


def test_lists_gguf_files_sorted(tmp_path):
    catalog = make_catalog(tmp_path, "b-model.gguf", "a-model.gguf", "notes.txt")

    assert [m.id for m in catalog.list()] == ["a-model", "b-model"]


def test_missing_directory_lists_nothing(tmp_path):
    assert ModelCatalog(tmp_path / "nope").list() == []


def test_parses_quant_and_params_from_filename(tmp_path):
    catalog = make_catalog(tmp_path, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    model = catalog.list()[0]

    assert model.quant == "Q4_K_M"
    assert model.params == "7B"
    # The "2.5" in the model name must not be read as a parameter count.
    assert model.id == "Qwen2.5-7B-Instruct-Q4_K_M"


@pytest.mark.parametrize(
    ("filename", "quant", "params"),
    [
        ("Llama-3.2-3B-Instruct-IQ3_XXS.gguf", "IQ3_XXS", "3B"),
        ("phi-4-500M-F16.gguf", "F16", "500M"),
        ("mistral-7b-instruct.BF16.gguf", "BF16", "7B"),
        ("some-model.gguf", None, None),
    ],
)
def test_filename_parsing_variants(tmp_path, filename, quant, params):
    catalog = make_catalog(tmp_path, filename)

    model = catalog.list()[0]

    assert model.quant == quant
    assert model.params == params


def test_multipart_models_appear_once(tmp_path):
    catalog = make_catalog(
        tmp_path,
        "big-model-00001-of-00003.gguf",
        "big-model-00002-of-00003.gguf",
        "big-model-00003-of-00003.gguf",
    )

    models = catalog.list()

    assert [m.id for m in models] == ["big-model"]
    assert models[0].shards == 3
    # llama.cpp is handed the first shard and opens the rest itself.
    assert models[0].path.name == "big-model-00001-of-00003.gguf"


def test_partial_downloads_are_not_models(tmp_path):
    catalog = make_catalog(tmp_path, "real.gguf")
    (catalog.models_dir / "half-done.gguf.part").write_bytes(b"")

    assert [m.id for m in catalog.list()] == ["real"]


def test_resolve_by_id_filename_and_substring(tmp_path):
    catalog = make_catalog(tmp_path, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    assert catalog.resolve("Qwen2.5-7B-Instruct-Q4_K_M").id.startswith("Qwen")
    assert catalog.resolve("Qwen2.5-7B-Instruct-Q4_K_M.gguf").id.startswith("Qwen")
    assert catalog.resolve("qwen").id.startswith("Qwen")  # case-insensitive substring


def test_resolve_by_path(tmp_path):
    catalog = make_catalog(tmp_path, "a.gguf")
    elsewhere = tmp_path / "other.gguf"
    elsewhere.write_bytes(b"")

    assert catalog.resolve(str(elsewhere)).id == "other"


def test_ambiguous_substring_names_the_candidates(tmp_path):
    catalog = make_catalog(tmp_path, "qwen-3b.gguf", "qwen-7b.gguf")

    with pytest.raises(CatalogError, match="ambiguous"):
        catalog.resolve("qwen")


def test_unknown_reference_lists_what_exists(tmp_path):
    catalog = make_catalog(tmp_path, "only-this.gguf")

    with pytest.raises(CatalogError, match="have: only-this"):
        catalog.resolve("missing")


def test_empty_catalog_suggests_pulling(tmp_path):
    catalog = make_catalog(tmp_path)

    with pytest.raises(CatalogError, match="models pull"):
        catalog.resolve("anything")


def test_empty_reference_is_rejected(tmp_path):
    catalog = make_catalog(tmp_path, "a.gguf")

    with pytest.raises(CatalogError, match="no model specified"):
        catalog.resolve("")


def test_inspect_adds_gguf_metadata(tmp_path):
    catalog = make_catalog(tmp_path)
    (catalog.models_dir / "real.gguf").write_bytes(
        build_gguf(
            [
                ("general.architecture", STRING, "qwen2"),
                ("qwen2.block_count", UINT32, 28),
                ("qwen2.context_length", UINT32, 32768),
            ]
        )
    )

    model = catalog.inspect(catalog.resolve("real"))

    assert model.architecture == "qwen2"
    assert model.block_count == 28
    assert model.context_length == 32768
    assert model.inspected


def test_inspect_tolerates_unreadable_header(tmp_path):
    """A bad header falls back to filename metadata rather than failing a load."""

    catalog = make_catalog(tmp_path, "Broken-7B-Q4_K_M.gguf")

    model = catalog.inspect(catalog.resolve("Broken"))

    assert model.architecture is None
    assert model.params == "7B"  # filename metadata survives


def test_inspect_reuses_the_cached_result(tmp_path):
    catalog = make_catalog(tmp_path)
    path = catalog.models_dir / "real.gguf"
    path.write_bytes(build_gguf([("general.architecture", STRING, "llama")]))

    first = catalog.inspect(catalog.resolve("real"))
    second = catalog.inspect(catalog.resolve("real"))

    assert first.architecture == "llama"
    assert second is first  # same object, not a re-read


def test_inspect_cache_invalidates_when_the_file_changes(tmp_path):
    """Swapping a model file behind the same name must not serve stale metadata."""

    catalog = make_catalog(tmp_path)
    path = catalog.models_dir / "real.gguf"
    path.write_bytes(build_gguf([("general.architecture", STRING, "llama")]))
    assert catalog.inspect(catalog.resolve("real")).architecture == "llama"

    path.write_bytes(build_gguf([("general.architecture", STRING, "qwen2")]))

    assert catalog.inspect(catalog.resolve("real")).architecture == "qwen2"


def test_label_is_human_readable(tmp_path):
    catalog = make_catalog(tmp_path)
    # Sparse: st_size reports 2 GiB without writing 2 GiB. Writing it for real
    # filled /tmp and took the rest of the suite down with ENOSPC.
    gguf = catalog.models_dir / "Qwen2.5-7B-Q4_K_M.gguf"
    with gguf.open("wb") as fh:
        fh.truncate(2 * 1024**3)

    model = catalog.inspect(catalog.resolve("Qwen"))

    assert model.label() == "7B Q4_K_M · 2.0 GB"


# --- pulling ---------------------------------------------------------------


def make_download_client(payload: bytes, *, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_pull_downloads_and_reports_progress(tmp_path):
    catalog = ModelCatalog(tmp_path / "models")
    seen: list[tuple[int, int | None]] = []

    model = await catalog.pull(
        "acme/models",
        "tiny.gguf",
        client=make_download_client(b"z" * 4096),
        progress=lambda done, total: seen.append((done, total)),
    )

    assert model.id == "tiny"
    assert model.path.read_bytes() == b"z" * 4096
    assert seen[-1] == (4096, 4096)


async def test_pull_leaves_no_partial_file_on_failure(tmp_path):
    catalog = ModelCatalog(tmp_path / "models")
    catalog.models_dir.mkdir(parents=True)

    with pytest.raises(CatalogError, match="failed to download"):
        await catalog.pull("acme/models", "tiny.gguf", client=make_download_client(b"", status=404))

    # Neither a real model nor a stray .part left behind.
    assert list(catalog.models_dir.iterdir()) == []


async def test_pull_is_idempotent(tmp_path):
    catalog = make_catalog(tmp_path, "tiny.gguf")

    model = await catalog.pull(
        "acme/models", "tiny.gguf", client=make_download_client(b"new content")
    )

    # Already present: returned as-is rather than re-downloaded.
    assert model.path.read_bytes() == b""


async def test_pull_rejects_non_gguf(tmp_path):
    catalog = ModelCatalog(tmp_path / "models")

    with pytest.raises(CatalogError, match="not a .gguf"):
        await catalog.pull("acme/models", "README.md")
