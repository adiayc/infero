"""Persistent telemetry, against a database in tmp_path.

What differs from the in-memory store: survival across restarts, windowed
aggregates, EWMA bootstrap, retention, and prompt capture.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from infero.telemetry.base import RequestRecord
from infero.telemetry.sqlite import SQLiteTelemetryStore, open_readonly, retention_cutoff

NOW = 1_700_000_000.0


def rec(request_id: str, **kwargs) -> RequestRecord:
    defaults = {
        "ts": NOW,
        "backend": "local",
        "cost_usd": 0.0,
        "latency_ms": 100.0,
    }
    return RequestRecord(request_id=request_id, **{**defaults, **kwargs})


@pytest.fixture
def store(tmp_path):
    return SQLiteTelemetryStore(tmp_path / "telemetry.db")


# --- persistence -----------------------------------------------------------


async def test_records_survive_a_restart(tmp_path):
    """The whole reason for this store: spend that resets is not spend."""

    path = tmp_path / "t.db"
    first = SQLiteTelemetryStore(path)
    await first.record(rec("a", cost_usd=0.25))
    await first.aclose()

    second = SQLiteTelemetryStore(path)

    assert (await second.snapshot()).total_cost_usd == 0.25
    await second.aclose()


async def test_the_database_is_created_with_its_parent(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "nested" / "deeper" / "t.db")

    assert store.path.exists()
    await store.aclose()


async def test_latency_ewma_is_rebuilt_from_history(tmp_path):
    """A restarted server that forgot its latencies routes badly for a while."""

    path = tmp_path / "t.db"
    first = SQLiteTelemetryStore(path)
    for i in range(5):
        await first.record(rec(f"r{i}", latency_ms=900.0))
    await first.aclose()

    second = SQLiteTelemetryStore(path)
    stats = await second.backend_stats("local")

    assert stats.ewma_latency_ms == pytest.approx(900.0)
    assert stats.samples == 5
    await second.aclose()


async def test_inflight_does_not_survive_a_restart(tmp_path):
    """Inflight is per-process by definition; a stale count would misroute."""

    path = tmp_path / "t.db"
    first = SQLiteTelemetryStore(path)
    await first.incr_inflight("local")
    await first.aclose()

    second = SQLiteTelemetryStore(path)

    assert (await second.backend_stats("local")).inflight == 0
    await second.aclose()


# --- aggregates ------------------------------------------------------------


async def test_totals_are_summed_over_the_table(store):
    await store.record(rec("a", cost_usd=0.10, routing_saved_usd=0.9))
    await store.record(rec("b", cost_usd=0.05))

    snap = await store.snapshot()

    assert snap.total_requests == 2
    assert snap.total_cost_usd == pytest.approx(0.15)
    assert snap.total_routing_saved_usd == pytest.approx(0.9)
    await store.aclose()


async def test_a_window_excludes_older_records(store):
    await store.record(rec("old", ts=NOW - 86400 * 10, cost_usd=1.0))
    await store.record(rec("new", ts=NOW, cost_usd=0.25))

    windowed = await store.snapshot(since=NOW - 3600)

    assert windowed.total_requests == 1
    assert windowed.total_cost_usd == pytest.approx(0.25)
    await store.aclose()


async def test_spend_is_broken_down_per_backend(store):
    await store.record(rec("a", backend="haiku", cost_usd=0.02, prompt_tokens=100))
    await store.record(rec("b", backend="haiku", cost_usd=0.03, completion_tokens=50))
    await store.record(rec("c", backend="local", cost_usd=0.0))

    by_backend = (await store.snapshot()).by_backend

    assert by_backend["haiku"].requests == 2
    assert by_backend["haiku"].cost_usd == pytest.approx(0.05)
    assert by_backend["haiku"].prompt_tokens == 100
    assert by_backend["haiku"].completion_tokens == 50
    assert by_backend["local"].cost_usd == 0.0
    await store.aclose()


async def test_an_empty_database_aggregates_to_zero_not_null(store):
    """SUM over no rows is NULL in SQL; the API promises numbers."""

    snap = await store.snapshot()

    assert (snap.total_requests, snap.total_cost_usd) == (0, 0.0)
    assert snap.by_backend == {}
    await store.aclose()


async def test_query_returns_newest_first(store):
    await store.record(rec("old", ts=NOW - 100))
    await store.record(rec("new", ts=NOW))

    assert [r.request_id for r in await store.query()] == ["new", "old"]
    await store.aclose()


async def test_query_respects_a_window(store):
    """Latency percentiles over `--since 7d` must not sample older requests."""

    await store.record(rec("old", ts=NOW - 86400 * 10))
    await store.record(rec("new", ts=NOW))

    assert [r.request_id for r in await store.query(since=NOW - 3600)] == ["new"]
    await store.aclose()


async def test_every_real_call_moves_the_latency_ewma(store):
    """Every recorded request is a real backend call, so each feeds the EWMA."""

    await store.record(rec("a", latency_ms=800.0))
    await store.record(rec("b", latency_ms=200.0))

    stats = await store.backend_stats("local")
    assert stats.ewma_latency_ms == pytest.approx(0.3 * 200.0 + 0.7 * 800.0)
    await store.aclose()


# --- prompts ---------------------------------------------------------------


async def test_prompts_are_stored_and_deduplicated(store):
    await store.record(rec("a", prompt="how do I reset my password"))
    await store.record(rec("b", prompt="how do I reset my password"))
    await store.record(rec("c", prompt="what is a GGUF"))

    prompts = await store.prompts()

    assert sorted(prompts) == ["how do I reset my password", "what is a GGUF"]
    await store.aclose()


async def test_prompt_capture_can_be_turned_off(tmp_path):
    store = SQLiteTelemetryStore(tmp_path / "t.db", store_prompts=False)

    await store.record(rec("a", prompt="something private", cost_usd=0.5))

    assert await store.prompts() == []
    # The rest of the record is still there — only the text is dropped.
    assert (await store.snapshot()).total_cost_usd == 0.5
    await store.aclose()


async def test_prompts_respect_a_window(store):
    await store.record(rec("old", ts=NOW - 86400 * 10, prompt="ancient"))
    await store.record(rec("new", ts=NOW, prompt="recent"))

    assert await store.prompts(since=NOW - 3600) == ["recent"]
    await store.aclose()


async def test_prompts_are_limited(store):
    for i in range(10):
        await store.record(rec(f"r{i}", ts=NOW + i, prompt=f"prompt {i}"))

    assert len(await store.prompts(limit=3)) == 3
    await store.aclose()


# --- retention -------------------------------------------------------------


async def test_prune_drops_only_old_records(store):
    await store.record(rec("old", ts=NOW - 86400 * 100))
    await store.record(rec("new", ts=NOW))

    dropped = await store.prune(NOW - 86400 * 90)

    assert dropped == 1
    assert [r.request_id for r in await store.query()] == ["new"]
    await store.aclose()


# --- schema migration ------------------------------------------------------


async def test_a_database_from_before_ttft_is_upgraded_in_place(tmp_path):
    """Someone's real spend history must survive an upgrade.

    ``CREATE TABLE IF NOT EXISTS`` does nothing to a file that already exists,
    so without the migration every insert against an older database fails and
    the server comes up unable to record anything.
    """

    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE requests (
            request_id TEXT PRIMARY KEY, ts REAL NOT NULL, tier TEXT, backend TEXT, model TEXT,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0, latency_ms REAL NOT NULL DEFAULT 0,
            route_reason TEXT, eval_score REAL,
            routing_saved_usd REAL NOT NULL DEFAULT 0, prompt TEXT
        );
        """
    )
    legacy.execute(
        "INSERT INTO requests (request_id, ts, backend, cost_usd) VALUES ('old', ?, 'haiku', 0.25)",
        (NOW,),
    )
    legacy.commit()
    legacy.close()

    store = SQLiteTelemetryStore(path)
    await store.record(rec("new", cost_usd=0.75, ttft_ms=120.0))

    snap = await store.snapshot()
    assert snap.total_requests == 2
    assert snap.total_cost_usd == pytest.approx(1.0)  # the old spend is still there
    assert {r.request_id: r.ttft_ms for r in await store.query()} == {"new": 120.0, "old": None}
    await store.aclose()


async def test_ttft_is_stored_and_read_back(store):
    await store.record(rec("streamed", ttft_ms=42.5))
    await store.record(rec("buffered"))

    by_id = {r.request_id: r.ttft_ms for r in await store.query()}

    assert by_id["streamed"] == 42.5
    assert by_id["buffered"] is None  # not streamed, not instant
    await store.aclose()


def test_retention_of_zero_days_keeps_everything():
    assert retention_cutoff(0) is None
    assert retention_cutoff(-1) is None
    assert retention_cutoff(30) < time.time()


# --- read-only access ------------------------------------------------------


async def test_open_readonly_does_not_create_a_database(tmp_path):
    """The eval harness must not conjure an empty db from a wrong path."""

    absent = tmp_path / "nothing.db"

    assert await open_readonly(absent) is None
    assert not absent.exists()


async def test_open_readonly_reads_an_existing_database(tmp_path):
    path = tmp_path / "t.db"
    writer = SQLiteTelemetryStore(path)
    await writer.record(rec("a", prompt="hello"))
    await writer.aclose()

    reader = await open_readonly(path)

    assert await reader.prompts() == ["hello"]
    await reader.aclose()


async def test_a_reader_and_a_writer_can_share_the_file(tmp_path):
    """WAL: `infero eval --from-telemetry` runs while the server is serving."""

    path = tmp_path / "t.db"
    writer = SQLiteTelemetryStore(path)
    await writer.record(rec("a", prompt="first"))

    reader = await open_readonly(path)
    await writer.record(rec("b", prompt="second"))

    assert set(await reader.prompts()) == {"first", "second"}
    await reader.aclose()
    await writer.aclose()
