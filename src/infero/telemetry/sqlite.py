"""Persistent telemetry, on SQLite.

Three choices rather than obvious implementations:

**Totals are aggregated, not accumulated.** ``snapshot`` runs ``SUM`` over the
table rather than maintaining counter rows, which can silently disagree with the
requests behind them after a crash, a manual delete, or a prune. O(rows), but an
index on ``ts`` keeps windowed queries cheap at one developer's scale.

**EWMA and inflight stay in memory, but the EWMA is rebuilt at startup.** The
router reads both per request, so neither may become a disk round-trip. Inflight
is genuinely per-process; latency is not, and a restarted server that has
forgotten it routes badly until it relearns.

**WAL mode.** The server writes while ``infero eval --from-telemetry`` reads the
same file from another process.

sqlite3 is blocking, so statements run through :func:`asyncio.to_thread` — one
connection under a lock, since connections are not thread-safe by default.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from infero.telemetry.base import (
    BackendMetric,
    BackendStats,
    MetricsSnapshot,
    RequestRecord,
    TelemetryStore,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        TEXT PRIMARY KEY,
    ts                REAL NOT NULL,
    tier              TEXT,
    backend           TEXT,
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0,
    ttft_ms           REAL,
    route_reason      TEXT,
    eval_score        REAL,
    routing_saved_usd REAL NOT NULL DEFAULT 0,
    prompt            TEXT
);
CREATE INDEX IF NOT EXISTS ix_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS ix_requests_backend ON requests(backend);
"""

_COLUMNS = (
    "request_id",
    "ts",
    "tier",
    "backend",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "latency_ms",
    "ttft_ms",
    "route_reason",
    "eval_score",
    "routing_saved_usd",
    "prompt",
)

_INSERT = (
    f"INSERT OR REPLACE INTO requests ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_COLUMNS))})"
)

#: How many past records seed the latency EWMA at startup.
BOOTSTRAP_LIMIT = 200

#: Columns added after the first release, with the type to add them as.
#:
#: ``CREATE TABLE IF NOT EXISTS`` does nothing to a database that already
#: exists, so a new column has to be added explicitly or every insert against
#: an existing history fails. Nullable and additive only — an upgrade must not
#: be able to lose a request somebody already paid for.
ADDED_COLUMNS = {"ttft_ms": "REAL"}


class SQLiteTelemetryStore(TelemetryStore):
    def __init__(
        self,
        path: str | Path,
        *,
        ewma_alpha: float = 0.3,
        store_prompts: bool = True,
        bootstrap_limit: int = BOOTSTRAP_LIMIT,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._alpha = ewma_alpha
        self._store_prompts = store_prompts

        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # A reader in another process must neither block writes nor be blocked.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

        self._db_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._ewma: dict[str, float] = {}
        self._samples: dict[str, int] = defaultdict(int)
        self._inflight: dict[str, int] = defaultdict(int)
        self._bootstrap(bootstrap_limit)

    # -- setup -------------------------------------------------------------

    def _migrate(self) -> None:
        """Add any column this version knows about that the file does not."""

        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(requests)")}
        for column, sql_type in ADDED_COLUMNS.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE requests ADD COLUMN {column} {sql_type}")  # noqa: S608

    def _bootstrap(self, limit: int) -> None:
        """Seed the latency EWMA from history.

        Otherwise a restart falls back to declared defaults, which on a
        partially-offloaded local model are off by an order of magnitude.
        Replayed oldest-first so the smoothing lands where it would have.
        """

        rows = self._conn.execute(
            "SELECT backend, latency_ms FROM requests "
            "WHERE backend IS NOT NULL AND latency_ms > 0 "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for row in reversed(rows):
            self._observe_latency(row["backend"], row["latency_ms"])

    def _observe_latency(self, backend: str, latency_ms: float) -> None:
        prev = self._ewma.get(backend)
        self._ewma[backend] = (
            latency_ms if prev is None else self._alpha * latency_ms + (1 - self._alpha) * prev
        )
        self._samples[backend] += 1

    async def _run(self, fn, *args):
        async with self._db_lock:
            return await asyncio.to_thread(fn, *args)

    # -- writes ------------------------------------------------------------

    async def record(self, record: RequestRecord) -> None:
        prompt = record.prompt if self._store_prompts else None
        values = tuple(
            prompt if column == "prompt" else getattr(record, column) for column in _COLUMNS
        )

        def write() -> None:
            self._conn.execute(_INSERT, values)
            self._conn.commit()

        await self._run(write)

        # Every record is a real backend call, so each one feeds the EWMA.
        if record.backend and record.latency_ms > 0:
            async with self._state_lock:
                self._observe_latency(record.backend, record.latency_ms)

    async def prune(self, before: float) -> int:
        def delete() -> int:
            cursor = self._conn.execute("DELETE FROM requests WHERE ts < ?", (before,))
            self._conn.commit()
            return cursor.rowcount or 0

        return await self._run(delete)

    # -- reads -------------------------------------------------------------

    async def query(self, limit: int = 100, since: float | None = None) -> list[RequestRecord]:
        where, params = ("WHERE ts >= ?", [since]) if since is not None else ("", [])

        def read() -> list[sqlite3.Row]:
            return self._conn.execute(
                f"SELECT * FROM requests {where} ORDER BY ts DESC LIMIT ?",  # noqa: S608
                [*params, limit],
            ).fetchall()

        return [RequestRecord(**dict(row)) for row in await self._run(read)]

    async def snapshot(self, since: float | None = None) -> MetricsSnapshot:
        where, params = ("WHERE ts >= ?", (since,)) if since is not None else ("", ())

        def read() -> tuple[sqlite3.Row, list[sqlite3.Row]]:
            totals = self._conn.execute(
                f"""
                SELECT
                    COUNT(*)                       AS total_requests,
                    COALESCE(SUM(cost_usd), 0)     AS total_cost_usd,
                    COALESCE(SUM(routing_saved_usd), 0) AS total_routing_saved_usd
                FROM requests {where}
                """,
                params,
            ).fetchone()
            by_backend = self._conn.execute(
                f"""
                SELECT backend,
                       COUNT(*) AS requests,
                       COALESCE(SUM(cost_usd), 0) AS cost_usd,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM requests {where}{" AND" if where else "WHERE"} backend IS NOT NULL
                GROUP BY backend
                """,
                params,
            ).fetchall()
            return totals, by_backend

        totals, rows = await self._run(read)
        return MetricsSnapshot(
            total_requests=totals["total_requests"],
            total_cost_usd=totals["total_cost_usd"],
            total_routing_saved_usd=totals["total_routing_saved_usd"],
            by_backend={
                row["backend"]: BackendMetric(
                    requests=row["requests"],
                    cost_usd=row["cost_usd"],
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                )
                for row in rows
            },
        )

    async def prompts(self, limit: int = 100, since: float | None = None) -> list[str]:
        """Distinct recent prompts, newest first.

        Deduplicated by ``MAX(ts)`` per prompt: asking the same question thirty
        times would otherwise seed a thirty-case eval set measuring one question.
        """

        clauses = ["prompt IS NOT NULL", "prompt != ''"]
        params: list[float] = []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        params.append(limit)

        def read() -> list[sqlite3.Row]:
            return self._conn.execute(
                f"SELECT prompt, MAX(ts) AS latest FROM requests "  # noqa: S608 - fixed clauses
                f"WHERE {' AND '.join(clauses)} "
                f"GROUP BY prompt ORDER BY latest DESC LIMIT ?",
                params,
            ).fetchall()

        return [row["prompt"] for row in await self._run(read)]

    # -- live signals ------------------------------------------------------

    async def backend_stats(self, name: str) -> BackendStats:
        return BackendStats(
            name=name,
            ewma_latency_ms=self._ewma.get(name),
            inflight=self._inflight[name],
            samples=self._samples[name],
        )

    async def incr_inflight(self, name: str) -> None:
        async with self._state_lock:
            self._inflight[name] += 1

    async def decr_inflight(self, name: str) -> None:
        async with self._state_lock:
            self._inflight[name] = max(0, self._inflight[name] - 1)

    async def aclose(self) -> None:
        async with self._db_lock:
            await asyncio.to_thread(self._conn.close)


async def open_readonly(path: str | Path) -> SQLiteTelemetryStore | None:
    """Open an existing telemetry database for reading, or ``None``.

    The eval harness must not conjure an empty database from a wrong path.
    """

    if not Path(path).exists():
        return None
    return SQLiteTelemetryStore(path, bootstrap_limit=0)


def default_path(state_dir: Path) -> Path:
    return state_dir / "telemetry.db"


def retention_cutoff(retention_days: int) -> float | None:
    """Timestamp before which records may be pruned, or ``None`` to keep all."""

    if retention_days <= 0:
        return None
    return time.time() - retention_days * 86400
