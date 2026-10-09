# AGENTS.md

Working notes for coding agents in this repo. `README.md` is for users; `SPEC.md`
is the source of truth for *why* anything is the way it is. This file is the
short version plus the rules that are easy to break.

**infero** is a local inference app: a CLI that loads and serves a GGUF on your
machine by supervising `llama-server`, routes to hosted providers when local
isn't the right tool, and can tell you — on your own prompts — whether the free
local model was good enough. It exposes an OpenAI-compatible HTTP API.

The package is `infero`. The repo directory is still `llm-gateway`; renaming it
is the author's job, not an agent's (SPEC #19).

## Commands

| Task    | Command                 |
| ------- | ----------------------- |
| install | `uv sync`               |
| test    | `uv run pytest`         |
| lint    | `uv run ruff check .`   |
| format  | `uv run ruff format .`  |
| serve   | `uv run infero serve`   |

Run all three of test/lint/format before calling work done. `make` wraps the
same commands for anyone who has it.

## Hard rules

- **Never run git.** No commit, push, branch, or `init` — not even when it seems
  helpful. This is not a git repository and the author keeps it that way.
- **Secrets live in the environment or `.env`.** `config.yaml` references a key
  by variable *name* only (`api_key_env: ANTHROPIC_API_KEY`). Never read a key
  into config, never print one, never log one. `.env` is not committed.
- **The test suite runs with no GPU, no model file, no API keys, no network, and
  no server.** Every one of those is a seam with a test double. If a change
  would require any of them, the change is wrong — find the seam.
- **Tests must never touch `~/.infero/`.** An autouse fixture redirects
  `INFERO_STATE_DIR` per test, and the gateway helpers pass in-memory telemetry.
  Both exist because tests once read and wrote the author's real spend history.
- **Never write real bytes just to make a file look big.** Use `fh.truncate(n)`
  for a sparse file of apparent size `n`; `st_size` reports it and the disk
  usage stays at zero. A test that wrote a genuine 2 GB file filled `/tmp` and
  took 34 unrelated tests down with `ENOSPC`.
- **Don't write token-generation code.** llama.cpp does inference; this project
  owns the lifecycle around it. Same for the other non-goals in SPEC §4: no RAG,
  no fine-tuning, no multi-tenant auth, no GUI.

## Layout

```
src/infero/
  cli/          the product surface: commands, daemon supervision, rendering
  app.py        FastAPI: /health, /v1/*, /admin/*, SSE
  childproc.py  Windows spawn flags: detach without drawing a window
  service.py    Gateway pipeline: route -> backend -> telemetry
  streaming.py  gateway events -> SSE frames (writing side)
  config.py     settings (.env) + config.yaml loader
  providers.py  hosted presets + price tables
  tokens.py     token estimation for pre-flight cost
  models/       OpenAI-compatible schemas + config models
  engine/       catalog, GGUF reader, GPU fit, llama-server supervisor
  backends/     Backend seam: llamacpp, anthropic, openai_compat, mock; sse.py reads
  routing/      Router seam + cost-ordered fallback router
  telemetry/    TelemetryStore seam + SQLite/in-memory stores, Prometheus, stats payload
  eval/         offline harness: dataset, scoring, runner, report
```

Seams (`Backend`, `Router`, `TelemetryStore`, `Scorer`) are the
extension points. Adding a provider should be config plus, at most, a preset in
`providers.py` — the router never special-cases a concrete backend.

## Conventions

- Python 3.11+, `from __future__ import annotations` at the top of every module.
- ruff, line length 100, rules `E,F,I,UP,B`. Typer/FastAPI parameter defaults are
  in `extend-immutable-calls`, so `typer.Option(...)` as a default is fine.
- pytest with `asyncio_mode = "auto"` — async tests need **no** marker.
- Test names are sentences about behaviour (`test_a_crashed_server_is_replaced`).
  A test docstring states *why the assertion matters*, never what the code does;
  if it would restate the test name, leave it out.
- Comments explain the non-obvious *why* — a platform quirk, a regression, an
  invariant. Delete anything that restates the code. Module docstrings are a few
  lines, not essays.
- Rich interprets `[...]` as markup, so env-var hints in Typer help text use
  parentheses: `(env: INFERO_CONFIG_PATH)`.
- Errors a user can act on get their own type and their own message. Provider
  wording ("invalid x-api-key") is passed through, never flattened to "502".

## Invariants worth protecting

Each of these is a bug that actually happened, or the reason a design looks odd.

- **Unknown is not zero, and unmeasured is not zero.** An unpriced model resolves
  to `pricing_source="unknown"` with `backend.priced is False`, so it is flagged
  in `infero keys` and warned about at startup — it is *not* refused, and its
  computed cost is still $0, which also makes it sort first under cost-ordered
  routing. An eval case with no golden answer scores `None`, rendered `—`, never
  0.0 (#12d). A request that was not streamed has `ttft_ms = None`, never 0
  (#15d). Averaging any of these as zero corrupts the one figure the feature
  exists to show.
- **Every request is its own telemetry record** (#9d). Reusing a response id made
  `request_id` non-unique and turned an insert into an overwrite of a prior
  record. Each call mints a fresh `chatcmpl-…` id and records under it.
- **Telemetry schema changes are additive, nullable, and migrated at open**
  (#15f). `CREATE TABLE IF NOT EXISTS` does nothing to a database that already
  has rows — add the column to `ADDED_COLUMNS` or every insert against real
  history fails. Never drop or rewrite a column: users have spend history.
- **Latency percentiles sample independently of the display feed.** `--feed 0`
  once computed p50 over one record and reported every backend at 0 ms.
- **A failed backend is skipped by the router during its cooldown** (#8R). The
  gateway calls `router.note_failure(name)` on `BackendError`, so the next
  request falls through to the next cheapest candidate instead of paying for the
  same failure again.
- **A missing API key means the backend is not built** — logged at info, not
  error (#17e). It is the expected state of a fresh checkout, which is why the
  example config can enable every provider.
- **No retries in a backend.** A 429 surfaces as `BackendError`; "try somewhere
  else" is a routing decision, not a transport one (#17g).
- **`ps`, `stats` and `stop` never auto-start a server**; only `run` and `chat`
  do (#5d). A command answering "what is running?" must not cause something to
  run. `models` and `keys` never contact a server at all.
- **`run` is pinned and direct; `chat` is routed** (#5e). A routed `run` could
  answer from a hosted provider instead of the model you named. The accepted cost
  is that `run` produces no telemetry.
- **The background server is stopped over HTTP, not by signal** (#5f). Its
  console is its own, so a console control event cannot reach it and `os.kill`
  degrades to `TerminateProcess`, skipping lifespan teardown and orphaning a
  multi-GB `llama-server` on the GPU.
- **Detach with `CREATE_NO_WINDOW`, never `DETACHED_PROCESS`** (#5g). Both
  detach, but `DETACHED_PROCESS` leaves the process with *no* console, and the
  first console program below it — the uv venv trampoline re-execing the
  interpreter — then makes Windows allocate one and draw it as a terminal
  window. Spawn every child through :mod:`infero.childproc`.
- **Streaming:** every backend is streamable via a buffered default (#15a); text
  is accumulated so a streamed reply is still priced and recorded (#15b); once
  the first SSE frame is sent the status code is gone, so expected failures
  become an in-band `error` frame while unexpected ones propagate as bugs (#15c).

## Environment

- Dev target: 8 GB RAM, RTX 3050 (4 GB VRAM), Windows 11. A 7B Q4 is 4.7 GB, so
  **partial GPU offload is the normal path**, not an edge case. `max_resident`
  defaults to 1 for the same reason.
- `llama-server` is discovered from `INFERO_LLAMA_SERVER`, then `vendor/bin/`,
  then `PATH`. The binaries are large and not committed.
- `~/.infero/` holds `server.json`, `server.log`, `chat_history`, `telemetry.db`.
- Config is `config.yaml` (copy `config.example.yaml`); settings and secrets come
  from `INFERO_*` env vars or `.env`.

## When decisions change

`SPEC.md` §2 is a numbered decision record with rationale, and §5 is the staged
roadmap. If you change a design choice — not just an implementation — add or
revise the numbered row with the reasoning, and update the roadmap entry. A
revised decision keeps the original alongside it (that is what the `R` suffix
means) so the trail stays intact. The bar for this project is that every choice
is defensible in conversation; an undocumented reversal fails that bar.
