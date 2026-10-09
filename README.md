# infero

Run open models on your own machine. Fall back to a paid API when local isn't
enough. Then find out which one you should have used.

infero supervises [`llama-server`](https://github.com/ggml-org/llama.cpp) to serve a
local GGUF model, and puts that model in the same pool as hosted providers
(Anthropic, OpenAI, Gemini, Groq, DeepSeek, Kimi, OpenRouter). One rule —
*use the cheapest backend that can answer* — decides every request. It speaks the
**OpenAI-compatible API**, so existing clients point at it unchanged.

It does not implement inference. llama.cpp does that; infero owns everything
around it: process lifecycle, routing, cost, and measurement.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync
cp config.example.yaml config.yaml
cp .env.example .env

uv run infero models pull unsloth/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf
uv run infero run qwen "why is the sky blue?"
```

The first call loads the model; later calls reuse the running process. You also
need a `llama-server` binary — put it in `vendor/bin/`, on `PATH`, or set
`INFERO_LLAMA_SERVER`. To build it yourself:

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON        # or -DGGML_VULKAN=ON, or no flag for CPU
cmake --build build --config Release -j$(nproc)
cp build/bin/llama-server /path/to/infero/vendor/bin/
```

## Commands

```
infero serve                    run the HTTP API in the foreground
infero models ls|pull|rm        manage local model files
infero run <model> [prompt]     load a model and generate with it
infero chat [--model tier]      interactive session, routed through the gateway
infero ps                       what is loaded right now
infero stop <model>|--all|--server
infero stats [--since 7d]       spend and latency
infero keys [--check]           provider keys, endpoints, and prices
infero eval --from-telemetry    compare backends on your own prompts
```

`models`, `run`, `ps`, `stats`, and `keys` take `--json`, so anything you can see
you can pipe.

**A background server holds the models.** `run` and `chat` start one on demand —
they cannot work without it, and a model that unloaded between commands would
make every invocation pay the load again. `ps`, `stats`, and `stop` never start
one: asking what is running should not cause something to run. `models` and
`keys` never contact a server at all. `--no-autostart` forbids it entirely.

`run` is pinned to the model you named and calls it directly — no routing, so it
cannot silently answer from somewhere else. `chat` goes through the full pipeline
and prints which backend answered and what it cost. The trade is that `run`
produces no telemetry.

Both stream by default. On a partially-offloaded local model that is the
difference between a reply and a wait.

## Routing

Each request resolves to a **tier** (from `gateway.tier`, a `model` naming a
tier or backend, or the configured default), then to the backends in that tier,
then to the cheapest one that is not currently excluded:

- candidates are ordered by estimated cost from the pricing table;
- a backend that just failed is skipped until its cooldown expires;
- a backend at `routing.max_inflight` concurrent requests is skipped.

If every candidate is excluded, the cheapest is still tried rather than refusing
the request — the failure surfaces from the backend call, which is the honest
signal. The response's `gateway` field names the chosen backend, why, and every
candidate's cost.

## Local models

Put a `.gguf` in `models/`, mark the backend `managed: true`, and infero starts
and supervises `llama-server` itself:

```yaml
backends:
  - name: local
    type: llamacpp
    tier: small
    managed: true
    model: Qwen2.5-7B-Instruct-Q4_K_M   # id, filename, path, or unique substring
```

GPU sizing is delegated to llama.cpp's `--fit`, which keeps
`engine.fit.target_mib` of VRAM free and picks a context size. A model larger
than your VRAM partially offloads instead of failing — on a 4 GB card a 7B Q4
*will* partially offload, and that is the normal path. Pin
`engine.fit.n_gpu_layers` or `ctx_size` to override.

One model stays resident by default (`engine.max_resident`); asking for another
evicts the least recently used. Every supervised process is stopped on shutdown.

To use a `llama-server` you started yourself, drop `managed` and set
`endpoint: http://127.0.0.1:8080`.

## Hosted providers

Two backend types cover every provider: `anthropic`, whose wire format genuinely
differs, and `openai_compat` for the rest. Naming a `provider` fills in the base
URL, key variable, and price table:

```yaml
backends:
  - name: haiku
    type: anthropic
    tier: small
    model: claude-haiku-4-5

  - name: gpt-mini
    type: openai_compat
    provider: openai          # openai | kimi | gemini | groq | deepseek | openrouter
    tier: small
    model: gpt-4.1-mini
```

Then set the key:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
uv run infero keys --check
```

**A backend whose key is unset is not built.** It never becomes a routing
candidate and nothing fails, so the example config can enable every provider and
setting one key is the only step.

Any other OpenAI-compatible endpoint works without a preset:

```yaml
- name: local-vllm
  type: openai_compat
  tier: medium
  base_url: http://10.0.0.5:8000/v1
  api_key_env: VLLM_API_KEY
  pricing: { input_per_1k: 0.0, output_per_1k: 0.0 }
```

### Prices

A built-in price table ships so cost reporting works immediately, but published
rates change and a stale price is worse than no price. `infero keys` reports
where each backend's price came from — `config`, `table`, or `unknown` — with
the table's as-of date.

A model with no known price is still built and still routable, but it is flagged:
it reports `priced: false`, warns at startup, and its cost computes as $0. Since
routing is cost-ordered, an unpriced backend will usually be chosen first — so if
that is wrong, give it a `pricing` block or disable it.

## Streaming

`stream: true` on `/v1/chat/completions` returns `text/event-stream`, from every
backend including Anthropic's own event format. Gateway metadata (`backend`,
`cost_usd`, `route_reason`) rides on the final chunk, alongside usage.

Text is accumulated while streaming so a streamed reply is still priced and
recorded like any other. Time-to-first-token is captured and shown per backend;
requests that were not streamed report it as absent, not `0`, because there was
no first token to measure.

Once the first frame is sent the status code is gone, so a backend that fails
mid-generation becomes an in-band `error` frame rather than a dropped
connection. The alternative — buffering until the reply is complete — would
discard the only reason to stream.

## Spend and history

Telemetry is persistent (SQLite, `~/.infero/telemetry.db`), because a spend
figure that resets on restart is not a spend figure.

```bash
uv run infero stats              # all time
uv run infero stats --since 7d   # a window
uv run infero stats --watch      # live
```

Totals are aggregated from the records with SQL rather than kept as running
counters, so the number cannot drift from the requests behind it. Every request
records what the priciest eligible backend would have charged, so
`routing_saved_usd` is a measured figure rather than an estimate.

Prompts are stored too, which is what makes the next section possible. That is a
local file that never leaves the machine; `telemetry.store_prompts: false` turns
it off and you keep counts, costs, and latencies.
`telemetry.retention_days` prunes old records at startup.

## Is the local model good enough?

This is the question the project exists to answer. `infero eval` runs an eval set
through each backend alone and through the full router, then prints cost,
quality, and latency side by side.

```bash
# your own prompts, replayed from history
uv run infero eval --from-telemetry --limit 25 --judge sonnet

# or a labelled set
uv run infero eval --eval-set evals/sample_support.json --judge sonnet
```

- **Eval sets** — JSON (`{name, cases:[...]}` or a bare list) or JSONL; each case
  is a chat prompt with an optional `reference` answer the judge is shown. See
  `evals/sample_support.json`.
- **Seeded sets** come from your telemetry, deduplicated, newest first. Case ids
  are a hash of the prompt, so a later run stays comparable case by case.
- **Scoring** — `--judge <backend>` is required; a strong model grades each
  answer and the scores aggregate uniformly.
- The harness builds its own gateway rather than reusing the running server, so
  it measures models rather than server state.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness, configured and active backends |
| `GET /v1/models` | configured backends, OpenAI list format |
| `POST /v1/chat/completions` | routed completion; `stream: true` for SSE, `503` if no backends |
| `GET /admin/stats?since_s=N` | totals, per-backend spend, latency percentiles, recent feed |
| `GET /admin/telemetry?limit=N` | recent per-request records |
| `GET /metrics` | Prometheus counters |

Engine admin, because resident models live inside the server process:

| Endpoint | Purpose |
| --- | --- |
| `GET /admin/engine` | server binary, models dir, GPUs, resident models |
| `POST /admin/engine/load` | `{"ref": "..."}` — load or reuse |
| `DELETE /admin/engine/models/{id}` | unload one, or `*` for all |
| `POST /admin/shutdown` | graceful stop, unloading models first |

Set `INFERO_API_KEY` to require `Authorization: Bearer <key>` on the write and
admin routes. It is off by default, which is right for a loopback-only server and
wrong for anything else.

## Design

Four seams are the extension points:

- **`Backend`** — llama.cpp, Anthropic, OpenAI-compatible, and mock. The router
  never special-cases a concrete backend.
- **`ModelManager`** — the only component that knows a subprocess exists: load,
  unload, LRU eviction, residency, and liveness. A process that has died is
  replaced on the next request rather than reported as resident.
- **`Router`** — cost-ordered fallback with a failure cooldown.
- **`TelemetryStore`** — SQLite or in-memory; feeds routing's load check, spend
  reporting, and eval seeding.

Adding a provider is config plus one preset in `providers.py`. The eval harness
additionally scores through a `Scorer` seam.

```
src/infero/
  cli/          commands, background server supervision, terminal rendering
  app.py        FastAPI app: /health, /v1/*, /admin/*, SSE
  service.py    the pipeline: route -> backend -> telemetry
  streaming.py  gateway stream events -> SSE frames
  backends/     Backend seam: llamacpp, anthropic, openai_compat, mock
  engine/       model catalog, GGUF reader, GPU fit, llama-server supervisor
  routing/      Router seam + cost-ordered fallback router
  telemetry/    TelemetryStore seam + SQLite, in-memory, Prometheus
  eval/         offline harness: dataset, scoring, runner, report
  providers.py  provider presets and price tables
  config.py     settings (.env) + config.yaml
  childproc.py  Windows spawn flags: detach without a console window
  tokens.py     token estimation for pre-flight cost
```

## Development

```bash
uv sync            # install
uv run pytest      # 369 tests
uv run ruff check .
uv run ruff format .
```

The suite runs with **no GPU, no model file, no API keys, no network, and no
server**. Each of those is a seam with a test double, so a full run needs nothing
but Python.

Design decisions and their reasoning — including the ones that were later
reversed — are recorded in `SPEC.md`, which is kept out of version control as
working notes rather than documentation.