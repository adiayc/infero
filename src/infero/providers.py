"""Known hosted providers: where they live, which key they need, what they cost.

**Presets.** Naming ``provider: openai`` fills in the base URL, key variable and
wire format; every field stays overridable for a proxy or regional endpoint.

**Pricing.** The table here is the fallback when config does not say. A shipped
price table goes stale, and a stale price is worse than none because it is
silently wrong, so staleness is kept visible: :data:`PRICES_AS_OF` dates the
table, :func:`resolve_pricing` reports where a price came from, and an unpriced
model is flagged rather than defaulted to zero.

Prices are USD per 1,000 tokens. Verify against the provider before trusting a
spend figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from infero.models.config_models import BackendConfig, Pricing

#: When the prices below were recorded. Shown by ``infero keys``.
PRICES_AS_OF = "2026-05"

ProviderKind = Literal["anthropic", "openai_compat"]

PricingSource = Literal["config", "table", "local", "unknown"]


def per_mtok(inp: float, out: float) -> Pricing:
    """Published rates are quoted per million tokens; we bill per thousand."""

    return Pricing(input_per_1k=inp / 1000.0, output_per_1k=out / 1000.0)


@dataclass(frozen=True)
class Provider:
    """A hosted API infero knows how to talk to."""

    name: str
    kind: ProviderKind
    base_url: str
    api_key_env: str
    docs: str
    # Model id -> price. Keys are matched exactly first, then as a prefix, so a
    # dated release like ``claude-sonnet-4-5-20250929`` inherits the family price.
    prices: dict[str, Pricing] = field(default_factory=dict)
    # A cheap authenticated GET used by ``infero keys --check``. Must not spend.
    probe_path: str = "/models"

    def price_for(self, model: str | None) -> Pricing | None:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        # Longest matching prefix wins, so "claude-opus-4-1" beats "claude-opus".
        candidates = [key for key in self.prices if model.startswith(key)]
        if not candidates:
            return None
        return self.prices[max(candidates, key=len)]


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        docs="https://docs.anthropic.com/en/api/getting-started",
        probe_path="/v1/models",
        prices={
            "claude-opus-4": per_mtok(15.0, 75.0),
            "claude-sonnet-4": per_mtok(3.0, 15.0),
            "claude-haiku-4": per_mtok(1.0, 5.0),
            "claude-3-7-sonnet": per_mtok(3.0, 15.0),
            "claude-3-5-haiku": per_mtok(0.80, 4.0),
            "claude-3-5-sonnet": per_mtok(3.0, 15.0),
        },
    ),
    "openai": Provider(
        name="openai",
        kind="openai_compat",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        docs="https://platform.openai.com/docs/api-reference",
        prices={
            "gpt-4.1-nano": per_mtok(0.10, 0.40),
            "gpt-4.1-mini": per_mtok(0.40, 1.60),
            "gpt-4.1": per_mtok(2.0, 8.0),
            "gpt-4o-mini": per_mtok(0.15, 0.60),
            "gpt-4o": per_mtok(2.50, 10.0),
            "o4-mini": per_mtok(1.10, 4.40),
            "o3-mini": per_mtok(1.10, 4.40),
            "o3": per_mtok(2.0, 8.0),
        },
    ),
    "kimi": Provider(
        name="kimi",
        kind="openai_compat",
        base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        docs="https://platform.moonshot.ai/docs/api/chat",
        prices={
            "kimi-k2": per_mtok(0.60, 2.50),
            "moonshot-v1-8k": per_mtok(0.20, 2.00),
            "moonshot-v1-32k": per_mtok(1.00, 3.00),
            "moonshot-v1-128k": per_mtok(2.00, 5.00),
        },
    ),
    "gemini": Provider(
        name="gemini",
        kind="openai_compat",
        # Google's OpenAI-compatible surface, taken rather than adding a third
        # wire format for one provider (#17R).
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        docs="https://ai.google.dev/gemini-api/docs/openai",
        prices={
            "gemini-2.5-flash-lite": per_mtok(0.10, 0.40),
            "gemini-2.5-flash": per_mtok(0.30, 2.50),
            "gemini-2.5-pro": per_mtok(1.25, 10.0),
            "gemini-2.0-flash": per_mtok(0.10, 0.40),
        },
    ),
    # The three below are same wire format, different base URL — why #17R chose
    # a generic class over per-provider ones.
    "deepseek": Provider(
        name="deepseek",
        kind="openai_compat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        docs="https://api-docs.deepseek.com/",
        prices={
            "deepseek-chat": per_mtok(0.27, 1.10),
            "deepseek-reasoner": per_mtok(0.55, 2.19),
        },
    ),
    "groq": Provider(
        name="groq",
        kind="openai_compat",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        docs="https://console.groq.com/docs/openai",
        prices={
            "llama-3.3-70b-versatile": per_mtok(0.59, 0.79),
            "llama-3.1-8b-instant": per_mtok(0.05, 0.08),
        },
    ),
    "openrouter": Provider(
        name="openrouter",
        kind="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        docs="https://openrouter.ai/docs/api-reference",
        # A marketplace sets prices per upstream model; nothing honest to
        # hardcode. Put it in config.
        prices={},
    ),
}


class ProviderError(ValueError):
    """A backend names a provider or endpoint that cannot be resolved."""


def get_provider(name: str) -> Provider:
    known = ", ".join(sorted(PROVIDERS))
    try:
        return PROVIDERS[name.lower()]
    except KeyError:
        raise ProviderError(f"unknown provider '{name}' (known: {known})") from None


def resolve_provider(config: BackendConfig) -> Provider | None:
    """The preset behind a backend, if it names one.

    A backend may instead give ``base_url`` + ``api_key_env`` directly — the
    escape hatch for an endpoint infero has never heard of.
    """

    if config.provider:
        return get_provider(config.provider)
    # A type alone is enough for Anthropic: there is exactly one of it.
    if config.type == "anthropic":
        return PROVIDERS["anthropic"]
    return None


def resolve_base_url(config: BackendConfig) -> str:
    provider = resolve_provider(config)
    base = config.base_url or (provider.base_url if provider else None)
    if not base:
        raise ProviderError(f"backend '{config.name}' needs a 'provider' or an explicit 'base_url'")
    return base.rstrip("/")


def resolve_key_env(config: BackendConfig) -> str | None:
    """Which environment variable holds this backend's key."""

    if config.api_key_env:
        return config.api_key_env
    provider = resolve_provider(config)
    return provider.api_key_env if provider else None


def resolve_pricing(config: BackendConfig) -> tuple[Pricing, PricingSource]:
    """This backend's prices, and where they came from.

    The source is returned because "we don't know" and "it's free" are different
    facts that a bare :class:`Pricing` cannot tell apart — and conflating them is
    how a spend report reads $0.00 for a month of paid API calls.
    """

    if config.pricing is not None:
        return config.pricing, "config"
    if config.type in {"llamacpp", "mock"}:
        # Free for real, not unknown.
        return Pricing(), "local"

    provider = resolve_provider(config)
    if provider is not None:
        priced = provider.price_for(config.model)
        if priced is not None:
            return priced, "table"
    return Pricing(), "unknown"


def pricing_warning(config: BackendConfig, source: PricingSource) -> str | None:
    """A message for a backend whose cost cannot be computed, or ``None``."""

    if source != "unknown":
        return None
    return (
        f"backend '{config.name}': no price known for model '{config.model}' — "
        f"cost will report as $0. Set 'pricing' in config.yaml "
        f"(built-in table is from {PRICES_AS_OF})"
    )
