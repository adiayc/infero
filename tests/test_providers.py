"""Provider presets and price resolution.

Not the table's contents — those go stale — but that the app can always say
*where* a price came from, and never reports a price it lacks as free.
"""

from __future__ import annotations

import pytest

from infero.models.config_models import BackendConfig, Pricing
from infero.providers import (
    PROVIDERS,
    ProviderError,
    get_provider,
    per_mtok,
    pricing_warning,
    resolve_base_url,
    resolve_key_env,
    resolve_pricing,
    resolve_provider,
)


def backend(**kwargs) -> BackendConfig:
    defaults = {"name": "b", "type": "openai_compat", "tier": "small"}
    return BackendConfig(**{**defaults, **kwargs})


# --- the table -------------------------------------------------------------


def test_published_rates_are_converted_from_per_million():
    """Providers quote per 1M tokens; the app bills per 1k."""

    assert per_mtok(3.0, 15.0) == Pricing(input_per_1k=0.003, output_per_1k=0.015)


def test_an_exact_model_id_is_priced():
    assert PROVIDERS["openai"].price_for("gpt-4o-mini") == per_mtok(0.15, 0.60)


def test_a_dated_release_inherits_its_family_price():
    """Providers append release dates; the table should not need a row each."""

    priced = PROVIDERS["anthropic"].price_for("claude-sonnet-4-5-20250929")

    assert priced == per_mtok(3.0, 15.0)


def test_the_longest_matching_prefix_wins():
    """'gpt-4.1-mini' must not be priced as 'gpt-4.1'."""

    assert PROVIDERS["openai"].price_for("gpt-4.1-mini") == per_mtok(0.40, 1.60)
    assert PROVIDERS["openai"].price_for("gpt-4.1") == per_mtok(2.0, 8.0)


def test_an_unknown_model_has_no_price():
    assert PROVIDERS["openai"].price_for("gpt-99-turbo") is None
    assert PROVIDERS["openai"].price_for(None) is None


def test_every_provider_declares_a_key_variable_and_base_url():
    for name, provider in PROVIDERS.items():
        assert provider.api_key_env, name
        assert provider.base_url.startswith("https://"), name
        assert provider.kind in {"anthropic", "openai_compat"}, name


# --- resolution ------------------------------------------------------------


def test_unknown_provider_names_the_known_ones():
    with pytest.raises(ProviderError, match="anthropic"):
        get_provider("altavista")


def test_anthropic_needs_no_provider_field():
    """There is exactly one Anthropic, so the type alone identifies it."""

    assert resolve_provider(backend(type="anthropic")).name == "anthropic"


def test_a_bare_openai_compat_backend_has_no_preset():
    assert resolve_provider(backend(base_url="http://x.test/v1")) is None


def test_base_url_comes_from_the_preset():
    assert resolve_base_url(backend(provider="groq")) == "https://api.groq.com/openai/v1"


def test_an_explicit_base_url_overrides_the_preset():
    """Pointing a known provider at a proxy or regional endpoint must work."""

    resolved = resolve_base_url(backend(provider="openai", base_url="http://proxy.test/v1/"))

    assert resolved == "http://proxy.test/v1"  # trailing slash normalized


def test_a_backend_with_neither_provider_nor_base_url_is_rejected():
    with pytest.raises(ProviderError, match="base_url"):
        resolve_base_url(backend())


def test_key_variable_falls_back_to_the_preset():
    assert resolve_key_env(backend(provider="kimi")) == "MOONSHOT_API_KEY"


def test_an_explicit_key_variable_wins():
    assert resolve_key_env(backend(provider="openai", api_key_env="MY_KEY")) == "MY_KEY"


# --- pricing ---------------------------------------------------------------


def test_config_pricing_beats_the_table():
    config = backend(provider="openai", model="gpt-4o", pricing=Pricing(input_per_1k=9.0))

    pricing, source = resolve_pricing(config)

    assert (pricing.input_per_1k, source) == (9.0, "config")


def test_the_table_supplies_a_price_when_config_does_not():
    pricing, source = resolve_pricing(backend(provider="openai", model="gpt-4o"))

    assert (pricing, source) == (per_mtok(2.50, 10.0), "table")


def test_local_backends_are_free_not_unknown():
    """A GGUF on your own disk genuinely costs nothing per token."""

    pricing, source = resolve_pricing(backend(type="llamacpp", model="qwen"))

    assert (pricing, source) == (Pricing(), "local")


def test_an_unpriced_hosted_model_is_unknown_not_free():
    """The distinction that keeps a spend report from silently reading $0."""

    _pricing, source = resolve_pricing(backend(provider="openai", model="gpt-99-turbo"))

    assert source == "unknown"


def test_a_marketplace_provider_defers_pricing_to_config():
    """OpenRouter's prices are per-model and upstream; hardcoding would lie."""

    _pricing, source = resolve_pricing(backend(provider="openrouter", model="anything"))

    assert source == "unknown"


def test_unknown_pricing_produces_an_actionable_warning():
    config = backend(name="gpt-x", provider="openai", model="gpt-99")

    message = pricing_warning(config, "unknown")

    assert message is not None
    assert "gpt-99" in message
    assert "pricing" in message  # says what to set


@pytest.mark.parametrize("source", ["config", "table", "local"])
def test_a_known_price_warns_about_nothing(source):
    assert pricing_warning(backend(), source) is None
