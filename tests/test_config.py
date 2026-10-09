"""Config loading + model tests."""

from __future__ import annotations

from textwrap import dedent

from infero.config import load_config
from infero.models.config_models import AppConfig


def test_load_config_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.tiers == ["small", "medium", "large"]
    assert cfg.backends == []


def test_load_config_parses_backends(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        dedent(
            """
            tiers: [small, medium]
            backends:
              - name: local-llama
                type: llamacpp
                tier: small
                endpoint: http://127.0.0.1:8080
              - name: mock-hosted
                type: mock
                tier: medium
                pricing:
                  input_per_1k: 0.25
                  output_per_1k: 1.25
                enabled: false
            routing:
              default_tier: small
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert len(cfg.backends) == 2
    assert cfg.backends[0].type == "llamacpp"
    assert cfg.backends[1].pricing.output_per_1k == 1.25


def test_backends_for_tier_filters_enabled(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        dedent(
            """
            backends:
              - name: a
                type: mock
                tier: small
              - name: b
                type: mock
                tier: small
                enabled: false
              - name: c
                type: mock
                tier: medium
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    small = cfg.backends_for_tier("small")
    assert [b.name for b in small] == ["a"]  # b is disabled, c is a different tier
