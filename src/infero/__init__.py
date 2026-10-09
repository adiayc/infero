"""infero: run models locally, or route to a hosted provider."""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """Console entry point. Kept because stale installs still resolve here."""

    from infero.cli.main import main as cli_main

    cli_main()
