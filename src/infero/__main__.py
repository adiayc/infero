"""``python -m infero`` — same entry point as the ``infero`` script.

Load-bearing: the CLI starts the background server with ``sys.executable -m
infero serve``, which works without the console script being on PATH.
"""

from __future__ import annotations

from infero.cli.main import main

main()
