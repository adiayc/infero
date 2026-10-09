# Common flows. Windows users without `make`: run the underlying
# `uv ...` commands directly — they're listed in the README.

.PHONY: install serve run test lint fmt

install:
	uv sync

serve:
	uv run infero serve

# Kept: `make run` used to start the server.
run: serve

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
