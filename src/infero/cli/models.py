"""``infero models`` — the local GGUF collection.

Deliberately server-free: these read and write the models directory directly, so
``models ls`` is instant, works before anything has been run, and never starts a
server to answer what is on disk.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from infero.cli.context import guard, run_async, state
from infero.cli.render import console, emit_json, models_table, note, ok
from infero.config import load_config
from infero.engine.catalog import LocalModel, ModelCatalog

app = typer.Typer(no_args_is_help=True, add_completion=False)


def catalog_for(ctx: typer.Context) -> ModelCatalog:
    st = state(ctx)
    return ModelCatalog(load_config(st.config_path).engine.models_dir)


def _as_dict(model: LocalModel) -> dict:
    payload = asdict(model)
    payload["path"] = str(model.path)
    payload["size_gb"] = round(model.size_gb, 2)
    return payload


@app.command("ls")
def list_models(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """List the models in the models directory."""

    catalog = catalog_for(ctx)
    # Header reads are sub-millisecond, so architecture and context length are
    # worth showing by default rather than hiding behind a flag.
    with guard():
        models = [catalog.inspect(m) for m in catalog.list()]

    if json_out:
        emit_json([_as_dict(m) for m in models])
        return

    if not models:
        note(f"no models in {catalog.models_dir}/ — fetch one with 'infero models pull'")
        return
    console.print(models_table(_as_dict(m) for m in models))


@app.command("pull")
def pull(
    ctx: typer.Context,
    repo_id: str = typer.Argument(..., help="Hugging Face repo, e.g. Qwen/Qwen2.5-GGUF"),
    filename: str | None = typer.Argument(None, help="the .gguf file within that repo"),
    revision: str = typer.Option("main", "--revision", help="branch, tag, or commit"),
    force: bool = typer.Option(False, "--force", help="re-download even if it already exists"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Download a GGUF from Hugging Face.

    Accepts ``repo/name file.gguf`` or the shorthand ``repo/name:file.gguf``.
    """

    if filename is None:
        if ":" not in repo_id:
            raise typer.BadParameter(
                "give a filename, or use the repo:file.gguf shorthand", param_hint="FILENAME"
            )
        repo_id, filename = repo_id.split(":", 1)

    catalog = catalog_for(ctx)

    async def _pull():
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(filename, total=None)

            def advance(done: int, total: int | None) -> None:
                progress.update(task, completed=done, total=total)

            return await catalog.pull(
                repo_id, filename, revision=revision, progress=advance, overwrite=force
            )

    model = run_async(_pull())
    if json_out:
        emit_json(_as_dict(model))
        return
    ok(f"{model.id} → {model.path} ({model.size_gb:.1f} GB)")


@app.command("rm")
def remove(
    ctx: typer.Context,
    ref: str = typer.Argument(..., help="model id, filename, path, or unique substring"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt"),
) -> None:
    """Delete a model file from disk."""

    catalog = catalog_for(ctx)
    with guard():
        model = catalog.resolve(ref)
    targets = _files_for(model)
    total_gb = sum(p.stat().st_size for p in targets if p.exists()) / 1024**3

    if not yes:
        listing = "\n  ".join(str(p) for p in targets)
        console.print(f"about to delete {len(targets)} file(s), {total_gb:.1f} GB:\n  {listing}")
        if not typer.confirm("continue?"):
            raise typer.Abort()

    for path in targets:
        path.unlink(missing_ok=True)
    ok(f"deleted {model.id} ({total_gb:.1f} GB)")


def _files_for(model: LocalModel) -> list[Path]:
    """Every file that makes up a model.

    A split model is listed once but stored as N shards; deleting the first
    alone would leave gigabytes of unusable remainder.
    """

    if model.shards <= 1:
        return [model.path]
    return sorted(model.path.parent.glob(f"{model.id}-*-of-*.gguf"))
