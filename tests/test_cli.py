"""The command line.

No server, GPU, model file, or network. Commands that need a server are pointed
at a dead port, so the "no server running" paths are exercised for real.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from infero.cli.main import app

runner = CliRunner()


def output(result) -> str:
    """Everything the command printed, wherever it went.

    click keeps stdout and stderr apart; most assertions here don't care which
    side a line came out of.
    """

    text = result.stdout or ""
    try:
        text += result.stderr or ""
    except ValueError:  # pragma: no cover - stderr not separately captured
        pass
    return text


def invoke(cli_home, *args: str):
    return runner.invoke(app, ["--config", str(cli_home.config_path), *args])


# --- top level -------------------------------------------------------------


def test_version_prints_and_exits(cli_home):
    result = invoke(cli_home, "--version")

    assert result.exit_code == 0
    assert "infero" in output(result)


def test_bare_invocation_shows_help(cli_home):
    result = runner.invoke(app, [])

    # no_args_is_help: exit 2 with usage, never a traceback.
    assert "Usage" in output(result)


# --- models ----------------------------------------------------------------


def test_models_ls_on_an_empty_directory_suggests_pull(cli_home):
    result = invoke(cli_home, "models", "ls")

    assert result.exit_code == 0
    assert "models pull" in output(result)


def test_models_ls_lists_what_is_there(cli_home):
    cli_home.add_model("Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    result = invoke(cli_home, "models", "ls")

    text = output(result)
    assert "Qwen2.5-7B-Instruct-Q4_K_M" in text
    assert "Q4_K_M" in text
    assert "7B" in text


def test_models_ls_json_is_parseable(cli_home):
    cli_home.add_model("tiny-1B-Q4_K_M.gguf", size=2048)

    result = invoke(cli_home, "models", "ls", "--json")

    payload = json.loads(result.stdout)
    assert payload[0]["id"] == "tiny-1B-Q4_K_M"
    assert payload[0]["quant"] == "Q4_K_M"
    assert payload[0]["size_bytes"] == 2048


def test_models_ls_json_is_empty_not_prose_when_nothing_is_installed(cli_home):
    result = invoke(cli_home, "models", "ls", "--json")

    assert json.loads(result.stdout) == []


def test_models_pull_accepts_the_repo_colon_file_shorthand(cli_home, monkeypatch):
    seen: dict = {}

    async def fake_pull(self, repo_id, filename, **kwargs):
        seen.update(repo_id=repo_id, filename=filename, **kwargs)
        return self._from_path(cli_home.add_model("pulled.gguf"))

    monkeypatch.setattr("infero.engine.catalog.ModelCatalog.pull", fake_pull)

    result = invoke(cli_home, "models", "pull", "acme/models:tiny.gguf")

    assert result.exit_code == 0
    assert seen["repo_id"] == "acme/models"
    assert seen["filename"] == "tiny.gguf"


def test_models_pull_without_a_filename_explains_the_shorthand(cli_home):
    result = invoke(cli_home, "models", "pull", "acme/models")

    assert result.exit_code != 0
    assert "shorthand" in output(result)


def test_models_rm_deletes_after_confirmation(cli_home):
    path = cli_home.add_model("doomed-1B-Q4_K_M.gguf")

    result = invoke(cli_home, "models", "rm", "doomed", "--yes")

    assert result.exit_code == 0
    assert not path.exists()


def test_models_rm_removes_every_shard(cli_home):
    """Deleting only the first shard would leave the rest as unusable ballast."""

    shards = [cli_home.add_model(f"big-0000{i}-of-00003.gguf") for i in (1, 2, 3)]

    invoke(cli_home, "models", "rm", "big", "--yes")

    assert not any(p.exists() for p in shards)


def test_models_rm_declined_leaves_the_file(cli_home):
    path = cli_home.add_model("keep-1B.gguf")

    result = runner.invoke(
        app, ["--config", str(cli_home.config_path), "models", "rm", "keep"], input="n\n"
    )

    assert result.exit_code != 0  # Abort
    assert path.exists()


def test_models_rm_unknown_reference_fails_cleanly(cli_home):
    cli_home.add_model("present.gguf")

    result = invoke(cli_home, "models", "rm", "absent", "--yes")

    assert result.exit_code == 1
    assert "no model matching" in output(result)


def test_ambiguous_reference_names_the_candidates(cli_home):
    cli_home.add_model("qwen-3b.gguf")
    cli_home.add_model("qwen-7b.gguf")

    result = invoke(cli_home, "models", "rm", "qwen", "--yes")

    assert result.exit_code == 1
    assert "ambiguous" in output(result)


# --- commands that need a server ------------------------------------------


def test_ps_without_a_server_is_not_an_error(cli_home):
    """ "Nothing is running" is a successful answer, not a failure."""

    result = invoke(cli_home, "ps")

    assert result.exit_code == 0
    assert "no infero server running" in output(result)


def test_ps_json_without_a_server_is_still_json(cli_home):
    result = invoke(cli_home, "ps", "--json")

    assert json.loads(result.stdout) == {"server": None, "resident": []}


def test_stats_without_a_server_is_not_an_error(cli_home):
    result = invoke(cli_home, "stats")

    assert result.exit_code == 0
    assert "no infero server running" in output(result)


def test_stop_requires_something_to_stop(cli_home):
    result = invoke(cli_home, "stop")

    assert result.exit_code != 0
    assert "--all" in output(result)


def test_stop_a_model_without_a_server_says_so(cli_home):
    result = invoke(cli_home, "stop", "qwen")

    assert result.exit_code == 0
    assert "no infero server running" in output(result)


def test_stop_server_when_none_is_running(cli_home):
    result = invoke(cli_home, "stop", "--server")

    assert result.exit_code == 0
    assert "no server was running" in output(result)


def test_run_refuses_json_without_a_prompt(cli_home):
    """A session has no single document to emit, so the flags conflict."""

    result = invoke(cli_home, "run", "qwen", "--json")

    assert result.exit_code != 0
    assert "needs a prompt" in output(result)


def test_run_without_autostart_does_not_start_anything(cli_home):
    result = runner.invoke(
        app,
        ["--config", str(cli_home.config_path), "--no-autostart", "run", "qwen", "hello"],
    )

    assert result.exit_code == 1
    assert "no infero server" in output(result)
    assert not (cli_home.state_dir / "server.json").exists()


# --- keys ------------------------------------------------------------------


def test_keys_reports_a_missing_provider_key(cli_home):
    result = invoke(cli_home, "keys")

    text = output(result)
    assert "INFERO_TEST_PROVIDER_KEY" in text
    assert "unset" in text


def test_keys_reports_a_present_one_without_printing_it(cli_home, monkeypatch):
    monkeypatch.setenv("INFERO_TEST_PROVIDER_KEY", "sk-super-secret-value")

    result = invoke(cli_home, "keys")

    text = output(result)
    assert "set" in text
    assert "sk-super-secret-value" not in text  # presence only, never the value


def test_keys_json_reports_presence_only(cli_home, monkeypatch):
    monkeypatch.setenv("INFERO_TEST_PROVIDER_KEY", "sk-secret")

    result = invoke(cli_home, "keys", "--json")

    payload = json.loads(result.stdout)
    rows = {r["backend"]: r for r in payload["backends"]}
    assert rows["hosted"]["set"] is True
    assert "sk-secret" not in result.stdout
    assert rows["local"]["env_var"] is None  # a local model needs no key
    assert payload["prices_as_of"]  # a price is only meaningful with a date


@pytest.mark.parametrize("command", [["ps"], ["stats"], ["models", "ls"], ["keys"]])
def test_read_commands_all_accept_json(cli_home, command):
    result = invoke(cli_home, *command, "--json")

    assert result.exit_code == 0
    json.loads(result.stdout)  # every read command emits a parseable document
