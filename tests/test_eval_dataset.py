"""Eval-set loading (JSON object, bare list, JSONL)."""

from __future__ import annotations

from textwrap import dedent

from infero.eval.dataset import load_eval_set


def test_load_json_object(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(
        dedent(
            """
            {"name": "s", "cases": [
              {"id": "a", "messages": [{"role": "user", "content": "q"}], "reference": "r"}
            ]}
            """
        ),
        encoding="utf-8",
    )
    es = load_eval_set(p)
    assert es.name == "s"
    assert es.cases[0].id == "a"
    assert es.cases[0].reference == "r"


def test_load_json_bare_list(tmp_path):
    p = tmp_path / "e.json"
    p.write_text('[{"id": "a", "messages": [{"role": "user", "content": "q"}]}]', encoding="utf-8")
    es = load_eval_set(p)
    assert es.name == "e"  # falls back to filename
    assert len(es.cases) == 1


def test_load_jsonl(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"id": "a", "messages": [{"role": "user", "content": "q1"}]}\n'
        '{"id": "b", "messages": [{"role": "user", "content": "q2"}]}\n',
        encoding="utf-8",
    )
    es = load_eval_set(p)
    assert [c.id for c in es.cases] == ["a", "b"]
