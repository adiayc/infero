"""Eval-set format, loading, and seeding.

An eval set is a named list of cases. Each case is a chat prompt (messages) with
an optional golden ``reference`` answer and a free-form ``category``. Loadable
from JSON (``{name, cases: [...]}`` or a bare list) or JSONL (one case per line).

It can also be built from your own telemetry (:func:`from_prompts`), which is
the point of storing prompts at all: "bring your own eval set" is the friction
that stops most people ever running one, and the prompts you actually send are
the better test anyway. Those cases have no golden reference, so the LLM judge
carries the comparison.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from infero.models.openai import Message


class EvalCase(BaseModel):
    id: str
    messages: list[Message]
    reference: str | None = None
    category: str | None = None


class EvalSet(BaseModel):
    name: str = "eval"
    cases: list[EvalCase]


def load_eval_set(path: str | Path) -> EvalSet:
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix == ".jsonl":
        cases = [EvalCase.model_validate_json(line) for line in text.splitlines() if line.strip()]
        return EvalSet(name=p.stem, cases=cases)

    data = json.loads(text)
    if isinstance(data, list):
        return EvalSet(name=p.stem, cases=[EvalCase.model_validate(c) for c in data])
    return EvalSet.model_validate(data)


def from_prompts(prompts: Iterable[str], *, name: str = "telemetry") -> EvalSet:
    """Build an eval set from raw prompt strings.

    Case ids are a hash of the prompt rather than a counter, so re-seeding after
    more traffic keeps stable ids for prompts that were already there — which is
    what makes two eval runs comparable case by case.
    """

    cases = []
    for prompt in prompts:
        text = prompt.strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        cases.append(EvalCase(id=digest, messages=[Message(role="user", content=text)]))
    return EvalSet(name=name, cases=cases)
