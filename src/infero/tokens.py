"""Cheap, dependency-free token estimation.

A rough chars/4 heuristic, for the router's forward-looking guesses and the
mock's synthetic counts. Real backends return exact usage.
"""

from __future__ import annotations

from infero.models.openai import Message

_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_prompt_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.content) + _PER_MESSAGE_OVERHEAD for m in messages)
