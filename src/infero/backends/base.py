"""The ``Backend`` seam.

Local llama.cpp, the mock, Anthropic, and every OpenAI-compatible provider
implement this one interface. The router never special-cases a concrete
backend — they see only tier, cost estimation, ``generate``, and ``stream``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from infero.models.config_models import BackendConfig, Pricing
from infero.models.openai import ChatCompletionRequest
from infero.providers import PricingSource, resolve_pricing


class BackendError(Exception):
    """A backend could not serve a request, with a reason worth showing.

    Carries the provider's own explanation ("invalid x-api-key"), which is the
    part a user can act on; a bare HTTP error would flatten it to "502".
    """


@dataclass(frozen=True)
class Delta:
    """One increment of a streamed reply."""

    text: str


class BackendResponse:
    """A backend's raw result, before the app wraps it in OpenAI shape.

    Also the terminal event of a stream, so the bookkeeping that follows a
    generation — usage, cost, telemetry — is written once for both paths.
    """

    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
        model: str | None = None,
    ) -> None:
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.model = model


#: What a streaming backend yields: any number of deltas, then exactly one
#: response. Consumers should treat a stream that ends without the response as
#: a truncated one.
StreamEvent = Delta | BackendResponse


class Backend(ABC):
    """Uniform interface over any inference provider."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        # Once, not per request.
        self.pricing, self.pricing_source = resolve_pricing(config)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tier(self) -> str:
        return self.config.tier

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> BackendResponse:
        """Run inference for a request and return the raw result."""
        raise NotImplementedError

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamEvent]:
        """Yield :class:`Delta`s, then one :class:`BackendResponse` to close.

        The terminal event is required, and carries the usage the deltas cannot:
        it is where cost and telemetry come from.

        This default generates whole and emits the result as a single delta, so
        every backend is streamable and the gateway needs no capability check. A
        backend that overrides it makes the reply *incremental*; one that does
        not is still correct, just no faster to first token.
        """

        result = await self.generate(request)
        if result.content:
            yield Delta(result.content)
        yield result

    def estimate_cost(self, prompt_tokens: int, expected_output_tokens: int) -> float:
        """Cost from the resolved pricing table, for router scoring.

        Output length is a guess pre-request; billed cost is recomputed after
        the response from real token counts.
        """

        p: Pricing = self.pricing
        return (
            prompt_tokens / 1000.0 * p.input_per_1k
            + expected_output_tokens / 1000.0 * p.output_per_1k
        )

    @property
    def priced(self) -> bool:
        """Whether this backend's cost figures mean anything."""

        source: PricingSource = self.pricing_source
        return source != "unknown"

    async def health(self) -> bool:
        """Whether this backend is reachable/usable. Overridden per backend."""
        return True


def parse_openai_chat(data: dict, fallback_model: str | None = None) -> BackendResponse:
    """Read an OpenAI-shaped chat completion body.

    Identical for llama.cpp and every hosted OpenAI-compatible provider, which
    is what let #17R collapse six vendors into one class.
    """

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    return BackendResponse(
        content=message.get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        finish_reason=choice.get("finish_reason") or "stop",
        model=data.get("model") or fallback_model,
    )


class OpenAIChunkReader:
    """Accumulates OpenAI-format stream chunks into a terminal response.

    Feed each decoded chunk to :meth:`read`; it returns the text delta, if any.
    The other fields arrive scattered — ``finish_reason`` on the choice,
    ``usage`` usually on a final choice-less chunk — so they are collected as
    seen and assembled by :meth:`result` at the end.

    Shared by llama.cpp and every OpenAI-compatible provider, for the same
    reason :func:`parse_openai_chat` is.
    """

    def __init__(self, fallback_model: str | None = None) -> None:
        self._model = fallback_model
        self._finish_reason: str | None = None
        self._usage: dict = {}
        self._text: list[str] = []

    def read(self, chunk: dict) -> str:
        if chunk.get("model"):
            self._model = chunk["model"]
        if chunk.get("usage"):
            self._usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]
        text = ((choice.get("delta") or {}).get("content")) or ""
        if text:
            self._text.append(text)
        return text

    @property
    def text(self) -> str:
        return "".join(self._text)

    def result(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> BackendResponse:
        """The terminal event. Token counts are a fallback for a silent provider."""

        return BackendResponse(
            content=self.text,
            prompt_tokens=int(self._usage.get("prompt_tokens") or prompt_tokens),
            completion_tokens=int(self._usage.get("completion_tokens") or completion_tokens),
            finish_reason=self._finish_reason or "stop",
            model=self._model,
        )
