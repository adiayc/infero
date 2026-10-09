"""Shared plumbing for backends that call somebody else's API.

Key resolution, error translation and the auth probe are identical whichever
wire format is in use, so the concrete classes carry only the format difference.

**A missing key means the backend is not built.** ``__init__`` raises and
``Gateway.from_config`` skips it, so an unkeyed provider is simply not a routing
candidate rather than one that fails every request it wins. ``infero keys`` says
why.

No retries: a 429 surfaces as a :class:`BackendError`. Retrying here would hide
latency the router is trying to measure, and "try a different backend" is a
routing decision, not a transport one. Deferred deliberately (SPEC backlog).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from infero.backends.base import Backend, BackendError
from infero.models.config_models import BackendConfig
from infero.providers import Provider, resolve_base_url, resolve_key_env, resolve_provider

# Hosted calls are network-bound and a long generation is legitimately slow;
# be patient on read, strict on connect.
REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
PROBE_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class MissingKeyError(ValueError):
    """A hosted backend is enabled but its API key is not set.

    Its own type because it is the *expected* state of a fresh checkout: the
    example config enables every provider, so "no key" is information rather
    than a configuration error.
    """


class HostedBackend(Backend):
    """A backend that talks to a remote provider with an API key."""

    def __init__(self, config: BackendConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config)
        self._client = client
        self.provider: Provider | None = resolve_provider(config)
        self.base_url = resolve_base_url(config)

        if not config.model:
            raise ValueError(f"backend '{config.name}' needs a 'model' to request")

        key_env = resolve_key_env(config)
        if not key_env:
            raise ValueError(
                f"backend '{config.name}' needs 'api_key_env' naming the variable "
                "that holds its API key"
            )
        self.key_env = key_env

        key = os.environ.get(key_env)
        if not key:
            raise MissingKeyError(
                f"{key_env} is not set — add it to .env or the environment "
                "('infero keys' lists them)"
            )
        self._key = key

    # -- per-format hooks --------------------------------------------------

    def auth_headers(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def probe_url(self) -> str:
        path = self.provider.probe_path if self.provider else "/models"
        return f"{self.base_url}{path}"

    # -- shared behavior ---------------------------------------------------

    async def health(self) -> bool:
        """Usable, as far as can be told without spending anything.

        Deliberately not a network call — ``GET /health`` is polled, and fanning
        that out to every provider would burn rate limit on a liveness check.
        ``infero keys --check`` is the live one.
        """

        return bool(self._key)

    async def probe(self) -> tuple[bool, str]:
        """Verify the key against the provider. Used by ``infero keys --check``.

        Hits the models endpoint — authenticated but free on every provider
        here, where a completion would cost money to answer a question about
        credentials.
        """

        try:
            response = await self._client.get(
                self.probe_url, headers=self.auth_headers(), timeout=PROBE_TIMEOUT
            )
        except httpx.HTTPError as exc:
            return False, f"unreachable: {exc}"
        if response.status_code == 200:
            return True, "ok"
        if response.status_code in (401, 403):
            return False, f"rejected: {extract_error(response)}"
        return False, f"HTTP {response.status_code}: {extract_error(response)}"

    @asynccontextmanager
    async def post_stream(self, path: str, payload: dict) -> AsyncIterator[httpx.Response]:
        """POST and hold the response open for reading as it arrives.

        Errors are raised here rather than surfacing as an empty stream: on a
        streaming response httpx has not read the body yet, so the provider's
        explanation has to be pulled in explicitly before it can be reported.
        """

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}{path}",
                json=payload,
                headers=self.auth_headers(),
                timeout=REQUEST_TIMEOUT,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise BackendError(f"{self.name}: {extract_error(response)}")
                yield response
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: could not reach {self.base_url}: {exc}") from exc

    async def post(self, path: str, payload: dict) -> dict:
        """POST to the provider and return the decoded body, or raise."""

        try:
            response = await self._client.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self.auth_headers(),
                timeout=REQUEST_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: could not reach {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise BackendError(f"{self.name}: {extract_error(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendError(f"{self.name}: unreadable response from {self.base_url}") from exc
        if not isinstance(data, dict):
            raise BackendError(f"{self.name}: unexpected response shape from {self.base_url}")
        return data


def extract_error(response: httpx.Response) -> str:
    """The provider's own explanation of a failure.

    Both wire formats nest it at ``error.message``, so one reader covers every
    provider. The status line alone would turn "invalid x-api-key" into "502".
    """

    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text.strip()[:300]}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return f"HTTP {response.status_code}: {error['message']}"
        if isinstance(error, str) and error:
            return f"HTTP {response.status_code}: {error}"
        if payload.get("message"):
            return f"HTTP {response.status_code}: {payload['message']}"
    return f"HTTP {response.status_code}"
