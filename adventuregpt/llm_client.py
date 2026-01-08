"""
LLM client abstraction for AdventureGPT.

Phase 1 goal:
    Keep agent logic independent of vendor SDKs.

Default implementation:
    OpenAI via the Responses API (NOT ChatCompletions).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .llm_types import Messages


class LLMClient(Protocol):
    """
    Minimal interface for an LLM capable of producing a text response.
    """

    def respond(self, messages: Messages, *, max_output_tokens: int) -> str:
        """
        Return a text response for the given messages.
        """


@dataclass(frozen=True)
class OpenAIResponsesConfig:
    """
    Configuration for OpenAI Responses calls.
    """

    model: str
    temperature: float = 0.0
    max_retries: int = 6
    retry_sleep_seconds: float = 10.0


class OpenAIResponsesClient:
    """
    OpenAI-backed LLM client using the Responses API.
    """

    def __init__(
        self,
        api_key: str,
        config: OpenAIResponsesConfig,
        *,
        on_usage: Optional[Callable[[dict], None]] = None,
    ):
        # Import lazily so tests/dry-runs don't require the dependency at import time.
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key)
        self._config = config
        self._on_usage = on_usage

    def respond(self, messages: Messages, *, max_output_tokens: int) -> str:
        """
        Call OpenAI Responses API and return the response text.
        """

        last_err: Optional[BaseException] = None
        for attempt in range(self._config.max_retries):
            try:
                resp = self._client.responses.create(
                    model=self._config.model,
                    input=messages,
                    temperature=self._config.temperature,
                    max_output_tokens=max_output_tokens,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None and self._on_usage is not None:
                    # Best-effort, schema-tolerant usage extraction.
                    u = {
                        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                    self._on_usage(u)

                # The SDK exposes a convenience accessor for text output.
                text = getattr(resp, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

                # Fallback: try best-effort extraction if output_text is empty.
                # (Keeps this resilient across minor SDK schema changes.)
                output = getattr(resp, "output", None)
                if output:
                    # output is a list of items; attempt to find any text content
                    # in a simple, defensive way.
                    chunks: list[str] = []
                    for item in output:
                        content = getattr(item, "content", None)
                        if not content:
                            continue
                        for part in content:
                            if getattr(part, "type", None) == "output_text":
                                chunks.append(getattr(part, "text", ""))
                    joined = "".join(chunks).strip()
                    if joined:
                        return joined

                return ""
            except Exception as e:  # noqa: BLE001 - provider SDK can raise varied exceptions
                last_err = e
                if attempt < self._config.max_retries - 1:
                    time.sleep(self._config.retry_sleep_seconds)
                    continue
                raise

        # Should never reach here, but keep mypy happy.
        if last_err:
            raise last_err
        raise RuntimeError("OpenAI request failed with unknown error.")


class CappedLLMClient:
    """
    Wrapper that caps `max_output_tokens` for all calls.

    This is useful to enforce a global budget from CLI/env configuration without
    changing all agent call-sites.
    """

    def __init__(self, inner: LLMClient, *, max_output_tokens_cap: int):
        self._inner = inner
        self._cap = int(max_output_tokens_cap)

    def respond(self, messages: Messages, *, max_output_tokens: int) -> str:
        capped = min(int(max_output_tokens), self._cap)
        return self._inner.respond(messages, max_output_tokens=capped)


class CachedLLMClient:
    """
    Simple in-memory cache for LLM responses within a process.

    Useful for eval iterations where identical prompts may repeat.
    """

    def __init__(self, inner: LLMClient, *, max_entries: int = 256):
        self._inner = inner
        self._max_entries = int(max_entries)
        self._cache: dict[tuple[int, int], str] = {}
        self._order: list[tuple[int, int]] = []

    def respond(self, messages: Messages, *, max_output_tokens: int) -> str:
        key = (hash(repr(messages)), int(max_output_tokens))
        if key in self._cache:
            return self._cache[key]
        text = self._inner.respond(messages, max_output_tokens=max_output_tokens)
        self._cache[key] = text
        self._order.append(key)
        if len(self._order) > self._max_entries:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return text

