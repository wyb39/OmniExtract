"""Per-workflow model token accounting.

DSPy exposes the provider's usage block on every LM history entry.  This
module normalises the common OpenAI/LiteLLM and Anthropic field names and
associates them with the report currently being produced.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterator, Mapping


def _value(container: Any, key: str) -> tuple[bool, Any]:
    if isinstance(container, Mapping):
        return key in container, container.get(key)
    if container is None:
        return False, None
    return hasattr(container, key), getattr(container, key, None)


def _token_count(container: Any, *keys: str) -> tuple[bool, int]:
    for key in keys:
        present, value = _value(container, key)
        if not present or value is None or isinstance(value, bool):
            continue
        try:
            return True, max(0, int(value))
        except (TypeError, ValueError):
            continue
    return False, 0


@dataclass
class TokenUsage:
    """Aggregated, provider-reported usage for a single report."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    _cache_read_reported_calls: int = 0
    _cache_creation_reported_calls: int = 0

    @classmethod
    def from_provider_usage(cls, usage: Any) -> "TokenUsage":
        _, input_tokens = _token_count(usage, "prompt_tokens", "input_tokens")
        _, output_tokens = _token_count(
            usage,
            "completion_tokens",
            "output_tokens",
        )

        cache_read_known, cached_input_tokens = _token_count(
            usage,
            "cache_read_input_tokens",
            "_cache_read_input_tokens",
            "cached_input_tokens",
            "cached_tokens",
        )
        if not cache_read_known:
            for details_key in ("prompt_tokens_details", "input_tokens_details"):
                present, details = _value(usage, details_key)
                if present and details is not None:
                    cache_read_known, cached_input_tokens = _token_count(
                        details,
                        "cached_tokens",
                        "cache_read_input_tokens",
                    )
                    if cache_read_known:
                        break

        cache_creation_known, cache_creation_input_tokens = _token_count(
            usage,
            "cache_creation_input_tokens",
            "_cache_creation_input_tokens",
        )
        return cls(
            model_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            _cache_read_reported_calls=int(cache_read_known),
            _cache_creation_reported_calls=int(cache_creation_known),
        )

    @classmethod
    def from_report_dict(cls, value: Any) -> "TokenUsage":
        if not isinstance(value, Mapping):
            return cls()
        calls = _token_count(value, "model_calls")[1]
        cached_present, cached = _token_count(value, "cached_input_tokens")
        creation_present, creation = _token_count(
            value,
            "cache_creation_input_tokens",
        )
        return cls(
            model_calls=calls,
            input_tokens=_token_count(value, "input_tokens")[1],
            output_tokens=_token_count(value, "output_tokens")[1],
            cached_input_tokens=cached,
            cache_creation_input_tokens=creation,
            _cache_read_reported_calls=calls if cached_present else 0,
            _cache_creation_reported_calls=calls if creation_present else 0,
        )

    def add(self, other: "TokenUsage") -> "TokenUsage":
        self.model_calls += other.model_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self._cache_read_reported_calls += other._cache_read_reported_calls
        self._cache_creation_reported_calls += (
            other._cache_creation_reported_calls
        )
        return self

    def copy(self) -> "TokenUsage":
        return TokenUsage(
            model_calls=self.model_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            _cache_read_reported_calls=self._cache_read_reported_calls,
            _cache_creation_reported_calls=self._cache_creation_reported_calls,
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": (
                self.cached_input_tokens
                if self.model_calls > 0
                and self._cache_read_reported_calls == self.model_calls
                else None
            ),
            "cache_creation_input_tokens": (
                self.cache_creation_input_tokens
                if self.model_calls > 0
                and self._cache_creation_reported_calls == self.model_calls
                else None
            ),
        }


class TokenUsageTracker:
    def __init__(self) -> None:
        self._usage = TokenUsage()
        self._lock = RLock()

    def record(self, usage: Any) -> None:
        normalised = TokenUsage.from_provider_usage(usage)
        with self._lock:
            self._usage.add(normalised)

    def snapshot(self) -> TokenUsage:
        with self._lock:
            return self._usage.copy()


_CURRENT_TRACKER: ContextVar[TokenUsageTracker | None] = ContextVar(
    "omniextract_token_usage_tracker",
    default=None,
)
_ACTIVE_TRACKERS: dict[int, TokenUsageTracker] = {}
_ACTIVE_LOCK = RLock()


@contextmanager
def track_token_usage() -> Iterator[TokenUsageTracker]:
    """Collect LM calls made while one service operation builds its report."""

    existing = _CURRENT_TRACKER.get()
    if existing is not None:
        yield existing
        return

    tracker = TokenUsageTracker()
    token = _CURRENT_TRACKER.set(tracker)
    identity = id(tracker)
    with _ACTIVE_LOCK:
        _ACTIVE_TRACKERS[identity] = tracker
    try:
        yield tracker
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_TRACKERS.pop(identity, None)
        _CURRENT_TRACKER.reset(token)


def current_token_usage() -> TokenUsage | None:
    tracker = _CURRENT_TRACKER.get()
    if tracker is None:
        return None
    return tracker.snapshot()


def record_provider_usage(usage: Any) -> None:
    """Record one response, including calls made by DSPy's worker threads."""

    tracker = _CURRENT_TRACKER.get()
    if tracker is None:
        # DSPy's optimizers create their own worker threads and do not copy
        # contextvars.  A sole active workflow is still unambiguous.
        with _ACTIVE_LOCK:
            active = list(_ACTIVE_TRACKERS.values())
        if len(active) == 1:
            tracker = active[0]
    if tracker is not None:
        tracker.record(usage)


__all__ = [
    "TokenUsage",
    "TokenUsageTracker",
    "current_token_usage",
    "record_provider_usage",
    "track_token_usage",
]
