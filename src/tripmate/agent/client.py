import logging
import re
from collections.abc import Callable
from threading import Event
from time import perf_counter
from typing import Any, Never, Protocol, cast

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    PermissionDeniedError,
    RateLimitError,
)

from tripmate.agent.models import (
    AgentConfigurationError,
    AgentProviderError,
    ModelReply,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
)
from tripmate.config import Settings
from tripmate.observability import correlation_fields

logger = logging.getLogger(__name__)

Message = dict[str, Any]


class ModelClient(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_token: Callable[[str], None],
        cancelled: Event,
    ) -> ModelReply: ...

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelReply: ...


class GroqModelClient:
    def __init__(self, settings: Settings) -> None:
        if settings.groq_api_key is None:
            raise AgentConfigurationError("TripMate AI service is not configured.")
        self._model = settings.groq_model
        self._client = Groq(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
        )

    @staticmethod
    def _retry_after(exc: APIError) -> int | None:
        response = getattr(exc, "response", None)
        value = response.headers.get("retry-after") if response is not None else None
        if value is None:
            return None
        try:
            seconds = int(value.strip())
        except (AttributeError, TypeError, ValueError):
            return None
        return seconds if 0 <= seconds <= 86400 else None

    @classmethod
    def _raise_provider_error(cls, exc: APIError) -> Never:
        if isinstance(exc, RateLimitError):
            error: AgentProviderError = ProviderRateLimitError(cls._retry_after(exc))
        elif isinstance(exc, APITimeoutError):
            error = ProviderTimeoutError("The AI provider timed out.")
        elif isinstance(exc, APIConnectionError):
            error = ProviderConnectionError("The AI provider could not be reached.")
        elif isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            error = ProviderAuthenticationError("The AI provider is not configured correctly.")
        elif isinstance(exc, APIStatusError) and exc.status_code >= 500:
            error = ProviderUnavailableError("The AI provider is temporarily unavailable.")
        elif isinstance(exc, (BadRequestError, APIStatusError)):
            error = ProviderRequestError("The AI provider rejected the request.")
        else:
            error = ProviderUnavailableError("The AI provider could not complete the request.")
        logger.error(
            "provider_error",
            extra={
                "event_fields": {
                    **correlation_fields(),
                    "provider_error_category": error.code,
                    "status_code": getattr(exc, "status_code", None),
                }
            },
        )
        raise error from None

    @staticmethod
    def _log_duration(started: float) -> None:
        logger.info(
            "provider_call_finished",
            extra={
                "event_fields": {
                    **correlation_fields(),
                    "provider_duration_ms": round((perf_counter() - started) * 1000, 2),
                }
            },
        )

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelReply:
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="auto",
                parallel_tool_calls=False,
                reasoning_format="hidden",
                temperature=0,
            )
        except APIError as exc:
            self._raise_provider_error(exc)
        finally:
            self._log_duration(started)
        if len(response.choices) != 1 or response.choices[0].finish_reason not in (
            "stop",
            "tool_calls",
        ):
            raise AgentProviderError("The AI provider returned an incomplete response.")
        message = response.choices[0].message
        return ModelReply(
            content=message.content,
            tool_calls=tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in (message.tool_calls or [])
            ),
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_token: Callable[[str], None],
        cancelled: Event,
    ) -> ModelReply:
        started = perf_counter()
        calls: dict[int, dict[str, str]] = {}
        content = ""
        pending = ""
        finish = None
        emitted = False
        try:
            with self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="auto",
                parallel_tool_calls=False,
                reasoning_format="hidden",
                temperature=0,
                stream=True,
            ) as stream:
                for chunk in stream:
                    if cancelled.is_set():
                        raise AgentProviderError("Request cancelled.")
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) != 1 or finish is not None:
                        raise AgentProviderError("The AI provider returned an invalid stream.")
                    choice = chunk.choices[0]
                    delta = choice.delta
                    # Reasoning fields are deliberately never read or forwarded.
                    for part in delta.tool_calls or []:
                        if part.index != 0 or emitted:
                            raise AgentProviderError(
                                "The AI provider returned mixed or parallel output."
                            )
                        call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                        call["id"] += part.id or ""
                        if part.function:
                            call["name"] += part.function.name or ""
                            call["arguments"] += part.function.arguments or ""
                        if sum(map(len, call.values())) > 16000:
                            raise AgentProviderError(
                                "The AI provider returned an oversized tool call."
                            )
                    if delta.content and not calls:
                        content += delta.content
                        pending += delta.content
                        # Hold incomplete markup across delta boundaries before checking tags.
                        if re.search(r"<\s*/?\s*(think|analysis|reasoning)\b", pending, re.I):
                            raise AgentProviderError(
                                "The AI provider returned no usable final answer."
                            )
                        cut = (
                            pending.rfind("<")
                            if pending.rfind("<") > pending.rfind(">")
                            else len(pending)
                        )
                        if cut:
                            on_token(pending[:cut])
                            emitted = True
                            pending = pending[cut:]
                    finish = choice.finish_reason
                if finish not in ("stop", "tool_calls") or bool(calls) != (finish == "tool_calls"):
                    raise AgentProviderError("The AI provider returned an incomplete response.")
                if pending and not calls:
                    on_token(pending)
        except APIError as exc:
            self._raise_provider_error(exc)
        finally:
            self._log_duration(started)
        return ModelReply(
            content=None if calls else content,
            tool_calls=tuple(ToolCall(**value) for value in calls.values()),
        )

    def close(self) -> None:
        self._client.close()
