import json
import logging
from collections.abc import AsyncIterator
from threading import Event
from time import perf_counter

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive

from tripmate.agent.events import ErrorEvent
from tripmate.agent.models import (
    AgentConfigurationError,
    AgentProviderError,
    AgentResult,
    InvalidAgentInputError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from tripmate.api.schemas import ChatRequest, ErrorResponse, HealthResponse
from tripmate.observability import bind_request, reset_request

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request) -> HealthResponse:
    return HealthResponse(service=request.app.state.settings.app_name)


def _error_details(exc: Exception) -> tuple[int, ErrorResponse]:
    if isinstance(exc, ProviderRateLimitError):
        return 429, ErrorResponse(
            detail="TripMate is temporarily rate limited. Please try again shortly.",
            code="rate_limited",
            retry_after_seconds=exc.retry_after_seconds,
        )
    if isinstance(exc, ProviderTimeoutError):
        return 502, ErrorResponse(
            detail="TripMate's AI service timed out. Please try again.", code="provider_timeout"
        )
    if isinstance(exc, (ProviderConnectionError, ProviderUnavailableError, ProviderRequestError)):
        return 502, ErrorResponse(
            detail="TripMate could not reach its AI service. Please try again.",
            code="provider_unavailable",
        )
    if isinstance(exc, (AgentConfigurationError, ProviderAuthenticationError)):
        return 503, ErrorResponse(
            detail="TripMate AI service is not currently available.", code="service_unavailable"
        )
    if isinstance(exc, InvalidAgentInputError):
        return 422, ErrorResponse(
            detail="Please provide a valid travel question.", code="invalid_request"
        )
    if isinstance(exc, AgentProviderError):
        return 502, ErrorResponse(
            detail="TripMate could not complete the AI request.", code="provider_unavailable"
        )
    return 500, ErrorResponse(
        detail="TripMate could not complete that request.", code="internal_error"
    )


def _error_response(exc: Exception, request_id: str) -> JSONResponse:
    status, error = _error_details(exc)
    headers = {"X-Request-ID": request_id}
    if isinstance(exc, ProviderRateLimitError) and exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(content=error.model_dump(mode="json"), status_code=status, headers=headers)


def _stream_error(exc: Exception) -> ErrorEvent:
    _, error = _error_details(exc)
    allowed = {
        "rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "service_unavailable",
        "internal_error",
    }
    code = error.code if error.code in allowed else "internal_error"
    return ErrorEvent(
        code=code,
        message=error.detail,
        retry_after_seconds=error.retry_after_seconds,
    )


@router.post("/chat", response_model=AgentResult, tags=["chat"])
def chat(body: ChatRequest, request: Request) -> AgentResult | JSONResponse:
    started = perf_counter()
    request_id = str(request.state.request_id)
    fields = {"request_id": request_id, "thread_id": str(body.thread_id)}
    tokens = bind_request(request.state.request_id, body.thread_id)
    logger.info(
        "chat_request_received",
        extra={"event_fields": {**fields, "message_length": len(body.message)}},
    )
    try:
        agent = request.app.state.tripmate_agent
        if agent is None:
            raise AgentConfigurationError("Agent unavailable")
        result = AgentResult.model_validate(agent.invoke(body.message, thread_id=body.thread_id))
    except Exception as exc:
        status, error = _error_details(exc)
        if not isinstance(
            exc, (AgentProviderError, AgentConfigurationError, InvalidAgentInputError)
        ):
            logger.error("chat_unexpected_error", extra={"event_fields": fields})
        logger.warning(
            "chat_request_failed",
            extra={
                "event_fields": {
                    **fields,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "http_status": status,
                    "provider_error_category": error.code
                    if isinstance(exc, AgentProviderError)
                    else None,
                }
            },
        )
        return _error_response(exc, request_id)
    else:
        logger.info(
            "chat_request_completed",
            extra={
                "event_fields": {
                    **fields,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "tools_used": result.tools_used,
                    "agent_status": result.status,
                    "http_status": 200,
                }
            },
        )
        return result
    finally:
        reset_request(tokens)


class ChatStreamingResponse(StreamingResponse):
    def __init__(self, content: AsyncIterator[str], cancelled: Event) -> None:
        self.cancelled = cancelled
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def listen_for_disconnect(self, receive: Receive) -> None:
        try:
            await super().listen_for_disconnect(receive)
        finally:
            self.cancelled.set()


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(body: ChatRequest, request: Request) -> Response:
    agent = request.app.state.tripmate_agent
    if agent is None:
        return _error_response(
            AgentConfigurationError("Agent unavailable"), str(request.state.request_id)
        )

    cancelled = Event()

    async def events() -> AsyncIterator[str]:
        iterator = agent.stream(body.message, thread_id=body.thread_id, cancelled=cancelled)
        sentinel = object()
        outcome = "stream_cancelled"
        started = perf_counter()
        fields = {
            "request_id": str(request.state.request_id),
            "thread_id": str(body.thread_id),
        }
        tools_used: list[str] = []
        agent_status = None
        logger.info("stream_started", extra={"event_fields": fields})

        def next_event():
            tokens = bind_request(request.state.request_id, body.thread_id)
            try:
                return next(iterator, sentinel)
            finally:
                reset_request(tokens)

        async def watch_disconnect() -> None:
            while not cancelled.is_set():
                if await request.is_disconnected():
                    cancelled.set()
                    return
                await anyio.sleep(0.1)

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(watch_disconnect)
                try:
                    while not cancelled.is_set():
                        event = await run_in_threadpool(next_event)
                        if event is sentinel or cancelled.is_set():
                            break
                        if event.type == "done":
                            outcome = "stream_completed"
                            tools_used = event.result.tools_used
                            agent_status = event.result.status
                        yield (
                            "data: "
                            + json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
                            + "\n\n"
                        )
                except Exception as exc:
                    outcome = "stream_failed"
                    error = _stream_error(exc)
                    logger.error(
                        "stream_failed",
                        extra={
                            "event_fields": {
                                **fields,
                                "provider_error_category": error.code
                                if isinstance(exc, AgentProviderError)
                                else None,
                            }
                        },
                    )
                    yield "data: " + error.model_dump_json() + "\n\n"
                finally:
                    cancelled.set()
                    group.cancel_scope.cancel()
        finally:
            cancelled.set()
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(iterator.close)
            logger.info(
                outcome,
                extra={
                    "event_fields": {
                        **fields,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "tools_used": tools_used,
                        "agent_status": agent_status,
                    }
                },
            )

    return ChatStreamingResponse(events(), cancelled)
