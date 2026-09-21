from contextvars import ContextVar, Token
from uuid import UUID

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)


def bind_request(request_id: UUID, thread_id: UUID) -> tuple[Token, Token]:
    return _request_id.set(str(request_id)), _thread_id.set(str(thread_id))


def reset_request(tokens: tuple[Token, Token]) -> None:
    request_token, thread_token = tokens
    _request_id.reset(request_token)
    _thread_id.reset(thread_token)


def correlation_fields() -> dict[str, str]:
    fields = {}
    if request_id := _request_id.get():
        fields["request_id"] = request_id
    if thread_id := _thread_id.get():
        fields["thread_id"] = thread_id
    return fields
