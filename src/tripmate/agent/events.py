from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tripmate.agent.models import AgentResult, ToolName


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StartEvent(StreamEvent):
    type: Literal["start"] = "start"
    thread_id: UUID


class ToolCallEvent(StreamEvent):
    type: Literal["tool_call"] = "tool_call"
    tool: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)


class ToolResultEvent(StreamEvent):
    type: Literal["tool_result"] = "tool_result"
    tool: ToolName | None = None
    status: Literal["success", "error"]


class TokenEvent(StreamEvent):
    type: Literal["token"] = "token"
    content: str


class DoneEvent(StreamEvent):
    type: Literal["done"] = "done"
    result: AgentResult


class ErrorEvent(StreamEvent):
    type: Literal["error"] = "error"
    message: str = "TripMate could not complete the request."
    code: Literal[
        "rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "service_unavailable",
        "internal_error",
    ] = "internal_error"
    retry_after_seconds: int | None = None
