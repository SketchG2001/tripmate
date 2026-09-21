from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from tripmate.agent.models import MAX_MESSAGE_LENGTH


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, strict=True, extra="forbid")
    thread_id: UUID = Field(default_factory=uuid4, strict=False)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ErrorResponse(BaseModel):
    detail: str
    code: str
    retry_after_seconds: int | None = None
