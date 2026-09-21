from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

MAX_MESSAGE_LENGTH = 4000
ToolName = Literal["search_destination_guide", "get_weather_forecast"]


class AgentConfigurationError(RuntimeError):
    """The agent cannot be constructed with the supplied configuration."""


class AgentProviderError(RuntimeError):
    """The model service failed or returned an unusable response."""

    code = "provider_unavailable"


class ProviderRateLimitError(AgentProviderError):
    code = "rate_limited"

    def __init__(self, retry_after_seconds: int | None = None) -> None:
        super().__init__("The AI provider is temporarily rate limited.")
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeoutError(AgentProviderError):
    code = "provider_timeout"


class ProviderConnectionError(AgentProviderError):
    code = "provider_unavailable"


class ProviderAuthenticationError(AgentProviderError):
    code = "service_unavailable"


class ProviderRequestError(AgentProviderError):
    code = "provider_request_failed"


class ProviderUnavailableError(AgentProviderError):
    code = "provider_unavailable"


class InvalidAgentInputError(ValueError):
    """A user message is missing or too long."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelReply:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: Literal[
        "agent_started", "tool_call", "tool_result", "tool_error", "limit_reached", "agent_complete"
    ]
    tool: ToolName | None = None
    arguments: dict[str, str] = Field(default_factory=dict)
    message: str | None = None


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thread_id: UUID = Field(default_factory=uuid4)
    answer: str
    tools_used: list[ToolName] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    status: Literal["completed", "limit_reached"] = "completed"
