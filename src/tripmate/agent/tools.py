from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tripmate.agent.models import MAX_MESSAGE_LENGTH, ToolCall, ToolName
from tripmate.rag.models import InvalidQueryError
from tripmate.tools import get_weather_forecast
from tripmate.tools.weather import InvalidWeatherPeriodError, UnsupportedDestinationError


class GuideArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class WeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    city: str = Field(min_length=1, max_length=80)
    date_or_month: str = Field(min_length=1, max_length=32)


class ToolInputError(ValueError):
    """A model-selected tool or its arguments are not allowed."""


class AgentTools:
    def __init__(
        self, destination_search: Callable[[str], list[str]], cities: Sequence[str]
    ) -> None:
        self._destination_search = destination_search
        self._cities = tuple(cities)

    def schemas(self) -> list[dict[str, Any]]:
        supported = ", ".join(self._cities)
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_destination_guide",
                    "description": (
                        "Retrieve source sections about entry, seasons, customs, packing "
                        f"or safety. Only for {supported}. Include the explicit supported "
                        "city in the query; never invent a city."
                    ),
                    "parameters": GuideArguments.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": (
                        f"Look up mock monthly seasonal conditions for {supported}. "
                        "Not live weather. Requires a city and English month/abbreviation, "
                        "YYYY-MM or YYYY-MM-DD. Ask for missing inputs."
                    ),
                    "parameters": WeatherArguments.model_json_schema(),
                },
            },
        ]

    def validate(self, call: ToolCall) -> tuple[ToolName, dict[str, str]]:
        try:
            if call.name == "search_destination_guide":
                return "search_destination_guide", GuideArguments.model_validate_json(
                    call.arguments
                ).model_dump()
            if call.name == "get_weather_forecast":
                return "get_weather_forecast", WeatherArguments.model_validate_json(
                    call.arguments
                ).model_dump()
        except ValidationError:
            raise ToolInputError(
                "Invalid tool arguments. Use exactly the fields and types in the tool schema."
            ) from None
        raise ToolInputError("Unknown tool. Only the provided tools may be called.")

    def execute(self, name: ToolName, arguments: dict[str, str]) -> list[str] | dict[str, Any]:
        if name == "search_destination_guide":
            return self._destination_search(arguments["query"])
        if name == "get_weather_forecast":
            return get_weather_forecast(**arguments).model_dump(mode="json")
        raise ToolInputError("Unknown tool.")


DOMAIN_ERRORS = (InvalidQueryError, InvalidWeatherPeriodError, UnsupportedDestinationError)
