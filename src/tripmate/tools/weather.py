import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tripmate.tools.weather_data import WEATHER_DATA

logger = logging.getLogger(__name__)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class UnsupportedDestinationError(ValueError):
    """City is missing or not supported by the mock weather dataset."""


class InvalidWeatherPeriodError(ValueError):
    """Period is not a supported English month or valid ISO calendar date."""


class WeatherForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    city: str = Field(min_length=1)
    month: str = Field(min_length=1)
    conditions: str = Field(min_length=1)
    temp_range_c: tuple[int, int]
    source: Literal["mock_monthly_climatology"] = "mock_monthly_climatology"

    @field_validator("temp_range_c")
    @classmethod
    def validate_temperature_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] > value[1]:
            raise ValueError("Minimum temperature must not exceed maximum temperature")
        return value


def _parse_month(period: str) -> int:
    if isinstance(period, str):
        normalized = period.strip().casefold()
        for number, name in enumerate(MONTH_NAMES, start=1):
            if normalized in (name.casefold(), name[:3].casefold()):
                return number
        parts = normalized.split("-")
        widths = (4, 2, 2) if len(parts) == 3 else (4, 2)
        if len(parts) == len(widths) and all(
            len(part) == width and part.isascii() and part.isdigit()
            for part, width in zip(parts, widths, strict=True)
        ):
            try:
                year, month = int(parts[0]), int(parts[1])
                day = int(parts[2]) if len(parts) == 3 else 1
                return date(year, month, day).month
            except ValueError:
                pass
    raise InvalidWeatherPeriodError(
        "Use an English month name/three-letter abbreviation, YYYY-MM, or valid YYYY-MM-DD"
    )


def get_weather_forecast(city: str, date_or_month: str) -> WeatherForecast:
    fields = {
        "city": city.strip()[:80] if isinstance(city, str) else None,
        "requested_period": date_or_month.strip()[:32] if isinstance(date_or_month, str) else None,
    }
    logger.info("weather_lookup", extra={"event_fields": fields})
    try:
        normalized_city = city.strip().casefold() if isinstance(city, str) else ""
        canonical = next(
            (name for name in WEATHER_DATA if name.casefold() == normalized_city), None
        )
        if canonical is None:
            raise UnsupportedDestinationError(
                f"Choose a supported weather destination: {', '.join(WEATHER_DATA)}"
            )
        month_number = _parse_month(date_or_month)
    except (UnsupportedDestinationError, InvalidWeatherPeriodError) as exc:
        logger.warning(
            "weather_lookup_error",
            extra={"event_fields": {**fields, "error_type": type(exc).__name__}},
        )
        raise
    conditions, temperatures = WEATHER_DATA[canonical][month_number - 1]
    forecast = WeatherForecast(
        city=canonical,
        month=MONTH_NAMES[month_number - 1],
        conditions=conditions,
        temp_range_c=temperatures,
    )
    logger.info(
        "weather_lookup_complete",
        extra={
            "event_fields": {
                **fields,
                "city": forecast.city,
                "resolved_month": forecast.month,
                "source": forecast.source,
            }
        },
    )
    return forecast
