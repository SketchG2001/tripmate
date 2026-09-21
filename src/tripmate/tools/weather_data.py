from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

MonthlyProfile: TypeAlias = tuple[str, tuple[int, int]]

WEATHER_DATA: Mapping[str, tuple[MonthlyProfile, ...]] = MappingProxyType(
    {
        "Bangkok": (
            ("Warm, mostly dry, with relatively lower humidity", (22, 32)),
            ("Hot, mostly dry, with increasing humidity", (24, 33)),
            ("Hot and humid, with occasional showers", (26, 35)),
            ("Very hot and humid, with possible thunderstorms", (27, 36)),
            ("Hot and humid, with frequent showers", (26, 35)),
            ("Hot and humid, with monsoon-like showers", (26, 34)),
            ("Hot, humid, and rainy, with monsoon-like conditions", (26, 33)),
            ("Hot and humid, with frequent heavy showers", (26, 33)),
            ("Hot and humid, with frequent heavy rain", (25, 32)),
            ("Hot and humid, with rain easing later in the month", (25, 32)),
            ("Warm, with decreasing rain and humidity", (24, 32)),
            ("Warm and generally dry", (22, 31)),
        ),
        "Barcelona": (
            ("Cool, with occasional rain", (5, 14)),
            ("Cool, with occasional rain and mild afternoons", (6, 15)),
            ("Mild days and cool evenings, with occasional showers", (8, 17)),
            ("Mild, with a mix of sunshine and showers", (10, 19)),
            ("Warm and often sunny, with occasional showers", (14, 23)),
            ("Warm to hot and generally dry", (18, 27)),
            ("Hot, sunny, and generally dry", (21, 30)),
            ("Hot and generally dry, with occasional thunderstorms", (22, 30)),
            ("Warm, with occasional rain or thunderstorms", (18, 26)),
            ("Mild to warm, with more frequent showers", (14, 22)),
            ("Cool to mild, with occasional rain", (9, 17)),
            ("Cool, with occasional rain and chilly evenings", (6, 14)),
        ),
        "Reykjavik": (
            ("Cold and windy, with possible rain or snow", (-3, 3)),
            ("Cold and windy, with possible snow and icy conditions", (-3, 3)),
            ("Cold and changeable, with possible rain or snow", (-2, 4)),
            ("Chilly and changeable, with rain or occasional snow", (0, 6)),
            ("Cool and breezy, with intermittent rain", (3, 10)),
            ("Cool to mild, with variable skies and occasional rain", (7, 13)),
            ("Mild and changeable, with occasional rain", (9, 15)),
            ("Mild and breezy, with occasional rain", (8, 14)),
            ("Cool and windy, with frequent showers", (5, 11)),
            ("Chilly and windy, with rain and possible snow", (2, 7)),
            ("Cold and windy, with possible rain or snow", (-1, 4)),
            ("Cold and windy, with possible rain or snow", (-2, 4)),
        ),
        "Tokyo": (
            ("Cool and generally dry, with chilly nights", (2, 10)),
            ("Cool and mostly dry, with occasional rain or snow", (3, 11)),
            ("Cool to mild, with occasional rain", (6, 14)),
            ("Mild, with a mix of sunshine and showers", (11, 19)),
            ("Warm, with occasional showers", (16, 24)),
            ("Warm and humid, with frequent seasonal rain", (20, 27)),
            ("Hot and humid, with showers and possible thunderstorms", (24, 31)),
            ("Hot and humid, with possible heavy rain and storms", (25, 33)),
            ("Warm and humid, with possible heavy rain and storms", (21, 28)),
            ("Mild, with occasional rain", (15, 22)),
            ("Cool to mild and often dry", (9, 17)),
            ("Cool and generally dry, with chilly nights", (4, 12)),
        ),
    }
)
