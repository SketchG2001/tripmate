from dataclasses import dataclass


class DestinationDataError(ValueError):
    """Destination data is absent or malformed."""


class InvalidQueryError(ValueError):
    """A query is not a nonempty string."""


@dataclass(frozen=True, slots=True)
class DestinationChunk:
    city: str
    section: str
    source: str
    content: str

    def as_text(self) -> str:
        return f"{self.city} / {self.section} [{self.source}]\n{self.content}"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk: DestinationChunk
    score: float
