import logging
import re
from pathlib import Path

from tripmate.rag.models import DestinationChunk, DestinationDataError

logger = logging.getLogger(__name__)
TITLE = re.compile(r"DESTINATION GUIDE:\s*([^,\n]+),\s*([^\n]+)")


def _parse(text: str, source: str) -> list[DestinationChunk]:
    lines = text.strip().splitlines()
    if not lines:
        raise DestinationDataError(f"{source}: empty destination document")
    title = TITLE.fullmatch(lines[0].strip())
    if title is None:
        raise DestinationDataError(f"{source}: expected DESTINATION GUIDE: city, country title")
    city = title[1].strip().title()
    if not city or not title[2].strip():
        raise DestinationDataError(f"{source}: destination city and country must be nonempty")
    chunks: list[DestinationChunk] = []
    heading: str | None = None
    body: list[str] = []
    seen: set[str] = set()

    def finish_section() -> None:
        if heading is None:
            return
        content = "\n".join(body).strip()
        if not content:
            raise DestinationDataError(f"{source}: empty section {heading}")
        chunks.append(DestinationChunk(city, heading, source, content))

    for raw in lines[1:]:
        line = raw.strip()
        # The supplied guides use standalone uppercase headings. Preserve body text.
        if line and line.isupper() and re.fullmatch(r"[A-Z][A-Z &/\-]*", line):
            finish_section()
            if line in seen:
                raise DestinationDataError(f"{source}: duplicate section {line}")
            seen.add(line)
            heading, body = line, []
        elif heading is not None:
            body.append(raw)
        elif line and not line.startswith("Note:"):
            raise DestinationDataError(f"{source}: unexpected text before first section")
    finish_section()
    if not chunks:
        raise DestinationDataError(f"{source}: no meaningful sections found")
    return chunks


def load_destination_chunks(directory: Path) -> tuple[DestinationChunk, ...]:
    try:
        if not directory.is_dir():
            raise DestinationDataError("Destination directory does not exist or is not a directory")
        files = sorted(directory.glob("*.txt"))
        if not files:
            raise DestinationDataError("Destination directory contains no .txt files")
        chunks: list[DestinationChunk] = []
        cities: set[str] = set()
        for path in files:
            document = _parse(path.read_text(encoding="utf-8-sig"), path.name)
            city = document[0].city.casefold()
            if city in cities:
                raise DestinationDataError(f"{path.name}: duplicate destination identity")
            cities.add(city)
            chunks.extend(document)
    except (OSError, UnicodeError) as exc:
        logger.error(
            "destination_data_error", extra={"event_fields": {"reason": type(exc).__name__}}
        )
        raise DestinationDataError("Cannot read destination documents as UTF-8") from exc
    except DestinationDataError:
        logger.error("destination_data_error")
        raise
    logger.info(
        "destination_documents_loaded",
        extra={"event_fields": {"document_count": len(files), "chunk_count": len(chunks)}},
    )
    return tuple(chunks)
