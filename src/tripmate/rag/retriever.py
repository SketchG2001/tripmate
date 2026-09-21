import logging
import re
from collections.abc import Sequence

import numpy as np

from tripmate.config import Settings
from tripmate.rag.embeddings import Embedder, SentenceTransformerEmbedder, VectorArray
from tripmate.rag.loader import load_destination_chunks
from tripmate.rag.models import DestinationChunk, InvalidQueryError, RetrievalResult

logger = logging.getLogger(__name__)


def _validate_top_k(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("top_k must be a positive integer")


def _normalize(vectors: VectorArray, shape: tuple[int, int]) -> VectorArray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Embeddings must be finite with shape {shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embedding vectors must have finite nonzero norms")
    return array / norms


class DestinationRetriever:
    def __init__(
        self, chunks: Sequence[DestinationChunk], embedder: Embedder, *, top_k: int
    ) -> None:
        _validate_top_k(top_k)
        if not chunks:
            raise ValueError("Cannot build an empty destination index")
        self._chunks = tuple(chunks)
        self._embedder = embedder
        self._top_k = top_k
        self._dimension = embedder.dimension
        if self._dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._cities = tuple(sorted({chunk.city for chunk in self._chunks}))
        self._patterns = {
            city: re.compile(rf"(?<!\w){re.escape(city)}(?!\w)", re.IGNORECASE)
            for city in self._cities
        }
        self._vectors = _normalize(
            embedder.encode_documents([chunk.as_text() for chunk in self._chunks]),
            (len(self._chunks), self._dimension),
        )
        self._vectors.flags.writeable = False
        logger.info(
            "destination_index_built",
            extra={
                "event_fields": {"chunk_count": len(self._chunks), "dimension": self._dimension}
            },
        )

    @property
    def supported_destinations(self) -> tuple[str, ...]:
        return self._cities

    def supports_destination(self, city: str) -> bool:
        return city.strip().casefold() in {name.casefold() for name in self._cities}

    def detect_destinations(self, query: str) -> tuple[str, ...]:
        return tuple(city for city, pattern in self._patterns.items() if pattern.search(query))

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Query must be a nonempty string")
        query = query.strip()
        limit = self._top_k if top_k is None else top_k
        _validate_top_k(limit)
        cities = self.detect_destinations(query)
        fields = {"query_length": len(query), "top_k": limit, "city_filter": cities}
        logger.info("destination_search", extra={"event_fields": fields})
        candidates = np.array(
            [i for i, chunk in enumerate(self._chunks) if not cities or chunk.city in cities]
        )
        vector = _normalize(
            np.asarray(self._embedder.encode_query(query))[None, :], (1, self._dimension)
        )[0]
        scores = np.clip(self._vectors[candidates] @ vector, -1.0, 1.0)
        # Stable ties follow the loader's filename/section order.
        order = np.argsort(-scores, kind="stable")[:limit]
        results = [
            RetrievalResult(self._chunks[int(candidates[i])], float(scores[i])) for i in order
        ]
        logger.info(
            "destination_search_complete",
            extra={
                "event_fields": {
                    **fields,
                    "result_sections": [result.chunk.section for result in results],
                    "result_scores": [result.score for result in results],
                }
            },
        )
        return results

    def search_destination_guide(self, query: str) -> list[str]:
        """Business/tool contract bound to this explicitly constructed service."""
        return [result.chunk.as_text() for result in self.search(query)]


def build_destination_retriever(
    settings: Settings, *, embedder: Embedder | None = None, local_files_only: bool = False
) -> DestinationRetriever:
    chunks = load_destination_chunks(settings.destination_data_dir)
    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model, local_files_only=local_files_only
        )
    return DestinationRetriever(chunks, embedder, top_k=settings.rag_top_k)
