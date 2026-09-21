from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

VectorArray = NDArray[np.float64]


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> VectorArray: ...

    def encode_query(self, query: str) -> VectorArray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, local_files_only: bool = False) -> None:
        # Importing TripMate never imports torch or downloads/loads a model.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name, device="cpu", local_files_only=local_files_only
        )
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("Embedding model must expose a positive vector dimension")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> VectorArray:
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    def encode_query(self, query: str) -> VectorArray:
        return self.encode_documents([query])[0]
