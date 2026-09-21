import sqlite3
from functools import cache
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

from tripmate.agent.client import GroqModelClient
from tripmate.agent.core import TripMateAgent
from tripmate.agent.models import AgentConfigurationError, AgentResult
from tripmate.agent.persistence import checkpoint_serializer
from tripmate.config import Settings
from tripmate.rag.embeddings import SentenceTransformerEmbedder
from tripmate.rag.loader import load_destination_chunks
from tripmate.rag.retriever import DestinationRetriever

__all__ = ["AgentResult", "TripMateAgent", "build_tripmate_agent"]


def build_tripmate_agent(settings: Settings) -> TripMateAgent:
    if settings.groq_api_key is None:
        raise AgentConfigurationError("TripMate AI service is not configured.")
    chunks = load_destination_chunks(settings.destination_data_dir)

    initialization_lock = Lock()

    @cache
    def retriever() -> DestinationRetriever:
        return DestinationRetriever(
            chunks, SentenceTransformerEmbedder(settings.embedding_model), top_k=settings.rag_top_k
        )

    def search(query: str) -> list[str]:
        with initialization_lock:
            service = retriever()
        return service.search_destination_guide(query)

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(connection, serde=checkpoint_serializer())
        checkpointer.setup()
        client = GroqModelClient(settings)

        def close() -> None:
            try:
                client.close()
            finally:
                connection.close()

        try:
            return TripMateAgent(
                client,
                search,
                sorted({chunk.city for chunk in chunks}),
                max_tool_calls=settings.agent_max_tool_calls,
                close_client=close,
                checkpointer=checkpointer,
            )
        except BaseException:
            client.close()
            raise
    except BaseException:
        connection.close()
        raise
