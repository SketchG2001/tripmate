import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tripmate.agent import TripMateAgent, build_tripmate_agent
from tripmate.api.routes import router
from tripmate.config import Settings, get_settings
from tripmate.logging_config import configure_logging

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in {"/api/chat", "/api/chat/stream"}:
            await self.app(scope, receive, send)
            return
        supplied = next(
            (
                value.decode("latin-1")
                for name, value in scope["headers"]
                if name == b"x-request-id"
            ),
            None,
        )
        try:
            request_id = UUID(supplied) if supplied is not None else uuid4()
        except (TypeError, ValueError):
            request_id = uuid4()
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                headers = [item for item in headers if item[0].lower() != b"x-request-id"]
                headers.append((b"x-request-id", str(request_id).encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def create_app(
    settings: Settings | None = None,
    *,
    agent_factory: Callable[[Settings], TripMateAgent] | None = None,
) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    factory = agent_factory if agent_factory is not None else build_tripmate_agent

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(app.state.settings.log_level)
        try:
            app.state.tripmate_agent = await run_in_threadpool(factory, app.state.settings)
        except Exception as exc:
            logger.error(
                "agent_initialization_failed",
                extra={
                    "event_fields": {
                        "error_type": type(exc).__name__,
                    }
                },
            )
        logger.info("application_started")
        try:
            yield
        finally:
            agent = app.state.tripmate_agent
            app.state.tripmate_agent = None
            if agent is not None:
                await run_in_threadpool(agent.close)
            logger.info("application_stopped")

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.tripmate_agent = None
    app.add_middleware(RequestIDMiddleware)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
