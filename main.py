"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.database.session import create_engine, create_session_factory
from app.security.middleware import add_security_middleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        engine = create_engine(settings)
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        application.state.rag_service = None
        application.state.document_indexing_service = None
        try:
            yield
        finally:
            await engine.dispose()

    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    add_security_middleware(application, settings)
    application.include_router(router)
    return application


app = create_app()
