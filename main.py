"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings)

    application = FastAPI(title=settings.app_name)
    application.include_router(router)
    return application


app = create_app()
