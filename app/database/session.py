"""Async SQLAlchemy engine and session helpers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.database.exceptions import DatabaseConfigurationError, DatabaseConnectionError
from app.database.models import VECTOR_DIMENSION


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create an async engine after checking schema/vector dimension compatibility."""
    current_settings = settings or get_settings()
    if current_settings.embedding_dimension != VECTOR_DIMENSION:
        raise DatabaseConfigurationError(
            f"Database vector schema requires dimension {VECTOR_DIMENSION}; "
            f"received {current_settings.embedding_dimension}."
        )
    return create_async_engine(current_settings.async_database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the application async session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def check_database_connection(engine: AsyncEngine) -> None:
    """Verify PostgreSQL connectivity and surface a safe application error."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as error:
        raise DatabaseConnectionError("Database connectivity check failed.") from error


async def enable_pgvector(engine: AsyncEngine) -> None:
    """Enable the pgvector extension for local initialization workflows."""
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as error:
        raise DatabaseConnectionError("Unable to enable the pgvector extension.") from error


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session whose caller controls the transaction boundary."""
    async with session_factory() as session:
        yield session