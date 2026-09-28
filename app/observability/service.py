"""Failure-isolated tracing service for the RAG pipeline."""

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from app.core.config import Settings
from app.observability.tracing import LangSmithBackend, TraceBackend

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RAGObservability:
    """Optional trace wrapper that never controls application behavior."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        backend: TraceBackend | None = None,
        environment: str = "development",
        project: str = "enterprise-rag",
        pipeline_version: str = "phase-11",
    ) -> None:
        self.enabled = enabled and backend is not None
        self.backend = backend
        self.environment = environment
        self.project = project
        self.pipeline_version = pipeline_version

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGObservability":
        """Build a LangSmith backend only when explicitly enabled and credentialed."""
        active = settings.observability_enabled and settings.langsmith_tracing
        if not active:
            return cls(environment=settings.app_env, project=settings.langsmith_project or "enterprise-rag")
        if not settings.langsmith_api_key or not settings.langsmith_api_key.get_secret_value():
            logger.warning("LangSmith tracing requested but API key is unavailable; tracing disabled")
            return cls(environment=settings.app_env, project=settings.langsmith_project or "enterprise-rag")
        try:
            from langsmith import Client

            client = Client(
                api_key=settings.langsmith_api_key.get_secret_value(),
                api_url=settings.langsmith_endpoint or None,
            )
            backend = LangSmithBackend(
                client, settings.langsmith_project or "enterprise-rag"
            )
            return cls(
                enabled=True,
                backend=backend,
                environment=settings.app_env,
                project=settings.langsmith_project or "enterprise-rag",
            )
        except Exception:
            logger.warning("LangSmith initialization failed; tracing disabled")
            return cls(environment=settings.app_env, project=settings.langsmith_project or "enterprise-rag")

    async def run(
        self,
        *,
        name: str,
        operation: Callable[[], Awaitable[T]],
        inputs: dict[str, Any],
        output: Callable[[T, float], dict[str, Any]],
        stage: str,
        run_type: str = "chain",
    ) -> T:
        """Run an operation unchanged, recording a safe summary if tracing works."""
        if not self.enabled or self.backend is None:
            return await operation()
        metadata = {
            "environment": self.environment,
            "pipeline_version": self.pipeline_version,
            "retrieval_mode": "hybrid",
            "stage": stage,
            "project": self.project,
        }
        tags = ["rag", "phase-11", "production-rag"]
        if stage not in tags:
            tags.append(stage)
        started = time.perf_counter()
        try:
            span = self.backend.start(
                name=name,
                run_type=run_type,
                inputs=inputs,
                tags=tags,
                metadata=metadata,
            )
        except Exception:
            logger.warning("LangSmith span creation failed", exc_info=True)
            return await operation()
        try:
            result = await operation()
        except BaseException as error:
            try:
                span.end(error=error)
            except Exception:
                logger.warning("LangSmith span completion failed", exc_info=True)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            span.end(outputs=output(result, elapsed_ms))
        except Exception:
            logger.warning("LangSmith span completion failed", exc_info=True)
        return result