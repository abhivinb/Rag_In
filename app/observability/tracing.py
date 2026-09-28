"""LangSmith tracing backend and safe span lifecycle."""

import logging
from collections.abc import Mapping
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TraceSpan(Protocol):
    def end(self, *, outputs: dict[str, Any] | None = None, error: BaseException | None = None) -> None:
        """Complete a trace span."""


class TraceBackend(Protocol):
    def start(
        self,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        tags: list[str],
        metadata: Mapping[str, Any],
    ) -> TraceSpan:
        """Start a trace span."""


class LangSmithSpan:
    """Safe handle around LangSmith's official trace context manager."""

    def __init__(self, context_manager, tracing_context_manager, run) -> None:
        self._context_manager = context_manager
        self._tracing_context = tracing_context_manager
        self._run = run
        self._finished = False

    def end(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._finished:
            return
        try:
            if error is None:
                self._run.end(outputs=outputs or {})
            else:
                self._run.end(error=type(error).__name__)
        finally:
            self._finished = True
            try:
                self._context_manager.__exit__(None, None, None)
            finally:
                self._tracing_context.__exit__(None, None, None)


class LangSmithBackend:
    """Create LangSmith trace spans using an injected SDK client."""

    def __init__(self, client, project_name: str) -> None:
        self.client = client
        self.project_name = project_name

    def start(
        self,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        tags: list[str],
        metadata: Mapping[str, Any],
    ) -> LangSmithSpan:
        from langsmith import trace

        from langsmith import tracing_context

        tracing_scope = tracing_context(
            enabled=True,
            project_name=self.project_name,
            client=self.client,
        )
        tracing_scope.__enter__()
        try:
            context_manager = trace(
                name,
                run_type=run_type,
                inputs=inputs,
                project_name=self.project_name,
                tags=tags,
                metadata=dict(metadata),
                client=self.client,
            )
            run = context_manager.__enter__()
            return LangSmithSpan(context_manager, tracing_scope, run)
        except Exception:
            tracing_scope.__exit__(None, None, None)
            raise