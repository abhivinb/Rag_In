"""Bounded context construction from hybrid retrieval results."""

from collections.abc import Sequence

from app.rag.exceptions import ContextConstructionError
from app.retrieval.models import HybridRetrievalResult


class ContextBuilder:
    """Format complete retrieved chunks within a deterministic character budget."""

    def __init__(self, max_context_chars: int = 20_000) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero.")
        self.max_context_chars = max_context_chars

    def build(self, results: Sequence[HybridRetrievalResult]) -> str:
        """Build clearly delimited context without merging source content."""
        context, _ = self.build_with_sources(results)
        return context

    def build_with_sources(
        self, results: Sequence[HybridRetrievalResult]
    ) -> tuple[str, list[HybridRetrievalResult]]:
        """Build context and return exactly the results represented in it."""
        if not results:
            return "", []
        sections: list[str] = []
        selected: list[HybridRetrievalResult] = []
        used = 0
        for index, result in enumerate(results, start=1):
            section = self._format_section(result, index)
            separator = "\n\n" if sections else ""
            available = self.max_context_chars - used - len(separator)
            if available <= 0:
                break
            if len(section) > available:
                if not sections:
                    section = _bounded_section(section, available)
                else:
                    break
            sections.append(section)
            selected.append(result)
            used += len(separator) + len(section)
        if not selected:
            raise ContextConstructionError("Retrieved context could not fit the configured limit.")
        return "\n\n".join(sections), selected

    def select(
        self, results: Sequence[HybridRetrievalResult]
    ) -> list[HybridRetrievalResult]:
        """Select the deterministic set of chunks represented in the context."""
        return self.build_with_sources(results)[1]

    @staticmethod
    def _format_section(result: HybridRetrievalResult, index: int) -> str:
        return (
            f"[Source {index}]\n"
            f"Document ID: {result.document_id}\n"
            f"Chunk ID: {result.chunk_id}\n"
            f"Chunk Index: {result.chunk_index}\n"
            f"Metadata: {result.metadata}\n"
            "Content:\n"
            f"{result.content}\n"
            f"[/Source {index}]"
        )


def _bounded_section(section: str, limit: int) -> str:
    """Bound one oversized source without silently cutting its middle."""
    marker = "\n[Source content truncated to context limit.]"
    if limit <= len(marker):
        raise ContextConstructionError("Context limit is too small for a source boundary.")
    return section[: limit - len(marker)] + marker