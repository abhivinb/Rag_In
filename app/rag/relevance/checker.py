"""Relevance checker abstraction and structured LLM implementation."""

import json
from typing import Protocol, Sequence

from app.rag.exceptions import LLMProviderError
from app.rag.providers.base import LLMProvider
from app.rag.relevance.models import RelevanceCheckResult
from app.rag.relevance.prompts import build_relevance_prompt
from app.retrieval.models import HybridRetrievalResult


class RelevanceChecker(Protocol):
    """Check whether candidates are sufficient for a query."""

    async def check(
        self, query: str, candidates: Sequence[HybridRetrievalResult]
    ) -> RelevanceCheckResult:
        """Return a bounded structured relevance judgment."""


class LLMRelevanceChecker:
    """Parse a provider response as a strict relevance result."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def check(
        self, query: str, candidates: Sequence[HybridRetrievalResult]
    ) -> RelevanceCheckResult:
        context = "\n\n".join(
            f"Chunk ID: {candidate.chunk_id}\nContent:\n{candidate.content}"
            for candidate in candidates
        )
        system_prompt, user_prompt = build_relevance_prompt(query, context)
        try:
            raw = await self.provider.generate(system_prompt, user_prompt)
            payload = json.loads(raw)
            return RelevanceCheckResult.model_validate(payload)
        except Exception as error:
            raise LLMProviderError("Retrieval relevance checking failed.") from error