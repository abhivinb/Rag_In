"""Query rewriting abstractions and LLM implementation."""

from typing import Protocol

from app.rag.exceptions import LLMProviderError
from app.rag.providers.base import LLMProvider
from app.rag.query.models import QueryRewriteResult
from app.rag.query.prompts import build_query_rewrite_prompt


class QueryRewriter(Protocol):
    """Rewrite one query without answering it."""

    async def rewrite(self, query: str) -> str:
        """Return a retrieval-oriented query."""


class LLMQueryRewriter:
    """Use the existing LLM provider abstraction for one rewrite call."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def rewrite(self, query: str) -> str:
        system_prompt, user_prompt = build_query_rewrite_prompt(query)
        try:
            rewritten = await self.provider.generate(system_prompt, user_prompt)
        except Exception as error:
            raise LLMProviderError("Query rewriting failed.") from error
        rewritten = rewritten.strip()
        if not rewritten:
            raise LLMProviderError("Query rewriting returned an empty query.")
        return QueryRewriteResult(
            original_query=query, rewritten_query=rewritten
        ).rewritten_query