"""Provider abstraction for answer generation."""

from typing import Protocol


class LLMProvider(Protocol):
    """Minimal async interface consumed by RAGService."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one complete answer."""