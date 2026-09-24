"""OpenAI Chat Completions provider for grounded answers."""

import logging

from openai import AsyncOpenAI

from app.rag.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class OpenAILLMProvider:
    """Generate non-streaming answers through the official async SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An OpenAI API key is required for the LLM provider.")
        self.model = model
        self.temperature = temperature
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call Chat Completions without logging prompts or credentials."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMProviderError("The LLM returned an empty answer.")
            return content.strip()
        except LLMProviderError:
            raise
        except Exception as error:
            logger.exception("LLM provider request failed")
            raise LLMProviderError("The LLM provider request failed.") from error