"""OpenAI provider contract tests with a fake async client."""

import pytest

from app.rag.providers.openai import OpenAILLMProvider


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            content = "  grounded answer  "

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


@pytest.mark.asyncio
async def test_openai_provider_uses_separate_prompts_and_strips_answer() -> None:
    client = FakeClient()
    provider = OpenAILLMProvider(
        api_key="test-key",
        model="gpt-test",
        temperature=0.0,
        client=client,
    )

    answer = await provider.generate("system", "user")

    call = client.chat.completions.calls[0]
    assert answer == "grounded answer"
    assert call["model"] == "gpt-test"
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]