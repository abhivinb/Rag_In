"""LLM judge factories for offline DeepEval runs."""

from typing import Any

from app.core.config import Settings


class OpenAICompatibleJudge:
    """DeepEval model adapter for Groq and compatible chat-completions APIs."""

    def __init__(self, *, model: str, api_key: str, base_url: str, temperature: float = 0.0) -> None:
        from deepeval.models import DeepEvalBaseLLM

        class JudgeModel(DeepEvalBaseLLM):
            def __init__(inner_self) -> None:
                inner_self.api_key = api_key
                inner_self.base_url = base_url.rstrip("/")
                inner_self.temperature = temperature
                inner_self.async_client = None
                super(JudgeModel, inner_self).__init__(model)

            def load_model(inner_self):
                from openai import OpenAI

                return OpenAI(api_key=inner_self.api_key, base_url=inner_self.base_url)

            def get_model_name(inner_self, *args, **kwargs) -> str:
                return inner_self.name

            def generate(inner_self, prompt: str, schema: Any = None):
                request: dict[str, Any] = {
                    "model": inner_self.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": inner_self.temperature,
                }
                if schema is not None:
                    request["response_format"] = {"type": "json_object"}
                response = inner_self.model.chat.completions.create(**request)
                content = response.choices[0].message.content or ""
                if schema is not None:
                    return schema.model_validate_json(content)
                return content

            async def a_generate(inner_self, prompt: str, schema: Any = None):
                if inner_self.async_client is None:
                    from openai import AsyncOpenAI

                    inner_self.async_client = AsyncOpenAI(
                        api_key=inner_self.api_key,
                        base_url=inner_self.base_url,
                    )
                request: dict[str, Any] = {
                    "model": inner_self.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": inner_self.temperature,
                }
                if schema is not None:
                    request["response_format"] = {"type": "json_object"}
                response = await inner_self.async_client.chat.completions.create(**request)
                content = response.choices[0].message.content or ""
                if schema is not None:
                    return schema.model_validate_json(content), 0.0
                return content, 0.0

            def supports_json_mode(inner_self) -> bool:
                return True

        self.model = JudgeModel()


def evaluation_judge_from_settings(settings: Settings):
    """Return the configured DeepEval judge model or the default model name."""
    provider = settings.eval_llm_provider.strip().lower()
    if provider == "openai":
        return settings.eval_llm_model
    if provider != "groq":
        raise ValueError(f"Unsupported evaluation LLM provider: {settings.eval_llm_provider}")
    if not settings.eval_llm_api_key or not settings.eval_llm_api_key.get_secret_value():
        raise ValueError("EVAL_LLM_API_KEY is required when EVAL_LLM_PROVIDER=groq")
    return OpenAICompatibleJudge(
        model=settings.eval_llm_model,
        api_key=settings.eval_llm_api_key.get_secret_value(),
        base_url=settings.eval_llm_base_url or "https://api.groq.com/openai/v1",
    ).model
