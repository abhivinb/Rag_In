"""HTTP contracts for conversation-aware chat requests."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.conversation.models import ConversationMessage
from app.rag.models import SourceReference


class ChatRequest(BaseModel):
    """One chat turn plus the client-side conversation history."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    messages: list[ConversationMessage] = Field(default_factory=list, max_length=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError("query contains unsupported control characters")
        return normalized


class ChatResponse(BaseModel):
    """One assistant turn and the conversation identifier."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    answer: str = Field(min_length=1)
    sources: list[SourceReference]
    messages: list[ConversationMessage]
