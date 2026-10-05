"""Optional PostgreSQL-backed conversation memory."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.models import ConversationMessage
from app.database.models import ConversationMessageRecord, ConversationRecord


class ConversationRepository:
    """Persist bounded chat history when conversation memory is enabled."""

    async def get_messages(
        self, session: AsyncSession, conversation_id: str, limit: int
    ) -> list[ConversationMessage]:
        result = await session.scalars(
            select(ConversationMessageRecord)
            .where(ConversationMessageRecord.conversation_id == conversation_id)
            .order_by(ConversationMessageRecord.created_at.desc())
            .limit(limit)
        )
        records = list(result)
        records.reverse()
        return [ConversationMessage(role=record.role, content=record.content) for record in records]

    async def append(
        self,
        session: AsyncSession,
        conversation_id: str,
        messages: Sequence[ConversationMessage],
    ) -> None:
        now = datetime.now(UTC)
        conversation = await session.get(ConversationRecord, conversation_id)
        if conversation is None:
            session.add(ConversationRecord(
                conversation_id=conversation_id,
                created_at=now,
                updated_at=now,
            ))
        else:
            conversation.updated_at = now
        session.add_all([
            ConversationMessageRecord(
                conversation_id=conversation_id,
                role=message.role,
                content=message.content,
                created_at=now,
            )
            for message in messages
        ])
        await session.commit()
