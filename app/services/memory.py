from uuid import UUID, uuid4

import asyncpg

from app.models.schemas import ConversationMessage


class MemoryService:
    def __init__(self, conversation_window: int) -> None:
        self._conversation_window = conversation_window

    async def ensure_conversation(self, pool: asyncpg.Pool, conversation_id: UUID | None) -> UUID:
        if conversation_id is None:
            conversation_id = uuid4()
            await pool.execute(
                """
                INSERT INTO conversations (conversation_id)
                VALUES ($1)
                """,
                conversation_id,
            )
            return conversation_id

        exists = await pool.fetchval(
            "SELECT 1 FROM conversations WHERE conversation_id = $1",
            conversation_id,
        )
        if not exists:
            await pool.execute(
                "INSERT INTO conversations (conversation_id) VALUES ($1)",
                conversation_id,
            )
        return conversation_id

    async def add_message(self, pool: asyncpg.Pool, conversation_id: UUID, role: str, content: str) -> None:
        await pool.execute(
            """
            INSERT INTO messages (message_id, conversation_id, role, content)
            VALUES ($1, $2, $3, $4)
            """,
            uuid4(),
            conversation_id,
            role,
            content,
        )

    async def get_recent_messages(
        self,
        pool: asyncpg.Pool,
        conversation_id: UUID,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        message_limit = limit or self._conversation_window
        rows = await pool.fetch(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id,
            message_limit,
        )

        ordered = list(reversed(rows))
        return [
            ConversationMessage(
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in ordered
        ]
