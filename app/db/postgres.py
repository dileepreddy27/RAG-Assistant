from typing import Optional

import asyncpg


class PostgresDB:
    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")
        return self._pool

    async def connect(self, dsn: str) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=10,
            command_timeout=60,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


db = PostgresDB()
