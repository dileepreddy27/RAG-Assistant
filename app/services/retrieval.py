from typing import Any
from uuid import UUID

import asyncpg

from app.core.config import Settings
from app.services.embeddings import EmbeddingService
from app.services.reranker import RerankerService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService | None,
        settings: Settings,
    ) -> None:
        self._embedding_service = embedding_service
        self._reranker_service = reranker_service
        self._settings = settings

    async def search(
        self,
        pool: asyncpg.Pool,
        question: str,
        document_ids: list[UUID] | None,
        top_k: int,
        use_hybrid: bool,
        use_reranker: bool | None,
    ) -> list[dict[str, Any]]:
        query_vector = self._embedding_service.embed_query(question)
        vector_literal = self._embedding_service.to_pgvector_literal(query_vector)

        should_rerank = use_reranker if use_reranker is not None else self._settings.enable_reranker
        fetch_k = min(max(top_k * 3, top_k), 30) if should_rerank else top_k

        rows = await pool.fetch(
            """
            SELECT
                chunk_id,
                document_id,
                chunk_text,
                metadata,
                (1 - (embedding <=> $1::vector)) AS vector_score,
                ts_rank_cd(search_vector, websearch_to_tsquery('english', $2)) AS keyword_score,
                CASE
                    WHEN $6::boolean THEN
                        ((1 - (embedding <=> $1::vector)) * $3::float +
                         ts_rank_cd(search_vector, websearch_to_tsquery('english', $2)) * $4::float)
                    ELSE
                        (1 - (embedding <=> $1::vector))
                END AS score
            FROM chunks
            WHERE ($5::uuid[] IS NULL OR document_id = ANY($5))
            ORDER BY score DESC
            LIMIT $7
            """,
            vector_literal,
            question,
            self._settings.vector_weight,
            self._settings.keyword_weight,
            document_ids,
            use_hybrid,
            fetch_k,
        )

        candidates = [dict(row) for row in rows]
        if should_rerank and self._reranker_service is not None:
            return self._reranker_service.rerank(question, candidates, top_k)

        return candidates[:top_k]
