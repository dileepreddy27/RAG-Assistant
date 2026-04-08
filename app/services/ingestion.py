import json
from uuid import UUID, uuid4

import asyncpg

from app.services.chunking import TextChunker
from app.services.embeddings import EmbeddingService
from app.utils.document_parser import ParsedDocument


class IngestionService:
    def __init__(self, chunker: TextChunker, embedding_service: EmbeddingService) -> None:
        self._chunker = chunker
        self._embedding_service = embedding_service

    async def ingest_documents(
        self,
        pool: asyncpg.Pool,
        documents: list[ParsedDocument],
    ) -> tuple[list[UUID], int]:
        if not documents:
            return [], 0

        inserted_documents: list[UUID] = []
        indexed_chunks = 0

        for document in documents:
            document_id = uuid4()
            metadata = dict(document.metadata)
            metadata["filename"] = document.filename

            await pool.execute(
                """
                INSERT INTO documents (document_id, filename, metadata)
                VALUES ($1, $2, $3::jsonb)
                """,
                document_id,
                document.filename,
                json.dumps(metadata),
            )

            chunks = self._chunker.chunk_document(document.text, metadata)
            if not chunks:
                inserted_documents.append(document_id)
                continue

            vectors = self._embedding_service.embed_texts([chunk.text for chunk in chunks])
            records = []
            for chunk, vector in zip(chunks, vectors):
                records.append(
                    (
                        uuid4(),
                        document_id,
                        chunk.text,
                        self._embedding_service.to_pgvector_literal(vector),
                        json.dumps(chunk.metadata),
                    )
                )

            await pool.executemany(
                """
                INSERT INTO chunks (chunk_id, document_id, chunk_text, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                """,
                records,
            )

            inserted_documents.append(document_id)
            indexed_chunks += len(records)

        return inserted_documents, indexed_chunks
