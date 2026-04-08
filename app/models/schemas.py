from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    document_ids: list[UUID]
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    conversation_id: UUID | None = None
    document_ids: list[UUID] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_hybrid_search: bool = True
    use_reranker: bool | None = None


class SourceChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    conversation_id: UUID
    sources: list[SourceChunk]
    retrieved_chunks: int


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    conversation_id: UUID
    messages: list[ConversationMessage]
