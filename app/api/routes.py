from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import (
    ConversationHistoryResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from app.utils.document_parser import parse_upload_file

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    parsed_documents = []
    for upload in files:
        try:
            parsed_documents.append(await parse_upload_file(upload))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: {error}") from error

    document_ids, chunk_count = await request.app.state.ingestion_service.ingest_documents(
        request.app.state.db_pool,
        parsed_documents,
    )

    return UploadResponse(document_ids=document_ids, chunks_indexed=chunk_count)


@router.post("/chat/query", response_model=QueryResponse)
async def query_documents(request: Request, payload: QueryRequest) -> QueryResponse:
    pool = request.app.state.db_pool
    settings = request.app.state.settings
    memory_service = request.app.state.memory_service

    conversation_id = await memory_service.ensure_conversation(pool, payload.conversation_id)

    history = await memory_service.get_recent_messages(pool, conversation_id)
    await memory_service.add_message(pool, conversation_id, role="user", content=payload.question)

    retrieved_chunks = await request.app.state.retrieval_service.search(
        pool=pool,
        question=payload.question,
        document_ids=payload.document_ids,
        top_k=payload.top_k or settings.top_k,
        use_hybrid=payload.use_hybrid_search,
        use_reranker=payload.use_reranker,
    )

    answer = await request.app.state.llm_service.generate_answer(
        question=payload.question,
        history=history,
        retrieved_chunks=retrieved_chunks,
    )

    await memory_service.add_message(pool, conversation_id, role="assistant", content=answer)

    sources = [
        SourceChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["chunk_text"],
            score=float(row["score"]),
            metadata=row.get("metadata") or {},
        )
        for row in retrieved_chunks
    ]

    return QueryResponse(
        answer=answer,
        conversation_id=conversation_id,
        sources=sources,
        retrieved_chunks=len(sources),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(request: Request, conversation_id: UUID) -> ConversationHistoryResponse:
    messages = await request.app.state.memory_service.get_recent_messages(
        request.app.state.db_pool,
        conversation_id,
        limit=50,
    )
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=messages)
