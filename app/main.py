from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.postgres import db
from app.services.chunking import TextChunker
from app.services.embeddings import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.llm_answer import LLMAnswerService
from app.services.memory import MemoryService
from app.services.reranker import RerankerService
from app.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.debug)

    await db.connect(settings.supabase_db_url)

    embedding_service = EmbeddingService(settings.embedding_model_name)
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    reranker = RerankerService(settings.reranker_model_name) if settings.enable_reranker else None

    app.state.settings = settings
    app.state.db_pool = db.pool
    app.state.ingestion_service = IngestionService(chunker, embedding_service)
    app.state.retrieval_service = RetrievalService(embedding_service, reranker, settings)
    app.state.memory_service = MemoryService(settings.conversation_window)
    app.state.llm_service = LLMAnswerService(settings)

    yield

    await db.close()


settings = get_settings()
app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "RAG System API is running",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


app.include_router(router, prefix=settings.api_prefix)
