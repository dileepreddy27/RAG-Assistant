# RAG-Assistant (FastAPI + Supabase pgvector)

A production-style Retrieval-Augmented Generation (RAG) backend where users upload documents and ask questions. Answers are generated using retrieved document context.

## What this project does

- Uploads documents (`PDF`, `DOCX`, `TXT`, `MD`)
- Splits documents into chunks (LlamaIndex)
- Generates embeddings (SentenceTransformers)
- Stores vectors and metadata in Supabase Postgres (`pgvector`)
- Retrieves relevant chunks using hybrid search (vector + keyword)
- Optionally reranks results with cross-encoder reranker
- Generates context-grounded answers with LangChain + LLM
- Maintains conversation history (memory) per `conversation_id`

## Tech stack

- Backend: FastAPI
- LLM framework: LangChain
- Chunking: LlamaIndex (`SentenceSplitter`)
- Embeddings: SentenceTransformers (`all-MiniLM-L6-v2`)
- Vector DB: Supabase Postgres + `pgvector`
- Database client: `asyncpg`

## Project structure

```text
rag-system-llm-project/
  app/
    api/
      routes.py
    core/
      config.py
      logging.py
    db/
      postgres.py
    models/
      schemas.py
    services/
      chunking.py
      embeddings.py
      ingestion.py
      llm_answer.py
      memory.py
      reranker.py
      retrieval.py
    utils/
      document_parser.py
    main.py
  scripts/
    init_db.sql
  tests/
    test_chunking.py
    test_embedding_literal.py
  .env.example
  .gitignore
  requirements.txt
  README.md
```

## How it works

### 1. Document ingestion pipeline

1. User uploads files via `POST /api/v1/documents/upload`
2. Files are parsed into text (`PDF/DOCX/TXT/MD`)
3. Text is chunked using LlamaIndex sentence splitter
4. Embeddings are generated for each chunk
5. Chunks are inserted into `chunks` table with vector + metadata

### 2. Question-answer pipeline

1. User asks via `POST /api/v1/chat/query`
2. Query embedding is generated
3. Hybrid retrieval runs:
   - vector similarity (`pgvector` cosine)
   - keyword relevance (`tsvector` / `ts_rank_cd`)
4. Optional reranker re-scores top candidates
5. Retrieved chunks + recent conversation messages are sent to LLM
6. Answer is returned with source chunk references

### 3. Conversation memory

- Conversations stored in `conversations`
- Messages stored in `messages`
- Recent message window is attached to each answer request

## API endpoints

- `GET /` -> root status message
- `GET /api/v1/health` -> health check
- `POST /api/v1/documents/upload` -> upload one or more files
- `POST /api/v1/chat/query` -> ask a question over indexed docs
- `GET /api/v1/conversations/{conversation_id}` -> fetch conversation history

Interactive docs:

- `http://127.0.0.1:8000/docs`

## Local setup (Windows / PowerShell)

### 1. Go to project

```powershell
cd "C:\Users\dilee\OneDrive\Desktop\Personal-Projects\rag-system-llm-project"
```

### 2. Create virtual environment

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Fill `.env` values:

- `SUPABASE_DB_URL`
- `OPENAI_API_KEY`
- optional model/config parameters

### 5. Initialize database schema

Run `scripts/init_db.sql` in Supabase SQL Editor.

### 6. Run API server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/v1/health`

## Example request flow

### A) Upload docs

Use Swagger (`/docs`) -> `POST /api/v1/documents/upload` -> upload files.

### B) Ask question

`POST /api/v1/chat/query`

```json
{
  "question": "Summarize the main points of the uploaded document.",
  "conversation_id": null,
  "document_ids": null,
  "top_k": 6,
  "use_hybrid_search": true,
  "use_reranker": false
}
```

### C) Continue same conversation

Re-use returned `conversation_id` in next query call.

## Retrieval and ranking details

Final hybrid score is computed as:

- `vector_score = 1 - cosine_distance`
- `keyword_score = ts_rank_cd(search_vector, tsquery)`
- `hybrid_score = vector_weight * vector_score + keyword_weight * keyword_score`

If reranking is enabled, cross-encoder scores reorder top candidates before answer generation.

## Testing

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Troubleshooting

### `ERR_CONNECTION_REFUSED`

- Server is not running or crashed.
- Restart uvicorn and keep terminal open.

### `{"detail":"Not Found"}` at `/`

- Fixed in this project by adding root route.
- Use `/docs` and `/api/v1/health` for API checks.

### `InvalidPasswordError` for Supabase

- DB password in `SUPABASE_DB_URL` is incorrect.
- Reset DB password in Supabase and update `.env`.

### `getaddrinfo failed`

- Hostname in DB URL is wrong or DNS issue.
- Verify direct connection URI from Supabase.

## Security and git notes

- `.env` is ignored by git in `.gitignore`.
- Do not commit secrets or API keys.
- Rotate secrets immediately if exposed.

## Suggested improvements

- Add authentication/authorization
- Add streaming responses (SSE/WebSocket)
- Add async background ingestion for large files
- Add RAG evaluation pipeline (precision/groundedness)
- Add multi-tenant document isolation
