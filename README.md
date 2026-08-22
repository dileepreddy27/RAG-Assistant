# RAG Document Intelligence Platform | Python and FastAPI

A backend-focused Retrieval-Augmented Generation (RAG) project for asking questions across uploaded documents. The application parses files, creates searchable text chunks, stores embeddings in Supabase Postgres with pgvector, retrieves relevant evidence, and sends that evidence to an LLM to produce a source-aware response.

> **Project status:** portfolio prototype. The core ingestion, retrieval, answer-generation, and conversation-memory paths are implemented. Authentication, tenant isolation, production telemetry, deployment infrastructure, and comprehensive evaluation are not yet included.

## User problem

Teams often have useful information spread across PDFs, Word files, notes, and Markdown documents. Keyword search can miss semantically related passages, while a general-purpose LLM cannot reliably answer questions about private documents it has never seen.

This project demonstrates a document-question-answering workflow that:

1. indexes user-provided documents;
2. retrieves passages related to a question;
3. supplies those passages and recent chat history to an LLM; and
4. returns the answer together with retrieved source chunks.

## Architecture

```text
PDF / DOCX / TXT / MD
          |
          v
 FastAPI upload endpoint
          |
          v
Document parser -> LlamaIndex SentenceSplitter
          |
          v
SentenceTransformers embeddings
          |
          v
Supabase Postgres
  - document metadata
  - text chunks
  - pgvector embeddings
  - full-text search vector
          |
          v
Question -> query embedding
          |
          v
Hybrid retrieval
  - cosine similarity
  - PostgreSQL keyword rank
  - optional cross-encoder reranking
          |
          v
Retrieved chunks + recent conversation messages
          |
          v
LangChain prompt -> OpenAI chat model
          |
          v
Answer + document/chunk references
```

### Main components

| Area | Implementation |
| --- | --- |
| API | FastAPI routes for health, upload, query, and conversation history |
| Parsing | PDF, DOCX, TXT, and Markdown text extraction |
| Chunking | LlamaIndex `SentenceSplitter` with configurable size and overlap |
| Embeddings | SentenceTransformers, defaulting to `all-MiniLM-L6-v2` (384 dimensions) |
| Storage | Supabase Postgres tables for documents, chunks, conversations, and messages |
| Retrieval | pgvector cosine similarity plus PostgreSQL full-text ranking |
| Reranking | Optional SentenceTransformers cross-encoder |
| Generation | LangChain prompt pipeline using `ChatOpenAI` |
| Memory | Recent user and assistant messages persisted in Postgres |

## Retrieval pipeline

The query path is implemented in `app/services/retrieval.py`:

1. Encode the question with the configured SentenceTransformers model.
2. Search document chunks using cosine similarity through pgvector.
3. When hybrid search is enabled, combine vector similarity with PostgreSQL `ts_rank_cd` keyword relevance.
4. Optionally fetch a larger candidate set and rerank it with a cross-encoder.
5. Restrict retrieval to selected document IDs when supplied.
6. Pass the highest-ranked chunks to the answer service, subject to a configurable context-character limit.

The current hybrid score is:

```text
(vector_score * VECTOR_WEIGHT) + (keyword_score * KEYWORD_WEIGHT)
```

The default weights are 0.75 for vector similarity and 0.25 for keyword rank. These defaults are configurable; they have not been benchmarked against a labeled retrieval dataset.

## Verified capabilities

The following capabilities are directly supported by the repository code:

- **Multi-format, multi-document ingestion:** accepts PDF, DOCX, TXT, and Markdown files; chunks each document; generates embeddings; and stores chunk metadata and vectors.
- **Configurable hybrid retrieval:** supports semantic-only or combined semantic/keyword retrieval, optional document filtering, configurable `top_k`, and optional cross-encoder reranking.
- **Persistent conversational querying:** stores conversations and messages in Postgres, includes recent history in the prompt, and returns retrieved chunks with document and chunk identifiers.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | API status and links |
| GET | `/api/v1/health` | Basic health response |
| POST | `/api/v1/documents/upload` | Upload one or more supported documents |
| POST | `/api/v1/chat/query` | Retrieve context and generate an answer |
| GET | `/api/v1/conversations/{conversation_id}` | Read recent conversation messages |

FastAPI provides interactive OpenAPI documentation at `http://127.0.0.1:8000/docs` when the application is running.

Example query:

```json
{
  "question": "What are the main obligations described in these documents?",
  "conversation_id": null,
  "document_ids": null,
  "top_k": 6,
  "use_hybrid_search": true,
  "use_reranker": false
}
```

## Local setup

### Prerequisites

- Python 3.10 or newer
- A Supabase project with a Postgres connection string
- An OpenAI API key
- Git

### 1. Clone and create an environment

```powershell
git clone https://github.com/dileepreddy27/RAG-Assistant.git
cd RAG-Assistant
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux equivalents:

```bash
git clone https://github.com/dileepreddy27/RAG-Assistant.git
cd RAG-Assistant
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and provide at least:

```env
SUPABASE_DB_URL=postgresql://...
OPENAI_API_KEY=...
LLM_MODEL=gpt-4o-mini
```

The local `.env` file is excluded by `.gitignore`. Do not commit credentials.

### 3. Initialize Supabase

Run `scripts/init_db.sql` in the Supabase SQL Editor. It creates:

- the pgvector extension;
- document and chunk tables;
- a 384-dimensional vector column;
- IVFFlat, GIN, and relational indexes;
- conversation and message tables; and
- a trigger that updates conversation timestamps.

If the embedding model changes, update the SQL vector dimension to match the model output.

### 4. Start the API

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

macOS/Linux:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then visit:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/api/v1/health`

## Tests and evidence

The repository includes two focused unit tests:

- `tests/test_chunking.py` checks that long text is split and source metadata is retained.
- `tests/test_embedding_literal.py` checks pgvector literal serialization.

Run them with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

A GitHub Actions workflow is also configured to install dependencies, run targeted Flake8 checks, and execute pytest on pushes and pull requests to `main`.

Current evidence is intentionally limited: the repository does not yet include API integration tests, database-backed retrieval tests, LLM response tests, retrieval-quality benchmarks, load tests, or a published evaluation report.

## Authentication and observability

### Authentication

Authentication and authorization are **not implemented**. The API currently has no user identity, role checks, document ownership, or tenant-level access controls. It should only be run in a trusted development environment until those controls are added.

### Observability

Implemented:

- configurable standard Python logging;
- Uvicorn request/startup logs; and
- a basic health endpoint.

Not implemented:

- structured request correlation;
- metrics or dashboards;
- distributed tracing;
- error monitoring;
- token/cost tracking; or
- retrieval and answer-quality telemetry.

## Demo status

There is currently **no hosted public demo** and no frontend UI in this repository. The runnable demonstration is the local FastAPI Swagger interface at `/docs`, after Supabase and API credentials are configured.

A short screen recording showing document upload, returned document IDs, a grounded query, source chunks, and conversation continuation would materially improve recruiter review.

## Known gaps and next steps

- Add Supabase Auth or another identity provider and enforce document ownership.
- Add row-level security or explicit tenant filters for every database operation.
- Add API integration tests with a disposable Postgres/pgvector instance.
- Add a labeled retrieval evaluation set and report Recall@K, MRR, and groundedness checks.
- Add retries, timeouts, rate limits, file-size limits, and background ingestion.
- Add structured logs, traces, metrics, and LLM token/cost reporting.
- Add a frontend or hosted demo and a reproducible sample dataset.
- Add containerization, deployment configuration, and a recognized open-source license.

## Recruiter review summary

This repository demonstrates a readable separation of API, ingestion, embedding, retrieval, generation, and memory concerns. It is best evaluated as a working RAG architecture prototype rather than a production-ready platform: the core code paths are present, while security, deployment, evaluation, and operational hardening remain explicit future work.
