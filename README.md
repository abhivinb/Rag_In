# Enterprise RAG Knowledge Assistant

## Project Overview

Phase 1 establishes the production-oriented foundation for an enterprise knowledge assistant: a FastAPI application, typed environment configuration, centralized logging, PostgreSQL/pgvector infrastructure, Docker support, and automated tests.

RAG behavior is intentionally out of scope for this phase. Retrieval, generation, agents, embeddings, evaluation, and deployment integrations will be added in later phases.

## Phase 2: Document Ingestion

Phase 2 adds a format-independent ingestion service for PDF, DOCX, and UTF-8 TXT files. It validates the source, extracts text, applies conservative whitespace cleanup, generates stable metadata, and returns a normalized Pydantic `Document` ready for Phase 3 chunking.

```text
File -> Validation -> Type Detection -> Loader -> Text Extraction
	-> Text Cleaning -> Metadata -> Normalized Document
```

Supported formats are `.pdf`, `.docx`, and `.txt`. The ingestion layer does not implement chunking, embeddings, vector search, database persistence, LLM calls, or RAG generation.

Example usage:

```python
from app.ingestion.service import IngestionService

document = IngestionService().ingest("data/raw/txt/example.txt")
print(document.document_id)
print(document.text)
```

## Phase 3: Document Chunking

Phase 3 converts a normalized `Document` into ordered `DocumentChunk` objects. Chunking is independent from embeddings, databases, and retrieval.

```text
Document -> Validate Configuration -> Recursive Chunking
		 -> Propagate Metadata -> Validate Chunks -> DocumentChunk[]
```

`ChunkingConfig` supports:

- `chunk_size`: maximum chunk content length; default `1000`
- `chunk_overlap`: characters repeated between adjacent chunks; default `150`
- `minimum_chunk_size`: preferred minimum boundary size; default `50`

Configuration requires `chunk_size > 0`, `0 <= chunk_overlap < chunk_size`, and `0 <= minimum_chunk_size <= chunk_size`. The recursive strategy prefers paragraph, line, sentence, and whitespace boundaries before using a hard character split. Empty documents return no chunks.

Chunks preserve document identity, source metadata, page-number lists, and character offsets. Chunk IDs are deterministic hashes of `document_id`, `chunk_index`, and content.

Example usage:

```python
from app.ingestion.chunking import ChunkingConfig, ChunkingService

chunks = ChunkingService(
	config=ChunkingConfig(chunk_size=800, chunk_overlap=100, minimum_chunk_size=40)
).chunk(document)
```

Phase 3 ends at `Document -> DocumentChunk[]`; embeddings and vector storage are future phases.

## Architecture Placeholder

```text
Client -> FastAPI API -> Services -> PostgreSQL + pgvector
```

This diagram is a placeholder for the later RAG architecture.

## Local Setup

Python 3.12 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a strong local value for `POSTGRES_PASSWORD` in `.env` before starting PostgreSQL.

## Environment Configuration

Configuration is loaded from `.env` through Pydantic Settings. The available variables are documented in `.env.example`:

- `APP_NAME`
- `APP_ENV`
- `LOG_LEVEL`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`

`.env` is ignored by Git and must never contain committed credentials.

## Running PostgreSQL

Start the pgvector-enabled PostgreSQL and FastAPI services:

```powershell
docker compose up -d
docker compose ps
```

Stop the service while retaining its persistent volume:

```powershell
docker compose down
```

Remove the service and its database volume:

```powershell
docker compose down -v
```

## Running FastAPI

Start the development server from the project root:

```powershell
uvicorn main:app --reload
```

Check the health endpoint at <http://127.0.0.1:8000/health>.

## Running Tests

```powershell
python -m pytest
```

The test suite uses temporary files and does not require PostgreSQL or developer-machine documents.

## Docker Commands

Build the application image:

```powershell
docker build -t enterprise-rag-api .
```

Run the API container on port 8000:

```powershell
docker run --rm --env-file .env -p 8000:8000 enterprise-rag-api
```

The Compose API service connects to PostgreSQL using the `postgres` service hostname and waits for the PostgreSQL healthcheck before starting.
