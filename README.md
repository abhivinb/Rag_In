# Enterprise RAG Knowledge Assistant

## Project Overview

Phase 1 establishes the production-oriented foundation for an enterprise knowledge assistant: a FastAPI application, typed environment configuration, centralized logging, PostgreSQL/pgvector infrastructure, Docker support, and automated tests.

RAG behavior is intentionally out of scope for this phase. Retrieval, generation, agents, evaluation, and deployment integrations will be added in later phases.

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

## Phase 4: Embeddings and Vector Store

Phase 4 generates embeddings for `DocumentChunk` objects and persists documents, chunks, metadata, and vectors in PostgreSQL with pgvector. Embedding generation and persistence are separate abstractions so the provider can be replaced without rewriting the repository.

```text
DocumentChunk[] -> EmbeddingService -> EmbeddingProvider
				-> EmbeddingResult[] -> VectorRepository
				-> PostgreSQL + pgvector
```

The default provider is the official OpenAI SDK, configured with `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `EMBEDDING_BATCH_SIZE`, and `EMBEDDING_MAX_RETRIES`. The provider does not log keys, document content, or vectors. Returned dimensions are validated before persistence; the database schema uses `Vector(1536)` and startup rejects a mismatched configured dimension.

The schema contains `documents` and `document_chunks`. Core fields remain relational, flexible metadata is JSONB, chunks reference documents with a foreign key, and stable document/chunk IDs make upserts idempotent. No vector similarity index is created yet.

Apply the initial Alembic migration after PostgreSQL is running:

```powershell
docker compose up -d postgres
alembic upgrade head
```

Run unit tests with:

```powershell
python -m pytest -q
```

Run PostgreSQL/pgvector integration tests explicitly against the local Compose service:

```powershell
$env:RUN_DB_INTEGRATION = "1"
python -m pytest tests/integration/test_database.py -q
```

Vector similarity search, retrieval, reranking, and RAG generation are not implemented. They belong to Phase 5.

## Phase 5: Vector Similarity Retrieval

Phase 5 embeds a user query with the existing Phase 4 `EmbeddingService`, executes cosine similarity in PostgreSQL through pgvector, applies safe document/metadata filters, then applies the similarity threshold and database-side top-k limit.

```text
User Query -> Query Embedding -> pgvector Cosine Similarity
		   -> Filters -> Threshold -> Top-K -> RetrievalResult[]
```

Retrieval configuration uses `RETRIEVAL_TOP_K` (default `5`) and `RETRIEVAL_SIMILARITY_THRESHOLD` (default `0.0`). Supported filters are `document_id`, `file_type`, and `source_type`. Similarity is returned as `1 - cosine_distance`, normalized to `0.0..1.0`, with deterministic `chunk_id` tie ordering. Empty or whitespace-only queries raise an error; valid queries with no matching chunks return `[]`.

Example usage:

```python
from app.retrieval import RetrievalConfig, RetrievalFilter, RetrievalService

results = await RetrievalService(
	embedding_service,
	vector_retrieval_repository,
	RetrievalConfig(top_k=5, similarity_threshold=0.7),
).retrieve(session, "What is the retention policy?", RetrievalFilter(file_type="pdf"))
```

Hybrid retrieval, BM25, reranking, vector indexes, LLM generation, and RAG answer generation are intentionally deferred to later phases.

## Phase 8: Core RAG Generation

Phase 8 builds the stateless grounded answering pipeline on top of Phase 6 hybrid retrieval:

```text
User Query -> Hybrid Retrieval -> Bounded Context -> Grounded Prompts
		   -> LLM Provider -> Answer + Source References
```

The default provider is OpenAI Chat Completions through the existing `openai` dependency. Configure `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, and `RAG_MAX_CONTEXT_CHARS`. Retrieved chunks are treated as untrusted data, and the system prompt requires answers to be supported by the supplied context. When retrieval returns no results, the LLM is not called and a deterministic no-context response is returned.

Sources are derived directly from retrieved chunks and include document/chunk identifiers, metadata, chunk ordering, and retrieval score. Context construction includes complete chunks until the configured character limit; one oversized first chunk is bounded with an explicit truncation marker.

The pipeline is non-streaming, stateless, and has no conversation memory, agents, query rewriting, or public API endpoint. Cross-Encoder reranking is intentionally deferred and is not implemented here.

## Phase 10: Offline RAG Evaluation

Phase 10 is an offline engineering workflow for measuring the current Phase 9 → Phase 6 → Phase 8 pipeline against explicit, reviewable JSON evaluation samples.

```text
Evaluation Dataset -> RAG Pipeline -> Actual Answer + Retrieved Context
				   -> DeepEval Metrics -> Evaluation Report
```

The dataset format contains `id`, `question`, `expected_answer`, and `expected_contexts`. Runtime retrieved contexts remain separate from expected ground-truth contexts. The supported DeepEval metrics are Faithfulness, Answer Relevancy, Contextual Relevancy, Contextual Precision, and Contextual Recall. Thresholds and the evaluation judge model are configured with `EVAL_*` settings.

Normal tests use fake RAG and metric components and make no network calls. An optional real evaluation run is available only after wiring a configured RAG application into `evaluation/run_evaluation.py`:

```powershell
$env:RUN_REAL_EVAL = "1"
python evaluation/run_evaluation.py
```

Evaluation scores are model- and dataset-dependent measurements, not absolute truth. Reports can be written to `evaluation/results/latest.json` by callers and that directory is ignored by Git. Evaluation is not part of the production request path, does not gate startup or CI, and does not implement reranking, agents, LangGraph, or monitoring.

## Phase 9: Query Rewriting and Retrieval Relevance

Phase 9 adds a bounded query-improvement and retrieval-sufficiency layer around Phase 6 and Phase 8:

```text
User Query -> Optional Query Rewrite -> Hybrid Retrieval
		   -> Relevance Check -> Optional One Retry -> RAG Generation
```

Query rewriting is controlled by `QUERY_REWRITE_ENABLED` and reuses the existing LLM provider abstraction. Rewrite failures fall back to the original query. The relevance checker returns structured `relevant`, model-generated `confidence`, reason, and relevant chunk IDs. A result is accepted only when `relevant` is true and confidence is at least `RAG_RELEVANCE_THRESHOLD` (default `0.70`). Confidence is a model judgment, not a calibrated probability.

The pipeline performs at most two retrieval attempts and never enters an unrestricted loop. Empty retrieval skips relevance checking. If both attempts are absent or insufficient, the service returns a deterministic safe no-answer with no sources and does not call answer generation. Sources come only from the final accepted result set.

The relevance checker determines whether retrieved context is sufficient for answering; it does not rank or reorder individual chunks. Cross-Encoder reranking remains intentionally deferred. No agents, query-rewriting loops, conversation memory, streaming, or public API endpoint are included.

## Phase 6: Hybrid Retrieval

Phase 6 provides a baseline hybrid retriever using pgvector semantic similarity plus PostgreSQL native full-text ranking. Vector retrieval is strong for semantic meaning, while keyword retrieval is useful for exact terms and identifiers.

```text
User Query -> Vector Retrieval
		   -> PostgreSQL Full-Text Retrieval
		   -> Score Normalization -> Weighted Fusion -> Filters -> Top-K
```

Keyword retrieval uses `to_tsvector('english', content)`, `plainto_tsquery('english', query)`, and PostgreSQL `ts_rank_cd`. This is PostgreSQL native full-text ranking, not a separate BM25 engine. Each score source is min-max normalized over its candidate set, then combined with `HYBRID_VECTOR_WEIGHT` and `HYBRID_KEYWORD_WEIGHT`. `HYBRID_CANDIDATE_MULTIPLIER` controls how many candidates each source contributes before final top-k selection.

The vector similarity threshold remains a vector-candidate filter and is never applied to the final hybrid score. Supported filters remain `document_id`, `file_type`, and `source_type`. Matching candidates are merged by `chunk_id`, ties are ordered deterministically, and no-result queries return an empty list.

Phase 6 adds a functional PostgreSQL GIN index through Alembic migration `0002_add_chunk_content_fts_index`. No vector index is introduced yet. Reranking, hybrid extensions, BM25-specific engines, LLM generation, and RAG answer generation remain out of scope.

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
- `OPENAI_API_KEY`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `EMBEDDING_BATCH_SIZE`
- `EMBEDDING_MAX_RETRIES`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_SIMILARITY_THRESHOLD`

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
