# Campus Knowledge Agent

Python FastAPI backend with the **original static frontend** in `app/static/` and Bun/SvelteKit tooling available in the repo.

## Main app path

- Primary UI: `app/static/index.html`, `app/static/app.js`, `app/static/styles.css`
- Served by FastAPI root route: `http://127.0.0.1:8000/`

The static UI was kept mostly unchanged and only patched where needed for:
- `/api/chat` submission shape
- provider/model controls
- source card rendering
- loading and error handling

## Backend architecture

- `app/main.py` - API routes and static serving
- `app/agent.py` - tool-calling orchestration (OpenAI tools + Ollama fallback routing)
- `app/retriever.py` - SQLite-backed indexing + hybrid retrieval
- `app/db.py` - SQLite schema and persistence layer
- `app/models.py` - request/response models
- `app/prompts.py` - strict grounding system prompt and tool schemas
- `scripts/build_embeddings.py` - corpus indexing + embedding build utility

## Runtime grounding rules

- Runtime retrieval indexes only `data/corpus/*.md` plus `data/corpus/index.json`.
- `index.json` URL mapping is used for source attribution.
- No DOCX/spec documents are indexed or used at runtime.
- If retrieval evidence is insufficient, response is:
  `I could not find that information in the indexed corpus.`

## SQLite data model

Implemented tables:
- `documents`
- `document_chunks`
- `chunk_embeddings`
- `conversations`
- `messages`
- `citations`
- `analytics_events`
- `document_chunks_fts` (FTS5 virtual table)

Retrieval uses:
1. SQLite FTS5 keyword search
2. Semantic similarity from normalized float32 embeddings stored in SQLite (`chunk_embeddings.embedding_blob`)
3. Deterministic hybrid merge (`final_score = weighted normalized FTS + normalized vector`)

## Setup

### 1) Install frontend tooling
```bash
bun install
```

### 2) Create Python env and install backend deps
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure environment
```bash
cp .env.example .env
```

### 4) Start backend
```bash
bun run backend:dev
```

Open `http://127.0.0.1:8000/`.

## Corpus and embeddings

Sample corpus is under `data/corpus/` as flat markdown files + `index.json`.

Build/update embeddings:
```bash
source .venv/bin/activate
bun run backend:embeddings
```

Force rebuild:
```bash
FORCE_REBUILD_EMBEDDINGS=true bun run backend:embeddings
```

## API endpoints

- `GET /health`
- `POST /api/chat`
- `GET /api/starter-questions`
- `GET /api/analytics`
- `POST /api/admin/build-embeddings`
- `POST /api/index`
- `GET /api/search?query=...&top_k=...`
  - includes `retrieval_debug` payload by default for demo/debug inspection
- `GET /api/conversations/{conversation_id}`

`POST /api/chat` supports `debug=true` (query param or request body) to include:
- retrieval mode
- tooling mode
- whether real tool calls were used
- top chunk scores (`fts_score`, `vector_score`, `final_score`)

No-answer threshold is configurable with `MIN_RETRIEVAL_FINAL_SCORE` in `.env`.

## NPM/Bun scripts

- `bun run backend:dev` - run FastAPI server
- `bun run backend:embeddings` - index corpus + build embeddings
- `bun run backend:index` - run incremental corpus indexing only
- `bun run test:backend` - run Python tests
- `bun run check` - SvelteKit type checks (tooling only)

## Tests

```bash
source .venv/bin/activate
bun run test:backend
```

Included backend tests:
- `tests/test_retriever.py`
- `tests/test_sqlite_retrieval.py`
