# Campus Knowledge Agent

A local, tool-calling web application for answering student questions against the **actual Cal Poly Pomona corpus** from the MISSA IT Competition 2026 case.

This version now includes the highest-impact upgrade: **optional embedding-based semantic search** combined with lexical retrieval using **reciprocal rank fusion (RRF)**.

## What changed in this upgrade

- Added real embedding-based semantic retrieval over the uploaded CPP corpus.
- Added hybrid ranking: lexical retrieval + semantic retrieval fused together.
- Added local embedding cache files so corpus embeddings only need to be generated once.
- Added configuration for embedding model, dimensions, batching, and ranking weights.
- Added an admin endpoint and CLI script to build embeddings.
- Expanded health output so you can verify whether semantic search is active.

## Corpus stats used in this build

- Pages: 8,042
- Indexed chunks: ~115,375

## What it includes

- Conversational chat UI
- Tool-calling agent using the OpenAI Responses API
- Corpus search tool over Markdown pages
- Grounded responses with explicit source attribution
- Multi-turn conversation memory
- Bonus features:
  - multiple specialized tools (`search_corpus`, `get_page_by_source`, `suggest_starter_questions`)
  - starter questions in the UI
  - **hybrid semantic + lexical retrieval**
  - basic analytics endpoint and dashboard card

## Repository layout

```text
app/
  main.py              FastAPI app + API routes
  agent.py             Tool-calling agent loop
  analytics.py         Simple in-memory analytics
  models.py            Pydantic models
  prompts.py           System prompt and tool schemas
  retriever.py         Corpus loading, chunking, hybrid retrieval, embedding cache
  static/
    index.html         Chat UI
    app.js             Front-end behavior
    styles.css         Styling
scripts/
  build_embeddings.py  One-time embedding index builder
.data/
  cpp_corpus/          Actual MISSA case corpus (from uploaded ZIP)
.env.example
requirements.txt
```

## Requirements

- Python 3.10+
- An OpenAI API key
- The CPP corpus folder containing:
  - Markdown files (`.md`)
  - `index.json` mapping original URLs to markdown filenames

Example `index.json`:

```json
{
  "https://www.cpp.edu/admissions/index.shtml": "admissions_index.md",
  "https://www.cpp.edu/financial-aid/index.shtml": "financial_aid_index.md"
}
```

## Quick start

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Copy `.env.example` to `.env` and add your key.
5. Point `CORPUS_DIR` at the real corpus.
6. Build embeddings once.
7. Start the server.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/build_embeddings.py
uvicorn app.main:app --reload
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/build_embeddings.py
uvicorn app.main:app --reload
```

The app will be available at `http://127.0.0.1:8000`.

## Environment variables

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
CORPUS_DIR=./data/cpp_corpus
MAX_TOOL_RESULTS=6
TOP_K_CHUNKS=8
SESSION_TTL_MINUTES=90
ENABLE_SEMANTIC_SEARCH=true
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=512
EMBEDDING_BATCH_SIZE=128
BUILD_EMBEDDINGS_ON_STARTUP=false
LEXICAL_SCORE_WEIGHT=0.65
EMBEDDING_SCORE_WEIGHT=0.35
RRF_K=60
```

## How semantic search works

### 1) Corpus processing

- Reads `index.json`
- Loads Markdown pages
- Strips common scrape noise where possible
- Splits pages into heading-aware chunks
- Builds an inverted index and IDF map for lexical retrieval

### 2) Embedding index

- Uses OpenAI embeddings (default: `text-embedding-3-small`)
- Stores normalized vectors in a local cache under:
  - `data/cpp_corpus/.embedding_cache/`
- Reuses the cache on future runs if the corpus fingerprint matches

### 3) Hybrid retrieval

For each query, the retriever:

- Runs lexical retrieval over all chunks
- Runs semantic retrieval using cosine similarity over cached embeddings
- Fuses both rankings using **reciprocal rank fusion (RRF)**
- Returns the best grounded chunks to the model

This materially improves recall for vague, paraphrased, or indirect questions.

## Building embeddings

### CLI

```bash
python scripts/build_embeddings.py
```

To force a rebuild:

```bash
FORCE_REBUILD_EMBEDDINGS=true python scripts/build_embeddings.py
```

### API endpoint

```bash
curl -X POST "http://127.0.0.1:8000/api/admin/build-embeddings"
```

To force a rebuild:

```bash
curl -X POST "http://127.0.0.1:8000/api/admin/build-embeddings?force_rebuild=true"
```

## Health check

`GET /health` now returns:

- pages
- chunks
- semantic search enabled/disabled
- whether the embedding cache is loaded
- embedding model and dimensions
- semantic search status message

## API routes

- `GET /health` - liveness check plus corpus and semantic search status
- `GET /api/starter-questions` - starter prompts
- `GET /api/analytics` - usage stats
- `POST /api/admin/build-embeddings` - build or rebuild semantic index
- `POST /api/chat` - chat with the agent

## Case requirements mapping

- **Chat Interface**: implemented in `app/static/index.html`
- **Corpus Search Tool**: implemented in `app/retriever.py` and exposed as tool calls
- **Grounded Responses**: enforced by system prompt and structured source return
- **Source Attribution**: returned with every answer and displayed in UI
- **Multi-turn Conversation**: session-based memory in backend

## Bonus features included

- Multiple specialized tools
- Starter questions
- **Semantic search via embeddings**
- Hybrid reranking with reciprocal rank fusion
- Basic analytics dashboard card

## Remaining limitations

- Embedding generation still requires an OpenAI API key and incurs API cost.
- The semantic index is local-file based, not backed by FAISS or a vector database.
- Similarity search currently uses an in-memory NumPy matrix, so large corpora consume meaningful RAM once embeddings are loaded.
- Some scraped Markdown files still contain noisy boilerplate.
- Sessions and analytics remain in-memory and reset on restart.
- I did not run live end-to-end model calls in this environment because that requires your API key.

## Why this upgrade matters

This was the biggest missing piece from the earlier version. It raises the ceiling on the **Implemented Features and Functionality** portion of the case because the assistant can now retrieve relevant information even when the student's wording does not closely match the source page text.

## Demo tips

Use questions like:

- What are the freshman admission requirements?
- How do I change my major?
- Where is Student Health Services located?
- What dining options are on campus?
- What are the admission requirements for the Computer Science master's program?
- I want to switch my field of study. What steps do I need to take?
