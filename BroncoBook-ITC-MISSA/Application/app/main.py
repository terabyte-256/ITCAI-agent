from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentService
from .analytics import AnalyticsStore
from .db import SQLiteStore
from .models import ChatRequest, ChatResponse, IndexRequest
from .retriever import CorpusRetriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = os.getenv("CORPUS_DIR", str(BASE_DIR / "data" / "corpus"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "8"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "90"))
DATABASE_URL = os.getenv("DATABASE_URL", str(BASE_DIR / "data" / "campus_agent.db"))

store = SQLiteStore(DATABASE_URL)
retriever = CorpusRetriever(CORPUS_DIR, top_k_default=TOP_K_CHUNKS, store=store)
agent = AgentService(retriever, store=store, ttl_minutes=SESSION_TTL_MINUTES)
analytics = AnalyticsStore()

app = FastAPI(title="Campus Knowledge Agent", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "provider_default": "openai" if os.getenv("OPENAI_API_KEY") else "ollama",
        "database_url": DATABASE_URL,
        **retriever.corpus_stats(),
    }


@app.get("/api/starter-questions")
def starter_questions() -> dict:
    return {"questions": retriever.suggest_starters()}


@app.get("/api/analytics")
def analytics_snapshot() -> dict:
    return analytics.snapshot().model_dump()


@app.post("/api/admin/build-embeddings")
def build_embeddings(
    force_rebuild: bool = False,
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
) -> dict:
    try:
        built = retriever.build_embeddings(
            force_rebuild=force_rebuild,
            provider=provider,
            model=model,
        )
        return {
            "ok": built,
            "provider": provider,
            "model": model,
            **retriever.corpus_stats(),
        }
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/index")
def index_corpus(request: IndexRequest) -> dict:
    try:
        summary = retriever.index_corpus(force=request.force_reindex)
        embedding_summary = None
        if request.build_embeddings:
            embedding_summary = retriever.build_embeddings(
                force_rebuild=request.force_reindex,
                provider=request.embedding_provider,
                model=request.embedding_model,
            )
        return {
            "ok": True,
            "index_summary": summary,
            "embedding_built": embedding_summary,
            **retriever.corpus_stats(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/search")
def search(
    query: Optional[str] = Query(default=None, min_length=1),
    q: Optional[str] = Query(default=None, min_length=1),
    top_k: int = Query(6, ge=1, le=20),
    provider: str = "openai",
    model: Optional[str] = None,
    debug: bool = Query(default=True),
) -> dict:
    try:
        query_used = (query or q or "").strip()
        if not query_used:
            raise HTTPException(status_code=422, detail="query (or q) is required")
        results = retriever.search_corpus(query_used, top_k=top_k, provider=provider, embedding_model=model)
        return {
            "query": query_used,
            "count": len(results),
            "results": [item.model_dump() for item in results],
            "retrieval_debug": {
                "enabled": debug,
                "mode": results[0].retrieval_method if results else "none",
                "query_used": query_used,
                "tooling_mode": "direct_retrieval",
                "used_tool_calls": False,
                "top_chunks": [
                    {
                        "chunk_id": item.chunk_id,
                        "title": item.title,
                        "heading_path": item.section,
                        "original_url": item.source_url,
                        "fts_score": item.fts_score,
                        "vector_score": item.vector_score,
                        "final_score": item.final_score if item.final_score is not None else item.score,
                    }
                    for item in (results[:8] if debug else [])
                ],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict:
    payload = store.get_conversation(conversation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return payload


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, debug: Optional[bool] = Query(default=None)) -> ChatResponse:
    try:
        debug_enabled = request.debug if debug is None else debug
        response = agent.chat(
            request.message,
            request.conversation_id,
            provider=request.provider,
            model=request.model,
            debug=debug_enabled,
        )
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer_lower = response.answer.lower()
    answered = not any(
        phrase in answer_lower
        for phrase in [
            "i could not find",
            "i couldn't find",
            "not in the corpus",
            "cannot be found",
            "don't have enough information",
        ]
    )
    analytics.record_query(
        request.message,
        answered=answered,
        source_count=len(response.sources),
        tool_calls=len(response.tool_trace),
    )
    return response
