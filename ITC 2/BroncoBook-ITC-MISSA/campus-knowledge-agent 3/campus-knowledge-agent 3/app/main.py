from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentService
from .analytics import AnalyticsStore
from .models import ChatRequest, ChatResponse
from .retriever import CorpusRetriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = os.getenv("CORPUS_DIR", str(BASE_DIR / "data" / "sample_corpus"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "8"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "90"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:8000")

retriever = CorpusRetriever(CORPUS_DIR, top_k_default=TOP_K_CHUNKS)
agent = AgentService(retriever, ttl_minutes=SESSION_TTL_MINUTES)
analytics = AnalyticsStore()

app = FastAPI(title="Campus Knowledge Agent", version="1.0.0")
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
        "ollama_host": OLLAMA_HOST,
        **retriever.corpus_stats(),
    }


@app.get("/api/starter-questions")
def starter_questions() -> dict:
    return {"questions": retriever.suggest_starters()}


@app.post("/api/admin/build-embeddings")
def build_embeddings(force_rebuild: bool = False) -> dict:
    try:
        built = retriever.build_embeddings(force_rebuild=force_rebuild)
        return {"ok": built, **retriever.corpus_stats()}
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/analytics")
def analytics_snapshot() -> dict:
    return analytics.snapshot().model_dump()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = agent.chat(request.message, request.conversation_id)
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