from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    title: str
    source_url: str
    markdown_file: str
    section: Optional[str] = None
    snippet: str


class RetrievalDebugChunk(BaseModel):
    chunk_id: str
    title: str
    heading_path: Optional[str] = None
    original_url: str
    fts_score: Optional[float] = None
    vector_score: Optional[float] = None
    final_score: Optional[float] = None


class RetrievalDebugInfo(BaseModel):
    enabled: bool = False
    mode: str = "hybrid"
    query_used: str
    tooling_mode: str
    used_tool_calls: bool = False
    top_chunks: List[RetrievalDebugChunk] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[SourceItem] = Field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    retrieval_debug: Optional[RetrievalDebugInfo] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    title: str
    source_url: str
    markdown_file: str
    section: Optional[str] = None
    snippet: str
    content: str
    retrieval_method: str = "fts"
    fts_score: Optional[float] = None
    vector_score: Optional[float] = None
    final_score: Optional[float] = None
    lexical_score: Optional[float] = None
    semantic_score: Optional[float] = None


class AnalyticsSnapshot(BaseModel):
    total_queries: int
    unanswered_queries: int
    tool_calls: int
    avg_sources_per_answer: float
    top_queries: List[str]


class IndexRequest(BaseModel):
    force_reindex: bool = False
    build_embeddings: bool = False
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
