from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    title: str
    source_url: str
    markdown_file: str
    section: Optional[str] = None
    snippet: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[SourceItem] = Field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    title: str
    source_url: str
    markdown_file: str
    section: Optional[str] = None
    snippet: str
    content: str
    retrieval_method: str = "lexical"
    lexical_score: Optional[float] = None
    semantic_score: Optional[float] = None


class AnalyticsSnapshot(BaseModel):
    total_queries: int
    unanswered_queries: int
    tool_calls: int
    avg_sources_per_answer: float
    top_queries: List[str]
