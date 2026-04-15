from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List

from .models import AnalyticsSnapshot


@dataclass
class AnalyticsStore:
    total_queries: int = 0
    unanswered_queries: int = 0
    tool_calls: int = 0
    source_counts: List[int] = field(default_factory=list)
    queries: Counter = field(default_factory=Counter)

    def record_query(self, query: str, answered: bool, source_count: int, tool_calls: int) -> None:
        self.total_queries += 1
        if not answered:
            self.unanswered_queries += 1
        self.tool_calls += tool_calls
        self.source_counts.append(source_count)
        normalized = " ".join(query.lower().split())
        self.queries[normalized] += 1

    def snapshot(self) -> AnalyticsSnapshot:
        avg = sum(self.source_counts) / len(self.source_counts) if self.source_counts else 0.0
        return AnalyticsSnapshot(
            total_queries=self.total_queries,
            unanswered_queries=self.unanswered_queries,
            tool_calls=self.tool_calls,
            avg_sources_per_answer=round(avg, 2),
            top_queries=[q for q, _ in self.queries.most_common(5)],
        )
