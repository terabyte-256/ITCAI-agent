from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

from .db import SQLiteStore
from .models import ChatResponse, RetrievalDebugChunk, RetrievalDebugInfo, SearchResult, SourceItem
from .prompts import SYSTEM_PROMPT, TOOLS
from .retriever import CorpusRetriever

NOT_FOUND_MESSAGE = "I could not find that information in the indexed corpus."
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ANSWER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
    "i",
    "we",
    "our",
    "can",
    "do",
    "does",
    "about",
    "into",
    "if",
}
DOMAIN_GENERIC_TERMS = {
    "campus",
    "university",
    "college",
    "school",
    "student",
    "students",
    "bronco",
    "broncos",
    "cpp",
    "cal",
    "poly",
    "pomona",
    "question",
    "questions",
    "information",
    "resource",
    "resources",
    "available",
    "requirements",
    "requirement",
    "resource",
    "services",
    "service",
}
TERM_SYNONYMS = {
    "freshman": ["freshmen", "first", "year", "first-year"],
    "freshmen": ["freshman", "first", "year", "first-year"],
    "first-year": ["freshman", "freshmen", "first", "year"],
    "admission": ["admissions"],
    "admissions": ["admission"],
}


class ToolDispatcher:
    def __init__(
        self,
        *,
        retriever: CorpusRetriever,
        provider: str,
        model: Optional[str],
        max_results: int,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.model = model
        self.max_results = max_results

    def search_corpus(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        limit = max(1, min(int(top_k), self.max_results))
        results = self.retriever.search_corpus(
            query=query,
            top_k=limit,
            provider=self.provider,
            embedding_model=self.model,
        )
        return {
            "tool": "search_corpus",
            "query": query,
            "top_k": limit,
            "mode": results[0].retrieval_method if results else "none",
            "results": [result.model_dump() for result in results],
        }

    def get_chunk_context(self, chunk_ids: List[str]) -> Dict[str, Any]:
        unique_ids = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        chunks = self.retriever.get_chunk_context(unique_ids)
        return {
            "tool": "get_chunk_context",
            "chunk_ids": unique_ids,
            "chunks": [item.model_dump() for item in chunks],
        }

    def list_sources_for_answer(self, chunk_ids: List[str]) -> Dict[str, Any]:
        unique_ids = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        sources = self.retriever.list_sources_for_answer(unique_ids)
        return {
            "tool": "list_sources_for_answer",
            "chunk_ids": unique_ids,
            "sources": [item.model_dump() for item in sources],
        }

    def dispatch(self, tool_name: str, arguments: str | Dict[str, Any]) -> Dict[str, Any]:
        args: Dict[str, Any]
        if isinstance(arguments, str):
            args = json.loads(arguments or "{}")
        else:
            args = arguments
        if tool_name == "search_corpus":
            return self.search_corpus(str(args["query"]), int(args.get("top_k", 5)))
        if tool_name == "get_chunk_context":
            return self.get_chunk_context([str(item) for item in args.get("chunk_ids", [])])
        if tool_name == "list_sources_for_answer":
            return self.list_sources_for_answer([str(item) for item in args.get("chunk_ids", [])])
        raise ValueError(f"Unknown tool: {tool_name}")


class AgentService:
    def __init__(self, retriever: CorpusRetriever, store: SQLiteStore, ttl_minutes: int = 90) -> None:
        self.retriever = retriever
        self.store = store
        self.ttl_minutes = ttl_minutes
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.max_tool_results = int(os.getenv("MAX_TOOL_RESULTS", "6"))
        self.min_retrieval_score = float(os.getenv("MIN_RETRIEVAL_FINAL_SCORE", "0.25"))
        self.ollama_host = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")

    def _is_retrieval_confident(self, results: List[SearchResult]) -> bool:
        if not results:
            return False
        top = results[0].final_score if results[0].final_score is not None else results[0].score
        return float(top) >= self.min_retrieval_score

    def _build_retrieval_debug(
        self,
        *,
        enabled: bool,
        mode: str,
        query_used: str,
        tooling_mode: str,
        used_tool_calls: bool,
        results: List[SearchResult],
    ) -> Optional[RetrievalDebugInfo]:
        if not enabled:
            return None
        return RetrievalDebugInfo(
            enabled=True,
            mode=mode,
            query_used=query_used,
            tooling_mode=tooling_mode,
            used_tool_calls=used_tool_calls,
            top_chunks=[
                RetrievalDebugChunk(
                    chunk_id=item.chunk_id,
                    title=item.title,
                    heading_path=item.section,
                    original_url=item.source_url,
                    fts_score=item.fts_score,
                    vector_score=item.vector_score,
                    final_score=item.final_score if item.final_score is not None else item.score,
                )
                for item in results[:8]
            ],
        )

    def _deterministic_grounded_fallback(self, results: List[SearchResult]) -> str:
        return self._extract_grounded_answer("", results)

    def _sentence_score(
        self,
        sentence: str,
        result: SearchResult,
        query_terms: List[str],
        focus_terms: List[str],
        user_message: str,
    ) -> float:
        lowered = sentence.lower()
        contextual_text = "\n".join(
            part for part in [result.title.lower(), (result.section or "").lower(), lowered] if part
        )
        if not lowered.strip():
            return 0.0
        overlap = sum(1 for term in query_terms if term in contextual_text)
        if query_terms and overlap == 0:
            return 0.0
        focus_overlap = 0
        for term in focus_terms:
            related_terms = [term, *TERM_SYNONYMS.get(term, [])]
            if any(related in contextual_text for related in related_terms):
                focus_overlap += 1
        title_bonus = sum(1 for term in query_terms if term in result.title.lower())
        heading_bonus = sum(1 for term in query_terms if result.section and term in result.section.lower())
        final_score = result.final_score if result.final_score is not None else result.score
        return (
            focus_overlap * 6.0
            +
            overlap * 3.0
            + title_bonus * 0.35
            + heading_bonus * 0.35
            + float(final_score) * 4.0
            + self._intent_adjustment(user_message, sentence, result)
        )

    def _normalize_sentence(self, sentence: str) -> str:
        cleaned = re.sub(r"^#{1,6}\s*", "", sentence.strip())
        cleaned = re.sub(r"^[A-Za-z]{1,3}\s+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^([A-Za-z][A-Za-z'-]*)\s+\1\b\s*", r"\1 ", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _strip_section_prefix(self, sentence: str, section: Optional[str]) -> str:
        if not section:
            return sentence
        section_leaf = section.split(" > ")[-1].strip()
        if not section_leaf:
            return sentence
        pattern = re.compile(rf"^{re.escape(section_leaf)}\s+", re.IGNORECASE)
        return pattern.sub("", sentence).strip()

    def _query_phrases(self, user_message: str) -> List[str]:
        tokens = [term for term in self.retriever._tokenize(user_message) if term not in ANSWER_STOPWORDS]
        phrases: List[str] = []
        for index in range(len(tokens) - 1):
            phrase = f"{tokens[index]} {tokens[index + 1]}"
            if tokens[index] in DOMAIN_GENERIC_TERMS and tokens[index + 1] in DOMAIN_GENERIC_TERMS:
                continue
            phrases.append(phrase)
        return list(dict.fromkeys(phrases))

    def _intent_adjustment(self, user_message: str, sentence: str, result: SearchResult) -> float:
        query_lower = user_message.lower()
        sentence_lower = sentence.lower()
        section_lower = (result.section or "").lower()
        title_lower = result.title.lower()
        combined = f"{title_lower}\n{section_lower}\n{sentence_lower}"

        adjustment = 0.0
        asks_for_resources = "resource" in query_lower or "available" in query_lower
        asks_for_deadline = "deadline" in query_lower or "date" in query_lower or "when" in query_lower

        if asks_for_resources and "deadline" in combined:
            adjustment -= 8.0
        if asks_for_resources and any(term in combined for term in ["applying", "apply", "fafsa", "cadaa"]):
            adjustment += 4.0
        if asks_for_deadline and "deadline" in combined:
            adjustment += 4.0
        query_phrases = self._query_phrases(user_message)
        phrase_matches = [phrase for phrase in query_phrases if phrase in combined]
        if phrase_matches:
            adjustment += 6.0 * len(phrase_matches[:2])
        elif query_phrases:
            adjustment -= 6.0

        return adjustment

    def _sentence_quality_score(self, sentence: str) -> float:
        score = 0.0
        if sentence and sentence[0].isupper():
            score += 1.0
        if sentence.endswith((".", "!", "?")):
            score += 1.0
        if "##" not in sentence and "#" not in sentence:
            score += 1.0
        if len(sentence.split()) >= 6:
            score += 0.5
        if re.search(r"\b([A-Za-z]{1,3})\s+\1\b", sentence):
            score -= 2.0
        return score

    def _is_near_duplicate_sentence(self, sentence: str, selected: List[str]) -> bool:
        sentence_key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        for existing in selected:
            existing_key = re.sub(r"[^a-z0-9]+", " ", existing.lower()).strip()
            if sentence_key == existing_key:
                return True
            if sentence_key in existing_key or existing_key in sentence_key:
                return True
        return False

    def _focus_terms(self, user_message: str) -> List[str]:
        terms = [
            term for term in self.retriever._tokenize(user_message)
            if term not in ANSWER_STOPWORDS and term not in DOMAIN_GENERIC_TERMS and len(term) >= 4
        ]
        unique_terms = list(dict.fromkeys(terms))
        return unique_terms

    def _results_match_focus_terms(self, user_message: str, results: List[SearchResult]) -> bool:
        focus_terms = self._focus_terms(user_message)
        if not focus_terms:
            return True
        top_results = results[:3]
        if not top_results:
            return False

        corpus_text = "\n".join(
            "\n".join(
                filter(
                    None,
                    [
                        result.title,
                        result.section or "",
                        result.snippet or "",
                        result.content or "",
                    ],
                )
            ).lower()
            for result in top_results
        )

        matched_terms = []
        for term in focus_terms:
            related_terms = [term, *TERM_SYNONYMS.get(term, [])]
            if any(related in corpus_text for related in related_terms):
                matched_terms.append(term)
        if not matched_terms:
            return False

        if len(focus_terms) == 1:
            return True

        return len(matched_terms) >= max(1, min(len(focus_terms), 2))

    def _expanded_query_terms(self, user_message: str) -> List[str]:
        base_terms = [term for term in self.retriever._tokenize(user_message) if term not in ANSWER_STOPWORDS]
        expanded: List[str] = []
        for term in base_terms:
            expanded.append(term)
            expanded.extend(TERM_SYNONYMS.get(term, []))
        return list(dict.fromkeys(expanded))

    def _compose_answer(self, selected: List[str]) -> str:
        cleaned = [sentence.strip() for sentence in selected if sentence.strip()]
        if not cleaned:
            return NOT_FOUND_MESSAGE
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return " ".join(cleaned)
        return "\n".join(f"- {sentence}" for sentence in cleaned)

    def _extract_grounded_answer(self, user_message: str, results: List[SearchResult]) -> str:
        if not self._is_retrieval_confident(results):
            return NOT_FOUND_MESSAGE
        if user_message and not self._results_match_focus_terms(user_message, results):
            return NOT_FOUND_MESSAGE

        query_terms = self._expanded_query_terms(user_message)
        focus_terms = self._focus_terms(user_message)
        if not query_terms:
            query_terms = self.retriever._tokenize(user_message)

        candidates: List[tuple[float, int, float, str]] = []
        seen = set()
        for result_index, result in enumerate(results[:5]):
            raw_segments: List[str] = []
            if result.content:
                raw_segments.extend(segment.strip() for segment in SENTENCE_SPLIT_RE.split(result.content) if segment.strip())
            if result.snippet:
                raw_segments.extend(segment.strip() for segment in SENTENCE_SPLIT_RE.split(result.snippet) if segment.strip())

            for sentence in raw_segments:
                normalized = self._normalize_sentence(sentence)
                normalized = self._strip_section_prefix(normalized, result.section)
                if len(normalized) < 24:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                score = self._sentence_score(normalized, result, query_terms, focus_terms, user_message)
                if score <= 0:
                    continue
                seen.add(key)
                candidates.append((score, result_index, self._sentence_quality_score(normalized), normalized))

        candidates.sort(key=lambda item: (-item[0], item[1], -item[2], item[3]))
        if candidates:
            best_result_index = min(
                candidates,
                key=lambda item: (-item[0], item[1], -item[2], item[3]),
            )[1]
            best_result_candidates = [item for item in candidates if item[1] == best_result_index]
            if best_result_candidates:
                candidates = best_result_candidates
        selected: List[str] = []
        if candidates:
            top_score = candidates[0][0]
            for score, _, _, sentence in candidates:
                if score < max(1.0, top_score - 2.5):
                    continue
                if self._is_near_duplicate_sentence(sentence, selected):
                    continue
                selected.append(sentence)
                if len(selected) >= 2:
                    break

        if not selected:
            return NOT_FOUND_MESSAGE

        return self._compose_answer(selected)

    def _finalize_answer(self, user_message: str, candidate_answer: str, results: List[SearchResult]) -> str:
        if user_message and not self._results_match_focus_terms(user_message, results):
            return NOT_FOUND_MESSAGE
        deterministic = self._extract_grounded_answer(user_message, results)
        if deterministic == NOT_FOUND_MESSAGE:
            return NOT_FOUND_MESSAGE
        return deterministic

    def _record_analytics(
        self,
        *,
        provider: str,
        model: Optional[str],
        event_type: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store.add_analytics_event(
            event_type=event_type,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    def _save_turn(
        self,
        *,
        conversation_id: str,
        user_message: str,
        answer: str,
        provider: str,
        model: Optional[str],
        sources: List[SourceItem],
    ) -> None:
        self.store.add_message(conversation_id, "user", user_message, provider=provider, model=model)
        assistant_message_id = self.store.add_message(conversation_id, "assistant", answer, provider=provider, model=model)
        self.store.add_citations(assistant_message_id, [source.model_dump() for source in sources])

    def _build_openai_input(self, conversation_id: str, user_message: str) -> List[Dict[str, Any]]:
        history = self.store.get_recent_messages(conversation_id, limit=12)
        input_items: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}
        ]
        for item in history:
            input_items.append(
                {
                    "role": item["role"],
                    "content": [{"type": "input_text", "text": item["content"]}],
                }
            )
        input_items.append({"role": "user", "content": [{"type": "input_text", "text": user_message}]})
        return input_items

    def _generate_openai_grounded_answer(
        self,
        *,
        user_message: str,
        context_chunks: List[SearchResult],
        model: str,
    ) -> str:
        if self.openai_client is None:
            return NOT_FOUND_MESSAGE
        context_text = "\n\n".join(
            (
                f"[{idx + 1}] chunk_id={item.chunk_id}\n"
                f"title={item.title}\n"
                f"section={item.section or '(none)'}\n"
                f"url={item.source_url}\n"
                f"content={item.content}"
            )
            for idx, item in enumerate(context_chunks)
        )
        response = self.openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Use only the retrieved corpus context below.\n\n"
                                f"Context:\n{context_text}\n\n"
                                f"Question: {user_message}"
                            ),
                        }
                    ],
                },
            ],
        )
        return (response.output_text or "").strip() or NOT_FOUND_MESSAGE

    def _chat_openai(self, user_message: str, conversation_id: str, model: Optional[str], debug: bool) -> ChatResponse:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        started = time.time()
        active_model = model or self.openai_model
        dispatcher = ToolDispatcher(
            retriever=self.retriever,
            provider="openai",
            model=None,
            max_results=self.max_tool_results,
        )
        input_items = self._build_openai_input(conversation_id, user_message)
        tool_trace: List[Dict[str, Any]] = []
        last_results: List[SearchResult] = []
        usage_input = None
        usage_output = None
        used_tool_calls = False
        tooling_mode = "openai_tool_calls"

        for _ in range(6):
            response = self.openai_client.responses.create(
                model=active_model,
                input=input_items,
                tools=TOOLS,
            )
            response_items = response.output or []
            usage_input = getattr(response.usage, "input_tokens", None) if getattr(response, "usage", None) else None
            usage_output = getattr(response.usage, "output_tokens", None) if getattr(response, "usage", None) else None
            input_items.extend([item.model_dump() for item in response_items])

            function_calls = [item for item in response_items if item.type == "function_call"]
            if not function_calls:
                answer_text = (response.output_text or "").strip() or NOT_FOUND_MESSAGE
                if not used_tool_calls:
                    tooling_mode = "openai_forced_retrieval_fallback"
                    forced = dispatcher.search_corpus(user_message, self.max_tool_results)
                    last_results = [SearchResult(**item) for item in forced.get("results", [])]
                    if self._is_retrieval_confident(last_results):
                        answer_text = self._generate_openai_grounded_answer(
                            user_message=user_message,
                            context_chunks=last_results,
                            model=active_model,
                        )
                    else:
                        answer_text = NOT_FOUND_MESSAGE

                sources = self.retriever.to_sources(last_results)
                if not self._is_retrieval_confident(last_results):
                    answer_text = NOT_FOUND_MESSAGE
                    sources = []
                if not sources:
                    answer_text = NOT_FOUND_MESSAGE
                else:
                    answer_text = self._finalize_answer(user_message, answer_text, last_results)

                self._save_turn(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    answer=answer_text,
                    provider="openai",
                    model=active_model,
                    sources=sources,
                )
                elapsed_ms = int((time.time() - started) * 1000)
                total_tokens = (usage_input or 0) + (usage_output or 0) if usage_input is not None else None
                self._record_analytics(
                    provider="openai",
                    model=active_model,
                    event_type="chat_completion",
                    prompt_tokens=usage_input,
                    completion_tokens=usage_output,
                    total_tokens=total_tokens,
                    latency_ms=elapsed_ms,
                    metadata={
                        "conversation_id": conversation_id,
                        "tool_calls": len(tool_trace),
                        "source_count": len(sources),
                        "tooling_mode": tooling_mode,
                    },
                )
                return ChatResponse(
                    answer=answer_text,
                    conversation_id=conversation_id,
                    sources=sources,
                    tool_trace=tool_trace,
                    provider="openai",
                    model=active_model,
                    retrieval_debug=self._build_retrieval_debug(
                        enabled=debug,
                        mode=last_results[0].retrieval_method if last_results else "none",
                        query_used=user_message,
                        tooling_mode=tooling_mode,
                        used_tool_calls=used_tool_calls,
                        results=last_results,
                    ),
                )

            for call in function_calls:
                used_tool_calls = True
                tool_output = dispatcher.dispatch(call.name, call.arguments)
                tool_trace.append({"tool": call.name, "arguments": json.loads(call.arguments or "{}")})
                if call.name == "search_corpus":
                    last_results = [SearchResult(**item) for item in tool_output.get("results", [])]
                elif call.name == "get_chunk_context":
                    last_results = [SearchResult(**item) for item in tool_output.get("chunks", [])]
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(tool_output),
                    }
                )

        self._save_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            answer=NOT_FOUND_MESSAGE,
            provider="openai",
            model=active_model,
            sources=[],
        )
        elapsed_ms = int((time.time() - started) * 1000)
        self._record_analytics(
            provider="openai",
            model=active_model,
            event_type="chat_fallback",
            prompt_tokens=usage_input,
            completion_tokens=usage_output,
            total_tokens=(usage_input or 0) + (usage_output or 0) if usage_input is not None else None,
            latency_ms=elapsed_ms,
            metadata={"conversation_id": conversation_id, "tool_calls": len(tool_trace)},
        )
        return ChatResponse(
            answer=NOT_FOUND_MESSAGE,
            conversation_id=conversation_id,
            sources=[],
            tool_trace=tool_trace,
            provider="openai",
            model=active_model,
            retrieval_debug=self._build_retrieval_debug(
                enabled=debug,
                mode=last_results[0].retrieval_method if last_results else "none",
                query_used=user_message,
                tooling_mode=tooling_mode,
                used_tool_calls=used_tool_calls,
                results=last_results,
            ),
        )

    def _chat_ollama(self, user_message: str, conversation_id: str, model: Optional[str], debug: bool) -> ChatResponse:
        started = time.time()
        active_model = model or self.ollama_model
        dispatcher = ToolDispatcher(
            retriever=self.retriever,
            provider="ollama",
            model=None,
            max_results=self.max_tool_results,
        )
        tool_trace: List[Dict[str, Any]] = []

        search_output = dispatcher.search_corpus(user_message, self.max_tool_results)
        tool_trace.append({"tool": "search_corpus", "arguments": {"query": user_message, "top_k": self.max_tool_results}})
        search_results = [SearchResult(**item) for item in search_output.get("results", [])]
        if not self._is_retrieval_confident(search_results):
            answer = NOT_FOUND_MESSAGE
            self._save_turn(
                conversation_id=conversation_id,
                user_message=user_message,
                answer=answer,
                provider="ollama",
                model=active_model,
                sources=[],
            )
            elapsed_ms = int((time.time() - started) * 1000)
            self._record_analytics(
                provider="ollama",
                model=active_model,
                event_type="chat_completion",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_ms=elapsed_ms,
                metadata={
                    "conversation_id": conversation_id,
                    "tool_calls": len(tool_trace),
                    "source_count": 0,
                    "tooling_mode": "ollama_deterministic_fallback",
                },
            )
            return ChatResponse(
                answer=answer,
                conversation_id=conversation_id,
                sources=[],
                tool_trace=tool_trace,
                provider="ollama",
                model=active_model,
                retrieval_debug=self._build_retrieval_debug(
                    enabled=debug,
                    mode=search_results[0].retrieval_method if search_results else "none",
                    query_used=user_message,
                    tooling_mode="ollama_deterministic_fallback",
                    used_tool_calls=False,
                    results=search_results,
                ),
            )

        chunk_ids = [result.chunk_id for result in search_results]
        context_output = dispatcher.get_chunk_context(chunk_ids)
        tool_trace.append({"tool": "get_chunk_context", "arguments": {"chunk_ids": chunk_ids}})
        sources_output = dispatcher.list_sources_for_answer(chunk_ids)
        tool_trace.append({"tool": "list_sources_for_answer", "arguments": {"chunk_ids": chunk_ids}})
        sources = [SourceItem(**item) for item in sources_output.get("sources", [])]

        history = self.store.get_recent_messages(conversation_id, limit=8)
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        context_items = context_output.get("chunks", [])
        context_text = "\n\n".join(
            (
                f"[{idx + 1}] chunk_id={item.get('chunk_id')}\n"
                f"title={item.get('title')}\n"
                f"section={item.get('section') or '(none)'}\n"
                f"url={item.get('source_url')}\n"
                f"content={item.get('content')}"
            )
            for idx, item in enumerate(context_items)
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Conversation history:\n"
            f"{history_text or '(none)'}\n\n"
            "Retrieved corpus context:\n"
            f"{context_text}\n\n"
            f"User question:\n{user_message}"
        )

        payload = json.dumps(
            {
                "model": active_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(f"{self.ollama_host}/api/chat", data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        tooling_mode = "ollama_deterministic_fallback"
        try:
            with urlopen(request, timeout=12) as raw:
                response_payload = json.loads(raw.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            response_payload = {}
            tooling_mode = "forced_retrieval_fallback"

        answer = response_payload.get("message", {}).get("content", "").strip() or self._extract_grounded_answer(user_message, search_results)
        answer = self._finalize_answer(user_message, answer, search_results)
        if not sources:
            answer = NOT_FOUND_MESSAGE

        self._save_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            answer=answer,
            provider="ollama",
            model=active_model,
            sources=sources,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        prompt_tokens = response_payload.get("prompt_eval_count")
        completion_tokens = response_payload.get("eval_count")
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None else None
        self._record_analytics(
            provider="ollama",
            model=active_model,
            event_type="chat_completion",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
            metadata={
                "conversation_id": conversation_id,
                "tool_calls": len(tool_trace),
                "source_count": len(sources),
                "tooling_mode": tooling_mode,
            },
        )
        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            tool_trace=tool_trace,
            provider="ollama",
            model=active_model,
            retrieval_debug=self._build_retrieval_debug(
                enabled=debug,
                mode=search_results[0].retrieval_method if search_results else "none",
                query_used=user_message,
                tooling_mode=tooling_mode,
                used_tool_calls=False,
                results=search_results,
            ),
        )

    def chat(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        debug: bool = False,
    ) -> ChatResponse:
        selected_provider = provider.strip().lower() if provider else "openai"
        if selected_provider not in {"openai", "ollama"}:
            selected_provider = "openai"
        if selected_provider == "openai" and self.openai_client is None:
            selected_provider = "ollama"

        conversation_id = self.store.ensure_conversation(
            conversation_id=conversation_id,
            title=user_message[:80],
        )
        if selected_provider == "openai":
            return self._chat_openai(user_message, conversation_id, model, debug)
        return self._chat_ollama(user_message, conversation_id, model, debug)
