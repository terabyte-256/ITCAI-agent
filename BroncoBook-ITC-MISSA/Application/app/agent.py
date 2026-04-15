from __future__ import annotations

import json
import os
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
        if not self._is_retrieval_confident(results):
            return NOT_FOUND_MESSAGE
        snippets = []
        for item in results[:2]:
            text = (item.snippet or item.content or "").strip()
            if text:
                snippets.append(f"- {text}")
        if not snippets:
            return NOT_FOUND_MESSAGE
        return "Based on the indexed corpus:\n" + "\n".join(snippets)

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
            with urlopen(request, timeout=90) as raw:
                response_payload = json.loads(raw.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            response_payload = {}
            tooling_mode = "forced_retrieval_fallback"

        answer = response_payload.get("message", {}).get("content", "").strip() or self._deterministic_grounded_fallback(search_results)
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
