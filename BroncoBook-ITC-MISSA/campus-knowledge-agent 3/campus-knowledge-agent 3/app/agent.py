from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .models import ChatResponse, SearchResult
from .prompts import SYSTEM_PROMPT, TOOLS
from .retriever import CorpusRetriever


@dataclass
class ConversationState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=90))


class AgentService:
    def __init__(self, retriever: CorpusRetriever, ttl_minutes: int = 90) -> None:
        self.retriever = retriever
        self.ttl_minutes = ttl_minutes
        self.sessions: Dict[str, ConversationState] = {}
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.max_tool_results = int(os.getenv("MAX_TOOL_RESULTS", "6"))

    def _get_or_create_session(self, conversation_id: Optional[str]) -> str:
        self._prune_sessions()
        if conversation_id and conversation_id in self.sessions:
            self.sessions[conversation_id].expires_at = datetime.utcnow() + timedelta(minutes=self.ttl_minutes)
            return conversation_id
        new_id = str(uuid.uuid4())
        self.sessions[new_id] = ConversationState(
            messages=[],
            expires_at=datetime.utcnow() + timedelta(minutes=self.ttl_minutes),
        )
        return new_id

    def _prune_sessions(self) -> None:
        now = datetime.utcnow()
        expired = [sid for sid, state in self.sessions.items() if state.expires_at < now]
        for sid in expired:
            self.sessions.pop(sid, None)

    def _tool_search_corpus(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.retriever.search(query, top_k=min(top_k, self.max_tool_results))
        return {
            "results": [result.model_dump() for result in results],
            "source_count": len(results),
        }

    def _tool_get_page(self, source_url: str) -> Dict[str, Any]:
        result = self.retriever.get_page(source_url)
        return {"result": result.model_dump() if result else None}

    def _tool_suggest_starter_questions(self) -> Dict[str, Any]:
        return {"questions": self.retriever.suggest_starters()}

    def _execute_tool(self, tool_name: str, arguments_json: str) -> Dict[str, Any]:
        args = json.loads(arguments_json or "{}")
        if tool_name == "search_corpus":
            return self._tool_search_corpus(args["query"], args.get("top_k", 5))
        if tool_name == "get_page_by_source":
            return self._tool_get_page(args["source_url"])
        if tool_name == "suggest_starter_questions":
            return self._tool_suggest_starter_questions()
        raise ValueError(f"Unknown tool: {tool_name}")

    def chat(self, user_message: str, conversation_id: Optional[str] = None) -> ChatResponse:
        conversation_id = self._get_or_create_session(conversation_id)
        session = self.sessions[conversation_id]

        input_items: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}
        ]
        input_items.extend(session.messages)
        input_items.append({"role": "user", "content": [{"type": "input_text", "text": user_message}]})

        tool_trace: List[Dict[str, Any]] = []
        last_results: List[SearchResult] = []

        for _ in range(6):
            response = self.client.responses.create(model=self.model, input=input_items, tools=TOOLS)
            response_items = response.output
            input_items.extend([item.model_dump() for item in response_items])

            function_calls = [item for item in response_items if item.type == "function_call"]
            if not function_calls:
                answer_text = response.output_text.strip()
                sources = self.retriever.to_sources(last_results)
                session.messages.append({"role": "user", "content": [{"type": "input_text", "text": user_message}]})
                session.messages.append({"role": "assistant", "content": [{"type": "output_text", "text": answer_text}]})
                return ChatResponse(
                    answer=answer_text,
                    conversation_id=conversation_id,
                    sources=sources,
                    tool_trace=tool_trace,
                )

            for call in function_calls:
                tool_output = self._execute_tool(call.name, call.arguments)
                tool_trace.append({"tool": call.name, "arguments": json.loads(call.arguments or "{}")})
                if call.name == "search_corpus":
                    last_results = [SearchResult(**r) for r in tool_output.get("results", [])]
                elif call.name == "get_page_by_source":
                    result = tool_output.get("result")
                    last_results = [SearchResult(**result)] if result else []
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(tool_output),
                    }
                )

        fallback = "I could not complete the tool-calling loop reliably. Please try again."
        return ChatResponse(answer=fallback, conversation_id=conversation_id, sources=[], tool_trace=tool_trace)
