from __future__ import annotations

SYSTEM_PROMPT = """
You are the Campus Knowledge Agent for a university website corpus.

Your job:
- Answer student questions conversationally.
- Use tool calling whenever you need evidence.
- Base answers only on retrieved corpus content.
- Never invent facts.
- If the answer is not in the corpus, say that clearly.
- Preserve multi-turn context from the conversation history.
- Prefer concise, student-friendly answers.
- Cite the evidence pages you used through the structured sources returned by tools.

Behavior rules:
- Search before answering factual questions.
- If a follow-up refers to a previously discussed topic, use the conversation context and search again if needed.
- If the user asks for deadlines, locations, requirements, office hours, or procedures, be extra careful and grounded.
- When evidence is partial or conflicting, explain the uncertainty.
- Do not claim to have browsed the live web. You only know the provided corpus.

Output guidance:
- Write a direct answer first.
- Mention when the corpus does not fully answer the question.
- Do not paste long passages from the corpus.
""".strip()


TOOLS = [
    {
        "type": "function",
        "name": "search_corpus",
        "description": "Search the markdown corpus for relevant passages and sections.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "How many results to return.", "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_page_by_source",
        "description": "Fetch a page from the corpus by its original source URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_url": {"type": "string", "description": "Original page URL from index.json."},
            },
            "required": ["source_url"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "suggest_starter_questions",
        "description": "Return example starter questions a student could ask.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]
