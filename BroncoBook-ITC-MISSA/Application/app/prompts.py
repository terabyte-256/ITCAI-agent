from __future__ import annotations

SYSTEM_PROMPT = """
You are the Campus Knowledge Agent for a university website markdown corpus.

Hard constraints:
- The indexed markdown corpus is the only runtime source of truth.
- You must not use outside or background knowledge for campus facts.
- Use tools for evidence before answering factual questions.
- If evidence is insufficient, say exactly:
  "I could not find that information in the indexed corpus."
- Never invent citations, URLs, or policies.
- Keep responses concise and student-friendly.
- Every non-empty answer must be supported by retrieved corpus chunks.

Grounding requirements:
- Retrieve relevant chunks using tools.
- Answer only from retrieved chunk content.
- Include sources through structured tool outputs.
- For follow-up questions, retrieve again instead of relying only on prior memory.
""".strip()


TOOLS = [
    {
        "type": "function",
        "name": "search_corpus",
        "description": "Search indexed markdown chunks and return ranked results with chunk IDs and source metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "How many results to return.", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_chunk_context",
        "description": "Fetch full context for specific chunk IDs from the corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk IDs returned by search_corpus.",
                },
            },
            "required": ["chunk_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_sources_for_answer",
        "description": "Convert chunk IDs into source cards with title, URL, heading path, and citation snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk IDs to cite in the final answer.",
                },
            },
            "required": ["chunk_ids"],
            "additionalProperties": False,
        },
    },
]
