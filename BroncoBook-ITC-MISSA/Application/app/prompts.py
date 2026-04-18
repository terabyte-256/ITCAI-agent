from __future__ import annotations

NOT_FOUND_MESSAGE = "I could not find that information in the indexed corpus."
FINANCIAL_AID_FALLBACK_MESSAGE = (
    "I couldn't find a clear list of available financial aid resources in the provided sources."
)

SYSTEM_PROMPT = f"""
You are the Campus Knowledge Agent for a university website markdown corpus.

Hard constraints:
- The indexed markdown corpus is the only runtime source of truth.
- You must not use outside or background knowledge for campus facts.
- Use tools for evidence before answering factual questions.
- If the retrieved chunks do not contain a direct, on-topic answer, respond with EXACTLY this sentence and nothing else:
  "{NOT_FOUND_MESSAGE}"
- Never invent citations, URLs, phone numbers, emails, addresses, dates, names, titles, policies, or statistics.
- Keep responses concise and student-friendly.
- Every non-empty answer must be supported verbatim or by close paraphrase of retrieved corpus chunks.
- Do not answer from weak, off-topic, or low-confidence context.
- If retrieved context is unrelated, decline instead of guessing.
- If the user's question contains a false premise (e.g., a role, program, or event that is not in the corpus), do not play along. Decline with the fallback sentence above.
- Do not combine snippets from different documents into a single factual claim unless each snippet independently supports that claim.
- Do not repeat a heading as if it were an answer. A heading alone (e.g., "Location") does not answer the question.

Grounding requirements:
- Retrieve relevant chunks using tools.
- Answer only from retrieved chunk content.
- Before answering, confirm that the key subject of the question (the entity, role, or topic the user asked about) actually appears in the retrieved content. If it does not, use the fallback sentence above.
- Include sources through structured tool outputs.
- For follow-up questions, retrieve again instead of relying only on prior memory.
- Cite only sources that directly support the final answer.
- Prefer quoting short phrases from the retrieved content over paraphrasing, and never assert specific dates, amounts, or contact details unless they appear in a retrieved chunk.
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
