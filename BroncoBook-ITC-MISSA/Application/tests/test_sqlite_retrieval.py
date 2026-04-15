import json
import math
from pathlib import Path
from urllib.error import URLError

import app.agent as agent_module
from app.agent import AgentService, ToolDispatcher
from app.db import SQLiteStore
from app.models import SearchResult
from app.retriever import CorpusRetriever


def _build_test_retriever(tmp_path: Path) -> tuple[SQLiteStore, CorpusRetriever]:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "admissions.md").write_text(
        "# Admissions\n\n## Deadlines\n\nFreshman application deadline is Nov 30.\n\n## Requirements\n\nSubmit transcripts.",
        encoding="utf-8",
    )
    (corpus_dir / "aid.md").write_text(
        "# Financial Aid\n\n## Deadlines\n\nPriority FAFSA deadline is March 2.\n\n## Contact\n\nVisit One Stop.",
        encoding="utf-8",
    )
    (corpus_dir / "index.json").write_text(
        json.dumps(
            {
                "https://example.edu/admissions": "admissions.md",
                "https://example.edu/aid": "aid.md",
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "agent.db"
    store = SQLiteStore(str(db_path))
    retriever = CorpusRetriever(str(corpus_dir), top_k_default=5, store=store)
    return store, retriever


def test_stable_retrieval_ordering(tmp_path: Path):
    _, retriever = _build_test_retriever(tmp_path)
    first = retriever.search_corpus("deadline", top_k=5, provider="openai")
    second = retriever.search_corpus("deadline", top_k=5, provider="openai")
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_normalized_vector_storage(tmp_path: Path):
    store, retriever = _build_test_retriever(tmp_path)
    row = store.fetchone("SELECT id FROM document_chunks ORDER BY id LIMIT 1")
    assert row is not None
    chunk_id = str(row["id"])

    store.upsert_embedding(
        chunk_id=chunk_id,
        provider="openai",
        embedding_model=retriever.embedding_model,
        vector=[3.0, 4.0],
    )
    vectors = store.fetch_embeddings(provider="openai", embedding_model=retriever.embedding_model)
    vec = vectors[chunk_id]
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)


def test_hybrid_ranking_shape(tmp_path: Path):
    store, retriever = _build_test_retriever(tmp_path)
    chunk_rows = store.fetchall("SELECT id FROM document_chunks ORDER BY id")
    assert len(chunk_rows) >= 2
    first_id = str(chunk_rows[0]["id"])
    second_id = str(chunk_rows[1]["id"])
    store.upsert_embedding(
        chunk_id=first_id,
        provider="openai",
        embedding_model=retriever.embedding_model,
        vector=[1.0, 0.0],
    )
    store.upsert_embedding(
        chunk_id=second_id,
        provider="openai",
        embedding_model=retriever.embedding_model,
        vector=[0.0, 1.0],
    )
    retriever._embed_texts = lambda texts, provider, model: [[1.0, 0.0] for _ in texts]  # type: ignore[method-assign]

    results = retriever.hybrid_search("admissions deadline", top_k=3, provider="openai")
    assert results
    item = results[0]
    assert item.fts_score is not None
    assert item.final_score is not None
    assert item.retrieval_method in {"hybrid", "fts"}
    assert any(result.vector_score is not None for result in results)


def test_tool_output_shapes(tmp_path: Path):
    _, retriever = _build_test_retriever(tmp_path)
    dispatcher = ToolDispatcher(retriever=retriever, provider="openai", model=None, max_results=6)

    search_output = dispatcher.search_corpus("FAFSA deadline", top_k=3)
    assert {"tool", "query", "top_k", "mode", "results"} <= set(search_output.keys())
    assert search_output["tool"] == "search_corpus"
    assert search_output["results"]

    chunk_ids = [row["chunk_id"] for row in search_output["results"]]
    context_output = dispatcher.get_chunk_context(chunk_ids)
    assert {"tool", "chunk_ids", "chunks"} <= set(context_output.keys())
    assert context_output["tool"] == "get_chunk_context"

    sources_output = dispatcher.list_sources_for_answer(chunk_ids)
    assert {"tool", "chunk_ids", "sources"} <= set(sources_output.keys())
    assert sources_output["tool"] == "list_sources_for_answer"


def test_no_answer_threshold_logic(tmp_path: Path):
    store, retriever = _build_test_retriever(tmp_path)
    agent = AgentService(retriever, store=store)
    agent.min_retrieval_score = 0.9
    low = SearchResult(
        chunk_id="chunk-low",
        document_id="doc-low",
        score=0.2,
        title="Low",
        source_url="https://example.edu/low",
        markdown_file="low.md",
        section=None,
        snippet="",
        content="",
        retrieval_method="hybrid",
        fts_score=0.2,
        vector_score=0.1,
        final_score=0.2,
    )
    assert agent._is_retrieval_confident([low]) is False


def test_provider_consistent_tool_result_shape(tmp_path: Path):
    _, retriever = _build_test_retriever(tmp_path)
    openai_dispatcher = ToolDispatcher(retriever=retriever, provider="openai", model=None, max_results=5)
    ollama_dispatcher = ToolDispatcher(retriever=retriever, provider="ollama", model=None, max_results=5)

    openai_output = openai_dispatcher.search_corpus("deadline", top_k=3)
    ollama_output = ollama_dispatcher.search_corpus("deadline", top_k=3)
    assert set(openai_output.keys()) == set(ollama_output.keys())
    assert openai_output["results"] and ollama_output["results"]
    assert set(openai_output["results"][0].keys()) == set(ollama_output["results"][0].keys())


def test_ollama_unreachable_falls_back_to_grounded_response(tmp_path: Path, monkeypatch):
    store, retriever = _build_test_retriever(tmp_path)
    agent = AgentService(retriever, store=store)
    monkeypatch.setattr(agent_module, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))
    response = agent.chat("What is the FAFSA deadline?", provider="ollama", debug=True)
    assert response.answer
    assert response.sources
    assert response.retrieval_debug is not None
    assert response.retrieval_debug.tooling_mode == "forced_retrieval_fallback"
