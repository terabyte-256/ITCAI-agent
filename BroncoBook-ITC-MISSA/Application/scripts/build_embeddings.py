from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.db import SQLiteStore
from app.retriever import CorpusRetriever


def main() -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent.parent
    corpus_dir = os.getenv("CORPUS_DIR", str(base_dir / "data" / "corpus"))
    db_path = os.getenv("DATABASE_URL", str(base_dir / "data" / "campus_agent.db"))
    top_k = int(os.getenv("TOP_K_CHUNKS", "8"))
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    model = os.getenv("EMBEDDING_MODEL")
    force_rebuild = os.getenv("FORCE_REBUILD_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}
    force_reindex = os.getenv("FORCE_REINDEX_CORPUS", "false").lower() in {"1", "true", "yes"}

    store = SQLiteStore(db_path)
    retriever = CorpusRetriever(corpus_dir, top_k_default=top_k, store=store)
    index_summary = retriever.index_corpus(force=force_reindex)
    built = retriever.build_embeddings(
        force_rebuild=force_rebuild,
        provider=provider,
        model=model,
    )
    stats = retriever.corpus_stats()
    print(
        {
            "indexed": index_summary,
            "embedding_built": built,
            "provider": provider,
            "model": model,
            **stats,
        }
    )


if __name__ == "__main__":
    main()

