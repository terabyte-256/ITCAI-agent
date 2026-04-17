from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.retriever import CorpusRetriever


def main() -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent.parent
    corpus_dir = os.getenv("CORPUS_DIR", str(base_dir / "data" / "cpp_corpus"))
    top_k = int(os.getenv("TOP_K_CHUNKS", "8"))
    retriever = CorpusRetriever(corpus_dir, top_k_default=top_k)
    built = retriever.build_embeddings(force_rebuild=os.getenv("FORCE_REBUILD_EMBEDDINGS", "false").lower() in {"1", "true", "yes"})
    stats = retriever.corpus_stats()
    print({"built": built, **stats})


if __name__ == "__main__":
    main()
