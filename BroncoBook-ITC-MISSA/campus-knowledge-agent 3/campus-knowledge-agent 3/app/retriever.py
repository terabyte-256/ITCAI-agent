from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
from openai import OpenAI

from .models import SearchResult, SourceItem

WORD_RE = re.compile(r"[A-Za-z0-9]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
URL_SOURCE_RE = re.compile(r"^\*\*Source:\*\*\s+(https?://\S+)")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "what", "when", "where", "which",
    "with", "you", "your", "i", "we", "our", "can", "do", "does", "about", "into", "if",
}
NOISE_LINE_RE = re.compile(
    r"^(?:\*\*Source:\*\*|---|\* \[!\[|\* \[Home\]|\[apply\]|\[visit\]|\[info\]|\[give\]|\[mycpp\]|Search$|menu$|apply$|visit$|info$|give$|mycpp$)"
)


@dataclass
class Chunk:
    chunk_id: str
    title: str
    source_url: str
    markdown_file: str
    section: Optional[str]
    content: str

    def semantic_text(self) -> str:
        parts = [self.title]
        if self.section:
            parts.append(self.section)
        parts.append(self.content)
        return "\n\n".join(part.strip() for part in parts if part and part.strip())


class CorpusRetriever:
    def __init__(self, corpus_dir: str, top_k_default: int = 8) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.top_k_default = top_k_default
        self.index_map = self._load_index()
        self.url_to_title: Dict[str, str] = {}
        self.page_lookup: Dict[str, Path] = {}
        self.chunks: List[Chunk] = []
        self.term_doc_freq: Dict[str, int] = {}
        self.chunk_term_freqs: List[Dict[str, int]] = []

        self.enable_semantic_search = os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() in {"1", "true", "yes"}
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
        self.embedding_score_weight = float(os.getenv("EMBEDDING_SCORE_WEIGHT", "0.35"))
        self.lexical_score_weight = float(os.getenv("LEXICAL_SCORE_WEIGHT", "0.65"))
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.auto_build_embeddings = os.getenv("BUILD_EMBEDDINGS_ON_STARTUP", "false").lower() in {"1", "true", "yes"}

        self.embedding_client: Optional[OpenAI] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        self.embedding_cache_loaded = False
        self.embedding_cache_path: Optional[Path] = None
        self.embedding_metadata_path: Optional[Path] = None
        self.embedding_status_message = "semantic search disabled"

        self._build()
        self._configure_embedding_paths()
        self._load_or_prepare_embeddings()

    def _load_index(self) -> Dict[str, str]:
        index_path = self.corpus_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index.json in corpus dir: {self.corpus_dir}")
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("index.json must be an object mapping URL -> markdown filename")
        return data

    def _tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in WORD_RE.finditer(text)]

    def _slug_to_title(self, source_url: str, fallback: str) -> str:
        parsed = urlparse(source_url)
        path = unquote(parsed.path).strip("/")
        if not path:
            return "Cal Poly Pomona"
        parts = [p for p in path.split("/") if p and p not in {"index.shtml", "index.html"}]
        if not parts:
            return "Cal Poly Pomona"
        cleaned = parts[-1].replace(".shtml", "").replace(".html", "").replace("-", " ").replace("_", " ")
        return cleaned.title() if cleaned else fallback

    def _extract_title(self, text: str, source_url: str, fallback: str) -> str:
        seen_source = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if URL_SOURCE_RE.match(line):
                seen_source = True
                continue
            if line == "---":
                continue
            heading = HEADING_RE.match(line)
            if heading:
                title = heading.group(2).strip().strip("# ")
                if title:
                    return title
            if seen_source and len(line) <= 120 and not NOISE_LINE_RE.match(line) and not line.startswith(("*", "+", "![", "[")):
                return line
        return self._slug_to_title(source_url, fallback)

    def _extract_main_body(self, text: str) -> str:
        lines = text.splitlines()
        heading_positions = [i for i, line in enumerate(lines) if HEADING_RE.match(line.strip())]
        if heading_positions:
            start = heading_positions[0]
            if start > 20:
                return "\n".join(lines[start:]).strip()
        filtered: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered.append("")
                continue
            if NOISE_LINE_RE.match(stripped):
                continue
            filtered.append(line)
        return "\n".join(filtered).strip()

    def _split_markdown(self, text: str, title: str, source_url: str, markdown_file: str) -> List[Chunk]:
        body = self._extract_main_body(text)
        lines = body.splitlines()
        chunks: List[Chunk] = []
        current_heading: Optional[str] = None
        buffer: List[str] = []
        idx = 0

        def flush() -> None:
            nonlocal idx, buffer
            content = "\n".join(buffer).strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_id=f"{markdown_file}::chunk::{idx}",
                        title=title,
                        source_url=source_url,
                        markdown_file=markdown_file,
                        section=current_heading,
                        content=content,
                    )
                )
                idx += 1
            buffer = []

        for line in lines:
            stripped = line.strip()
            heading = HEADING_RE.match(stripped)
            if heading:
                flush()
                current_heading = heading.group(2).strip()
                buffer.append(line)
                continue
            buffer.append(line)
            if len("\n".join(buffer)) > 1200:
                flush()
        flush()

        if not chunks:
            content = body[:2000] if body else text[:2000]
            chunks.append(
                Chunk(
                    chunk_id=f"{markdown_file}::chunk::0",
                    title=title,
                    source_url=source_url,
                    markdown_file=markdown_file,
                    section=None,
                    content=content,
                )
            )
        return chunks

    def _build(self) -> None:
        all_term_freqs: List[Dict[str, int]] = []
        for source_url, md_filename in self.index_map.items():
            path = self.corpus_dir / md_filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = self._extract_title(text, source_url=source_url, fallback=md_filename)
            self.url_to_title[source_url] = title
            self.page_lookup[source_url] = path
            file_chunks = self._split_markdown(text, title, source_url, md_filename)
            self.chunks.extend(file_chunks)

        for chunk in self.chunks:
            tokens = self._tokenize(f"{chunk.title} {chunk.section or ''} {chunk.content}")
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            all_term_freqs.append(tf)
            for token in tf:
                self.term_doc_freq[token] = self.term_doc_freq.get(token, 0) + 1
        self.chunk_term_freqs = all_term_freqs

    def _corpus_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.corpus_dir.resolve()).encode("utf-8"))
        digest.update(str(len(self.index_map)).encode("utf-8"))
        digest.update(str(len(self.chunks)).encode("utf-8"))
        for source_url, md_filename in list(sorted(self.index_map.items()))[:512]:
            path = self.corpus_dir / md_filename
            stat_part = "missing"
            if path.exists():
                stat = path.stat()
                stat_part = f"{stat.st_size}:{int(stat.st_mtime)}"
            digest.update(f"{source_url}|{md_filename}|{stat_part}".encode("utf-8"))
        return digest.hexdigest()[:16]

    def _configure_embedding_paths(self) -> None:
        cache_dir = self.corpus_dir / ".embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._corpus_fingerprint()
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.embedding_model)
        base_name = f"{safe_model}-{self.embedding_dimensions}d-{fingerprint}"
        self.embedding_cache_path = cache_dir / f"{base_name}.npy"
        self.embedding_metadata_path = cache_dir / f"{base_name}.json"

    def _embedding_dimensions_arg(self) -> Dict[str, int]:
        if self.embedding_model.startswith("text-embedding-3"):
            return {"dimensions": self.embedding_dimensions}
        return {}

    def _get_embedding_client(self) -> Optional[OpenAI]:
        if self.embedding_client is not None:
            return self.embedding_client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        self.embedding_client = OpenAI(api_key=api_key)
        return self.embedding_client

    def _load_or_prepare_embeddings(self) -> None:
        if not self.enable_semantic_search:
            self.embedding_status_message = "semantic search disabled by configuration"
            return
        if self.embedding_cache_path and self.embedding_cache_path.exists():
            self.embedding_matrix = np.load(self.embedding_cache_path)
            self.embedding_cache_loaded = True
            self.embedding_status_message = f"semantic index loaded from cache: {self.embedding_cache_path.name}"
            return
        if self.auto_build_embeddings:
            built = self.build_embeddings(force_rebuild=False)
            if built:
                self.embedding_status_message = f"semantic index built: {self.embedding_cache_path.name if self.embedding_cache_path else 'cache'}"
                return
        if self._get_embedding_client() is None:
            self.embedding_status_message = "semantic search available only after embeddings are built with an OpenAI API key"
        else:
            self.embedding_status_message = "semantic search configured but cache missing; run the embedding build step"

    def _idf(self, term: str) -> float:
        df = self.term_doc_freq.get(term, 0)
        n = max(len(self.chunks), 1)
        return math.log((1 + n) / (1 + df)) + 1.0

    def _score_chunk(self, query_terms: List[str], chunk: Chunk, tf: Dict[str, int], raw_query: str) -> float:
        score = 0.0
        for term in query_terms:
            score += tf.get(term, 0) * self._idf(term)

        haystack = f"{chunk.title}\n{chunk.section or ''}\n{chunk.content}".lower()
        title_text = chunk.title.lower()
        section_text = (chunk.section or "").lower()
        raw_query_lower = raw_query.lower().strip()

        if raw_query_lower and raw_query_lower in haystack:
            score += 8.0
        if raw_query_lower and raw_query_lower in title_text:
            score += 10.0
        if raw_query_lower and raw_query_lower in section_text:
            score += 6.0
        for term in query_terms:
            if term in title_text:
                score += 2.5
            if term in section_text:
                score += 1.5
        if chunk.source_url.endswith("/index.shtml") or chunk.source_url.endswith("/index.html"):
            score += 0.75
        if len(query_terms) >= 3 and "/index." not in chunk.source_url:
            score += 0.5
        return score

    def _lexical_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_terms = [t for t in self._tokenize(query) if t not in STOPWORDS]
        if not query_terms:
            query_terms = self._tokenize(query)
        scored: List[Tuple[int, float]] = []
        for idx, (chunk, tf) in enumerate(zip(self.chunks, self.chunk_term_freqs)):
            score = self._score_chunk(query_terms, chunk, tf, query)
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(top_k * 5, 40)]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        client = self._get_embedding_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is required to build embeddings.")
        vectors: List[List[float]] = []
        extra_args = self._embedding_dimensions_arg()
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                **extra_args,
            )
            vectors.extend(item.embedding for item in response.data)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms

    def build_embeddings(self, force_rebuild: bool = False) -> bool:
        if not self.enable_semantic_search:
            return False
        if self.embedding_cache_path is None:
            self._configure_embedding_paths()
        assert self.embedding_cache_path is not None
        assert self.embedding_metadata_path is not None
        if self.embedding_cache_path.exists() and not force_rebuild:
            self.embedding_matrix = np.load(self.embedding_cache_path)
            self.embedding_cache_loaded = True
            self.embedding_status_message = f"semantic index loaded from cache: {self.embedding_cache_path.name}"
            return True

        texts = [chunk.semantic_text()[:8000] for chunk in self.chunks]
        self.embedding_matrix = self._embed_texts(texts)
        np.save(self.embedding_cache_path, self.embedding_matrix)
        metadata = {
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
            "chunk_count": len(self.chunks),
            "corpus_fingerprint": self._corpus_fingerprint(),
            "cache_file": self.embedding_cache_path.name,
        }
        self.embedding_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.embedding_cache_loaded = True
        self.embedding_status_message = f"semantic index built and saved to {self.embedding_cache_path.name}"
        return True

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.embedding_matrix is None:
            return []
        query_vector = self._embed_texts([query])[0]
        similarities = self.embedding_matrix @ query_vector
        candidate_count = min(max(top_k * 5, 40), len(self.chunks))
        top_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
        ranked = sorted(((int(idx), float(similarities[idx])) for idx in top_indices), key=lambda x: x[1], reverse=True)
        return ranked

    def _rrf_fuse(self, lexical: List[Tuple[int, float]], semantic: List[Tuple[int, float]]) -> List[Tuple[int, float, float, float]]:
        fused: Dict[int, Dict[str, float]] = {}
        for rank, (idx, score) in enumerate(lexical, start=1):
            entry = fused.setdefault(idx, {"rrf": 0.0, "lexical": 0.0, "semantic": 0.0})
            entry["rrf"] += self.lexical_score_weight * (1.0 / (self.rrf_k + rank))
            entry["lexical"] = max(entry["lexical"], score)
        for rank, (idx, score) in enumerate(semantic, start=1):
            entry = fused.setdefault(idx, {"rrf": 0.0, "lexical": 0.0, "semantic": 0.0})
            entry["rrf"] += self.embedding_score_weight * (1.0 / (self.rrf_k + rank))
            entry["semantic"] = max(entry["semantic"], score)
        ranked = [
            (idx, values["rrf"], values["lexical"], values["semantic"])
            for idx, values in fused.items()
        ]
        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        return ranked

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        if not query.strip():
            return []
        k = top_k or self.top_k_default
        query_terms = [t for t in self._tokenize(query) if t not in STOPWORDS]
        if not query_terms:
            query_terms = self._tokenize(query)

        lexical = self._lexical_search(query, k)
        semantic: List[Tuple[int, float]] = []
        if self.embedding_matrix is not None:
            semantic = self._semantic_search(query, k)

        if semantic:
            ranked_candidates = self._rrf_fuse(lexical, semantic)
            candidate_indices = [idx for idx, _, _, _ in ranked_candidates]
            semantic_map = {idx: score for idx, score in semantic}
        else:
            ranked_candidates = [(idx, score, score, 0.0) for idx, score in lexical]
            candidate_indices = [idx for idx, _, _, _ in ranked_candidates]
            semantic_map = {}

        deduped: List[SearchResult] = []
        seen_ids = set()
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            lexical_score = self._score_chunk(query_terms, chunk, self.chunk_term_freqs[idx], query)
            semantic_score = semantic_map.get(idx, 0.0)
            final_score = lexical_score if not semantic else (lexical_score * self.lexical_score_weight) + (semantic_score * 20.0 * self.embedding_score_weight)
            deduped.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    score=round(final_score, 3),
                    title=chunk.title,
                    source_url=chunk.source_url,
                    markdown_file=chunk.markdown_file,
                    section=chunk.section,
                    snippet=self._snippet(chunk.content, query_terms),
                    content=chunk.content[:2200],
                    retrieval_method="hybrid" if semantic else "lexical",
                    lexical_score=round(lexical_score, 3),
                    semantic_score=round(float(semantic_score), 5) if semantic else None,
                )
            )
            if len(deduped) >= k:
                break
        return deduped

    def get_page(self, source_url: str) -> Optional[SearchResult]:
        path = self.page_lookup.get(source_url)
        if not path or not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = self.url_to_title.get(source_url, self._slug_to_title(source_url, path.name))
        body = self._extract_main_body(text)
        return SearchResult(
            chunk_id=f"{path.name}::page",
            score=1.0,
            title=title,
            source_url=source_url,
            markdown_file=path.name,
            section=None,
            snippet=self._snippet(body, [], window=400),
            content=body[:8000],
            retrieval_method="page_lookup",
            lexical_score=None,
            semantic_score=None,
        )

    def corpus_stats(self) -> Dict[str, int | bool | str]:
        return {
            "pages": len(self.index_map),
            "chunks": len(self.chunks),
            "semantic_search_enabled": self.enable_semantic_search,
            "embedding_cache_loaded": self.embedding_cache_loaded,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_status": self.embedding_status_message,
        }

    def suggest_starters(self) -> List[str]:
        preferred = [
            "What are the freshman admission requirements?",
            "How do I change my major?",
            "Where is Student Health Services located?",
            "What financial aid resources are available?",
            "What dining options are on campus?",
        ]
        return preferred

    def top_page_titles(self, limit: int = 8) -> List[str]:
        counts = Counter(chunk.title for chunk in self.chunks if chunk.title)
        return [title for title, _ in counts.most_common(limit)]

    def _snippet(self, text: str, query_terms: Iterable[str], window: int = 280) -> str:
        lowered = text.lower()
        positions = [lowered.find(term) for term in query_terms if term and lowered.find(term) != -1]
        if not positions:
            snippet = text[:window]
        else:
            start = max(0, min(positions) - 80)
            snippet = text[start : start + window]
        return re.sub(r"\s+", " ", snippet).strip()

    def to_sources(self, results: List[SearchResult]) -> List[SourceItem]:
        return [
            SourceItem(
                title=result.title,
                source_url=result.source_url,
                markdown_file=result.markdown_file,
                section=result.section,
                snippet=result.snippet,
            )
            for result in results
        ]
