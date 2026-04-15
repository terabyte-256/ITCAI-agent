from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

from .db import SQLiteStore
from .models import SearchResult, SourceItem

WORD_RE = re.compile(r"[A-Za-z0-9]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
URL_SOURCE_RE = re.compile(r"^\*\*Source:\*\*\s+(https?://\S+)")
STOPWORDS = {
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
NOISE_LINE_RE = re.compile(
    r"^(?:\*\*Source:\*\*|---|\* \[!\[|\* \[Home\]|\[apply\]|\[visit\]|\[info\]|\[give\]|\[mycpp\]|Search$|menu$|apply$|visit$|info$|give$|mycpp$)"
)


@dataclass
class ChunkDraft:
    chunk_index: int
    heading_path: Optional[str]
    content: str
    token_count: int


class CorpusRetriever:
    def __init__(
        self,
        corpus_dir: str,
        top_k_default: int = 8,
        store: Optional[SQLiteStore] = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir).resolve()
        self.top_k_default = top_k_default
        self.index_map = self._load_index()
        self.max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", "260"))

        self.enable_semantic_search = os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() in {"1", "true", "yes"}
        self.embedding_provider_default = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
        self.embedding_score_weight = float(os.getenv("EMBEDDING_SCORE_WEIGHT", "0.35"))
        self.lexical_score_weight = float(os.getenv("LEXICAL_SCORE_WEIGHT", "0.65"))
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.auto_build_embeddings = os.getenv("BUILD_EMBEDDINGS_ON_STARTUP", "false").lower() in {"1", "true", "yes"}

        self.ollama_host = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        database_url = os.getenv("DATABASE_URL", str(Path(__file__).resolve().parent.parent / "data" / "campus_agent.db"))
        self.store = store or SQLiteStore(database_url)
        self.embedding_client: Optional[OpenAI] = None
        self.embedding_status_message = "semantic search not initialized"

        self.index_corpus(force=False)
        self._load_or_prepare_embeddings()

    def _load_index(self) -> Dict[str, str]:
        index_path = self.corpus_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index.json in corpus dir: {self.corpus_dir}")
        with index_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("index.json must be an object mapping URL -> markdown filename")
        return {str(k): str(v) for k, v in data.items()}

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()

    def _tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in WORD_RE.finditer(text)]

    def _estimate_token_count(self, text: str) -> int:
        words = len(self._tokenize(text))
        return max(1, int(math.ceil(words * 1.3)))

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _slug_to_title(self, source_url: str, fallback: str) -> str:
        parsed = urlparse(source_url)
        path = unquote(parsed.path).strip("/")
        if not path:
            return "Campus Page"
        parts = [p for p in path.split("/") if p and p not in {"index.shtml", "index.html"}]
        if not parts:
            return "Campus Page"
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
            if heading and heading.group(1) == "#":
                candidate = heading.group(2).strip().strip("# ")
                if candidate:
                    return candidate
            if seen_source and len(line) <= 120 and not NOISE_LINE_RE.match(line):
                if not line.startswith(("*", "+", "![", "[")):
                    return line
        return self._slug_to_title(source_url, fallback)

    def _extract_main_body(self, text: str) -> str:
        lines = text.splitlines()
        filtered: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered.append("")
                continue
            if NOISE_LINE_RE.match(stripped):
                continue
            filtered.append(line)
        return self._normalize_whitespace("\n".join(filtered))

    def _split_by_headings(self, text: str) -> List[Tuple[Optional[str], str]]:
        lines = self._extract_main_body(text).splitlines()
        heading_stack: List[Optional[str]] = [None] * 6
        current_heading: Optional[str] = None
        buffer: List[str] = []
        sections: List[Tuple[Optional[str], str]] = []

        def flush() -> None:
            nonlocal buffer
            content = self._normalize_whitespace("\n".join(buffer))
            if content:
                sections.append((current_heading, content))
            buffer = []

        for line in lines:
            heading_match = HEADING_RE.match(line.strip())
            if heading_match:
                flush()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack[level - 1] = title
                for idx in range(level, len(heading_stack)):
                    heading_stack[idx] = None
                path_parts = [h for h in heading_stack if h]
                current_heading = " > ".join(path_parts) if path_parts else None
                buffer.append(line)
                continue
            buffer.append(line)

        flush()
        return sections

    def _split_section(self, heading_path: Optional[str], content: str) -> List[ChunkDraft]:
        paragraphs = [self._normalize_whitespace(p) for p in re.split(r"\n{2,}", content) if p.strip()]
        if not paragraphs:
            paragraphs = [content]

        chunks: List[ChunkDraft] = []
        buffer: List[str] = []
        token_budget = 0
        chunk_index = 0

        def flush() -> None:
            nonlocal buffer, token_budget, chunk_index
            normalized = self._normalize_whitespace("\n\n".join(buffer))
            if not normalized:
                buffer = []
                token_budget = 0
                return
            chunks.append(
                ChunkDraft(
                    chunk_index=chunk_index,
                    heading_path=heading_path,
                    content=normalized,
                    token_count=self._estimate_token_count(normalized),
                )
            )
            chunk_index += 1
            buffer = []
            token_budget = 0

        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_token_count(paragraph)
            if paragraph_tokens > self.max_chunk_tokens:
                words = paragraph.split()
                slice_size = max(80, int(self.max_chunk_tokens / 1.3))
                for start in range(0, len(words), slice_size):
                    piece = " ".join(words[start : start + slice_size])
                    chunks.append(
                        ChunkDraft(
                            chunk_index=chunk_index,
                            heading_path=heading_path,
                            content=piece,
                            token_count=self._estimate_token_count(piece),
                        )
                    )
                    chunk_index += 1
                continue

            if token_budget and token_budget + paragraph_tokens > self.max_chunk_tokens:
                flush()

            buffer.append(paragraph)
            token_budget += paragraph_tokens

        flush()
        return chunks

    def _chunk_document(self, text: str) -> List[ChunkDraft]:
        sections = self._split_by_headings(text)
        all_chunks: List[ChunkDraft] = []
        for heading_path, section_content in sections:
            all_chunks.extend(self._split_section(heading_path, section_content))
        # Reindex globally for deterministic ordering.
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx
        return all_chunks

    def _chunk_id(self, document_id: str, chunk_index: int, heading_path: Optional[str], content: str) -> str:
        digest = self._sha256(f"{document_id}:{chunk_index}:{heading_path or ''}:{content}")[:16]
        return f"{document_id}:{chunk_index}:{digest}"

    def index_corpus(self, force: bool = False) -> Dict[str, int | List[str]]:
        summary = {
            "mapped_files": len(self.index_map),
            "indexed_documents": 0,
            "skipped_documents": 0,
            "indexed_chunks": 0,
            "warnings": [],
        }

        corpus_root = self.corpus_dir.resolve()
        for source_url, markdown_file in self.index_map.items():
            if not markdown_file.endswith(".md"):
                summary["warnings"].append(f"Skipped non-markdown entry: {markdown_file}")
                continue

            path = (corpus_root / markdown_file).resolve()
            if corpus_root not in path.parents and path != corpus_root:
                summary["warnings"].append(f"Skipped path outside corpus dir: {markdown_file}")
                continue
            if not path.exists():
                summary["warnings"].append(f"Missing markdown file: {markdown_file}")
                continue

            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            normalized_text = self._normalize_whitespace(raw_text)
            checksum = self._sha256(normalized_text)
            title = self._extract_title(normalized_text, source_url, markdown_file)

            existing = self.store.fetchone(
                "SELECT id, checksum FROM documents WHERE file_path = ?",
                (markdown_file,),
            )
            if existing and str(existing["checksum"]) == checksum and not force:
                summary["skipped_documents"] += 1
                continue

            document_id = str(existing["id"]) if existing else str(uuid.uuid4())
            self.store.execute(
                """
                INSERT INTO documents (id, file_path, original_url, title, checksum)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    original_url = excluded.original_url,
                    title = excluded.title,
                    checksum = excluded.checksum,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                (document_id, markdown_file, source_url, title, checksum),
            )

            old_chunk_rows = self.store.fetchall(
                "SELECT id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            old_chunk_ids = [str(row["id"]) for row in old_chunk_rows]
            if old_chunk_ids:
                placeholders = ", ".join("?" for _ in old_chunk_ids)
                self.store.execute(
                    f"DELETE FROM document_chunks_fts WHERE chunk_id IN ({placeholders})",
                    old_chunk_ids,
                )
                self.store.execute(
                    f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                    old_chunk_ids,
                )
            self.store.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))

            chunks = self._chunk_document(normalized_text)
            if not chunks:
                chunks = [
                    ChunkDraft(
                        chunk_index=0,
                        heading_path=None,
                        content=normalized_text[:2000],
                        token_count=self._estimate_token_count(normalized_text[:2000]),
                    )
                ]

            for chunk in chunks:
                chunk_id = self._chunk_id(document_id, chunk.chunk_index, chunk.heading_path, chunk.content)
                metadata = {
                    "document_id": document_id,
                    "source_filename": markdown_file,
                    "heading_path": chunk.heading_path,
                    "chunk_index": chunk.chunk_index,
                    "source_url": source_url,
                    "title": title,
                }
                self.store.execute(
                    """
                    INSERT INTO document_chunks (id, document_id, chunk_index, heading_path, content, token_count, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk.chunk_index,
                        chunk.heading_path,
                        chunk.content,
                        chunk.token_count,
                        json.dumps(metadata),
                    ),
                )
                self.store.execute(
                    "INSERT INTO document_chunks_fts (chunk_id, content, heading_path) VALUES (?, ?, ?)",
                    (chunk_id, chunk.content, chunk.heading_path or ""),
                )

            summary["indexed_documents"] += 1
            summary["indexed_chunks"] += len(chunks)

        return summary

    def _get_embedding_client(self) -> Optional[OpenAI]:
        if self.embedding_client is not None:
            return self.embedding_client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        self.embedding_client = OpenAI(api_key=api_key)
        return self.embedding_client

    def _normalize_vector(self, vector: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm <= 1e-12:
            return [0.0 for _ in vector]
        return [float(v / norm) for v in vector]

    def _embed_openai(self, texts: List[str], model: str) -> List[List[float]]:
        client = self._get_embedding_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY is required to build OpenAI embeddings.")
        extra_args: Dict[str, int] = {}
        if model.startswith("text-embedding-3"):
            extra_args["dimensions"] = self.embedding_dimensions
        response = client.embeddings.create(model=model, input=texts, **extra_args)
        return [self._normalize_vector(item.embedding) for item in response.data]

    def _embed_ollama(self, texts: List[str], model: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
            request = Request(f"{self.ollama_host}/api/embeddings", data=payload, method="POST")
            request.add_header("Content-Type", "application/json")
            try:
                with urlopen(request, timeout=90) as raw:
                    body = json.loads(raw.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as exc:
                raise RuntimeError(f"Ollama embeddings request failed: {exc}") from exc
            vector = body.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("Ollama embeddings response missing vector.")
            vectors.append(self._normalize_vector([float(v) for v in vector]))
        return vectors

    def _embed_texts(self, texts: List[str], provider: str, model: str) -> List[List[float]]:
        if provider == "openai":
            return self._embed_openai(texts, model)
        if provider == "ollama":
            return self._embed_ollama(texts, model)
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def build_embeddings(
        self,
        force_rebuild: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bool:
        if not self.enable_semantic_search:
            self.embedding_status_message = "semantic search disabled by configuration"
            return False

        target_provider = (provider or self.embedding_provider_default or "openai").strip().lower()
        if target_provider == "ollama":
            target_model = model or self.ollama_embedding_model
        else:
            target_model = model or self.embedding_model

        if force_rebuild:
            self.store.execute(
                "DELETE FROM chunk_embeddings WHERE provider = ? AND embedding_model = ?",
                (target_provider, target_model),
            )

        chunk_rows = self.store.fetchall(
            """
            SELECT dc.id AS chunk_id, d.title AS title, dc.heading_path AS heading_path, dc.content AS content
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            ORDER BY d.file_path, dc.chunk_index
            """
        )
        if not chunk_rows:
            self.embedding_status_message = "no chunks available for embedding"
            return False

        existing_rows = self.store.fetchall(
            "SELECT chunk_id FROM chunk_embeddings WHERE provider = ? AND embedding_model = ?",
            (target_provider, target_model),
        )
        existing_chunk_ids = {str(row["chunk_id"]) for row in existing_rows}

        to_embed: List[Tuple[str, str]] = []
        for row in chunk_rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in existing_chunk_ids and not force_rebuild:
                continue
            semantic_text = "\n\n".join(
                part.strip()
                for part in [
                    str(row["title"]),
                    str(row["heading_path"] or ""),
                    str(row["content"]),
                ]
                if part and part.strip()
            )
            to_embed.append((chunk_id, semantic_text[:8000]))

        if not to_embed:
            self.embedding_status_message = f"semantic index already available for {target_provider}/{target_model}"
            return True

        for start in range(0, len(to_embed), self.embedding_batch_size):
            batch = to_embed[start : start + self.embedding_batch_size]
            vectors = self._embed_texts([text for _, text in batch], target_provider, target_model)
            for (chunk_id, _), vector in zip(batch, vectors, strict=False):
                self.store.upsert_embedding(
                    chunk_id=chunk_id,
                    provider=target_provider,
                    embedding_model=target_model,
                    vector=vector,
                )

        self.embedding_status_message = (
            f"semantic index built for {target_provider}/{target_model} ({len(to_embed)} chunks)"
        )
        return True

    def _load_or_prepare_embeddings(self) -> None:
        if not self.enable_semantic_search:
            self.embedding_status_message = "semantic search disabled by configuration"
            return
        if self.auto_build_embeddings:
            try:
                self.build_embeddings(force_rebuild=False)
                return
            except Exception as exc:
                self.embedding_status_message = f"embedding auto-build failed: {exc}"
                return
        self.embedding_status_message = "semantic search available after embedding build"

    def _dot_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b, strict=False)))

    def _score_chunk_text(self, query_terms: List[str], title: str, heading: str, content: str, source_url: str, raw_query: str) -> float:
        score = 0.0
        haystack = f"{title}\n{heading}\n{content}".lower()
        title_lower = title.lower()
        heading_lower = heading.lower()
        raw_query_lower = raw_query.lower().strip()

        for term in query_terms:
            score += haystack.count(term) * 1.2
        if raw_query_lower and raw_query_lower in haystack:
            score += 6.0
        if raw_query_lower and raw_query_lower in title_lower:
            score += 8.0
        if raw_query_lower and raw_query_lower in heading_lower:
            score += 4.0
        for term in query_terms:
            if term in title_lower:
                score += 1.7
            if term in heading_lower:
                score += 1.2
        if source_url.endswith("/index.shtml") or source_url.endswith("/index.html"):
            score += 0.6
        return score

    def _lexical_search(self, query: str, candidate_limit: int) -> List[Tuple[str, float]]:
        query_terms = [t for t in self._tokenize(query) if t not in STOPWORDS]
        if not query_terms:
            query_terms = self._tokenize(query)
        if not query_terms:
            return []

        fts_query = " OR ".join(query_terms)
        try:
            fts_rows = self.store.fetchall(
                """
                SELECT chunk_id, bm25(document_chunks_fts) AS bm25_score
                FROM document_chunks_fts
                WHERE document_chunks_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, candidate_limit),
            )
        except sqlite3.OperationalError:
            fallback = " OR ".join(f'"{token}"' for token in query_terms)
            fts_rows = self.store.fetchall(
                """
                SELECT chunk_id, bm25(document_chunks_fts) AS bm25_score
                FROM document_chunks_fts
                WHERE document_chunks_fts MATCH ?
                LIMIT ?
                """,
                (fallback, candidate_limit),
            )

        scored: List[Tuple[str, float]] = []
        for row in fts_rows:
            chunk_id = str(row["chunk_id"])
            bm25_score = float(row["bm25_score"] if row["bm25_score"] is not None else 0.0)
            lexical_score = 1.0 / (1.0 + abs(bm25_score))
            scored.append((chunk_id, lexical_score))

        if scored:
            return sorted(scored, key=lambda item: (-item[1], item[0]))

        rows = self.store.fetchall(
            """
            SELECT dc.id AS chunk_id, d.title AS title, dc.heading_path AS heading_path, dc.content AS content, d.original_url AS source_url
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            """
        )
        fallback_scored = []
        for row in rows:
            score = self._score_chunk_text(
                query_terms,
                str(row["title"]),
                str(row["heading_path"] or ""),
                str(row["content"]),
                str(row["source_url"]),
                query,
            )
            if score > 0:
                fallback_scored.append((str(row["chunk_id"]), score))
        fallback_scored.sort(key=lambda item: (-item[1], item[0]))
        return fallback_scored[:candidate_limit]

    def _semantic_search(
        self,
        query: str,
        candidate_limit: int,
        provider: str,
        embedding_model: str,
    ) -> List[Tuple[str, float]]:
        if not self.enable_semantic_search:
            return []
        vectors_by_chunk = self.store.fetch_embeddings(provider=provider, embedding_model=embedding_model)
        if not vectors_by_chunk:
            return []

        query_vector = self._normalize_vector(self._embed_texts([query], provider=provider, model=embedding_model)[0])
        scored: List[Tuple[str, float]] = []
        for chunk_id, vector in vectors_by_chunk.items():
            score = max(-1.0, min(1.0, self._dot_similarity(query_vector, vector)))
            scored.append((chunk_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:candidate_limit]

    def _normalize_scores(self, score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        minimum = min(values)
        maximum = max(values)
        if maximum - minimum <= 1e-12:
            return {chunk_id: 1.0 for chunk_id in score_map}
        return {
            chunk_id: float((score - minimum) / (maximum - minimum))
            for chunk_id, score in score_map.items()
        }

    def _snippet(self, text: str, query_terms: Iterable[str], window: int = 280) -> str:
        lowered = text.lower()
        positions = [lowered.find(term) for term in query_terms if term and lowered.find(term) != -1]
        if not positions:
            snippet = text[:window]
        else:
            start = max(0, min(positions) - 80)
            snippet = text[start : start + window]
        return re.sub(r"\s+", " ", snippet).strip()

    def _fetch_chunk_details(self, chunk_ids: List[str]) -> Dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = self.store.fetchall(
            f"""
            SELECT
                dc.id AS chunk_id,
                dc.document_id AS document_id,
                dc.heading_path AS heading_path,
                dc.content AS content,
                d.title AS title,
                d.original_url AS source_url,
                d.file_path AS markdown_file
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.id IN ({placeholders})
            """,
            chunk_ids,
        )
        return {str(row["chunk_id"]): row for row in rows}

    def _build_scored_results(
        self,
        *,
        chunk_ids: List[str],
        query: str,
        method: str,
        fts_scores: Dict[str, float],
        vector_scores: Dict[str, float],
        final_scores: Dict[str, float],
        top_k: int,
    ) -> List[SearchResult]:
        details = self._fetch_chunk_details(chunk_ids)
        query_terms = [t for t in self._tokenize(query) if t not in STOPWORDS] or self._tokenize(query)
        output: List[SearchResult] = []
        for chunk_id in chunk_ids:
            row = details.get(chunk_id)
            if row is None:
                continue
            content = str(row["content"])
            fts_score = float(fts_scores.get(chunk_id, 0.0))
            vector_score = float(vector_scores.get(chunk_id, 0.0)) if chunk_id in vector_scores else None
            final_score = float(final_scores.get(chunk_id, fts_score))
            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(row["document_id"]),
                    score=round(final_score, 6),
                    title=str(row["title"]),
                    source_url=str(row["source_url"]),
                    markdown_file=str(row["markdown_file"]),
                    section=row["heading_path"],
                    snippet=self._snippet(content, query_terms),
                    content=content[:2600],
                    retrieval_method=method,
                    fts_score=round(fts_score, 6) if chunk_id in fts_scores else None,
                    vector_score=round(vector_score, 6) if vector_score is not None else None,
                    final_score=round(final_score, 6),
                    lexical_score=round(fts_score, 6) if chunk_id in fts_scores else None,
                    semantic_score=round(vector_score, 6) if vector_score is not None else None,
                )
            )
            if len(output) >= top_k:
                break
        return output

    def fts_search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        if not query.strip():
            return []
        k = top_k or self.top_k_default
        lexical_pairs = self._lexical_search(query, candidate_limit=max(k * 5, 40))
        lexical_raw = {chunk_id: score for chunk_id, score in lexical_pairs}
        lexical_norm = self._normalize_scores(lexical_raw)
        ordered_ids = sorted(lexical_norm.keys(), key=lambda chunk_id: (-lexical_norm[chunk_id], chunk_id))
        return self._build_scored_results(
            chunk_ids=ordered_ids,
            query=query,
            method="fts",
            fts_scores=lexical_norm,
            vector_scores={},
            final_scores=lexical_norm,
            top_k=k,
        )

    def vector_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []
        k = top_k or self.top_k_default
        retrieval_provider = (provider or self.embedding_provider_default or "openai").strip().lower()
        retrieval_model = embedding_model or (
            self.ollama_embedding_model if retrieval_provider == "ollama" else self.embedding_model
        )
        semantic_pairs = self._semantic_search(query, candidate_limit=max(k * 5, 40), provider=retrieval_provider, embedding_model=retrieval_model)
        semantic_raw = {chunk_id: (score + 1.0) / 2.0 for chunk_id, score in semantic_pairs}
        semantic_norm = self._normalize_scores(semantic_raw)
        ordered_ids = sorted(semantic_norm.keys(), key=lambda chunk_id: (-semantic_norm[chunk_id], chunk_id))
        return self._build_scored_results(
            chunk_ids=ordered_ids,
            query=query,
            method="vector",
            fts_scores={},
            vector_scores=semantic_norm,
            final_scores=semantic_norm,
            top_k=k,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []
        k = top_k or self.top_k_default
        candidate_limit = max(k * 5, 40)
        retrieval_provider = (provider or self.embedding_provider_default or "openai").strip().lower()
        retrieval_model = embedding_model or (
            self.ollama_embedding_model if retrieval_provider == "ollama" else self.embedding_model
        )

        lexical_pairs = self._lexical_search(query, candidate_limit=candidate_limit)
        lexical_norm = self._normalize_scores({chunk_id: score for chunk_id, score in lexical_pairs})
        semantic_pairs: List[Tuple[str, float]] = []
        if self.enable_semantic_search:
            try:
                semantic_pairs = self._semantic_search(
                    query,
                    candidate_limit=candidate_limit,
                    provider=retrieval_provider,
                    embedding_model=retrieval_model,
                )
            except Exception:
                semantic_pairs = []
        vector_norm = self._normalize_scores({chunk_id: (score + 1.0) / 2.0 for chunk_id, score in semantic_pairs})

        if vector_norm:
            total_weight = self.lexical_score_weight + self.embedding_score_weight
            lexical_weight = self.lexical_score_weight / total_weight if total_weight > 0 else 0.5
            vector_weight = self.embedding_score_weight / total_weight if total_weight > 0 else 0.5
            method = "hybrid"
        else:
            lexical_weight = 1.0
            vector_weight = 0.0
            method = "fts"

        all_ids = sorted(set(lexical_norm.keys()) | set(vector_norm.keys()))
        final_scores = {
            chunk_id: (lexical_weight * lexical_norm.get(chunk_id, 0.0)) + (vector_weight * vector_norm.get(chunk_id, 0.0))
            for chunk_id in all_ids
        }
        ordered_ids = sorted(
            all_ids,
            key=lambda chunk_id: (
                -final_scores.get(chunk_id, 0.0),
                -lexical_norm.get(chunk_id, 0.0),
                -vector_norm.get(chunk_id, 0.0),
                chunk_id,
            ),
        )
        return self._build_scored_results(
            chunk_ids=ordered_ids,
            query=query,
            method=method,
            fts_scores=lexical_norm,
            vector_scores=vector_norm,
            final_scores=final_scores,
            top_k=k,
        )

    def search_corpus(
        self,
        query: str,
        top_k: Optional[int] = None,
        provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[SearchResult]:
        return self.hybrid_search(query, top_k=top_k, provider=provider, embedding_model=embedding_model)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[SearchResult]:
        return self.search_corpus(query, top_k=top_k, provider=provider, embedding_model=embedding_model)

    def get_chunk_context(self, chunk_ids: List[str]) -> List[SearchResult]:
        if not chunk_ids:
            return []
        details = self._fetch_chunk_details(chunk_ids)
        output: List[SearchResult] = []
        for chunk_id in chunk_ids:
            row = details.get(chunk_id)
            if row is None:
                continue
            content = str(row["content"])
            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(row["document_id"]),
                    score=1.0,
                    title=str(row["title"]),
                    source_url=str(row["source_url"]),
                    markdown_file=str(row["markdown_file"]),
                    section=row["heading_path"],
                    snippet=self._snippet(content, []),
                    content=content[:2600],
                    retrieval_method="context_lookup",
                    fts_score=None,
                    vector_score=None,
                    final_score=1.0,
                    lexical_score=None,
                    semantic_score=None,
                )
            )
        return output

    def list_sources_for_answer(self, chunk_ids: List[str]) -> List[SourceItem]:
        return self.to_sources(self.get_chunk_context(chunk_ids))

    def get_page(self, source_url: str) -> Optional[SearchResult]:
        row = self.store.fetchone(
            """
            SELECT d.id AS document_id, d.title AS title, d.file_path AS markdown_file, d.original_url AS source_url
            FROM documents d
            WHERE d.original_url = ?
            """,
            (source_url,),
        )
        if row is None:
            return None
        chunk_rows = self.store.fetchall(
            """
            SELECT heading_path, content
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (str(row["document_id"]),),
        )
        content = "\n\n".join(str(r["content"]) for r in chunk_rows)[:8000]
        return SearchResult(
            chunk_id=f"{row['document_id']}::page",
            document_id=str(row["document_id"]),
            score=1.0,
            title=str(row["title"]),
            source_url=str(row["source_url"]),
            markdown_file=str(row["markdown_file"]),
            section=None,
            snippet=self._snippet(content, [], window=400),
            content=content,
            retrieval_method="page_lookup",
            fts_score=None,
            vector_score=None,
            final_score=1.0,
            lexical_score=None,
            semantic_score=None,
        )

    def corpus_stats(self) -> Dict[str, int | bool | str]:
        page_row = self.store.fetchone("SELECT COUNT(*) AS count FROM documents")
        chunk_row = self.store.fetchone("SELECT COUNT(*) AS count FROM document_chunks")
        embedding_row = self.store.fetchone(
            "SELECT COUNT(*) AS count FROM chunk_embeddings WHERE provider = ?",
            (self.embedding_provider_default,),
        )
        return {
            "pages": int(page_row["count"] if page_row else 0),
            "chunks": int(chunk_row["count"] if chunk_row else 0),
            "semantic_search_enabled": self.enable_semantic_search,
            "embedding_cache_loaded": int(embedding_row["count"] if embedding_row else 0) > 0,
            "embedding_provider": self.embedding_provider_default,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_status": self.embedding_status_message,
        }

    def suggest_starters(self) -> List[str]:
        return [
            "What are the freshman admission requirements?",
            "How do I change my major?",
            "Where is Student Health Services located?",
            "What financial aid resources are available?",
            "What dining options are on campus?",
        ]

    def top_page_titles(self, limit: int = 8) -> List[str]:
        rows = self.store.fetchall(
            """
            SELECT title, COUNT(*) AS count
            FROM documents
            GROUP BY title
            ORDER BY count DESC, title ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [str(row["title"]) for row in rows]

    def to_sources(self, results: List[SearchResult]) -> List[SourceItem]:
        return [
            SourceItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                title=result.title,
                source_url=result.source_url,
                markdown_file=result.markdown_file,
                section=result.section,
                snippet=result.snippet,
            )
            for result in results
        ]
