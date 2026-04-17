from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    original_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading_path TEXT,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    vector_dim INTEGER,
    is_normalized INTEGER NOT NULL DEFAULT 1,
    embedding_blob BLOB,
    embedding_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
    UNIQUE(chunk_id, provider, embedding_model)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    original_url TEXT NOT NULL,
    heading_path TEXT,
    cited_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);
CREATE INDEX IF NOT EXISTS idx_documents_original_url ON documents(original_url);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_embeddings_provider_model ON chunk_embeddings(provider, embedding_model);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON chunk_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id);
CREATE INDEX IF NOT EXISTS idx_analytics_event ON analytics_events(event_type, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    heading_path,
    tokenize = 'porter unicode61'
);
"""


def normalize_vector(vector: Sequence[float]) -> List[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [float(v) / norm for v in vector]


def encode_embedding_blob(vector: Sequence[float]) -> bytes:
    values = [float(v) for v in vector]
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def decode_embedding_blob(blob: bytes, vector_dim: Optional[int]) -> List[float]:
    if not blob:
        return []
    if vector_dim is not None and vector_dim > 0:
        expected_bytes = vector_dim * 4
        if len(blob) != expected_bytes:
            return []
        return list(struct.unpack(f"<{vector_dim}f", blob))
    if len(blob) % 4 != 0:
        return []
    size = len(blob) // 4
    return list(struct.unpack(f"<{size}f", blob))


class SQLiteStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA_SQL)
        self._migrate_schema()
        self.conn.commit()

    def _column_names(self, table_name: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _migrate_schema(self) -> None:
        columns = self._column_names("chunk_embeddings")
        if "vector_dim" not in columns:
            self.conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN vector_dim INTEGER")
        if "is_normalized" not in columns:
            self.conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN is_normalized INTEGER NOT NULL DEFAULT 1")
        if "embedding_blob" not in columns:
            self.conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN embedding_blob BLOB")
        if "embedding_json" not in columns:
            self.conn.execute("ALTER TABLE chunk_embeddings ADD COLUMN embedding_json TEXT")

        rows = self.conn.execute(
            """
            SELECT id, embedding_json, embedding_blob, vector_dim, is_normalized
            FROM chunk_embeddings
            """
        ).fetchall()
        for row in rows:
            if row["embedding_blob"] is not None:
                continue
            raw_json = row["embedding_json"]
            if not raw_json:
                continue
            try:
                parsed = json.loads(str(raw_json))
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            normalized = normalize_vector([float(v) for v in parsed])
            self.conn.execute(
                """
                UPDATE chunk_embeddings
                SET embedding_blob = ?, vector_dim = ?, is_normalized = 1
                WHERE id = ?
                """,
                (encode_embedding_blob(normalized), len(normalized), str(row["id"])),
            )
        self.conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cursor

    def executemany(self, sql: str, values: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self.conn.executemany(sql, values)
            self.conn.commit()

    def fetchone(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    def fetchall(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    def upsert_embedding(
        self,
        *,
        chunk_id: str,
        provider: str,
        embedding_model: str,
        vector: Sequence[float],
    ) -> None:
        normalized = normalize_vector(vector)
        self.execute(
            """
            INSERT INTO chunk_embeddings (
                id, chunk_id, provider, embedding_model, vector_dim, is_normalized, embedding_blob, embedding_json
            ) VALUES (?, ?, ?, ?, ?, 1, ?, NULL)
            ON CONFLICT(chunk_id, provider, embedding_model) DO UPDATE SET
                vector_dim = excluded.vector_dim,
                is_normalized = 1,
                embedding_blob = excluded.embedding_blob,
                embedding_json = NULL,
                created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (
                str(uuid.uuid4()),
                chunk_id,
                provider,
                embedding_model,
                len(normalized),
                encode_embedding_blob(normalized),
            ),
        )

    def fetch_embeddings(
        self,
        *,
        provider: str,
        embedding_model: str,
    ) -> Dict[str, List[float]]:
        rows = self.fetchall(
            """
            SELECT chunk_id, vector_dim, is_normalized, embedding_blob, embedding_json
            FROM chunk_embeddings
            WHERE provider = ? AND embedding_model = ?
            """,
            (provider, embedding_model),
        )
        output: Dict[str, List[float]] = {}
        for row in rows:
            vector: List[float] = []
            if row["embedding_blob"] is not None:
                vector = decode_embedding_blob(bytes(row["embedding_blob"]), row["vector_dim"])
            elif row["embedding_json"]:
                try:
                    parsed = json.loads(str(row["embedding_json"]))
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, list):
                    vector = [float(v) for v in parsed]
            if not vector:
                continue
            if not bool(row["is_normalized"]):
                vector = normalize_vector(vector)
            output[str(row["chunk_id"])] = vector
        return output

    def ensure_conversation(self, conversation_id: Optional[str], title: Optional[str] = None) -> str:
        if conversation_id:
            row = self.fetchone("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if row:
                return str(row["id"])
        new_id = conversation_id or str(uuid.uuid4())
        self.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (new_id, title))
        return new_id

    def get_recent_messages(self, conversation_id: str, limit: int = 12) -> List[Dict[str, str]]:
        rows = self.fetchall(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ? AND role IN ('user', 'assistant')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows.reverse()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        self.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, provider, model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, provider, model),
        )
        self.execute(
            "UPDATE conversations SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (conversation_id,),
        )
        return message_id

    def add_citations(self, message_id: str, citations: List[Dict[str, Any]]) -> None:
        if not citations:
            return
        values = []
        for citation in citations:
            values.append(
                (
                    str(uuid.uuid4()),
                    message_id,
                    citation.get("chunk_id"),
                    citation.get("document_id"),
                    citation.get("source_url"),
                    citation.get("section"),
                    citation.get("snippet", ""),
                )
            )
        self.executemany(
            """
            INSERT INTO citations (
                id, message_id, chunk_id, document_id, original_url, heading_path, cited_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    def add_analytics_event(
        self,
        *,
        event_type: str,
        provider: Optional[str],
        model: Optional[str],
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        latency_ms: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO analytics_events (
                id, event_type, provider, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                event_type,
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                latency_ms,
                json.dumps(metadata or {}),
            ),
        )

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self.fetchone(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if row is None:
            return None

        message_rows = self.fetchall(
            """
            SELECT id, role, content, provider, model, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        out_messages: List[Dict[str, Any]] = []
        for message_row in message_rows:
            message_id = str(message_row["id"])
            citation_rows = self.fetchall(
                """
                SELECT chunk_id, document_id, original_url, heading_path, cited_text
                FROM citations
                WHERE message_id = ?
                ORDER BY created_at ASC
                """,
                (message_id,),
            )
            out_messages.append(
                {
                    "id": message_id,
                    "role": str(message_row["role"]),
                    "content": str(message_row["content"]),
                    "provider": message_row["provider"],
                    "model": message_row["model"],
                    "created_at": str(message_row["created_at"]),
                    "citations": [
                        {
                            "chunk_id": str(c["chunk_id"]),
                            "document_id": str(c["document_id"]),
                            "original_url": str(c["original_url"]),
                            "heading_path": c["heading_path"],
                            "cited_text": str(c["cited_text"]),
                        }
                        for c in citation_rows
                    ],
                }
            )

        return {
            "id": str(row["id"]),
            "title": row["title"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "messages": out_messages,
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
