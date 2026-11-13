"""Persistent pgvector storage with exact cosine search."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import time

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

from .models import Chunk
from .settings import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    EMBEDDING_DIMENSIONS,
    MAX_SEQUENCE_LENGTH,
)

ROOT = Path(__file__).resolve().parents[1]


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int):
        if not database_url:
            raise ValueError("DATABASE_URL is missing")

        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Use the configured embedding dimensions"
            )

        self.database_url = database_url
        self.dimensions = EMBEDDING_DIMENSIONS

    def connect(self):
        return psycopg.connect(
            self.database_url,
            connect_timeout=10,
            row_factory=dict_row,
        )

    def create_schema(self):
        with self.connect() as connection:
            connection.execute(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )

            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS day6_chunks (
                    strategy TEXT NOT NULL
                        CHECK (strategy IN ('A', 'B')),
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL CHECK (page > 0),
                    chunk_text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_revision TEXT NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
            """)

            row = connection.execute("""
                SELECT
                    format_type(atttypid, atttypmod)
                    AS vector_type
                FROM pg_attribute
                WHERE attrelid = 'day6_chunks'::regclass
                  AND attname = 'embedding'
                  AND NOT attisdropped
            """).fetchone()

            expected = f"vector({self.dimensions})"
            if row["vector_type"] != expected:
                raise ValueError(
                    "Existing vector column has different dimensions"
                )

    def vector_literal(self, vector):
        array = np.asarray(vector, dtype=np.float64)

        if (
            array.shape != (self.dimensions,)
            or not np.isfinite(array).all()
        ):
            raise ValueError("Invalid vector shape or values")

        if np.linalg.norm(array) < 1e-12:
            raise ValueError(
                "Cosine search cannot use a zero vector"
            )

        return "[" + ",".join(
            str(float(value)) for value in array
        ) + "]"

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors,
    ):
        """Replace the COMPLETE snapshot of A or B."""
        if strategy not in {"A", "B"}:
            raise ValueError("Store A/B only; C reuses B")

        if not chunks or len(chunks) != len(vectors):
            raise ValueError(
                "Need matching nonempty chunks and vectors"
            )

        ids = [chunk.chunk_id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")

        rows = []

        for chunk, vector in zip(chunks, vectors):
            if chunk.metadata.get("strategy") != strategy:
                raise ValueError("Chunk strategy mismatch")

            if (
                not chunk.text.strip()
                or not chunk.metadata.get("section")
            ):
                raise ValueError(
                    "Chunk text or section is missing"
                )

            rows.append((
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                Jsonb(chunk.metadata),
                EMBEDDING_MODEL,
                EMBEDDING_REVISION,
                self.vector_literal(vector),
            ))

        with self.connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"day6_chunks_{strategy}",),
            )

            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO day6_chunks (
                        strategy,
                        chunk_id,
                        document_id,
                        source,
                        page,
                        chunk_text,
                        metadata,
                        embedding_model,
                        embedding_revision,
                        embedding
                    )
                    VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector
                    )
                    ON CONFLICT (strategy, chunk_id)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        chunk_text = EXCLUDED.chunk_text,
                        metadata = EXCLUDED.metadata,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_revision = EXCLUDED.embedding_revision,
                        embedding = EXCLUDED.embedding
                """, rows)

            connection.execute("""
                DELETE FROM day6_chunks
                WHERE strategy = %s
                  AND NOT (chunk_id = ANY(%s::text[]))
            """, (strategy, ids))

    def count(self, strategy: str) -> int:
        if strategy == "C":
            strategy = "B"

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT count(*) AS total
                FROM day6_chunks
                WHERE strategy = %s
                """,
                (strategy,),
            ).fetchone()

        return row["total"]

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ):
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Unknown strategy")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 1000
        ):
            raise ValueError(
                "top_k must be an integer from 1 to 1000"
            )

        if filters and strategy != "C":
            raise ValueError(
                "Only configuration C applies filters"
            )

        stored_strategy = "B" if strategy == "C" else strategy

        params = [
            self.vector_literal(query_vector),
            stored_strategy,
            EMBEDDING_MODEL,
            EMBEDDING_REVISION,
        ]

        clauses = [
            "strategy = %s",
            "embedding_model = %s",
            "embedding_revision = %s",
        ]

        allowed_filters = {
            "section",
            "chapter",
            "language",
            "year",
            "version",
        }

        for key, value in (filters or {}).items():
            if key not in allowed_filters:
                raise ValueError(
                    f"Unsupported metadata filter: {key}"
                )

            if (
                not isinstance(value, (str, int))
                or isinstance(value, bool)
            ):
                raise ValueError(
                    "Filter values must be strings or integers"
                )

            clauses.append("metadata ->> %s = %s")
            params.extend([key, str(value)])

        params.append(top_k)

        query = """
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                chunk_text,
                metadata,
                embedding <=> %s::vector AS distance
            FROM day6_chunks
            WHERE
        """

        query += " AND ".join(clauses)
        query += " ORDER BY distance, chunk_id LIMIT %s"

        with self.connect() as connection:
            rows = connection.execute(
                query, params
            ).fetchall()

        return [
            {
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "text": row["chunk_text"],
                "score": 1.0 - row["distance"],
                "cosine_distance": row["distance"],
                "metadata": {
                    **row["metadata"],
                    "document_id": row["document_id"],
                    "source": row["source"],
                    "page": row["page"],
                },
            }
            for rank, row in enumerate(rows, 1)
        ]


def load_saved_embeddings(strategy):
    from .embeddings import validate_vectors

    folder = ROOT / "data/processed"

    chunks = [
        json.loads(line)
        for line in (
            folder / f"chunks_{strategy}.jsonl"
        ).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    meta = json.loads(
        (
            folder / f"embeddings_{strategy}.meta.json"
        ).read_text(encoding="utf-8")
    )

    path = folder / f"embeddings_{strategy}.npy"

    inputs = {
        "chunks": chunks,
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalize_embeddings": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "encoder_code_sha256": hashlib.sha256(
            (ROOT / "src/embeddings.py").read_bytes()
        ).hexdigest(),
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            inputs,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    if (
        meta.get("fingerprint") != fingerprint
        or meta.get("chunk_ids")
        != [chunk["chunk_id"] for chunk in chunks]
        or meta.get("vectors_sha256")
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Embedding cache is stale or mismatched; "
            "rerun src.embeddings"
        )

    vectors = validate_vectors(
        np.load(path, allow_pickle=False),
        len(chunks),
    )

    return [
        Chunk(**chunk) for chunk in chunks
    ], vectors


def main():
    from .embeddings import MyEmbedder

    load_dotenv(ROOT / ".env")

    store = PgVectorStore(
        os.getenv("DATABASE_URL"),
        EMBEDDING_DIMENSIONS,
    )
    store.create_schema()

    stats = {
        "at": datetime.now(timezone.utc).isoformat(),
        "metric": "cosine",
        "search": "exact; no approximate vector index",
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "strategies": {},
    }

    for strategy in ["A", "B"]:
        chunks, vectors = load_saved_embeddings(strategy)

        started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        seconds = time.perf_counter() - started

        before = store.count(strategy)

        # Repeat the full ingestion to check for duplicates.
        store.upsert(strategy, chunks, vectors)
        after = store.count(strategy)

        if before != len(chunks) or after != before:
            raise ValueError(
                "Re-ingestion duplicated or lost chunks"
            )

        stats["strategies"][strategy] = {
            "chunks": after,
            "first_upsert_seconds": seconds,
            "idempotency_verified": True,
        }

        print(
            f"{strategy}: {after} rows; re-ingestion OK"
        )

    question = (
        "According to Section 4.5, why does the Moon "
        "always keep the same face toward Earth?"
    )

    vector = MyEmbedder().embed_query(question)
    probes = []

    for strategy in ["A", "B", "C"]:
        filters = (
            {"section": "4.5"}
            if strategy == "C"
            else None
        )

        hits = store.search(
            strategy,
            vector,
            top_k=3,
            filters=filters,
        )

        if len(hits) != 3:
            raise ValueError(
                "Expected three retrieved chunks"
            )

        if filters and not all(
            hit["metadata"]["section"] == "4.5"
            for hit in hits
        ):
            raise ValueError("Metadata filtering failed")

        probes.append({
            "strategy": strategy,
            "question": question,
            "filter": filters,
            "hits": hits,
        })

        print(f"{strategy}: top-3 retrieved")

    stats["example_retrieval"] = probes

    output = ROOT / "results/store_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Store OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log = ROOT / "results/errors.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)

        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "store",
                "error": str(error),
            }) + "\n")

        raise