"""Real local embeddings with content-keyed caching."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .settings import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    EMBEDDING_DIMENSIONS,
    MAX_SEQUENCE_LENGTH,
    SEED,
)

ROOT = Path(__file__).resolve().parents[1]


def validate_vectors(vectors, count):
    array = np.asarray(vectors, dtype=np.float32)

    if array.shape != (count, EMBEDDING_DIMENSIONS):
        raise ValueError(
            f"Unexpected embedding shape: {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain non-finite values")

    norms = np.linalg.norm(array, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not unit-normalized")

    return array


class MyEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if model_name != EMBEDDING_MODEL:
            raise ValueError(
                "Use the shared model in src/settings.py"
            )

        self.model_name = model_name
        self.dimensions = EMBEDDING_DIMENSIONS

        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        np.random.seed(SEED)

        self.model = SentenceTransformer(
            model_name,
            revision=EMBEDDING_REVISION,
            device="cpu",
        )

        if (
            self.model.get_sentence_embedding_dimension()
            != self.dimensions
        ):
            raise ValueError(
                "Model dimension differs from settings"
            )

        if self.model.max_seq_length != MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Model sequence length differs from settings"
            )

    def _encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("Expected a list of strings")

        if not texts:
            return []

        if not all(
            isinstance(text, str) and text.strip()
            for text in texts
        ):
            raise ValueError(
                "Embedding inputs must be nonempty strings"
            )

        tokenized = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )

        if any(
            len(ids) > MAX_SEQUENCE_LENGTH
            for ids in tokenized["input_ids"]
        ):
            raise ValueError(
                "Input exceeds the model limit; "
                "no silent truncation is allowed"
            )

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return validate_vectors(
            vectors, len(texts)
        ).tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question])[0]


def main():
    started = time.perf_counter()
    embedder = MyEmbedder()

    stats = {
        "at": datetime.now(timezone.utc).isoformat(),
        "provider": "local Sentence Transformers",
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "device": "cpu",
        "normalized": True,
        "query_prefix": "",
        "document_prefix": "",
        "model_load_seconds": (
            time.perf_counter() - started
        ),
        "paid_api_cost_usd": 0,
        "cost_note": (
            "Local compute, electricity and network "
            "costs are not estimated."
        ),
        "strategies": {},
    }

    folder = ROOT / "data/processed"

    for strategy in ["A", "B"]:
        path = folder / f"chunks_{strategy}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                "Run python -m src.chunking first"
            )

        chunks = [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        if not chunks:
            raise ValueError(
                f"No chunks for {strategy}"
            )

        ids = [chunk["chunk_id"] for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")

        inputs = {
            "chunks": chunks,
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize_embeddings": True,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "encoder_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }

        fingerprint = hashlib.sha256(
            json.dumps(
                inputs,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        vector_path = (
            folder / f"embeddings_{strategy}.npy"
        )
        meta_path = (
            folder / f"embeddings_{strategy}.meta.json"
        )

        cached = False
        elapsed = time.perf_counter()

        if vector_path.exists() and meta_path.exists():
            meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )

            cached = (
                meta.get("fingerprint") == fingerprint
                and meta.get("vectors_sha256")
                == hashlib.sha256(
                    vector_path.read_bytes()
                ).hexdigest()
            )

        if cached:
            vectors = validate_vectors(
                np.load(
                    vector_path,
                    allow_pickle=False,
                ),
                len(chunks),
            )

        else:
            vectors = validate_vectors(
                embedder.embed_documents(
                    [chunk["text"] for chunk in chunks]
                ),
                len(chunks),
            )

            np.save(vector_path, vectors)

            meta = {
                "fingerprint": fingerprint,
                "model": EMBEDDING_MODEL,
                "revision": EMBEDDING_REVISION,
                "dimensions": EMBEDDING_DIMENSIONS,
                "normalized": True,
                "chunk_ids": ids,
                "vectors_sha256": hashlib.sha256(
                    vector_path.read_bytes()
                ).hexdigest(),
            }

            meta_path.write_text(
                json.dumps(meta, indent=2) + "\n",
                encoding="utf-8",
            )

        stats["strategies"][strategy] = {
            "chunks": len(chunks),
            "shape": list(vectors.shape),
            "cache_reused": cached,
            "encode_or_cache_load_seconds": (
                time.perf_counter() - elapsed
            ),
        }

        action = (
            "cache reused"
            if cached
            else "embeddings generated"
        )

        print(
            f"{strategy}: {vectors.shape}; {action}"
        )

    query = embedder.embed_query(
        "Why does the Moon always show "
        "the same face to Earth?"
    )

    print(f"Query dimensions: {len(query)}")

    output = ROOT / "results/embedding_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Embedding OK")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log = ROOT / "results/errors.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)

        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "embeddings",
                "error": str(error),
            }) + "\n")

        raise