"""Run development retrieval without exposing the test split."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

from dotenv import load_dotenv

from .embeddings import MyEmbedder
from .evaluation import evaluate_query, save_dev_results
from .ingestion import load_corpus
from .settings import EMBEDDING_DIMENSIONS
from .store import PgVectorStore, load_saved_embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_benchmark import validate_questions


def read_questions():
    path = ROOT / "benchmark/questions.jsonl"

    protocol = json.loads(
        (ROOT / "benchmark/protocol.json").read_text(
            encoding="utf-8"
        )
    )

    checksum = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if checksum != protocol["benchmark_sha256"]:
        raise ValueError(
            "Benchmark differs from the recorded protocol"
        )

    if (
        protocol["top_k"] != 3
        or protocol["metric"] != "cosine"
    ):
        raise ValueError(
            "Expected the predefined top-3 cosine benchmark"
        )

    questions = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    validate_questions(questions, load_corpus())
    return questions


def experiment_fingerprint():
    files = [
        "benchmark/questions.jsonl",
        "benchmark/protocol.json",
        "data/manifest.json",
    ]

    files += [
        f"src/{name}.py"
        for name in [
            "models",
            "settings",
            "ingestion",
            "chunking",
            "embeddings",
            "store",
        ]
    ]

    for strategy in ["A", "B"]:
        files += [
            f"data/processed/chunks_{strategy}.jsonl",
            f"data/processed/embeddings_{strategy}.npy",
            f"data/processed/embeddings_{strategy}.meta.json",
        ]

    manifest = json.loads(
        (ROOT / "data/manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )

    files += [
        source["local_path"]
        for source in manifest["sources"]
    ]

    hashes = {
        name: hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
        for name in files
    }

    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()


def run_dev():
    if (
        (ROOT / "artifacts/frozen_selection.json").exists()
        or (ROOT / "results/retrieval_test.jsonl").exists()
    ):
        raise ValueError(
            "This experiment is frozen or test has been "
            "exposed; do not run new development "
            "on this holdout"
        )

    load_dotenv(ROOT / ".env")

    questions = [
        question
        for question in read_questions()
        if question["split"] == "dev"
    ]

    if len(questions) != 15:
        raise ValueError("Expected 15 dev questions")

    fingerprint = experiment_fingerprint()

    store = PgVectorStore(
        os.getenv("DATABASE_URL"),
        EMBEDDING_DIMENSIONS,
    )
    store.create_schema()

    database_reload = {}

    for strategy in ["A", "B"]:
        chunks, vectors = load_saved_embeddings(strategy)

        started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)

        database_reload[strategy] = (
            time.perf_counter() - started
        )

        if store.count(strategy) != len(chunks):
            raise ValueError(
                "Database snapshot count mismatch"
            )

    encoder = MyEmbedder()

    # Encode only the question text.
    # Never encode gold answers, evidence or source pages.
    query_vectors = [
        encoder.embed_query(question["question"])
        for question in questions
    ]

    traces = []

    for strategy in ["A", "B", "C"]:
        for question, vector in zip(
            questions, query_vectors
        ):
            filters = (
                question["filter"]
                if strategy == "C"
                else None
            )

            started = time.perf_counter()

            hits = store.search(
                strategy,
                vector,
                top_k=3,
                filters=filters,
            )

            milliseconds = (
                time.perf_counter() - started
            ) * 1000

            if len(hits) != 3:
                raise ValueError(
                    f"Expected top-3 for "
                    f"{strategy}/{question['id']}"
                )

            trace = evaluate_query(question, hits)

            trace.update({
                "strategy": strategy,
                "applied_filter": filters,
                "retrieval_ms": milliseconds,
                "experiment_fingerprint": fingerprint,
                "run_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

            traces.append(trace)

    if fingerprint != experiment_fingerprint():
        raise ValueError(
            "Experiment inputs changed during retrieval"
        )

    metrics = save_dev_results(
        ROOT, traces, fingerprint
    )

    info = {
        "at": datetime.now(timezone.utc).isoformat(),
        "experiment_fingerprint": fingerprint,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "sentence-transformers",
                "transformers",
                "numpy",
                "psycopg",
                "PyMuPDF",
            ]
        },
        "database_reload_seconds": database_reload,
        "timing_note": (
            "Database reload only; this is not complete "
            "cold-ingestion time. "
            "C shares B's stored vectors."
        ),
        "query_traces": len(traces),
        "test_retrieval_executed": False,
    }

    (ROOT / "results/dev_run.json").write_text(
        json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in metrics:
        print(
            f"{row['strategy']}: "
            f"exact-quote Hit@1="
            f"{row['exact_quote_hit1_fraction']}; "
            f"Hit@3={row['exact_quote_hit3_fraction']}"
        )

    print("Dev traces: 45 | Audit records: 135")
    print("Human review: pending")
    print("Dev OK; test was not run.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        nargs="?",
        default="dev",
        choices=["dev"],
    )

    parser.parse_args()

    try:
        run_dev()

    except Exception as error:
        path = ROOT / "results/errors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "dev retrieval",
                "error": str(error),
            }) + "\n")

        raise


if __name__ == "__main__":
    main()