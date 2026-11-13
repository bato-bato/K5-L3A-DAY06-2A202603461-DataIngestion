"""Token-based fixed windows and paragraph-aware chunks."""

from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import hashlib
import json

from transformers import AutoTokenizer

from .models import Chunk, DocumentPage
from .settings import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    MAX_SEQUENCE_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        use_fast=True,
    )


def token_count(text: str) -> int:
    return len(get_tokenizer()(
        text,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"])


def validate_limits(size: int, overlap: int):
    if not isinstance(size, int) or not isinstance(overlap, int):
        raise ValueError("Chunk size and overlap must be integers")

    if not 0 <= overlap < size:
        raise ValueError("Need 0 <= overlap < chunk size")

    special = get_tokenizer().num_special_tokens_to_add()

    if size + special > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            "Chunk size exceeds the embedding sequence limit"
        )


def token_windows(text: str, size: int, overlap: int):
    offsets = get_tokenizer()(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )["offset_mapping"]

    start = 0

    while start < len(offsets):
        end = min(start + size, len(offsets))

        # Slices can tokenize differently at WordPiece boundaries.
        while end > start and token_count(
            text[offsets[start][0]:offsets[end - 1][1]]
        ) > size:
            end -= 1

        if end <= start:
            raise ValueError(
                "Cannot create a nonempty token window"
            )

        yield offsets[start][0], offsets[end - 1][1]

        if end == len(offsets):
            break

        next_start = end - overlap

        if next_start <= start:
            raise ValueError(
                "Overlap prevents the window from advancing"
            )

        start = next_start


def build_chunks(
    page: DocumentPage,
    strategy: str,
    size: int,
    overlap: int,
) -> list[Chunk]:
    validate_limits(size, overlap)

    paragraphs = page.metadata.get("paragraphs")
    if not paragraphs:
        raise ValueError(
            "Use ingestion.load_corpus() "
            "to retain paragraph metadata"
        )

    groups = defaultdict(list)

    for paragraph in paragraphs:
        groups[paragraph["section"]].append(paragraph["text"])

    chunks = []

    for section, texts in groups.items():
        text = " ".join(texts)

        if strategy == "A":
            spans = list(token_windows(text, size, overlap))

        elif strategy == "B":
            spans = []
            pending = None
            cursor = 0

            for paragraph in texts:
                start = cursor
                end = cursor + len(paragraph)
                cursor = end + 1

                if token_count(paragraph) > size:
                    if pending is not None:
                        spans.append(pending)
                        pending = None

                    spans.extend(
                        (start + a, start + b)
                        for a, b in token_windows(
                            paragraph, size, overlap
                        )
                    )

                elif pending is None:
                    pending = (start, end)

                elif token_count(
                    text[pending[0]:end]
                ) <= size:
                    pending = (pending[0], end)

                else:
                    spans.append(pending)
                    pending = (start, end)

            if pending is not None:
                spans.append(pending)

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        for index, (start, end) in enumerate(spans):
            content = text[start:end]
            count = token_count(content)

            if not content or count > size:
                raise ValueError(
                    "Invalid chunk content or token count"
                )

            identity = json.dumps(
                [
                    strategy,
                    page.document_id,
                    page.page,
                    section,
                    start,
                    end,
                    content,
                ],
                ensure_ascii=False,
            )

            suffix = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16]

            chunk_id = (
                f"{strategy}_{page.document_id}"
                f"_p{page.page:03}_s{section}"
                f"_{index:03}_{suffix}"
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=content,
                metadata={
                    "strategy": strategy,
                    "section": section,
                    "chapter": page.metadata["chapter"],
                    "pdf_page": page.metadata["pdf_page"],
                    "language": page.metadata["language"],
                    "attribution": page.metadata["attribution"],
                    "token_count": count,
                    "group_char_start": start,
                    "group_char_end": end,
                },
            ))

    return chunks


def fixed_size(
    page: DocumentPage,
    size: int,
    overlap: int,
) -> list[Chunk]:
    return build_chunks(page, "A", size, overlap)


def by_paragraph(
    page: DocumentPage,
    max_size: int,
) -> list[Chunk]:
    return build_chunks(
        page, "B", max_size, CHUNK_OVERLAP
    )


if __name__ == "__main__":
    from .ingestion import load_corpus

    pages = load_corpus()

    output = (
        Path(__file__).resolve().parents[1]
        / "data/processed"
    )
    output.mkdir(parents=True, exist_ok=True)

    for strategy in ["A", "B"]:
        chunks = []

        for page in pages:
            if strategy == "A":
                chunks.extend(fixed_size(
                    page, CHUNK_SIZE, CHUNK_OVERLAP
                ))
            else:
                chunks.extend(by_paragraph(
                    page, CHUNK_SIZE
                ))

        ids = [chunk.chunk_id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")

        path = output / f"chunks_{strategy}.jsonl"

        with path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(
                        asdict(chunk),
                        ensure_ascii=False,
                    ) + "\n"
                )

        maximum = max(
            chunk.metadata["token_count"]
            for chunk in chunks
        )

        print(
            f"{strategy}: {len(chunks)} chunks; "
            f"max tokens: {maximum}"
        )

    print("Chunking OK")