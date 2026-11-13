"""Load selected PDF pages using the source manifest."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata

import pymupdf

from .models import DocumentPage

ROOT = Path(__file__).resolve().parents[1]

SECTION_TITLES = {
    "3.1": "The Laws of Planetary Motion",
    "3.2": "Newton's Great Synthesis",
    "3.3": "Newton's Universal Law of Gravitation",
    "3.4": "Orbits in the Solar System",
    "3.5": "Motions of Satellites and Spacecraft",
    "3.6": "Gravity with More Than Two Bodies",
    "4.1": "Earth and Sky",
    "4.2": "The Seasons",
    "4.3": "Keeping Time",
    "4.4": "The Calendar",
    "4.5": "Phases and Motions of the Moon",
    "4.6": "Ocean Tides and the Moon",
    "4.7": "Eclipses of the Sun and Moon",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    text = text.translate(str.maketrans({
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ad": "",
        "\u0002": "",
    }))

    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    return " ".join(text.split())


def read_manifest() -> dict:
    path = ROOT / "data/manifest.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
) -> list[DocumentPage]:
    path = Path(path)

    entries = [
        item
        for item in read_manifest()["sources"]
        if item["source_id"] == document_id
    ]

    if len(entries) != 1:
        raise ValueError(
            f"Expected one manifest entry for {document_id}"
        )

    entry = entries[0]

    if source != entry["source_path_or_url"]:
        raise ValueError(
            f"Source URL differs from manifest: {document_id}"
        )

    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != entry["sha256"]:
        raise ValueError(f"PDF checksum mismatch: {path.name}")

    pages = []
    active_section = None

    with pymupdf.open(path) as pdf:
        locations = sorted(
            entry["pages"],
            key=lambda item: item["pdf_page"],
        )

        for location in locations:
            pdf_index = location["pdf_page"] - 1

            if not 0 <= pdf_index < len(pdf):
                raise ValueError(
                    f"Invalid PDF page: {location['pdf_page']}"
                )

            pdf_page = pdf[pdf_index]
            paragraphs = []

            for block in pdf_page.get_text("blocks", sort=True):
                if len(block) > 6 and block[6] != 0:
                    continue

                # Remove running headers and footers.
                if (
                    block[1] < 45
                    or block[3] > pdf_page.rect.height - 32
                ):
                    continue

                text = normalize_text(block[4])
                if not text:
                    continue

                heading = re.match(
                    r"^(\d+\.\d+)\s+(.+)$",
                    text,
                )

                if heading:
                    section_id = heading.group(1)
                    title = SECTION_TITLES.get(section_id)

                    if title and heading.group(2).startswith(title):
                        active_section = section_id

                if active_section is None:
                    raise ValueError(
                        "Section missing on book page "
                        f"{location['page']}"
                    )

                paragraphs.append({
                    "section": active_section,
                    "text": text,
                })

            text = " ".join(
                paragraph["text"]
                for paragraph in paragraphs
            )

            if len(text) < 100:
                raise ValueError(
                    "Insufficient text on book page "
                    f"{location['page']}"
                )

            pages.append(DocumentPage(
                document_id=document_id,
                source=source,
                page=location["page"],
                text=text,
                metadata={
                    "pdf_page": pdf_index + 1,
                    "chapter": active_section.split(".")[0],
                    "sections": sorted({
                        paragraph["section"]
                        for paragraph in paragraphs
                    }),
                    "paragraphs": paragraphs,
                    "language": "en",
                    "character_count": len(text),
                    "attribution": entry["attribution"],
                },
            ))

    return pages


def load_corpus() -> list[DocumentPage]:
    pages = []

    for source in read_manifest()["sources"]:
        pages.extend(load_pdf(
            ROOT / source["local_path"],
            source["source_id"],
            source["source_path_or_url"],
        ))

    if len(pages) < 30:
        raise ValueError(
            "The corpus needs at least 30 useful pages"
        )

    keys = [
        (page.document_id, page.page)
        for page in pages
    ]

    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source pages")

    return pages


if __name__ == "__main__":
    try:
        pages = load_corpus()

        print(f"Loaded pages: {len(pages)}")

        blocks = sum(
            len(page.metadata["paragraphs"])
            for page in pages
        )
        print(f"Text blocks: {blocks}")

        page105 = next(
            page
            for page in pages
            if page.document_id == "astronomy2e_ch04"
            and page.page == 105
        )
        print(
            "Book page 105 sections:",
            page105.metadata["sections"],
        )

        print("Ingestion OK")

    except Exception as error:
        log = ROOT / "results/errors.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)

        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "ingestion",
                "error": str(error),
            }) + "\n")

        raise