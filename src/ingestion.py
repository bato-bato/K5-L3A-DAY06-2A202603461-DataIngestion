from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS = {
    "astronomy2e_ch03": {
        "first_printed_page": 63,
        "chapter": "3",
        "sections": {
            "3.1": "The Laws of Planetary Motion",
            "3.2": "Newton's Great Synthesis",
            "3.3": "Newton's Universal Law of Gravitation",
            "3.4": "Orbits in the Solar System",
            "3.5": "Motions of Satellites and Spacecraft",
            "3.6": "Gravity with More Than Two Bodies",
        },
    },
    "astronomy2e_ch04": {
        "first_printed_page": 95,
        "chapter": "4",
        "sections": {
            "4.1": "Earth and Sky",
            "4.2": "The Seasons",
            "4.3": "Keeping Time",
            "4.4": "The Calendar",
            "4.5": "Phases and Motions of the Moon",
            "4.6": "Ocean Tides and the Moon",
            "4.7": "Eclipses of the Sun and Moon",
        },
    },
}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)

    text = text.translate(
        str.maketrans({
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u00ad": "",
            "\u0002": "",
        })
    )

    # Join words split by a hyphen at a PDF line break.
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    return " ".join(text.split())


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    all_pages = []
    all_paragraphs = []

    for source in manifest["sources"]:
        document_id = source["source_id"]
        settings = DOCUMENTS[document_id]
        pdf_path = ROOT / source["local_path"]

        checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if checksum != source["sha256"]:
            raise ValueError(
                f"PDF changed since download: {pdf_path.name}"
            )

        start, end = source["selected_printed_pages"]
        active_section = None
        source_page_records = []

        with pymupdf.open(pdf_path) as pdf:
            for printed_page in range(start, end + 1):
                pdf_index = (
                    printed_page - settings["first_printed_page"]
                )
                page = pdf[pdf_index]
                paragraphs = []

                for block in page.get_text("blocks", sort=True):
                    # Skip image blocks.
                    if len(block) > 6 and block[6] != 0:
                        continue

                    # Remove running headers and footers.
                    if (
                        block[1] < 45
                        or block[3] > page.rect.height - 32
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
                        expected_title = settings["sections"].get(
                            section_id
                        )

                        if (
                            expected_title
                            and heading.group(2).startswith(
                                normalize_text(expected_title)
                            )
                        ):
                            active_section = section_id

                    if active_section is None:
                        raise ValueError(
                            "Section not identified on "
                            f"page {printed_page}: {text[:80]}"
                        )

                    paragraphs.append({
                        "paragraph_id": (
                            f"{document_id}_p{printed_page:03}"
                            f"_b{len(paragraphs):03}"
                        ),
                        "document_id": document_id,
                        "source": source["source_path_or_url"],
                        "page": printed_page,
                        "pdf_page": pdf_index + 1,
                        "chapter": settings["chapter"],
                        "section": active_section,
                        "language": "en",
                        "text": text,
                        "attribution": (
                            "Copyright 2022 Rice University. "
                            "CC BY 4.0. "
                            "Access for free at openstax.org."
                        ),
                    })

                page_text = " ".join(
                    paragraph["text"]
                    for paragraph in paragraphs
                )

                if len(page_text) < 100:
                    raise ValueError(
                        "Too little extracted content on "
                        f"page {printed_page}; inspect the PDF."
                    )

                sections = sorted({
                    paragraph["section"]
                    for paragraph in paragraphs
                })

                page_record = {
                    "document_id": document_id,
                    "page": printed_page,
                    "pdf_page": pdf_index + 1,
                    "sections": sections,
                    "character_count": len(page_text),
                    "text": page_text,
                }

                all_pages.append(page_record)
                all_paragraphs.extend(paragraphs)

                source_page_records.append({
                    "page": printed_page,
                    "pdf_page": pdf_index + 1,
                    "normalized_character_count": len(page_text),
                    "sections": sections,
                })

        source["pages"] = source_page_records
        source["normalized_character_count"] = sum(
            page["normalized_character_count"]
            for page in source_page_records
        )

        print(
            f"{document_id}: "
            f"{len(source_page_records)} pages processed"
        )

    if len(all_pages) < 30:
        raise ValueError("The corpus needs at least 30 useful pages.")

    write_jsonl(
        ROOT / "data/processed/pages.jsonl",
        all_pages,
    )
    write_jsonl(
        ROOT / "data/processed/paragraphs.jsonl",
        all_paragraphs,
    )

    manifest["stage"] = "normalized pages and section metadata"
    manifest["processed_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    manifest["character_count_method"] = (
        "Normalized body text after running header/footer removal."
    )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nPages: {len(all_pages)}")
    print(f"Text blocks: {len(all_paragraphs)}")
    print("Saved to data/processed/")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        error_path = ROOT / "results/errors.jsonl"
        error_path.parent.mkdir(parents=True, exist_ok=True)

        with error_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "ingestion",
                "error": str(error),
            }) + "\n")

        raise