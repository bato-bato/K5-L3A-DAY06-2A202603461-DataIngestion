from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import hashlib
import json

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "source_id": "astronomy2e_ch03",
        "title": "Astronomy 2e — Chapter 3: Orbits and Gravity",
        "url": (
            "https://www.theexpertta.com/book-files/"
            "OpenStaxAstronomy2e/Chapter3_63-94.pdf"
        ),
        "filename": "chapter3.pdf",
        "first_printed_page": 63,
        "selected_pages": [64, 87],
    },
    {
        "source_id": "astronomy2e_ch04",
        "title": "Astronomy 2e — Chapter 4: Earth, Moon, and Sky",
        "url": (
            "https://www.theexpertta.com/book-files/"
            "OpenStaxAstronomy2e/Chapter4_95-134.pdf"
        ),
        "filename": "chapter4.pdf",
        "first_printed_page": 95,
        "selected_pages": [96, 125],
    },
]


def main():
    folders = ["data/raw", "data/processed", "results"]

    for folder in folders:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "data/manifest.json"
    previous_sources = {}

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        previous_sources = {
            item["source_id"]: item
            for item in previous.get("sources", [])
        }

    records = []
    total_selected_pages = 0

    for source in SOURCES:
        path = ROOT / "data/raw" / source["filename"]
        previous = previous_sources.get(source["source_id"], {})
        downloaded_at = previous.get("downloaded_at")

        if not path.exists():
            print(f"Downloading: {source['filename']}")

            request = Request(
                source["url"],
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urlopen(request, timeout=90) as response:
                content = response.read()

            if not content.startswith(b"%PDF"):
                raise RuntimeError(
                    f"The source did not return a PDF: {source['url']}"
                )

            path.write_bytes(content)
            downloaded_at = datetime.now(timezone.utc).isoformat()
        else:
            print(f"Using existing file: {source['filename']}")

        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        start, end = source["selected_pages"]

        with pymupdf.open(path) as pdf:
            page_records = []

            for printed_page in range(start, end + 1):
                pdf_index = (
                    printed_page - source["first_printed_page"]
                )

                if not 0 <= pdf_index < len(pdf):
                    raise ValueError(
                        f"Page mapping is invalid: {printed_page}"
                    )

                text = pdf[pdf_index].get_text()
                normalized_text = " ".join(text.split())

                if not normalized_text:
                    raise ValueError(
                        f"No extractable text on page {printed_page}"
                    )

                page_records.append(
                    {
                        "page": printed_page,
                        "pdf_page": pdf_index + 1,
                        "extracted_character_count": len(
                            normalized_text
                        ),
                    }
                )

            physical_pages = len(pdf)

        record = {
            "source_id": source["source_id"],
            "title": source["title"],
            "source_path_or_url": source["url"],
            "local_path": path.relative_to(ROOT).as_posix(),
            "official_book_url": (
                "https://openstax.org/details/books/astronomy-2e"
            ),
            "publisher": "OpenStax, Rice University",
            "edition": "2e",
            "language": "en",
            "license_or_permission": (
                "CC BY 4.0 for the matched 2022 PDF snapshot; "
                "retain source attribution."
            ),
            "attribution": (
                "Copyright 2022 Rice University. "
                "Access for free at openstax.org."
            ),
            "downloaded_at": downloaded_at,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "sha256": checksum,
            "physical_pdf_pages": physical_pages,
            "selected_printed_pages": [start, end],
            "selected_page_count": len(page_records),
            "page_counting_rule": (
                "Count selected printed pages once. "
                "Exclude chapter opening pages and end matter."
            ),
            "character_count_method": (
                "PyMuPDF extracted text with collapsed whitespace; "
                "running headers and footers are still included."
            ),
            "pages": page_records,
        }

        records.append(record)
        total_selected_pages += len(page_records)

        print(
            f"{source['filename']}: "
            f"{len(page_records)} selected pages"
        )

    manifest = {
        "stage": "source acquisition and initial text check",
        "total_selected_pages": total_selected_pages,
        "sources": records,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nTotal selected pages: {total_selected_pages}")
    print("Manifest saved to data/manifest.json")


if __name__ == "__main__":
    main()