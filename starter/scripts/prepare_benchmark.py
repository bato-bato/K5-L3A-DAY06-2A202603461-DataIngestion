"""Create the fixed English benchmark and verify its evidence."""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import load_corpus, normalize_text


# Each item contains:
# chapter, section, page, question, answer, evidence, filter, type.
ITEMS = [
    (3, "3.1", 68, "According to Kepler’s first law, what shape is a planet’s orbit, and where is the Sun located?", "The orbit is an ellipse, with the Sun at one focus.", "Each planet moves around the Sun in an orbit that is an ellipse, with the Sun at one focus of the ellipse.", None, "direct"),

    (3, "3.1", 68, "What can be said about the areas swept out by the line joining a planet and the Sun during equal time intervals?", "The line sweeps out equal areas in equal time intervals.", "The straight line joining a planet and the Sun sweeps out equal areas in space in equal intervals of time.", None, "paraphrase"),

    (3, "3.1", 68, "According to Section 3.1, how is the square of a planet’s orbital period related to its semimajor axis?", "The square of the orbital period is directly proportional to the cube of the semimajor axis.", "The square of a planet’s orbital period is directly proportional to the cube of the semimajor axis of its orbit.", {"section": "3.1"}, "filter"),

    (3, "3.1", 66, "What shape does an elliptical orbit become when its eccentricity is zero?", "A circle.", "This means that the eccentricity is zero and the ellipse is just a circle; thus, a circle can be called an ellipse of zero eccentricity.", None, "condition"),

    (3, "3.2", 70, "According to Newton’s first law, how does an object behave unless an external force changes its motion?", "It remains at rest or continues moving in a straight line at a constant speed.", "Every object will continue to be in a state of rest or move at a constant speed in a straight line unless it is compelled to change by an outside force.", None, "condition"),

    (3, "3.2", 71, "Which two quantities are multiplied to determine an object’s momentum?", "Its mass and velocity.", "Momentum then can be defined as an object’s mass times its velocity.", None, "paraphrase"),

    (3, "3.3", 75, "If a planet’s distance from the Sun doubles while both masses remain unchanged, what fraction of the original gravitational force remains?", "One quarter of the original force.", "In other words, if a planet were twice as far from the Sun, the force would be (1/2)2, or 1/4 as large.", None, "condition"),

    (3, "3.3", 75, "If you move to another planet, do your mass and weight change in the same way?", "No. Your mass remains unchanged, while your weight depends on the local gravitational force.", "Unlike your mass, which would remain the same on any planet or moon, your weight depends on the local force of gravity.", None, "paraphrase"),

    (4, "4.2", 99, "What is the main cause of Earth’s seasons?", "The approximately 23.5-degree tilt of Earth’s axis.", "As we shall show, the seasons are actually caused by the 23.5° tilt of Earth’s axis.", None, "direct"),

    (4, "4.2", 100, "Why does sunlight heat Earth’s surface more effectively when it strikes at a more direct angle?", "The sunlight is spread over a smaller area, making its heating more concentrated.", "If you shine the flashlight straight on, you get an intense spot of light on the wall. But if you hold the flashlight at an angle (if the wall “leans out” of the beam), then the spot of light is more spread out.", None, "paraphrase"),

    (4, "4.2", 105, "Why is the summer solstice not necessarily the hottest day of the year in the Northern Hemisphere?", "Air and water take time to absorb heat; they do not warm instantaneously.", "This is because our weather involves the air and water covering Earth’s surface, and these large reservoirs do not heat up instantaneously.", None, "exception"),

    (4, "4.3", 105, "According to Section 4.3, approximately how many minutes longer is a solar day than a sidereal day?", "About four minutes.", "So the solar day is about 4 minutes longer than the sidereal day.", {"section": "4.3"}, "filter"),

    (4, "4.5", 111, "Why does the illuminated portion of the Moon visible from Earth change during the month?", "The changing angle between the Sun and Moon changes how much of the Moon’s illuminated face we see from Earth.", "As we watch the Moon from our vantage point on Earth, how much of its face we see illuminated by sunlight depends on the angle the Sun makes with the Moon.", None, "paraphrase"),

    (4, "4.5", 114, "According to Section 4.5, why does the Moon always keep the same face toward Earth?", "Its rotation period equals its orbital period around Earth.", "The Moon rotates on its axis in exactly the same time that it takes to revolve about Earth. As a consequence, the Moon always keeps the same face turned toward Earth (Figure 4.15).", {"section": "4.5"}, "filter"),

    (4, "4.5", 115, "Is the far side of the Moon always dark?", "No. The Sun rises and sets on all sides of the Moon; the far side is not permanently dark.", "The back side is dark no more frequently than the front side. Since the Moon rotates, the Sun rises and sets on all sides of the Moon.", None, "exception"),

    (3, "3.4", 78, "What are the names of the points where a planet is closest to and farthest from the Sun?", "The closest point is perihelion; the farthest point is aphelion.", "The place where the planet is closest to the Sun (helios in Greek) and moves the fastest is called the perihelion of its orbit, and the place where it is farthest away and moves the most slowly is the aphelion.", None, "direct"),

    (3, "3.5", 82, "Why do some satellites in very low Earth orbits experience orbital decay?", "Atmospheric drag causes them to lose energy, leading to orbital decay.", "Some of the very low Earth orbits are not indefinitely stable because, as Earth’s atmosphere swells from time to time, a frictional drag is generated by the atmosphere on these satellites, eventually leading to a loss of energy and “decay” of the orbit.", None, "condition"),

    (4, "4.4", 109, "According to Section 4.4 on calendars, how many days are in a tropical year?", "365.2422 days.", "The basic period of revolution of Earth, called the tropical year, is 365.2422 days.", {"section": "4.4"}, "filter"),

    (4, "4.6", 118, "According to Section 4.6, during which lunar phases do neap tides occur, and why?", "During the first and last quarters, when the Sun’s tidal effect partially cancels the Moon’s tidal effect.", "When the Moon is at first quarter or last quarter (at right angles to the Sun’s direction), the tides produced by the Sun partially cancel the tides of the Moon, making them lower than usual. These are called neap tides.", {"section": "4.6"}, "filter"),

    (4, "4.7", 120, "Why do solar and lunar eclipses not occur every month?", "The Moon’s orbit is tilted by about 5 degrees relative to Earth’s orbital plane, so the Moon usually passes above or below that plane.", "However, as we mentioned, the Moon’s orbit is tilted relative to the plane of Earth’s orbit about the Sun by about 5° (imagine two hula hoops with a common center, but tilted a bit). As a result, during most months, the Moon is sufficiently above or below the ecliptic plane to avoid an eclipse.", None, "exception"),
]


def make_questions():
    questions = []

    for number, item in enumerate(ITEMS, 1):
        (
            chapter,
            section,
            page,
            question,
            answer,
            quote,
            filters,
            kind,
        ) = item

        source_id = f"astronomy2e_ch{chapter:02}"

        questions.append({
            "id": f"Q{number:02}",
            "split": "dev" if number <= 15 else "test",
            "question": question,
            "expected_answer": answer,
            "gold_source_id": source_id,
            "gold_page": page,
            "gold_section": section,
            "evidence_quote": quote,
            "gold_evidence": [{
                "source_id": source_id,
                "page": page,
                "quote": quote,
            }],
            "filter": filters,
            "question_type": kind,
        })

    return questions


def validate_questions(questions, pages):
    counts = Counter(q["split"] for q in questions)

    if (
        len(questions) != 20
        or counts != {"dev": 15, "test": 5}
    ):
        raise ValueError(
            "Expected exactly 15 dev and 5 test questions"
        )

    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")

    filtered = [
        q for q in questions if q["filter"]
    ]

    if (
        len(filtered) < 3
        or not any(q["split"] == "test" for q in filtered)
    ):
        raise ValueError(
            "Need at least 3 filter questions, "
            "including a test question"
        )

    lookup = {
        (page.document_id, page.page): page
        for page in pages
    }

    for q in questions:
        if (
            not q["question"].strip()
            or not q["expected_answer"].strip()
        ):
            raise ValueError(
                f"Empty question or answer: {q['id']}"
            )

        for gold in q["gold_evidence"]:
            page = lookup[
                (gold["source_id"], gold["page"])
            ]

            if (
                not gold["quote"].strip()
                or normalize_text(gold["quote"])
                not in normalize_text(page.text)
            ):
                raise ValueError(
                    f"Evidence not found: {q['id']}"
                )

        page = lookup[
            (q["gold_source_id"], q["gold_page"])
        ]

        if q["gold_section"] not in page.metadata["sections"]:
            raise ValueError(
                f"Wrong gold section: {q['id']}"
            )

        if q["filter"]:
            if set(q["filter"]) != {"section"}:
                raise ValueError(
                    f"Unexpected filter field: {q['id']}"
                )

            text = " ".join(
                paragraph["text"]
                for paragraph in page.metadata["paragraphs"]
                if paragraph["section"]
                == q["filter"]["section"]
            )

            if (
                normalize_text(q["evidence_quote"])
                not in normalize_text(text)
            ):
                raise ValueError(
                    "Filter does not match the evidence: "
                    f"{q['id']}"
                )

    return counts, filtered


def main():
    questions = make_questions()
    pages = load_corpus()

    counts, filtered = validate_questions(
        questions, pages
    )

    folder = ROOT / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "questions.jsonl"

    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        if existing != questions:
            raise ValueError(
                "An edited benchmark already exists; "
                "it was not overwritten"
            )

    else:
        path.write_text(
            "".join(
                json.dumps(q, ensure_ascii=False) + "\n"
                for q in questions
            ),
            encoding="utf-8",
        )

    checksum = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    protocol_path = folder / "protocol.json"

    if protocol_path.exists():
        protocol = json.loads(
            protocol_path.read_text(encoding="utf-8")
        )

        if protocol["benchmark_sha256"] != checksum:
            raise ValueError(
                "Benchmark differs from the recorded protocol"
            )

    else:
        protocol = {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "benchmark_sha256": checksum,
            "dev_ids": [
                q["id"] for q in questions
                if q["split"] == "dev"
            ],
            "test_ids": [
                q["id"] for q in questions
                if q["split"] == "test"
            ],
            "filter_ids": [
                q["id"] for q in filtered
            ],
            "split_method": (
                "Different textbook sections; "
                "related-topic leakage remains possible."
            ),
            "top_k": 3,
            "metric": "cosine",
            "selection_rule": (
                "Prefer dev Evidence Hit@3, then dev Hit@1, "
                "then lower ingestion time. "
                "Prefer B in an exact B/C tie."
            ),
            "automatic_scoring": (
                "Exact-quote scores are conservative proxies; "
                "accept verified equivalent evidence "
                "through human review."
            ),
            "selection_basis": (
                "Use reviewed evidence metrics if all dev "
                "labels are complete; otherwise explicitly "
                "record exact-quote proxy selection."
            ),
            "test_policy": (
                "Freeze the configuration before test "
                "retrieval. Do not tune on test results."
            ),
            "coherence_sample_rule": (
                "First text block of at least 120 characters "
                "on each of the first 10 selected source "
                "pages; use its midpoint as a shared anchor."
            ),
            "limitations": (
                "Only five test questions. Coherence "
                "sampling covers Chapter 3 only."
            ),
            "attribution": (
                "Copyright 2022 Rice University, CC BY 4.0. "
                "Access for free at openstax.org."
            ),
        }

        protocol_path.write_text(
            json.dumps(protocol, indent=2) + "\n",
            encoding="utf-8",
        )

    samples = []

    selected_pages = sorted(
        pages,
        key=lambda page: (page.document_id, page.page),
    )[:10]

    for number, page in enumerate(selected_pages, 1):
        paragraph = next(
            p for p in page.metadata["paragraphs"]
            if len(p["text"]) >= 120
        )

        group = [
            p for p in page.metadata["paragraphs"]
            if p["section"] == paragraph["section"]
        ]

        start = sum(
            len(p["text"]) + 1
            for p in group[:group.index(paragraph)]
        )

        samples.append({
            "sample_id": f"P{number:02}",
            "document_id": page.document_id,
            "page": page.page,
            "section": paragraph["section"],
            "group_anchor_char": (
                start + len(paragraph["text"]) // 2
            ),
            "source_paragraph": paragraph["text"],
        })

    sample_path = folder / "coherence_positions.json"

    if (
        sample_path.exists()
        and json.loads(
            sample_path.read_text(encoding="utf-8")
        ) != samples
    ):
        raise ValueError(
            "Existing coherence positions changed; "
            "they were not overwritten"
        )

    sample_path.write_text(
        json.dumps(
            samples,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    result = {
        "verified_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "benchmark_sha256": checksum,
        "questions": len(questions),
        "splits": dict(counts),
        "filter_questions": len(filtered),
        "evidence_verified": len(questions),
        "retrieval_executed": False,
    }

    output = ROOT / "results/benchmark_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Questions: {len(questions)}")
    print(
        f"Dev: {counts['dev']} | Test: {counts['test']}"
    )
    print(f"Filter questions: {len(filtered)}")
    print("Evidence verified: 20/20")
    print("Coherence positions: 10")
    print(
        "Benchmark ready; no test retrieval was executed."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        log = ROOT / "results/errors.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)

        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "benchmark preparation",
                "error": str(error),
            }) + "\n")

        raise