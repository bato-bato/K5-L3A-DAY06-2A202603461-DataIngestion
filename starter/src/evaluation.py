"""Exact-quote retrieval proxies and human-review forms."""

from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics

from .ingestion import normalize_text


def evaluate_query(question, hits):
    golds = question.get("gold_evidence") or [{
        "source_id": question["gold_source_id"],
        "page": question["gold_page"],
        "quote": question["evidence_quote"],
    }]

    results = []

    for hit in hits:
        metadata = hit["metadata"]

        matched = any(
            metadata.get("document_id") == gold["source_id"]
            and metadata.get("page") == gold["page"]
            and normalize_text(gold["quote"])
            in normalize_text(hit["text"])
            for gold in golds
        )

        results.append({
            **hit,
            "exact_quote_hit": bool(matched),
        })

    requested_filter = question.get("filter")
    compliant = None

    if requested_filter:
        compliant = len(results) == 3 and all(
            all(
                hit["metadata"].get(key) == value
                for key, value in requested_filter.items()
            )
            for hit in results
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_evidence": golds,
        "requested_filter": requested_filter,
        "results": results,
        "exact_quote_hit1": int(
            bool(results) and results[0]["exact_quote_hit"]
        ),
        "exact_quote_hit3": int(
            any(hit["exact_quote_hit"] for hit in results)
        ),
        "filter_compliant": compliant,
    }


def summarize(traces):
    groups = defaultdict(list)

    for trace in traces:
        groups[
            (trace["strategy"], trace["split"])
        ].append(trace)

    rows = []

    for (strategy, split), items in sorted(groups.items()):
        if len({
            item["question_id"] for item in items
        }) != len(items):
            raise ValueError("Duplicate question traces")

        n = len(items)
        hit1 = sum(item["exact_quote_hit1"] for item in items)
        hit3 = sum(item["exact_quote_hit3"] for item in items)

        filtered = [
            item for item in items
            if item["requested_filter"]
        ]
        compliant = sum(
            item["filter_compliant"]
            for item in filtered
        )

        rows.append({
            "strategy": strategy,
            "split": split,
            "questions": n,
            "exact_quote_hit1_n": hit1,
            "exact_quote_hit3_n": hit3,
            "exact_quote_hit1_fraction": f"{hit1}/{n}",
            "exact_quote_hit3_fraction": f"{hit3}/{n}",
            "exact_quote_hit1_rate": hit1 / n,
            "exact_quote_hit3_rate": hit3 / n,
            "filter_compliant_n": compliant,
            "filter_questions": len(filtered),
            "filter_compliance_fraction": (
                f"{compliant}/{len(filtered)}"
                if filtered else "N/A"
            ),
            "mean_retrieval_ms": statistics.mean(
                item["retrieval_ms"] for item in items
            ),
            "reviewed_hit1_n": None,
            "reviewed_hit3_n": None,
            "human_review_status": "pending",
        })

    return rows


def write_csv(path, rows, fields):
    with Path(path).open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def save_dev_results(root, traces, fingerprint):
    folder = Path(root) / "results"
    folder.mkdir(parents=True, exist_ok=True)

    audit = []

    for trace in traces:
        for hit in trace["results"]:
            audit.append({
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "rank": hit["rank"],
                "chunk_id": hit["chunk_id"],
                "experiment_fingerprint": fingerprint,
                "question": trace["question"],
                "expected_answer": trace["expected_answer"],
                "gold_evidence": json.dumps(
                    trace["gold_evidence"],
                    ensure_ascii=False,
                ),
                "chunk_text": hit["text"],
                "document_id": hit["metadata"]["document_id"],
                "source": hit["metadata"]["source"],
                "page": hit["metadata"]["page"],
                "section": hit["metadata"]["section"],
                "exact_quote_hit": int(hit["exact_quote_hit"]),
                "evidence_hit_0_1": "",
                "metadata_correct_0_1": "",
                "reason": "",
                "reviewer": "",
            })

    audit_path = folder / "dev_audit.csv"

    if audit_path.exists():
        with audit_path.open(
            encoding="utf-8-sig", newline=""
        ) as file:
            old = list(csv.DictReader(file))

        keys = [
            "strategy",
            "question_id",
            "rank",
            "chunk_id",
            "experiment_fingerprint",
        ]

        old_keys = [
            tuple(row[key] for key in keys)
            for row in old
        ]
        new_keys = [
            tuple(str(row[key]) for key in keys)
            for row in audit
        ]

        if old_keys != new_keys:
            raise ValueError(
                "Existing dev audit differs. Preserve it "
                "under another name before starting "
                "a new experiment."
            )
    else:
        write_csv(
            audit_path, audit, list(audit[0])
        )

    (folder / "retrieval_dev.jsonl").write_text(
        "".join(
            json.dumps(trace, ensure_ascii=False) + "\n"
            for trace in traces
        ),
        encoding="utf-8",
    )

    metrics = summarize(traces)

    write_csv(
        folder / "dev_metrics.csv",
        metrics,
        list(metrics[0]),
    )

    (folder / "dev_metrics.json").write_text(
        json.dumps({
            "metric_note": (
                "Conservative exact-quote proxies; "
                "equivalent sufficient evidence requires "
                "human review."
            ),
            "experiment_fingerprint": fingerprint,
            "metrics": metrics,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Development results",
        "",
        "Exact-quote scores are conservative proxies, "
        "not completed human-reviewed retrieval accuracy.",
        "",
        "| Strategy | Exact-quote Hit@1 | Exact-quote Hit@3 "
        "| Filter compliance | Human review |",
        "|---|---|---|---|---|",
    ]

    for row in metrics:
        lines.append(
            f"| {row['strategy']} "
            f"| {row['exact_quote_hit1_fraction']} "
            f"| {row['exact_quote_hit3_fraction']} "
            f"| {row['filter_compliance_fraction']} "
            "| Pending |"
        )

    lines += [
        "",
        "A complete single retrieved chunk must contain "
        "sufficient evidence to count as a reviewed hit. "
        "Accept verified equivalent evidence, including "
        "other valid source locations.",
        "",
        "Complete results/dev_audit.csv: evidence_hit_0_1 "
        "and metadata_correct_0_1 use 0 or 1; every judgment "
        "needs a reason and reviewer. Blank means pending, "
        "not zero.",
        "",
        "Retrieval timing includes database connection/query "
        "time; it excludes model loading and query embedding.",
        "",
        "Grounding and coherence require separate manual "
        "assessment. Do not infer their scores from quote "
        "matches.",
        "",
        "No test retrieval was executed by this development run.",
        "",
        "Textbook excerpts copyright 2022 Rice University, "
        "CC BY 4.0. Access for free at openstax.org.",
    ]

    (folder / "dev_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return metrics