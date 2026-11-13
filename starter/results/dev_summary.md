# Development results

Exact-quote scores are conservative proxies, not completed human-reviewed retrieval accuracy.

| Strategy | Exact-quote Hit@1 | Exact-quote Hit@3 | Filter compliance | Human review |
|---|---|---|---|---|
| A | 7/15 | 10/15 | 2/3 | Pending |
| B | 8/15 | 13/15 | 1/3 | Pending |
| C | 8/15 | 13/15 | 3/3 | Pending |

A complete single retrieved chunk must contain sufficient evidence to count as a reviewed hit. Accept verified equivalent evidence, including other valid source locations.

Complete results/dev_audit.csv: evidence_hit_0_1 and metadata_correct_0_1 use 0 or 1; every judgment needs a reason and reviewer. Blank means pending, not zero.

Retrieval timing includes database connection/query time; it excludes model loading and query embedding.

Grounding and coherence require separate manual assessment. Do not infer their scores from quote matches.

No test retrieval was executed by this development run.

Textbook excerpts copyright 2022 Rice University, CC BY 4.0. Access for free at openstax.org.
