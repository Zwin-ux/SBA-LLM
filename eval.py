"""Evaluation script for SBA Copilot retrieval.

Usage:
    python eval.py

Requires eval.jsonl with objects of the form:
    {"q": "question", "gold_ids": ["doc_id1", "doc_id2"]}
"""

from __future__ import annotations

import json
from statistics import mean

from agent_tools import rerank_hits, search_docs

EVAL_FILE = "eval.jsonl"


def main() -> None:
    with open(EVAL_FILE, "r", encoding="utf-8") as fh:
        evals = [json.loads(line) for line in fh if line.strip()]

    hits_at_5 = []
    reciprocal_ranks = []

    for entry in evals:
        question = entry["q"]
        gold = {g for g in entry.get("gold_ids", [])}

        retrieved = search_docs(question, k=40)
        reranked = rerank_hits(question, retrieved, topn=5)
        doc_ids = [hit.metadata.get("file") or hit.metadata.get("doc_id") or hit.metadata.get("doc") for hit in reranked]

        # Hit@5: at least one gold doc in top 5
        hit = any(doc_id in gold for doc_id in doc_ids)
        hits_at_5.append(1 if hit else 0)

        # MRR: reciprocal rank of first relevant doc
        rr = 0.0
        for idx, doc_id in enumerate(doc_ids, start=1):
            if doc_id in gold:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)

    print(f"Total questions: {len(evals)}")
    print(f"Hit@5: {mean(hits_at_5):.3f}")
    print(f"MRR@5: {mean(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
