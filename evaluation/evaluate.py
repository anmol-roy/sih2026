"""
Phase 2 Retrieval Evaluation
─────────────────────────────
Measures Recall@K for the hybrid retriever against the 30-question gold set.

Usage (from project root):
    python evaluation/evaluate.py
    python evaluation/evaluate.py --top-k 5
    python evaluation/evaluate.py --questions evaluation/questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval import HybridRetriever


# ---------------------------------------------------------------------------
# Match logic
# ---------------------------------------------------------------------------

def _section_matches(chunk_section: Optional[str], expected: Optional[str]) -> bool:
    """Check if the expected section label appears anywhere in the chunk section string."""
    if expected is None:
        return True   # No expectation → always passes
    if chunk_section is None:
        return False
    # Normalise: remove "Section " prefix for comparison
    norm_chunk = chunk_section.replace("Section ", "").strip()
    norm_expected = expected.replace("Section ", "").strip()
    return norm_expected.lower() in norm_chunk.lower()


def _subsection_matches(chunk_sub: Optional[str], expected: Optional[str]) -> bool:
    if expected is None:
        return True
    if chunk_sub is None:
        return False
    # "3(p)" in "3(p)" — normalise
    return expected.lower().replace(" ", "") in chunk_sub.lower().replace(" ", "")


def _document_matches(chunk_doc_id: str, expected_doc_id: str) -> bool:
    return expected_doc_id.lower() in chunk_doc_id.lower()


def _chunk_hits(chunk, q: dict) -> bool:
    """Return True if this chunk satisfies the question's expected fields."""
    doc_ok = _document_matches(chunk.document_id, q["expected_document_id"])
    sec_ok = _section_matches(chunk.section, q.get("expected_section"))
    sub_ok = _subsection_matches(chunk.subsection, q.get("expected_subsection"))
    return doc_ok and sec_ok and sub_ok


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(questions_path: str, top_k: int = 5) -> None:
    questions_file = Path(questions_path)
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}")
        sys.exit(1)

    with open(questions_file, encoding="utf-8") as fh:
        questions: list[dict] = json.load(fh)

    print(f"Loaded {len(questions)} questions from {questions_file}")
    print(f"Evaluating Recall@1 / Recall@3 / Recall@{top_k}\n")

    retriever = HybridRetriever(
        bm25_top_k=10,
        vector_top_k=10,
        final_top_k=top_k,
    )

    results: list[dict] = []

    for q in questions:
        result = retriever.retrieve(q["question"])
        chunks = result["chunks"]

        hit_at: Optional[int] = None
        for rank, chunk in enumerate(chunks, start=1):
            if _chunk_hits(chunk, q):
                hit_at = rank
                break

        results.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "hit_at": hit_at,
            "retrieved_docs": [c.document_id for c in chunks],
            "retrieved_sections": [c.section for c in chunks],
            "retrieved_subsections": [c.subsection for c in chunks],
        })

        status = f"✓ @{hit_at}" if hit_at else "✗"
        print(f"  [{q['id']}] {status:6s}  {q['question'][:65]}")

    # ── Compute Recall@K ──────────────────────────────────────────────────
    def recall_at(k: int) -> float:
        hits = sum(1 for r in results if r["hit_at"] is not None and r["hit_at"] <= k)
        return hits / len(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for diff in ["easy", "medium", "difficult"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        hits = sum(1 for r in subset if r["hit_at"] is not None)
        print(f"  {diff.capitalize():10s}  {hits}/{len(subset)}")

    print()
    print(f"  Recall@1  : {recall_at(1):.1%}  ({sum(1 for r in results if r['hit_at'] == 1)}/{len(results)})")
    print(f"  Recall@3  : {recall_at(3):.1%}  ({sum(1 for r in results if r['hit_at'] is not None and r['hit_at'] <= 3)}/{len(results)})")
    print(f"  Recall@{top_k}  : {recall_at(top_k):.1%}  ({sum(1 for r in results if r['hit_at'] is not None)}/{len(results)})")

    target = recall_at(top_k)
    if target >= 0.90:
        verdict = "✓ Phase 2 target met (Recall@K ≥ 90%)"
    elif target >= 0.70:
        verdict = "~ Getting close — keep improving the parser and metadata"
    else:
        verdict = "✗ Below target — review chunking and metadata extraction"
    print(f"\n  {verdict}")

    # ── Save detailed results ─────────────────────────────────────────────
    out_path = Path(questions_path).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "recall_at_1": recall_at(1),
                "recall_at_3": recall_at(3),
                f"recall_at_{top_k}": recall_at(top_k),
                "details": results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Detailed results → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IP-SAKTI retrieval")
    parser.add_argument(
        "--questions",
        default=str(Path(__file__).parent / "questions.json"),
        help="Path to questions.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Recall@K to target (default 5)",
    )
    args = parser.parse_args()
    evaluate(args.questions, args.top_k)
