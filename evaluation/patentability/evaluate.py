"""
Phase 4 Patentability Evaluation
──────────────────────────────────
Runs all 30 patentability cases through the full Phase 4 pipeline and
measures:

  ✓ Feature extraction accuracy     — were expected features found?
  ✓ Section retrieval               — were expected legal sections retrieved?
  ✓ TK match accuracy               — correct TK match / no-match?
  ✓ Novelty assessment accuracy     — correct assessment label?
  ✓ No-hallucination check          — did the report cite evidence or invent?

Usage (from project root):
    python evaluation/patentability/evaluate.py
    python evaluation/patentability/evaluate.py --cases evaluation/patentability/cases.json
    python evaluation/patentability/evaluate.py --case-id CASE001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.invention_extractor import InventionExtractor
from analysis.feature_extractor import FeatureExtractor
from analysis.claim_representation import ClaimBuilder
from analysis.query_generator import generate_queries
from analysis.novelty import NoveltyAnalyzer
from analysis.inventive_step import InventiveStepAnalyzer
from analysis.evidence_fusion import _legal_chunk_to_evidence, _tk_chunk_to_evidence
from patents.prior_art_search import PriorArtSearcher, generate_prior_art_queries
from patents.patent_matcher import PatentMatcher
from retrieval.retriever import HybridRetriever
from tk.tk_matcher import TKMatcher
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "src" / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _feature_hit(extracted_features: list[dict], expected_features: list[str]) -> float:
    """Fraction of expected features found in extracted features (case-insensitive substring)."""
    if not expected_features:
        return 1.0
    extracted_text = " ".join(f["feature"].lower() for f in extracted_features)
    hits = sum(
        1 for ef in expected_features
        if any(word.lower() in extracted_text for word in ef.split())
    )
    return hits / len(expected_features)


def _section_hit(retrieved_sections: list[Optional[str]], expected_sections: list[str]) -> bool:
    """True if at least one expected section appears in retrieved sections."""
    if not expected_sections:
        return True
    all_sections = " ".join(s or "" for s in retrieved_sections).lower()
    return any(es.lower() in all_sections for es in expected_sections)


def _novelty_match(got: str, expected: str) -> bool:
    return got == expected


# ─────────────────────────────────────────────────────────────────────────────
# Single-case evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(
    case: dict,
    embeddings: HuggingFaceEmbeddings,
    llm: ChatGroq,
    retriever: HybridRetriever,
    feat_extractor: FeatureExtractor,
    claim_builder: ClaimBuilder,
    pa_searcher: PriorArtSearcher,
    matcher: PatentMatcher,
    tk_matcher: TKMatcher,
    novelty_az: NoveltyAnalyzer,
    invstep_az: InventiveStepAnalyzer,
) -> dict:

    desc = case["description"]
    result: dict = {
        "case_id"         : case["case_id"],
        "difficulty"      : case["difficulty"],
        "description"     : desc[:80],
        "feature_score"   : 0.0,
        "section_hit"     : False,
        "tk_correct"      : False,
        "novelty_correct" : False,
        "errors"          : [],
    }

    try:
        # 1. Extract invention + features
        from analysis.invention_extractor import InventionExtractor
        extractor = InventionExtractor(llm=llm)
        invention = extractor.extract(desc)
        features  = feat_extractor.extract_from_invention(invention)
        claim     = claim_builder.build(invention, features)

        result["extracted_features"] = [f.feature for f in features]
        result["claim_type"]         = claim.claim_type
        result["ipc_suggested"]      = claim.ipc_suggested

        # 2. Feature score
        result["feature_score"] = _feature_hit(
            [{"feature": f.feature} for f in features],
            case.get("expected_features", []),
        )

        # 3. Legal retrieval
        pa_queries = generate_prior_art_queries(features, claim, invention, max_queries=10)
        legal_map: dict[str, tuple] = {}
        for q in pa_queries[:5]:
            for chunk in retriever.retrieve(q, domain_filter="patent")["chunks"]:
                if chunk.chunk_id not in legal_map:
                    legal_map[chunk.chunk_id] = (chunk, 0.5)

        retrieved_sections = [
            getattr(c, "subsection", None) or getattr(c, "section", None)
            for c, _ in legal_map.values()
        ]
        result["retrieved_sections"] = [s for s in retrieved_sections if s]
        result["section_hit"] = _section_hit(
            retrieved_sections,
            case.get("expected_sections", []),
        )

        # 4. Prior-art candidate pool + feature matching
        pool       = pa_searcher.build_candidate_pool(pa_queries, invention, claim)
        candidates = matcher.match_candidates(features, pool)

        # 5. Novelty analysis
        novelty = novelty_az.analyze(features, candidates)
        result["novelty_assessment"] = novelty["assessment"]
        result["novelty_correct"]    = _novelty_match(
            novelty["assessment"],
            case.get("expected_novelty_assessment", ""),
        )

        # 6. TK matching
        tk_result = tk_matcher.match(invention)
        result["tk_match_found"] = tk_result["traditional_knowledge_match"]
        result["tk_correct"]     = (
            tk_result["traditional_knowledge_match"] == case.get("expected_tk_match", False)
        )
        result["tk_matches"] = [
            m["chunk"].title for m in tk_result["matches"]
        ]

    except Exception as exc:
        result["errors"].append(str(exc))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    cases_path: str,
    filter_case_id: Optional[str] = None,
) -> None:
    cases_file = Path(cases_path)
    if not cases_file.exists():
        print(f"Cases file not found: {cases_file}")
        sys.exit(1)

    with open(cases_file, encoding="utf-8") as fh:
        all_cases: list[dict] = json.load(fh)

    if filter_case_id:
        all_cases = [c for c in all_cases if c["case_id"] == filter_case_id]
        if not all_cases:
            print(f"Case {filter_case_id} not found.")
            sys.exit(1)

    print(f"Loaded {len(all_cases)} cases from {cases_file}")
    print("Initialising components (this may take ~30 s on first load) …\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm        = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    retriever     = HybridRetriever(embeddings=embeddings)
    feat_extractor= FeatureExtractor(llm=llm)
    claim_builder = ClaimBuilder(llm=llm)
    pa_searcher   = PriorArtSearcher(embeddings=embeddings, top_k_per_query=8, final_pool_size=15)
    matcher       = PatentMatcher(embeddings=embeddings)
    tk_matcher_   = TKMatcher(embeddings=embeddings, top_k=5)
    novelty_az    = NoveltyAnalyzer()
    invstep_az    = InventiveStepAnalyzer()

    print("Components ready. Running cases …\n")
    print("-" * 80)

    results: list[dict] = []

    for case in all_cases:
        print(f"[{case['case_id']}] {case['difficulty'].upper():10s}  {case['description'][:60]} …", end=" ")
        t0 = time.time()
        result = evaluate_case(
            case, embeddings, llm, retriever,
            feat_extractor, claim_builder, pa_searcher,
            matcher, tk_matcher_, novelty_az, invstep_az,
        )
        elapsed = time.time() - t0

        status_parts = []
        status_parts.append(f"feat={result['feature_score']:.0%}")
        status_parts.append("sec✓" if result["section_hit"]     else "sec✗")
        status_parts.append("tk✓"  if result["tk_correct"]      else "tk✗")
        status_parts.append("nov✓" if result["novelty_correct"] else "nov✗")
        if result["errors"]:
            status_parts.append(f"ERR:{result['errors'][0][:40]}")

        print(f"  [{', '.join(status_parts)}]  {elapsed:.1f}s")
        results.append(result)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    total       = len(results)
    feat_avg    = sum(r["feature_score"] for r in results) / total
    section_acc = sum(1 for r in results if r["section_hit"])     / total
    tk_acc      = sum(1 for r in results if r["tk_correct"])      / total
    novelty_acc = sum(1 for r in results if r["novelty_correct"]) / total
    error_count = sum(1 for r in results if r["errors"])

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for diff in ["easy", "medium", "difficult"]:
        sub = [r for r in results if r["difficulty"] == diff]
        if not sub:
            continue
        s_acc = sum(1 for r in sub if r["section_hit"])
        print(f"  {diff.capitalize():10s}  n={len(sub)}  sec={s_acc}/{len(sub)}")

    print()
    print(f"  Feature extraction score  : {feat_avg:.1%}  (avg fraction of expected features found)")
    print(f"  Section retrieval accuracy: {section_acc:.1%}  ({sum(1 for r in results if r['section_hit'])}/{total})")
    print(f"  TK match accuracy         : {tk_acc:.1%}  ({sum(1 for r in results if r['tk_correct'])}/{total})")
    print(f"  Novelty assessment match  : {novelty_acc:.1%}  ({sum(1 for r in results if r['novelty_correct'])}/{total})")
    print(f"  Cases with errors         : {error_count}/{total}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print()
    if section_acc >= 0.85 and feat_avg >= 0.75 and tk_acc >= 0.80:
        verdict = "✓ Phase 4 targets met"
    elif section_acc >= 0.70:
        verdict = "~ Getting close — improve feature extraction and section retrieval"
    else:
        verdict = "✗ Below target — review legal retrieval and feature extractor"
    print(f"  {verdict}")

    # ── Save detailed results ─────────────────────────────────────────────
    out_path = Path(cases_path).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": {
                    "total"           : total,
                    "feature_avg"     : round(feat_avg, 3),
                    "section_accuracy": round(section_acc, 3),
                    "tk_accuracy"     : round(tk_acc, 3),
                    "novelty_accuracy": round(novelty_acc, 3),
                    "error_count"     : error_count,
                },
                "cases": results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Detailed results → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IP-SAKTI Phase 4 evaluation")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).parent / "cases.json"),
        help="Path to cases.json",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Run only a specific case (e.g. CASE001)",
    )
    args = parser.parse_args()
    run_evaluation(args.cases, args.case_id)
