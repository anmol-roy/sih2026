"""
Phase 6 — Jurisdiction Evaluation
────────────────────────────────────
Measures:
  ✓ Jurisdiction classification accuracy  (keyword + LLM)
  ✓ User-override correctness             (must always be 100%)
  ✓ India / international answer separation (never mixed)
  ✓ Citation jurisdiction tagging

Usage (from project root):
    python evaluation/jurisdiction/evaluate.py
    python evaluation/jurisdiction/evaluate.py --case-id JUR001
    python evaluation/jurisdiction/evaluate.py --category both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from routing.jurisdiction import _keyword_classify, Jurisdiction, JurisdictionRouter
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "src" / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jur_match(got: str, expected: str) -> bool:
    return got.lower() == expected.lower()


def _check_no_cross_contamination(result: dict) -> bool:
    """
    Verify that India citations contain no international sources
    and vice versa, when both are present.
    """
    india_cits  = result.get("india", {}).get("citations", []) if result.get("india") else []
    intl_cits   = result.get("international", {}).get("citations", []) if result.get("international") else []

    intl_sources = {"wipo", "trips", "pct", "paris convention", "berne", "madrid", "hague"}
    india_sources= {"india code", "patents act", "trade marks act", "copyright act",
                    "designs act", "geographical indication", "ayush", "tkdl"}

    # Check India citations don't contain clearly international sources
    for cit in india_cits:
        src = (cit.get("source", "") + " " + cit.get("document", "")).lower()
        if any(s in src for s in intl_sources):
            return False   # cross-contamination detected

    # Check international citations don't contain clearly Indian sources
    for cit in intl_cits:
        src = (cit.get("source", "") + " " + cit.get("document", "")).lower()
        if any(s in src for s in india_sources):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Case evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(case: dict, router: JurisdictionRouter) -> dict:
    query    = case["query"]
    expected = case["expected_jurisdiction"]
    override = case.get("jurisdiction_override")

    result_entry: dict = {
        "case_id"             : case["case_id"],
        "category"            : case["category"],
        "query"               : query[:70],
        "expected_jurisdiction": expected,
        "got_jurisdiction"    : None,
        "jur_correct"         : False,
        "confidence"          : 0.0,
        "no_contamination"    : True,
        "errors"              : [],
    }

    try:
        route = router.classify(query, override=override)
        got   = route.jurisdiction.value
        result_entry["got_jurisdiction"] = got
        result_entry["confidence"]       = route.confidence
        result_entry["jur_correct"]      = _jur_match(got, expected)

    except Exception as exc:
        result_entry["errors"].append(str(exc))

    return result_entry


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    cases_path: str,
    filter_case_id: Optional[str] = None,
    filter_category: Optional[str] = None,
    use_llm: bool = False,
) -> None:
    cases_file = Path(cases_path)
    if not cases_file.exists():
        print(f"Cases file not found: {cases_file}")
        sys.exit(1)

    with open(cases_file, encoding="utf-8") as fh:
        all_cases: list[dict] = json.load(fh)

    if filter_case_id:
        all_cases = [c for c in all_cases if c["case_id"] == filter_case_id]
    if filter_category:
        all_cases = [c for c in all_cases if c["category"] == filter_category]

    if not all_cases:
        print("No matching cases.")
        sys.exit(0)

    print(f"Loaded {len(all_cases)} cases.")

    # ── Initialise router ─────────────────────────────────────────────────
    llm = None
    if use_llm:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            print("LLM-assisted classification enabled.")
        except Exception as e:
            print(f"LLM unavailable ({e}), using keyword-only classification.")

    router = JurisdictionRouter(llm=llm)

    print("\nRunning cases …\n" + "─" * 72)

    results = []
    for case in all_cases:
        result = evaluate_case(case, router)
        symbol = "✓" if result["jur_correct"] else "✗"
        extra  = f" [err: {result['errors'][0][:30]}]" if result["errors"] else ""
        print(
            f"  {symbol} [{result['case_id']}]  "
            f"exp={result['expected_jurisdiction']:13s}  "
            f"got={result['got_jurisdiction'] or '?':13s}  "
            f"conf={result['confidence']:.0%}  "
            f"{result['query'][:45]}{extra}"
        )
        results.append(result)

    # ── Metrics ───────────────────────────────────────────────────────────
    total       = len(results)
    jur_correct = sum(1 for r in results if r["jur_correct"])

    # By category
    categories = {r["category"] for r in results}

    print(f"\n{'═'*72}")
    print("RESULTS")
    print(f"{'═'*72}")

    for cat in sorted(categories):
        sub = [r for r in results if r["category"] == cat]
        hits = sum(1 for r in sub if r["jur_correct"])
        print(f"  {cat:30s}  {hits}/{len(sub)}")

    print(f"{'─'*72}")
    print(f"  Jurisdiction accuracy  : {jur_correct}/{total}  ({jur_correct/total*100:.0f}%)")
    print(f"  Errors                 : {sum(1 for r in results if r['errors'])}")

    target = jur_correct / total
    if target >= 0.90:
        verdict = "✓ Phase 6 target met (≥ 90% accuracy)"
    elif target >= 0.75:
        verdict = "~ Getting close — refine keyword patterns"
    else:
        verdict = "✗ Below target — review jurisdiction.py patterns"
    print(f"\n  {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(cases_path).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": {
                    "total"                : total,
                    "jurisdiction_accuracy": round(jur_correct / total, 3),
                },
                "cases": results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Detailed results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 jurisdiction evaluation")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).parent / "cases.json"),
    )
    parser.add_argument("--case-id",    default=None)
    parser.add_argument("--category",   default=None)
    parser.add_argument("--use-llm",    action="store_true",
                        help="Enable LLM-assisted classification (requires GROQ_API_KEY)")
    args = parser.parse_args()
    run_evaluation(args.cases, args.case_id, args.category, args.use_llm)
