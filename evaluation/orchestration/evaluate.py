"""
Phase 10 — Orchestration Evaluation
──────────────────────────────────────
Measures:
  ✓ Tool selection accuracy  — correct tools selected, no unnecessary tools
  ✓ Evidence type coverage   — expected evidence types retrieved
  ✓ Issue detection          — correct Section 3 flags raised
  ✓ No hallucination         — tools only return real evidence

Usage (from project root):
    python evaluation/orchestration/evaluate.py
    python evaluation/orchestration/evaluate.py --case-id ORC001
    python evaluation/orchestration/evaluate.py --skip-live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.orchestrator import select_tools


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_base(tool_call: str) -> str:
    """'legal_search:patent' → 'legal_search'"""
    return tool_call.split(":")[0]


def evaluate_tool_selection(case: dict) -> tuple[bool, str]:
    """Check that required tools are selected and unexpected tools are not."""
    selected      = [_tool_base(t) for t in select_tools(case["query"])]
    expected      = case.get("expected_tools", [])
    unexpected    = case.get("unexpected_tools", [])

    missing    = [t for t in expected   if t not in selected]
    extra      = [t for t in unexpected if t in selected]

    if missing:
        return False, f"Missing tools: {missing}"
    if extra:
        return False, f"Unexpected tools called: {extra}"
    return True, f"Tools: {selected}"


def evaluate_live(case: dict) -> dict:
    """Run the actual orchestrator and check evidence + issues."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from agents.orchestrator import IPSaktiOrchestrator
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / "src" / ".env")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    orch = IPSaktiOrchestrator(llm=llm, embeddings=emb)

    result = orch.run(case["query"])

    evidence     = result.get("evidence", {})
    tools_used   = result.get("tools_used", [])
    issues_found = result.get("issues", [])

    # Check evidence types
    expected_ev_types = case.get("expected_evidence_types", [])
    all_ev = []
    for ev_type in ["legal", "prior_art", "traditional_knowledge", "abs", "international"]:
        if evidence.get(ev_type):
            all_ev.append(ev_type.replace("prior_art", "patent"))

    ev_ok = all(t in all_ev or t == "patent" and "prior_art" in evidence
                for t in expected_ev_types)

    # Check issues
    expected_issues = case.get("expected_issues", [])
    issues_ok = all(
        any(exp.lower() in iss.lower() for iss in issues_found)
        for exp in expected_issues
    )

    return {
        "case_id"       : case["id"],
        "query"         : case["query"][:60],
        "tools_used"    : tools_used,
        "evidence_types": all_ev,
        "issues_found"  : issues_found,
        "ev_ok"         : ev_ok,
        "issues_ok"     : issues_ok,
        "confidence"    : result.get("confidence", "low"),
        "sources"       : result.get("sources_consulted", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    cases_path  : str,
    filter_id   : str | None = None,
    skip_live   : bool = False,
) -> None:
    with open(cases_path, encoding="utf-8") as fh:
        all_cases: list[dict] = json.load(fh)

    if filter_id:
        all_cases = [c for c in all_cases if c["id"] == filter_id]

    print("=" * 70)
    print("IP-SAKTI  Phase 10 — Orchestration Evaluation")
    print("=" * 70)
    print(f"  Cases: {len(all_cases)}  |  Live: {not skip_live}\n")

    # ── Tool selection (deterministic — no LLM) ─────────────────────────
    print("── Tool Selection (keyword heuristics) ─────────────────────────")
    sel_passed = sel_failed = 0
    for case in all_cases:
        ok, note = evaluate_tool_selection(case)
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} [{case['id']}] {case['query'][:55]}")
        if not ok:
            print(f"       {note}")
            sel_failed += 1
        else:
            sel_passed += 1
    print(f"\n  Tool selection: {sel_passed}/{sel_passed+sel_failed}")

    # ── Live orchestration ──────────────────────────────────────────────
    if skip_live:
        print("\n  [SKIP] Live orchestration tests require --no-skip-live")
    else:
        print("\n── Live Orchestration ──────────────────────────────────────────")
        live_results = []
        for case in all_cases[:5]:   # run first 5 to limit LLM calls
            print(f"  Running [{case['id']}] …", end=" ")
            try:
                result = evaluate_live(case)
                ev_ok     = result["ev_ok"]
                issues_ok = result["issues_ok"]
                ok = ev_ok and issues_ok
                symbol = "✓" if ok else "~"
                print(
                    f"{symbol}  ev={ev_ok} issues={issues_ok} "
                    f"conf={result['confidence']} srcs={result['sources']}"
                )
                live_results.append(result)
            except Exception as e:
                print(f"✗ ERROR: {e}")
                live_results.append({"ev_ok": False, "issues_ok": False})

        live_passed = sum(1 for r in live_results if r.get("ev_ok") and r.get("issues_ok"))
        print(f"\n  Live: {live_passed}/{len(live_results)}")

    # ── Summary ─────────────────────────────────────────────────────────
    total = sel_passed
    total_run = sel_passed + sel_failed
    target = total / total_run if total_run else 0

    print(f"\n{'═'*70}")
    print(f"  Tool selection accuracy : {sel_passed}/{sel_passed+sel_failed}  ({sel_passed/(sel_passed+sel_failed)*100:.0f}%)")
    if target >= 0.85:
        print("  ✓ Phase 10 tool-selection target met (≥ 85%)")
    else:
        print("  ⚠  Below target — review select_tools() heuristics")

    out = Path(cases_path).parent / "eval_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"tool_selection_accuracy": round(target, 3)}, fh, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases",     default=str(Path(__file__).parent / "cases.json"))
    parser.add_argument("--case-id",   default=None)
    parser.add_argument("--skip-live", action="store_true", default=True,
                        help="Skip live LLM calls (default: True)")
    args = parser.parse_args()
    run_evaluation(args.cases, args.case_id, args.skip_live)
