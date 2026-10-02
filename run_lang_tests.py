"""
Language Agent — End-to-End API Test Script
Runs 5 test queries against /ask endpoint and saves structured results.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://127.0.0.1:8001"
OUTPUT_FILE = Path(__file__).parent / "lang_test_results.json"

TESTS = [
    (
        "Test 1 - English",
        "Can I patent my Ayurvedic formulation containing neem and turmeric in India?",
        {"detected": "en", "response_language": "en"},
    ),
    (
        "Test 2 - Hindi",
        "क्या मैं नीम और हल्दी से बनी अपनी आयुर्वेदिक फॉर्मूलेशन का भारत में पेटेंट करा सकता हूँ?",
        {"detected": "hi", "response_language": "hi"},
    ),
    (
        "Test 3 - Kannada",
        "ನನ್ನ ಬೇವು ಮತ್ತು ಅರಿಶಿನದ ಆಯುರ್ವೇದಿಕ್ ಫಾರ್ಮುಲೇಶನ್‌ಗೆ ಭಾರತದಲ್ಲಿ ಪೇಟೆಂಟ್ ಪಡೆಯಬಹುದೇ?",
        {"detected": "kn", "response_language": "kn"},
    ),
    (
        "Test 4 - Hinglish",
        "Mujhe neem aur turmeric se banaye formulation ka India mein patent lena hai.",
        {"detected": "hi"},
    ),
    (
        "Test 5 - English Tech Terms",
        "What does Section 3(p) of the Patents Act mean for traditional knowledge?",
        {"detected": "en", "must_preserve": ["Section 3(p)", "Patents Act", "traditional knowledge"]},
    ),
]


def call_ask(query: str, timeout: int = 300) -> dict:
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_health() -> bool:
    try:
        req = urllib.request.Request(BASE + "/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[HEALTH] FAIL: {e}")
        return False


def run_all():
    print("=" * 65)
    print("Language Agent — End-to-End API Tests")
    print(f"Target: {BASE}")
    print("=" * 65)

    if not check_health():
        print("\nServer not reachable. Aborting tests.")
        return

    results: dict = {}
    passed = failed = 0

    for name, query, expected in TESTS:
        print()
        print("─" * 65)
        print(f" {name}")
        print("─" * 65)
        t0 = time.time()
        status_details: list[str] = []
        test_pass = True

        try:
            data = call_ask(query)
            elapsed = time.time() - t0

            detected = data.get("detected_language")
            resp_lang = data.get("response_language")
            norm_q = data.get("normalized_question") or ""
            answer = data.get("answer") or ""
            citations = data.get("citations") or []
            confidence = data.get("confidence")
            sufficient = data.get("sufficient")

            print(f"  Latency            : {elapsed:.1f}s")
            print(f"  detected_language  : {detected}")
            print(f"  response_language  : {resp_lang}")
            print(f"  confidence         : {confidence}")
            print(f"  sufficient         : {sufficient}")
            print(f"  citations count    : {len(citations)}")
            print(f"  normalized_query   : {norm_q[:160]}{'...' if len(norm_q) > 160 else ''}")
            print(f"  answer (preview)   : {answer[:240]}{'...' if len(answer) > 240 else ''}")

            # --- Assertions ---
            if "detected" in expected and detected != expected["detected"]:
                status_details.append(
                    f"detected_language mismatch: got={detected}, expected={expected['detected']}"
                )
                test_pass = False
            if "response_language" in expected and resp_lang != expected["response_language"]:
                status_details.append(
                    f"response_language mismatch: got={resp_lang}, expected={expected['response_language']}"
                )
                test_pass = False
            if "must_preserve" in expected:
                missing = [t for t in expected["must_preserve"] if t not in answer]
                if missing:
                    status_details.append(f"Missing preserved terms in answer: {missing}")
                    test_pass = False
            if not answer:
                status_details.append("Empty answer")
                test_pass = False

            results[name] = {
                "input": query,
                "detected": detected,
                "normalized": norm_q,
                "response_language": resp_lang,
                "answer": answer,
                "citations_count": len(citations),
                "confidence": confidence,
                "sufficient": sufficient,
                "elapsed_sec": round(elapsed, 1),
                "status": "PASS" if test_pass else "FAIL",
                "issues": status_details,
            }
            if test_pass:
                passed += 1
                print("  Status             : PASS")
            else:
                failed += 1
                print("  Status             : FAIL")
                for issue in status_details:
                    print(f"    - {issue}")

        except urllib.error.HTTPError as e:
            elapsed = time.time() - t0
            body = e.read().decode("utf-8", errors="replace")[:300]
            print(f"  HTTPError {e.code} after {elapsed:.1f}s: {body}")
            results[name] = {
                "input": query,
                "status": f"HTTP {e.code}",
                "error_body": body,
                "elapsed_sec": round(elapsed, 1),
            }
            failed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
            results[name] = {
                "input": query,
                "status": f"ERROR: {type(e).__name__}",
                "error": str(e),
                "elapsed_sec": round(elapsed, 1),
            }
            failed += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    total = passed + failed
    print(f" TOTAL: {total}   PASSED: {passed}   FAILED: {failed}   "
          f"({passed / total * 100:.0f}%)" if total else "No tests ran.")
    print("=" * 65)

    # Save
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved → {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nCould not save results: {e}")


if __name__ == "__main__":
    run_all()
