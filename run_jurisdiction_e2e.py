"""
End-to-end tests for Jurisdiction Agent (Agent 3)
Tests the 10 test cases from 03.txt against the /ask endpoint
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test cases from 03.txt
TESTS = [
    {
        "name": "TEST 1 — India",
        "query": "Can I patent my Ayurvedic formulation in India?",
        "expected_jurisdiction": "india",
    },
    {
        "name": "TEST 2 — Indian Law",
        "query": "What does Indian patent law say about traditional knowledge?",
        "expected_jurisdiction": "india",
    },
    {
        "name": "TEST 3 — International",
        "query": "How can I protect my Ayurvedic invention internationally?",
        "expected_jurisdiction": "international",
    },
    {
        "name": "TEST 4 — PCT",
        "query": "How does the PCT process work for an invention?",
        "expected_jurisdiction": "international",
    },
    {
        "name": "TEST 5 — India + International",
        "query": "What is the difference between filing a patent in India and using the international PCT route?",
        "expected_jurisdiction": "both",
    },
    {
        "name": "TEST 6 — Explicit Both",
        "query": "I want to protect my Ayurvedic invention in India and internationally. What should I consider?",
        "expected_jurisdiction": "both",
    },
    {
        "name": "TEST 7 — Unclear",
        "query": "Can I patent my Ayurvedic formulation?",
        "expected_jurisdiction": "unknown",
    },
    {
        "name": "TEST 8 — Ayurveda + Foreign Jurisdiction",
        "query": "I developed an Ayurvedic formulation and want to protect it in Europe.",
        "expected_jurisdiction": "international",
    },
    {
        "name": "TEST 9 — Hindi",
        "query": "क्या मैं अपनी आयुर्वेदिक फॉर्मूलेशन का भारत में पेटेंट करा सकता हूँ?",
        "expected_jurisdiction": "india",
    },
    {
        "name": "TEST 10 — Kannada",
        "query": "ನನ್ನ ಆಯುರ್ವೇದಿಕ್ ಫಾರ್ಮುಲೇಶನ್‌ಗೆ ಅಂತಾರಾಷ್ಟ್ರೀಯವಾಗಿ ಪೇಟೆಂಟ್ ರಕ್ಷಣೆ ಪಡೆಯುವುದು ಹೇಗೆ?",
        "expected_jurisdiction": "international",
    },
]

def main():
    print("=" * 80)
    print("JURISDICTION AGENT END-TO-END TESTS")
    print("=" * 80)
    print()
    print("NOTE: These tests require the backend server to be running on http://localhost:8000")
    print()

    import requests

    base_url = "http://localhost:8000"
    passed = 0
    failed = 0
    results = []

    for test in TESTS:
        name = test["name"]
        query = test["query"]
        expected = test["expected_jurisdiction"]

        try:
            response = requests.post(
                f"{base_url}/ask",
                json={"query": query},
                timeout=120  # Increased timeout for RAG pipeline
            )
            response.raise_for_status()
            data = response.json()

            actual = data.get("jurisdiction", "unknown")
            jurisdiction_reason = data.get("jurisdiction_reason", "")
            scope_status = data.get("scope_status", "")
            detected_language = data.get("detected_language", "unknown")

            status = "PASS" if actual == expected else "FAIL"
            if actual == expected:
                passed += 1
            else:
                failed += 1

            results.append({
                "name": name,
                "input": query,
                "expected": expected,
                "actual": actual,
                "status": status,
                "jurisdiction_reason": jurisdiction_reason,
                "scope_status": scope_status,
                "detected_language": detected_language,
            })

            print(f"{status}: {name}")
            print(f"  Input: {query}")
            print(f"  Expected jurisdiction: {expected}")
            print(f"  Actual jurisdiction: {actual}")
            print(f"  Reason: {jurisdiction_reason}")
            print(f"  Scope status: {scope_status}")
            print(f"  Detected language: {detected_language}")
            print()

        except requests.exceptions.ConnectionError:
            print(f"ERROR: Could not connect to backend server at {base_url}")
            print("Please start the backend server first:")
            print("  cd d:/AnveshAI/server")
            print("  python -m uvicorn src.api.main:app --reload")
            print()
            return False
        except Exception as e:
            print(f"ERROR: {name}")
            print(f"  Exception: {e}")
            print()
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 80)

    # Save results to JSON
    with open("jurisdiction_e2e_results.json", "w") as f:
        json.dump({
            "total": len(TESTS),
            "passed": passed,
            "failed": failed,
            "tests": results
        }, f, indent=2)

    print(f"Results saved to jurisdiction_e2e_results.json")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
