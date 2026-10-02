"""
Unit tests for Jurisdiction Router (Agent 3)
Tests the 10 test cases from 03.txt
"""

import sys
from pathlib import Path

# Add server directory to path
server_dir = str(Path(__file__).parent)
sys.path.insert(0, server_dir)

# Set up the module namespace before loading
import importlib.util
spec = importlib.util.spec_from_file_location("jurisdiction", str(Path(__file__).parent / "src" / "routing" / "jurisdiction.py"))
jurisdiction_module = importlib.util.module_from_spec(spec)

# Add to sys.modules so imports within the module work
sys.modules['jurisdiction'] = jurisdiction_module

# Execute the module
spec.loader.exec_module(jurisdiction_module)

JurisdictionRouter = jurisdiction_module.JurisdictionRouter
Jurisdiction = jurisdiction_module.Jurisdiction

# Test cases from 03.txt
TESTS = [
    {
        "name": "TEST 1 — India",
        "query": "Can I patent my Ayurvedic formulation in India?",
        "expected": Jurisdiction.INDIA,
    },
    {
        "name": "TEST 2 — Indian Law",
        "query": "What does Indian patent law say about traditional knowledge?",
        "expected": Jurisdiction.INDIA,
    },
    {
        "name": "TEST 3 — International",
        "query": "How can I protect my Ayurvedic invention internationally?",
        "expected": Jurisdiction.INTERNATIONAL,
    },
    {
        "name": "TEST 4 — PCT",
        "query": "How does the PCT process work for an invention?",
        "expected": Jurisdiction.INTERNATIONAL,
    },
    {
        "name": "TEST 5 — India + International",
        "query": "What is the difference between filing a patent in India and using the international PCT route?",
        "expected": Jurisdiction.BOTH,
    },
    {
        "name": "TEST 6 — Explicit Both",
        "query": "I want to protect my Ayurvedic invention in India and internationally. What should I consider?",
        "expected": Jurisdiction.BOTH,
    },
    {
        "name": "TEST 7 — Unclear",
        "query": "Can I patent my Ayurvedic formulation?",
        "expected": Jurisdiction.UNKNOWN,
    },
    {
        "name": "TEST 8 — Ayurveda + Foreign Jurisdiction",
        "query": "I developed an Ayurvedic formulation and want to protect it in Europe.",
        "expected": Jurisdiction.INTERNATIONAL,
    },
    {
        "name": "TEST 9 — Hindi",
        "query": "क्या मैं अपनी आयुर्वेदिक फॉर्मूलेशन का भारत में पेटेंट करा सकता हूँ?",
        "expected": Jurisdiction.INDIA,
    },
    {
        "name": "TEST 10 — Kannada",
        "query": "ನನ್ನ ಆಯುರ್ವೇದಿಕ್ ಫಾರ್ಮುಲೇಶನ್‌ಗೆ ಅಂತಾರಾಷ್ಟ್ರೀಯವಾಗಿ ಪೇಟೆಂಟ್ ರಕ್ಷಣೆ ಪಡೆಯುವುದು ಹೇಗೆ?",
        "expected": Jurisdiction.INTERNATIONAL,
    },
]

def main():
    print("=" * 80)
    print("JURISDICTION AGENT UNIT TESTS")
    print("=" * 80)
    print()

    # Initialize router without LLM (keyword-only mode)
    router = JurisdictionRouter(llm=None, use_llm_threshold=1.0)

    passed = 0
    failed = 0

    for test in TESTS:
        name = test["name"]
        query = test["query"]
        expected = test["expected"]

        result = router.classify(query)
        actual = result.jurisdiction

        status = "PASS" if actual == expected else "FAIL"
        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}: {name}")
        print(f"  Input: {query}")
        print(f"  Expected: {expected.value}")
        print(f"  Actual: {actual.value}")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence}")
        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 80)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
