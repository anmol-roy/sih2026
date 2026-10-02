"""
Quick retrieval test — verifies the corpus is working.
Run: python tests/test_retrieval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.retriever import HybridRetriever

print("Loading retriever …")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ret = HybridRetriever(embeddings=emb)

tests = [
    ("What does Section 3(p) of the Patents Act say?",        "patent",    "3(p)"),
    ("What are the requirements for trademark registration?",  "trademark", None),
    ("What does the TRIPS Agreement say about patents?",       None,        None),
    ("Can I patent a traditional Ayurvedic formulation?",      "patent",    None),
]

all_pass = True
print("\n" + "=" * 65)
for query, domain, expected_section in tests:
    result = ret.retrieve(query, domain_filter=domain)
    chunks = result["chunks"]
    conf   = result["confidence"]

    found_section = None
    for c in chunks:
        sub = getattr(c, "subsection", "") or ""
        sec = getattr(c, "section", "") or ""
        if expected_section and expected_section.lower() in (sub + sec).lower():
            found_section = expected_section
            break

    section_ok = (expected_section is None) or (found_section is not None)
    status     = "✓" if (chunks and section_ok) else "✗"
    all_pass   = all_pass and bool(chunks and section_ok)

    print(f"\n{status} Query: {query[:60]}")
    print(f"  Confidence: {conf}  |  Chunks: {len(chunks)}")
    if chunks:
        top = chunks[0]
        print(f"  Top: {getattr(top,'title','?')} | {getattr(top,'section','')} {getattr(top,'subsection','')}")
        print(f"  Source: {getattr(top,'source','?')} | Jurisdiction: {getattr(top,'jurisdiction','?')}")
    if expected_section:
        print(f"  Section check ({expected_section}): {'✓ FOUND' if found_section else '✗ NOT FOUND'}")

print("\n" + "=" * 65)
print(f"Result: {'✓ All tests passed' if all_pass else '✗ Some tests failed'}")
print("=" * 65)
