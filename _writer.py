import pathlib

ROOT = pathlib.Path(r"d:\d\hackathon\sih2026")
src = ROOT / "src" / "ingest.py"

# Read existing file to confirm we can write
old = src.read_text(encoding="utf-8")
print("old lines:", len(old.splitlines()))

# Build new content by assembling parts
parts = []

# ---- header ----
header = [
    "from __future__ import annotations",
    "import argparse, json, sys",
    "from pathlib import Path",
    "from dotenv import load_dotenv",
    "from langchain_core.documents import Document",
    "from langchain_huggingface import HuggingFaceEmbeddings",
    "from langchain_qdrant import QdrantVectorStore",
    "from qdrant_client import QdrantClient",
    "from qdrant_client.models import Distance, VectorParams",
    "",
    "sys.path.insert(0, str(Path(__file__).parent.parent))",
    "from ingestion import BM25Store, LegalChunk, LegalDocumentParser, PatentChunk, PatentDocumentParser",
    "load_dotenv()",
    "",
    "EMBEDDING_MODEL   = \"sentence-transformers/all-MiniLM-L6-v2\"",
    "VECTOR_SIZE       = 384",
    "QDRANT_PATH       = str(Path(__file__).parent.parent / \"qdrant_db\")",
    "LEGAL_COLLECTION  = \"ip_sakti_legal\"",
    "PATENT_COLLECTION = \"ip_sakti_patents\"",
    "LEGAL_BM25_PATH   = Path(__file__).parent.parent / \"bm25_index.pkl\"",
    "PATENT_BM25_PATH  = Path(__file__).parent.parent / \"bm25_patent_index.pkl\"",
    "PROCESSED_DIR     = Path(__file__).parent.parent / \"data\" / \"processed\"",
    "BASE              = Path(__file__).parent.parent / \"data\" / \"raw\"",
]
parts.extend(header)
print("header ok")
src.write_text("\n".join(parts), encoding="utf-8")
print("wrote ok")
