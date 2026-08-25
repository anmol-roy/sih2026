"""
IP-SAKTI — Phase 3 Ingestion Pipeline
───────────────────────────────────────
Ingests two corpora:

1. LEGAL corpus  → Qdrant collection "ip_sakti_legal"  + bm25_index.pkl
   Acts, rules, AYUSH guidelines, TK documents

2. PATENT corpus → Qdrant collection "ip_sakti_patents" + bm25_patent_index.pkl
   Patent documents from IP India

Run from the project root:
    python src/ingest.py
    python src/ingest.py --only legal
    python src/ingest.py --only patents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion import (
    BM25Store,
    LegalChunk,
    LegalDocumentParser,
    PatentChunk,
    PatentDocumentParser,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE      = 384
QDRANT_PATH      = str(Path(__file__).parent.parent / "qdrant_db")

LEGAL_COLLECTION  = "ip_sakti_legal"
PATENT_COLLECTION = "ip_sakti_patents"

LEGAL_BM25_PATH   = Path(__file__).parent.parent / "bm25_index.pkl"
PATENT_BM25_PATH  = Path(__file__).parent.parent / "bm25_patent_index.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Legal document registry
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_REGISTRY: list[dict] = [
    {
        "path": "../data/raw/legal/patents/patents_act.pdf",
        "document_id": "patents_act_1970",
        "title": "The Patents Act, 1970",
        "source": "India Code",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1392",
        "document_type": "act",
        "domain": "patent",
        "authority_level": "primary",
        "version": "as amended up to 2005",
        "effective_date": "1972-04-20",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/legal/trademarks/trademarks_act.pdf",
        "document_id": "trademarks_act_1999",
        "title": "The Trade Marks Act, 1999",
        "source": "India Code",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2078",
        "document_type": "act",
        "domain": "trademark",
        "authority_level": "primary",
        "version": "current",
        "effective_date": "2003-09-15",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/legal/copyright/copyright_act.pdf",
        "document_id": "copyright_act_1957",
        "title": "The Copyright Act, 1957",
        "source": "India Code",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1367",
        "document_type": "act",
        "domain": "copyright",
        "authority_level": "primary",
        "version": "as amended up to 2012",
        "effective_date": "1958-01-21",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/legal/designs/designs_act.pdf",
        "document_id": "designs_act_2000",
        "title": "The Designs Act, 2000",
        "source": "India Code",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1369",
        "document_type": "act",
        "domain": "design",
        "authority_level": "primary",
        "version": "current",
        "effective_date": "2001-05-11",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/legal/geographical_indications/gi_act.pdf",
        "document_id": "gi_act_1999",
        "title": "The Geographical Indications of Goods Act, 1999",
        "source": "India Code",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1634",
        "document_type": "act",
        "domain": "gi",
        "authority_level": "primary",
        "version": "current",
        "effective_date": "2003-09-15",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/ayush/ayurveda/ayurveda_formulary.pdf",
        "document_id": "ayurveda_formulary",
        "title": "Ayurvedic Formulary of India",
        "source": "Ministry of AYUSH",
        "source_url": "https://ayush.gov.in",
        "document_type": "guideline",
        "domain": "ayush",
        "authority_level": "secondary",
        "version": "Part I, 2nd Edition",
        "effective_date": "2003-01-01",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/ayush/general/tkdl_overview.pdf",
        "document_id": "tkdl_overview",
        "title": "Traditional Knowledge Digital Library — Overview",
        "source": "CSIR / Ministry of AYUSH",
        "source_url": "https://www.tkdl.res.in",
        "document_type": "guideline",
        "domain": "ayush",
        "authority_level": "secondary",
        "version": "current",
        "effective_date": "2001-01-01",
        "last_verified": "2026-08-25",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Patent document registry
# Each entry must have at minimum: path, publication_number, title
# ─────────────────────────────────────────────────────────────────────────────

PATENT_REGISTRY: list[dict] = [
    # Add patent PDFs here as you collect them from IP India.
    # Example:
    # {
    #     "path": "../data/raw/public/ipindia/patents/IN202011012345.pdf",
    #     "publication_number": "IN202011012345",
    #     "title": "Herbal formulation for skin disorders",
    #     "applicant": "Example Pharma Ltd.",
    #     "classification": "A61K 36/00",
    #     "source": "IP India",
    #     "source_url": "https://iprsearch.ipindia.gov.in/",
    # },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(raw: str) -> Path:
    return (Path(__file__).parent / raw).resolve()


def _legal_to_doc(chunk: LegalChunk) -> Document:
    meta = chunk.to_metadata()
    meta["chunk_id"] = chunk.chunk_id
    return Document(page_content=chunk.text, metadata=meta)


def _patent_to_doc(chunk: PatentChunk) -> Document:
    meta = chunk.to_metadata()
    meta["chunk_id"] = chunk.chunk_id
    return Document(page_content=chunk.text, metadata=meta)


def _upload(
    docs: list[Document],
    vector_store: QdrantVectorStore,
    batch_size: int = 100,
) -> None:
    for i in range(0, len(docs), batch_size):
        vector_store.add_documents(docs[i : i + batch_size])
        print(f"  Uploaded {min(i + batch_size, len(docs))}/{len(docs)}", end="\r")
    print()


def _setup_collection(
    client: QdrantClient,
    name: str,
    recreate: bool = True,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if recreate:
            print(f"  Recreating collection '{name}' …")
            client.delete_collection(name)
        else:
            print(f"  Collection '{name}' already exists — skipping creation.")
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion functions
# ─────────────────────────────────────────────────────────────────────────────

def ingest_legal(embeddings: HuggingFaceEmbeddings, client: QdrantClient) -> None:
    print("\n── Legal Corpus ─────────────────────────────────────────────")
    _setup_collection(client, LEGAL_COLLECTION)

    vector_store = QdrantVectorStore(
        client=client, collection_name=LEGAL_COLLECTION, embedding=embeddings
    )

    all_chunks: list[LegalChunk] = []
    for entry in LEGAL_REGISTRY:
        pdf_path = _resolve(entry["path"])
        if not pdf_path.exists():
            print(f"  [SKIP] Not found: {pdf_path}")
            continue
        parser = LegalDocumentParser(doc_meta=entry)
        chunks = parser.parse_pdf(str(pdf_path))
        print(f"  {entry['document_id']:45s}  {len(chunks):>4d} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  ⚠  No legal documents parsed.")
        return

    print(f"\n  Total legal chunks : {len(all_chunks)}")
    print("  Uploading to Qdrant …")
    _upload([_legal_to_doc(c) for c in all_chunks], vector_store)

    print("  Building BM25 index …")
    bm25 = BM25Store()
    bm25.build(all_chunks)
    bm25.save(LEGAL_BM25_PATH)
    print(f"  ✓ Legal ingestion done  ({LEGAL_BM25_PATH})")


def ingest_patents(embeddings: HuggingFaceEmbeddings, client: QdrantClient) -> None:
    print("\n── Patent Corpus ────────────────────────────────────────────")

    if not PATENT_REGISTRY:
        print("  ⚠  PATENT_REGISTRY is empty.")
        print("  Add patent PDF entries to PATENT_REGISTRY in ingest.py")
        print("  and re-run: python src/ingest.py --only patents")
        return

    _setup_collection(client, PATENT_COLLECTION)

    vector_store = QdrantVectorStore(
        client=client, collection_name=PATENT_COLLECTION, embedding=embeddings
    )

    all_chunks: list[PatentChunk] = []
    for entry in PATENT_REGISTRY:
        pdf_path = _resolve(entry["path"])
        if not pdf_path.exists():
            print(f"  [SKIP] Not found: {pdf_path}")
            continue
        parser = PatentDocumentParser(doc_meta=entry)
        chunks = parser.parse_pdf(str(pdf_path))
        print(f"  {entry['publication_number']:30s}  {len(chunks):>4d} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  ⚠  No patent documents parsed.")
        return

    print(f"\n  Total patent chunks: {len(all_chunks)}")
    print("  Uploading to Qdrant …")
    _upload([_patent_to_doc(c) for c in all_chunks], vector_store)

    print("  Building BM25 index …")
    bm25 = BM25Store()
    bm25.build(all_chunks)
    bm25.save(PATENT_BM25_PATH)
    print(f"  ✓ Patent ingestion done  ({PATENT_BM25_PATH})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(only: str = "all") -> None:
    print("=" * 60)
    print("IP-SAKTI — Phase 3 Ingestion")
    print("=" * 60)

    print("\n[1/2] Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[2/2] Connecting to Qdrant …")
    client = QdrantClient(path=QDRANT_PATH)

    if only in ("all", "legal"):
        ingest_legal(embeddings, client)

    if only in ("all", "patents"):
        ingest_patents(embeddings, client)

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IP-SAKTI ingestion pipeline")
    parser.add_argument(
        "--only",
        choices=["all", "legal", "patents"],
        default="all",
        help="Which corpus to ingest (default: all)",
    )
    args = parser.parse_args()
    main(args.only)
