"""
IP-SAKTI — Phase 6 Ingestion Pipeline
───────────────────────────────────────
Three corpora:

1. LEGAL  (Indian)    → collection "ip_sakti_legal"   + bm25_index.pkl
   jurisdiction = "india"  for every chunk

2. INTERNATIONAL      → same collection "ip_sakti_legal" (domain = international)
   jurisdiction = "international" for every chunk

3. PATENTS            → collection "ip_sakti_patents"  + bm25_patent_index.pkl

Run from the project root:
    python src/ingest.py                       # all three
    python src/ingest.py --only legal
    python src/ingest.py --only international
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

EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE       = 384
QDRANT_PATH       = str(Path(__file__).parent.parent / "qdrant_db")

LEGAL_COLLECTION  = "ip_sakti_legal"
PATENT_COLLECTION = "ip_sakti_patents"

LEGAL_BM25_PATH   = Path(__file__).parent.parent / "bm25_index.pkl"
PATENT_BM25_PATH  = Path(__file__).parent.parent / "bm25_patent_index.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Indian Legal document registry  (jurisdiction = "india")
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
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
        "jurisdiction": "india",
        "authority_level": "secondary",
        "version": "current",
        "effective_date": "2001-01-01",
        "last_verified": "2026-08-25",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# International corpus registry  (jurisdiction = "international")
# Add PDFs to data/raw/international/ and register them here.
# ─────────────────────────────────────────────────────────────────────────────

INTERNATIONAL_REGISTRY: list[dict] = [
    {
        "path": "../data/raw/international/treaties/trips_agreement.pdf",
        "document_id": "trips_agreement",
        "title": "Agreement on Trade-Related Aspects of Intellectual Property Rights (TRIPS)",
        "source": "WTO / WIPO",
        "source_url": "https://www.wto.org/english/tratop_e/trips_e/trips_e.htm",
        "document_type": "treaty",
        "domain": "patent",           # TRIPS covers all IP; "patent" as primary
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "1994",
        "effective_date": "1995-01-01",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/international/pct/pct_overview.pdf",
        "document_id": "pct_overview",
        "title": "Patent Cooperation Treaty (PCT) — Overview",
        "source": "WIPO",
        "source_url": "https://www.wipo.int/pct/en/",
        "document_type": "treaty",
        "domain": "patent",
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "current",
        "effective_date": "1978-01-24",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/international/treaties/paris_convention.pdf",
        "document_id": "paris_convention",
        "title": "Paris Convention for the Protection of Industrial Property",
        "source": "WIPO",
        "source_url": "https://www.wipo.int/treaties/en/ip/paris/",
        "document_type": "treaty",
        "domain": "patent",
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "Stockholm Act 1967",
        "effective_date": "1884-07-07",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/international/treaties/berne_convention.pdf",
        "document_id": "berne_convention",
        "title": "Berne Convention for the Protection of Literary and Artistic Works",
        "source": "WIPO",
        "source_url": "https://www.wipo.int/treaties/en/ip/berne/",
        "document_type": "treaty",
        "domain": "copyright",
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "Paris Act 1971",
        "effective_date": "1886-09-09",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/international/wipo/madrid_system_overview.pdf",
        "document_id": "madrid_system",
        "title": "Madrid System for the International Registration of Marks — Overview",
        "source": "WIPO",
        "source_url": "https://www.wipo.int/madrid/en/",
        "document_type": "guideline",
        "domain": "trademark",
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "current",
        "effective_date": "1891-04-14",
        "last_verified": "2026-08-25",
    },
    {
        "path": "../data/raw/international/wipo/wipo_ip_basics.pdf",
        "document_id": "wipo_ip_basics",
        "title": "WIPO Intellectual Property Handbook",
        "source": "WIPO",
        "source_url": "https://www.wipo.int/edocs/pubdocs/en/intproperty/489/wipo_pub_489.pdf",
        "document_type": "guideline",
        "domain": "general",
        "jurisdiction": "international",
        "authority_level": "international_reference",
        "version": "2nd Edition",
        "effective_date": "2004-01-01",
        "last_verified": "2026-08-25",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Patent document registry
# ─────────────────────────────────────────────────────────────────────────────

PATENT_REGISTRY: list[dict] = [
    # Add patent PDFs here as you collect them from IP India.
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
    meta["chunk_id"]    = chunk.chunk_id
    meta["jurisdiction"]= getattr(chunk, "jurisdiction", "india")
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


def _setup_collection(client: QdrantClient, name: str, recreate: bool = True) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if recreate:
            print(f"  Recreating collection '{name}' …")
            client.delete_collection(name)
        else:
            print(f"  Collection '{name}' exists — skipping creation.")
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def _ingest_registry(
    registry: list[dict],
    label: str,
    embeddings: HuggingFaceEmbeddings,
    vector_store: QdrantVectorStore,
    bm25_path: Path,
    existing_chunks: list[LegalChunk] | None = None,
) -> list[LegalChunk]:
    """Parse all PDFs in registry, upload to Qdrant, return chunks."""
    all_chunks: list[LegalChunk] = list(existing_chunks or [])
    new_chunks: list[LegalChunk] = []

    for entry in registry:
        pdf_path = _resolve(entry["path"])
        if not pdf_path.exists():
            print(f"  [SKIP] Not found: {pdf_path.name}")
            continue
        parser = LegalDocumentParser(doc_meta=entry)
        chunks = parser.parse_pdf(str(pdf_path))
        jur    = entry.get("jurisdiction", "india")
        # Stamp jurisdiction on each chunk
        for c in chunks:
            object.__setattr__(c, "jurisdiction", jur) if hasattr(c, "__fields__") else None
            c.__dict__["jurisdiction"] = jur
        print(f"  [{jur:13s}] {entry['document_id']:40s}  {len(chunks):>4d} chunks")
        new_chunks.extend(chunks)

    if not new_chunks:
        print(f"  ⚠  No documents parsed for {label}.")
        return all_chunks

    print(f"  Uploading {len(new_chunks)} {label} chunks to Qdrant …")
    _upload([_legal_to_doc(c) for c in new_chunks], vector_store)

    all_chunks.extend(new_chunks)
    return all_chunks

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion functions
# ─────────────────────────────────────────────────────────────────────────────

def ingest_legal(
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
    recreate: bool = True,
) -> list[LegalChunk]:
    print("\n── Indian Legal Corpus ──────────────────────────────────────")
    if recreate:
        _setup_collection(client, LEGAL_COLLECTION, recreate=True)

    vector_store = QdrantVectorStore(
        client=client, collection_name=LEGAL_COLLECTION, embedding=embeddings
    )
    chunks = _ingest_registry(LEGAL_REGISTRY, "indian", embeddings, vector_store, LEGAL_BM25_PATH)
    print(f"  ✓  {len(chunks)} Indian legal chunks ready.")
    return chunks


def ingest_international(
    embeddings: HuggingFaceEmbeddings,
    client: QdrantClient,
    existing_chunks: list[LegalChunk] | None = None,
) -> list[LegalChunk]:
    print("\n── International Corpus ─────────────────────────────────────")

    # Collection must already exist (created by ingest_legal or setup)
    existing_names = {c.name for c in client.get_collections().collections}
    if LEGAL_COLLECTION not in existing_names:
        _setup_collection(client, LEGAL_COLLECTION, recreate=False)

    vector_store = QdrantVectorStore(
        client=client, collection_name=LEGAL_COLLECTION, embedding=embeddings
    )
    new_chunks = _ingest_registry(
        INTERNATIONAL_REGISTRY, "international",
        embeddings, vector_store, LEGAL_BM25_PATH,
        existing_chunks=existing_chunks,
    )
    return new_chunks


def ingest_patents(embeddings: HuggingFaceEmbeddings, client: QdrantClient) -> None:
    print("\n── Patent Corpus ────────────────────────────────────────────")

    if not PATENT_REGISTRY:
        print("  ⚠  PATENT_REGISTRY is empty.")
        print("  Add patent PDF entries to PATENT_REGISTRY in ingest.py")
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

    print(f"  Uploading {len(all_chunks)} patent chunks …")
    _upload([_patent_to_doc(c) for c in all_chunks], vector_store)

    bm25 = BM25Store()
    bm25.build(all_chunks)
    bm25.save(PATENT_BM25_PATH)
    print(f"  ✓ Patent ingestion done  ({PATENT_BM25_PATH})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(only: str = "all") -> None:
    print("=" * 62)
    print("IP-SAKTI — Phase 6 Ingestion")
    print("=" * 62)

    print("\n[1/2] Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[2/2] Connecting to Qdrant …")
    client = QdrantClient(path=QDRANT_PATH)

    all_legal_chunks: list[LegalChunk] = []

    if only in ("all", "legal"):
        all_legal_chunks = ingest_legal(embeddings, client, recreate=True)

    if only in ("all", "international"):
        # If "all", append international docs to same collection without recreating
        # If "international" only, we need the collection to exist
        if only == "international":
            existing = {c.name for c in client.get_collections().collections}
            if LEGAL_COLLECTION not in existing:
                _setup_collection(client, LEGAL_COLLECTION, recreate=False)
        all_legal_chunks = ingest_international(embeddings, client, all_legal_chunks)

    # Rebuild BM25 index with all chunks (Indian + international)
    if all_legal_chunks and only in ("all", "legal", "international"):
        print("\n  Rebuilding BM25 index with all legal chunks …")
        bm25 = BM25Store()
        bm25.build(all_legal_chunks)
        bm25.save(LEGAL_BM25_PATH)
        print(f"  BM25 index saved → {LEGAL_BM25_PATH}  ({len(all_legal_chunks)} chunks)")

    if only in ("all", "patents"):
        ingest_patents(embeddings, client)

    print("\n✓ Ingestion complete.")
    print(f"  Qdrant path : {QDRANT_PATH}")
    print(f"  BM25 index  : {LEGAL_BM25_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IP-SAKTI ingestion pipeline")
    parser.add_argument(
        "--only",
        choices=["all", "legal", "international", "patents"],
        default="all",
        help="Which corpus to ingest (default: all)",
    )
    args = parser.parse_args()
    main(args.only)
