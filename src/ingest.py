"""
IP-SAKTI - Final Ingestion Pipeline

Ingests all PDFs from data/raw/ into:
  Qdrant collection  ip_sakti_legal   (Indian corpus + international)
  Qdrant collection  ip_sakti_patents (patent documents)
  BM25 index         bm25_index.pkl
  data/processed/    one JSON file per document

Run from the project root:
    python src/ingest.py
    python src/ingest.py --only legal
    python src/ingest.py --only international
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion import BM25Store, LegalChunk, LegalDocumentParser, PatentChunk, PatentDocumentParser

load_dotenv()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE       = 384
QDRANT_PATH       = str(Path(__file__).parent.parent / "qdrant_db")
LEGAL_COLLECTION  = "ip_sakti_legal"
PATENT_COLLECTION = "ip_sakti_patents"
LEGAL_BM25_PATH   = Path(__file__).parent.parent / "bm25_index.pkl"
PATENT_BM25_PATH  = Path(__file__).parent.parent / "bm25_patent_index.pkl"
PROCESSED_DIR     = Path(__file__).parent.parent / "data" / "processed"
BASE              = Path(__file__).parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Indian legal corpus registry
# ---------------------------------------------------------------------------

LEGAL_REGISTRY: list[dict] = [
    {
        "path"           : BASE / "india/patents/01_Patents_Act_1970.pdf",
        "document_id"    : "patents_act_1970",
        "title"          : "The Patents Act, 1970",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1392",
        "document_type"  : "act",
        "domain"         : "patent",
        "jurisdiction"   : "india",
        "authority_level": "primary",
        "version"        : "as amended up to 2005",
        "effective_date" : "1972-04-20",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/patents/02_Patents_Rules_2003.pdf",
        "document_id"    : "patents_rules_2003",
        "title"          : "The Patents Rules, 2003",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1393",
        "document_type"  : "rule",
        "domain"         : "patent",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "as amended up to 2021",
        "effective_date" : "2003-05-20",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/trademarks/01_Trade_Marks_Act_1999.pdf",
        "document_id"    : "trademarks_act_1999",
        "title"          : "The Trade Marks Act, 1999",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/2078",
        "document_type"  : "act",
        "domain"         : "trademark",
        "jurisdiction"   : "india",
        "authority_level": "primary",
        "version"        : "current",
        "effective_date" : "2003-09-15",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/trademarks/02_Trade_Marks_Rules_2017.pdf",
        "document_id"    : "trademarks_rules_2017",
        "title"          : "The Trade Marks Rules, 2017",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/2079",
        "document_type"  : "rule",
        "domain"         : "trademark",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2017-03-06",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/design/02_Designs_Rules_2001.pdf",
        "document_id"    : "designs_rules_2001",
        "title"          : "The Designs Rules, 2001",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1370",
        "document_type"  : "rule",
        "domain"         : "design",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2001-05-11",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/gi/01_Geographical_Indications_Act_1999.pdf",
        "document_id"    : "gi_act_1999",
        "title"          : "The Geographical Indications of Goods Act, 1999",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1634",
        "document_type"  : "act",
        "domain"         : "gi",
        "jurisdiction"   : "india",
        "authority_level": "primary",
        "version"        : "current",
        "effective_date" : "2003-09-15",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/gi/02_Geographical_Indications_Rules_2002.pdf",
        "document_id"    : "gi_rules_2002",
        "title"          : "The Geographical Indications of Goods Rules, 2002",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1635",
        "document_type"  : "rule",
        "domain"         : "gi",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2003-03-08",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/copyright/01_Copyright_Act_1957.pdf",
        "document_id"    : "copyright_act_1957",
        "title"          : "The Copyright Act, 1957",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1367",
        "document_type"  : "act",
        "domain"         : "copyright",
        "jurisdiction"   : "india",
        "authority_level": "primary",
        "version"        : "as amended up to 2012",
        "effective_date" : "1958-01-21",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/copyright/02_Copyright_Rules_2013.pdf",
        "document_id"    : "copyright_rules_2013",
        "title"          : "The Copyright Rules, 2013",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/1368",
        "document_type"  : "rule",
        "domain"         : "copyright",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2013-03-14",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/biodiversity/01_Biological_Diversity_Act_2002.pdf",
        "document_id"    : "biodiversity_act_2002",
        "title"          : "The Biological Diversity Act, 2002",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/2046",
        "document_type"  : "act",
        "domain"         : "general",
        "jurisdiction"   : "india",
        "authority_level": "primary",
        "version"        : "current",
        "effective_date" : "2004-10-05",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "india/biodiversity/02_Biological_Diversity_Rules_2004.pdf",
        "document_id"    : "biodiversity_rules_2004",
        "title"          : "The Biological Diversity Rules, 2004",
        "source"         : "India Code",
        "source_url"     : "https://www.indiacode.nic.in/handle/123456789/2047",
        "document_type"  : "rule",
        "domain"         : "general",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2004-10-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "ayush/pharmacopoeia/ayurvedic_pharmacopoeia.pdf",
        "document_id"    : "ayurvedic_pharmacopoeia",
        "title"          : "Ayurvedic Pharmacopoeia of India",
        "source"         : "Ministry of AYUSH",
        "source_url"     : "https://pharmacopoeia.ayush.gov.in",
        "document_type"  : "guideline",
        "domain"         : "ayush",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2001-01-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "ayush/pharmacopoeia/Unani Pharmacopoeia of India Part II Vol 3.pdf",
        "document_id"    : "unani_pharmacopoeia_p2v3",
        "title"          : "Unani Pharmacopoeia of India - Part II, Vol 3",
        "source"         : "Ministry of AYUSH",
        "source_url"     : "https://pharmacopoeia.ayush.gov.in",
        "document_type"  : "guideline",
        "domain"         : "ayush",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2007-01-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "tkdl/TKDL_overview.pdf",
        "document_id"    : "tkdl_overview",
        "title"          : "Traditional Knowledge Digital Library - Overview",
        "source"         : "CSIR / Ministry of AYUSH",
        "source_url"     : "https://www.tkdl.res.in",
        "document_type"  : "guideline",
        "domain"         : "ayush",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2001-01-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "tkdl/02_TKDL_public_information.pdf",
        "document_id"    : "tkdl_public_info_2",
        "title"          : "TKDL - Public Information (2)",
        "source"         : "CSIR / Ministry of AYUSH",
        "source_url"     : "https://www.tkdl.res.in",
        "document_type"  : "guideline",
        "domain"         : "ayush",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2001-01-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "tkdl/03_TKDL_public_information.pdf",
        "document_id"    : "tkdl_public_info_3",
        "title"          : "TKDL - Public Information (3)",
        "source"         : "CSIR / Ministry of AYUSH",
        "source_url"     : "https://www.tkdl.res.in",
        "document_type"  : "guideline",
        "domain"         : "ayush",
        "jurisdiction"   : "india",
        "authority_level": "secondary",
        "version"        : "current",
        "effective_date" : "2001-01-01",
        "last_verified"  : "2026-08-25",
    },
]

# ---------------------------------------------------------------------------
# International corpus registry
# ---------------------------------------------------------------------------

INTERNATIONAL_REGISTRY: list[dict] = [
    {
        "path"           : BASE / "international/treaties/TRIPS.pdf",
        "document_id"    : "trips_agreement",
        "title"          : "Agreement on Trade-Related Aspects of Intellectual Property Rights (TRIPS)",
        "source"         : "WTO / WIPO",
        "source_url"     : "https://www.wto.org/english/tratop_e/trips_e/trips_e.htm",
        "document_type"  : "treaty",
        "domain"         : "patent",
        "jurisdiction"   : "international",
        "authority_level": "international_reference",
        "version"        : "1994",
        "effective_date" : "1995-01-01",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "international/pct/PCT_reference.pdf",
        "document_id"    : "pct_overview",
        "title"          : "Patent Cooperation Treaty (PCT) - Reference",
        "source"         : "WIPO",
        "source_url"     : "https://www.wipo.int/pct/en/",
        "document_type"  : "treaty",
        "domain"         : "patent",
        "jurisdiction"   : "international",
        "authority_level": "international_reference",
        "version"        : "current",
        "effective_date" : "1978-01-24",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "international/treaties/Paris_Convention.pdf",
        "document_id"    : "paris_convention",
        "title"          : "Paris Convention for the Protection of Industrial Property",
        "source"         : "WIPO",
        "source_url"     : "https://www.wipo.int/treaties/en/ip/paris/",
        "document_type"  : "treaty",
        "domain"         : "patent",
        "jurisdiction"   : "international",
        "authority_level": "international_reference",
        "version"        : "Stockholm Act 1967",
        "effective_date" : "1884-07-07",
        "last_verified"  : "2026-08-25",
    },
    {
        "path"           : BASE / "international/wipo/WIPO_IP_overview.pdf",
        "document_id"    : "wipo_ip_overview",
        "title"          : "WIPO Intellectual Property Overview",
        "source"         : "WIPO",
        "source_url"     : "https://www.wipo.int",
        "document_type"  : "guideline",
        "domain"         : "general",
        "jurisdiction"   : "international",
        "authority_level": "international_reference",
        "version"        : "current",
        "effective_date" : "2020-01-01",
        "last_verified"  : "2026-08-25",
    },
]

PATENT_REGISTRY: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legal_to_doc(chunk: LegalChunk) -> Document:
    meta = chunk.to_metadata()
    meta["chunk_id"]     = chunk.chunk_id
    meta["jurisdiction"] = getattr(chunk, "jurisdiction", "india")
    return Document(page_content=chunk.text, metadata=meta)


def _patent_to_doc(chunk: PatentChunk) -> Document:
    meta = chunk.to_metadata()
    meta["chunk_id"] = chunk.chunk_id
    return Document(page_content=chunk.text, metadata=meta)


def _upload(docs: list[Document], vs: QdrantVectorStore, batch: int = 100) -> None:
    for i in range(0, len(docs), batch):
        vs.add_documents(docs[i : i + batch])
        print(f"    uploaded {min(i+batch, len(docs))}/{len(docs)}", end="\r")
    print()


def _setup_collection(client: QdrantClient, name: str, recreate: bool = True) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        if recreate:
            client.delete_collection(name)
        else:
            return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def _save_processed(document_id: str, chunks: list[LegalChunk]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{document_id}.json"
    records = [c.model_dump() for c in chunks]
    out_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _ingest_registry(
    registry : list[dict],
    label    : str,
    vs       : QdrantVectorStore,
    existing : list[LegalChunk] | None = None,
) -> list[LegalChunk]:
    chunks: list[LegalChunk] = list(existing or [])
    for entry in registry:
        pdf = Path(entry["path"]) if not isinstance(entry["path"], Path) else entry["path"]
        if not pdf.exists():
            print(f"    [SKIP] {pdf.name}")
            continue
        parser = LegalDocumentParser(doc_meta={k: v for k, v in entry.items() if k != "path"})
        parsed = parser.parse_pdf(str(pdf))
        jur    = entry.get("jurisdiction", "india")
        for c in parsed:
            c.__dict__["jurisdiction"] = jur
        print(f"    [{jur:13s}] {entry['document_id']:40s}  {len(parsed):>4d} chunks")
        _save_processed(entry["document_id"], parsed)
        chunks.extend(parsed)
    if chunks and (len(chunks) > len(existing or [])):
        new = chunks[len(existing or []):]
        print(f"  Uploading {len(new)} new chunks for {label} ...")
        _upload([_legal_to_doc(c) for c in new], vs)
    return chunks


def ingest(only: str = "all") -> None:
    print("=" * 62)
    print("IP-SAKTI - Ingestion Pipeline")
    print("=" * 62)

    print("\n[1/3] Loading embedding model ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[2/3] Connecting to Qdrant ...")
    client = QdrantClient(path=QDRANT_PATH)

    all_legal: list[LegalChunk] = []

    if only in ("all", "legal"):
        print("\n-- Indian Legal Corpus --")
        _setup_collection(client, LEGAL_COLLECTION)
        vs_legal = QdrantVectorStore(
            client=client, collection_name=LEGAL_COLLECTION, embedding=embeddings
        )
        all_legal = _ingest_registry(LEGAL_REGISTRY, "indian", vs_legal)

    if only in ("all", "international"):
        print("\n-- International Corpus --")
        existing_cols = {c.name for c in client.get_collections().collections}
        if LEGAL_COLLECTION not in existing_cols:
            _setup_collection(client, LEGAL_COLLECTION, recreate=False)
        vs_intl = QdrantVectorStore(
            client=client, collection_name=LEGAL_COLLECTION, embedding=embeddings
        )
        all_legal = _ingest_registry(INTERNATIONAL_REGISTRY, "international",
                                     vs_intl, all_legal)

    if all_legal:
        print(f"\n[3/3] Building BM25 index ({len(all_legal)} chunks) ...")
        bm25 = BM25Store()
        bm25.build(all_legal)
        bm25.save(LEGAL_BM25_PATH)
        print(f"  BM25 saved -> {LEGAL_BM25_PATH}")

    if only in ("all", "patents") and PATENT_REGISTRY:
        print("\n-- Patent Corpus --")
        _setup_collection(client, PATENT_COLLECTION)
        vs_pat = QdrantVectorStore(
            client=client, collection_name=PATENT_COLLECTION, embedding=embeddings
        )
        pat_chunks: list[PatentChunk] = []
        for entry in PATENT_REGISTRY:
            pdf = Path(entry["path"])
            if not pdf.exists():
                print(f"    [SKIP] {pdf.name}")
                continue
            parser = PatentDocumentParser(
                doc_meta={k: v for k, v in entry.items() if k != "path"}
            )
            parsed = parser.parse_pdf(str(pdf))
            pat_chunks.extend(parsed)
        if pat_chunks:
            _upload([_patent_to_doc(c) for c in pat_chunks], vs_pat)
            bm25_p = BM25Store()
            bm25_p.build(pat_chunks)
            bm25_p.save(PATENT_BM25_PATH)

    print("\nIngestion complete.")
    stats = {
        "qdrant"      : QDRANT_PATH,
        "bm25"        : str(LEGAL_BM25_PATH),
        "processed"   : str(PROCESSED_DIR),
        "total_chunks": len(all_legal),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IP-SAKTI ingestion pipeline")
    ap.add_argument("--only", choices=["all", "legal", "international", "patents"],
                    default="all")
    args = ap.parse_args()
    ingest(args.only)
