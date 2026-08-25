import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


load_dotenv()


# -----------------------------
# 1. Load PDFs
# -----------------------------

pdf_files = [
    "data/patents_act.pdf",
    "data/drugs_cosmetics.pdf",
    "data/ayush.pdf",
]


documents = []

for pdf_file in pdf_files:

    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = pdf_file

    documents.extend(docs)


print("Pages loaded:", len(documents))


# -----------------------------
# 2. Chunk documents
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# -----------------------------
# 3. Embedding model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Create Qdrant database
# -----------------------------

client = QdrantClient(
    path="./qdrant_db"
)


collection_name = "ip_sakti_documents"


collections = client.get_collections().collections

collection_exists = any(
    collection.name == collection_name
    for collection in collections
)


if not collection_exists:

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


# -----------------------------
# 5. Store vectors
# -----------------------------

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vector_store.add_documents(chunks)


print("Documents stored in Qdrant.")