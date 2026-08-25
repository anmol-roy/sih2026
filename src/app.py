import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient


load_dotenv()


# -----------------------------
# 1. Embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 2. Connect to Qdrant
# -----------------------------

client = QdrantClient(
    path="./qdrant_db"
)


vector_store = QdrantVectorStore(
    client=client,
    collection_name="ip_sakti_documents",
    embedding=embeddings,
)


# -----------------------------
# 3. Create retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# -----------------------------
# 4. Create LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# -----------------------------
# 5. Ask question
# -----------------------------

question = input("\nAsk your question: ")


# Retrieve relevant documents

docs = retriever.invoke(question)


# -----------------------------
# 6. Create context
# -----------------------------

context = "\n\n".join(
    [
        f"""
Source: {doc.metadata.get("source_file")}
Page: {doc.metadata.get("page", "Unknown")}

{doc.page_content}
"""
        for doc in docs
    ]
)


# -----------------------------
# 7. Prompt LLM
# -----------------------------

prompt = f"""
You are an Indian intellectual-property legal information assistant.

Answer the user's question ONLY using the provided documents.

If the answer is not present in the documents, say:
"I could not find this information in the provided documents."

Do not invent legal information.

Always provide sources.

Documents:

{context}


User Question:

{question}


Answer:
"""


response = llm.invoke(prompt)


# -----------------------------
# 8. Print answer
# -----------------------------

print("\n==============================")
print("ANSWER")
print("==============================\n")

print(response.content)


# -----------------------------
# 9. Print citations
# -----------------------------

print("\n==============================")
print("SOURCES")
print("==============================\n")


for i, doc in enumerate(docs, start=1):

    print(f"[{i}] {doc.metadata.get('source_file')}")
    print(f"    Page: {doc.metadata.get('page', 'Unknown')}")
    print()